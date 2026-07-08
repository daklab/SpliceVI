#!/usr/bin/env python
"""
eval_multivisplice_basic.py

Evaluation-only pipeline for SPLICEVI:

1. Load TRAIN and TEST MuData from disk
2. Load a trained SPLICEVI model from disk
3. Run a configurable subset of evaluation blocks:
   - UMAPs
   - Unsupervised clustering + cluster consistency
   - Train split latent quality metrics
   - Test split latent quality metrics
   - Age R² aggregation + CSV
   - Masked-ATSE imputation on multiple masked TEST files

4. Save figures under a user-specified output directory

W&B logging is optional and controlled via CLI flags (typically from a shell script).
"""

import os
import argparse
from typing import Tuple, Optional, List, Dict

import scanpy as sc
import scvi
import mudata as mu
import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score,
    adjusted_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from scipy.stats import spearmanr, ttest_rel

from tqdm.auto import tqdm

from splicevi import SPLICEVI
from splicevi.mean_bayes import MeanBayes

import gc
import re


# ---------------------------------------------------------------------
# W&B helper
# ---------------------------------------------------------------------
def maybe_import_wandb():
    """Import wandb if available; otherwise return (None)."""
    try:
        import wandb
    except ImportError:
        wandb = None
    return wandb


# ---------------------------------------------------------------------
# obs mapping helper
# ---------------------------------------------------------------------
_MAPPING_DF = None  # global cache


def apply_obs_mapping_from_csv(mdata, mapping_csv: str):
    """
    Overwrite selected obs fields on mdata, mdata['rna'], and mdata['splicing']
    using a mapping CSV with one row per cell.

    Expected columns in the mapping CSV (example):
        'cell_id', 'cell_name', 'cell_ontology_class',
        'broad_cell_type', 'medium_cell_type', 'tissue', 'tissue_celltype'

    Matching strategy:
      1. If CSV has 'cell_name' and mdata.obs has 'cell_name', join on that.
      2. Else if CSV has 'cell_id' and mdata.obs has 'cell_id', join on that.
      3. Else, assume mdata.obs.index matches CSV['cell_id'].
    """
    global _MAPPING_DF

    if mapping_csv is None:
        print("[obs-mapping] No mapping CSV provided; skipping mapping.")
        return

    if _MAPPING_DF is None:
        print(f"[obs-mapping] Loading mapping from {mapping_csv}")
        _MAPPING_DF = pd.read_csv(mapping_csv)

    df = _MAPPING_DF.copy()

    # Decide join key
    join_on = None
    if "cell_name" in df.columns and "cell_name" in mdata.obs.columns:
        join_on = "cell_name"
    elif "cell_id" in df.columns and "cell_id" in mdata.obs.columns:
        join_on = "cell_id"
    else:
        # Fallback: assume index == cell_id
        if "cell_id" not in df.columns:
            raise ValueError(
                "[obs-mapping] Could not find a suitable join key. "
                "Need 'cell_name' or 'cell_id' in both mapping CSV and mdata.obs/index."
            )
        df = df.set_index("cell_id")
        mapping_idx = df.index
        common = mdata.obs.index.intersection(mapping_idx)
        if len(common) == 0:
            raise ValueError(
                "[obs-mapping] No overlap between mdata.obs.index and mapping CSV 'cell_id'."
            )

        print(
            f"[obs-mapping] Using mdata.obs.index ↔ CSV['cell_id'] "
            f"(overlap {len(common)}/{mdata.n_obs})"
        )

        df_reindexed = df.reindex(mdata.obs.index)
        for col in df_reindexed.columns:
            vals = df_reindexed[col].values
            mdata.obs[col] = vals
            if "rna" in mdata.mod:
                mdata["rna"].obs[col] = vals
            if "splicing" in mdata.mod:
                mdata["splicing"].obs[col] = vals

        missing = df_reindexed.isna().all(axis=1).sum()
        if missing > 0:
            print(
                f"[obs-mapping] WARNING: {missing} cells had no mapping row in CSV (all NaN)."
            )
        return

    # If we reach here, we have an explicit join_on column
    print(f"[obs-mapping] Joining on '{join_on}'")

    df = df.set_index(join_on)

    if join_on not in mdata.obs.columns:
        raise ValueError(
            f"[obs-mapping] Expected '{join_on}' in mdata.obs but it is missing."
        )

    key_vals = mdata.obs[join_on].astype(str)
    mapping_idx = df.index.astype(str)

    common = pd.Index(key_vals).intersection(mapping_idx)
    if len(common) == 0:
        raise ValueError(
            f"[obs-mapping] No overlap between mdata.obs['{join_on}'] "
            f"and CSV['{join_on}']."
        )

    print(
        f"[obs-mapping] Found {len(common)}/{mdata.n_obs} cells with mapping for '{join_on}'"
    )

    df_reindexed = df.reindex(key_vals)
    for col in df_reindexed.columns:
        vals = df_reindexed[col].values
        mdata.obs[col] = vals
        if "rna" in mdata.mod:
            mdata["rna"].obs[col] = vals
        if "splicing" in mdata.mod:
            mdata["splicing"].obs[col] = vals

    missing = df_reindexed.isna().all(axis=1).sum()
    if missing > 0:
        print(
            f"[obs-mapping] WARNING: {missing} cells had no mapping row in CSV (all NaN)."
        )


# ---------------------------------------------------------------------
# Evaluation helper: train/test split metrics
# ---------------------------------------------------------------------
AGE_R2_RECORDS = []
CROSS_FOLD_RECORDS = []
CROSS_FOLD_SIGNIFICANCE = []
CROSS_FOLD_CLASS_RECORDS = []
MIN_GROUP_N = 25  # minimum cells per tissue | celltype group


def evaluate_split(
    name: str,
    mdata,
    model,
    umap_color_key: str,
    cell_type_classification_key: str,
    Z_type: str = "joint",
    wandb=None,
    precomputed_Z: Optional[np.ndarray] = None,
):
    """
    Latent-quality evaluation for TRAIN / TEST splits:
      - PCA 90% variance
      - silhouette scores (broad & medium)
      - LR classification on medium cell type
      - Age R² overall + per tissue|celltype group
    """
    print(f"\n=== [EVAL] Evaluating {name.upper()} split for latent space '{Z_type}' ===")
    if precomputed_Z is not None:
        Z = precomputed_Z
    else:
        Z = model.get_latent_representation(adata=mdata, modality=Z_type)
    print(f"[EVAL/{name}-{Z_type}] Latent shape: {Z.shape}")

    # PCA 90% variance
    print(f"[EVAL/{name}-{Z_type}] Running PCA to explain 90% variance...")
    n_comp_max = min(Z.shape[0], Z.shape[1])
    pca = PCA(n_components=n_comp_max, svd_solver="full").fit(Z)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    pcs_90 = int(np.searchsorted(cum_var, 0.90) + 1)
    print(f"[EVAL/{name}-{Z_type}] PCs for 90% variance: {pcs_90}/{Z.shape[1]}")

    if wandb is not None:
        wandb.log(
            {
                f"real-{name}-{Z_type}/pca_n_components_90var": pcs_90,
                f"real-{name}-{Z_type}/pca_total_dim": Z.shape[1],
                f"real-{name}-{Z_type}/pca_var90_ratio": pcs_90 / Z.shape[1],
            }
        )

    # Silhouette scores
    print(f"[EVAL/{name}-{Z_type}] Computing silhouette scores...")
    labels_broad = mdata.obs[umap_color_key].astype(str).values
    sil_broad = silhouette_score(Z, labels_broad)
    labels_med = mdata.obs[cell_type_classification_key].astype(str).values
    sil_med = silhouette_score(Z, labels_med)

    print(f"[EVAL/{name}-{Z_type}] Silhouette ({umap_color_key}): {sil_broad:.4f}")
    print(
        f"[EVAL/{name}-{Z_type}] Silhouette ({cell_type_classification_key}): {sil_med:.4f}"
    )

    if wandb is not None:
        wandb.log(
            {
                f"real-{name}-{Z_type}/{umap_color_key}-silhouette_score": sil_broad,
                f"real-{name}-{Z_type}/{cell_type_classification_key}-silhouette_score": sil_med,
            }
        )

    # LR classification on medium cell type
    print(f"[EVAL/{name}-{Z_type}] Training logistic regression classifier...")
    Z_tr, Z_ev, y_tr, y_ev = train_test_split(
        Z, labels_med, test_size=0.2, random_state=0
    )
    clf = LogisticRegression(max_iter=1000).fit(Z_tr, y_tr)
    y_pred = clf.predict(Z_ev)

    acc = accuracy_score(y_ev, y_pred)
    prec = precision_score(y_ev, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_ev, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_ev, y_pred, average="weighted", zero_division=0)

    print(f"[EVAL/{name}-{Z_type}] LR accuracy:  {acc:.4f}")
    print(f"[EVAL/{name}-{Z_type}] LR precision: {prec:.4f}")
    print(f"[EVAL/{name}-{Z_type}] LR recall:    {rec:.4f}")
    print(f"[EVAL/{name}-{Z_type}] LR F1:        {f1:.4f}")

    if wandb is not None:
        wandb.log(
            {
                f"real-{name}-{Z_type}/accuracy": acc,
                f"real-{name}-{Z_type}/precision": prec,
                f"real-{name}-{Z_type}/recall": rec,
                f"real-{name}-{Z_type}/f1_score": f1,
            }
        )

    # Age regression tasks
    if "age_numeric" in mdata.obs:
        print(f"[EVAL/{name}-{Z_type}] Running age R² regression tasks...")
        ages_full = mdata.obs["age_numeric"].astype(float).values
        target_ages = np.array([3.0, 18.0, 24.0], dtype=float)
        mask_age = np.isin(ages_full, target_ages)
        n_kept = int(mask_age.sum())
        print(f"[EVAL/{name}-{Z_type}] Kept {n_kept}/{len(mask_age)} cells at ages {target_ages.tolist()}")

        if n_kept < MIN_GROUP_N:
            print(
                f"[EVAL/{name}-{Z_type}] Only {n_kept} cells with target ages; skipping age R² tasks."
            )
            return

        ages = ages_full[mask_age]
        Z_use = Z[mask_age, :]
        obs_local = mdata.obs.iloc[np.where(mask_age)[0]].copy()

        X_latent = StandardScaler().fit_transform(Z_use)
        X_tr, X_ev, y_tr, y_ev = train_test_split(
            X_latent, ages, test_size=0.2, random_state=0
        )

        # Global R²
        if np.std(y_tr) == 0.0 or np.std(y_ev) == 0.0:
            print(
                f"[EVAL/{name}-{Z_type}] Degenerate age variance after filtering; skipping global age R²."
            )
        else:
            ridge = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5).fit(X_tr, y_tr)
            r2_age = ridge.score(X_ev, y_ev)
            print(f"[EVAL/{name}-{Z_type}] Global age R²: {r2_age:.4f}")
            if wandb is not None:
                wandb.log(
                    {
                        f"real-{name}-{Z_type}/age_r2": r2_age,
                        f"real-{name}-{Z_type}/age_n_cells": n_kept,
                    }
                )

        # Per (tissue | cell_type) R²
        if "tissue" in obs_local:
            ct_key = cell_type_classification_key
            tissue_series = obs_local["tissue"].astype(str)
            ct_series = obs_local[ct_key].astype(str)
            pair = tissue_series + " | " + ct_series
            pair_unique = pair.unique()

            print(
                f"[EVAL/{name}-{Z_type}] Computing per-group age R² for {len(pair_unique)} tissue|cell_type pairs..."
            )

            for p in pair_unique:
                idx = np.where(pair.values == p)[0]
                if idx.size < MIN_GROUP_N:
                    continue

                Zg = X_latent[idx]
                yg = ages[idx]

                if np.std(yg) == 0.0:
                    continue

                Ztr, Zev, ytr, yev = train_test_split(
                    Zg, yg, test_size=0.2, random_state=0
                )
                if (
                    Ztr.shape[0] < 2
                    or Zev.shape[0] < 2
                    or np.std(ytr) == 0.0
                    or np.std(yev) == 0.0
                ):
                    continue

                try:
                    rg = RidgeCV(alphas=np.logspace(-2, 3, 20), cv=5).fit(Ztr, ytr)
                    r2g = rg.score(Zev, yev)
                except Exception:
                    continue

                AGE_R2_RECORDS.append(
                    {
                        "dataset": name,
                        "space": Z_type,
                        "pair": p,
                        "tissue": p.split(" | ", 1)[0],
                        "cell_type": p.split(" | ", 1)[1],
                        "r2": float(r2g),
                        "n": int(idx.size),
                    }
                )
    else:
        print(f"[EVAL/{name}-{Z_type}] No 'age_numeric' column found; skipping age R².")


def run_cross_fold_classification(
    split_name: str,
    mdata,
    latent_spaces: Dict[str, np.ndarray],
    targets: List[str],
    k_folds: int,
    classifiers: List[str],
    metrics: List[str],
    fig_dir: str,
    wandb=None,
    do_dummy: bool = False,
    do_label_permute: bool = False,
):
    """
    K-fold classification for multiple obs targets across latent spaces.

    Evaluates Logistic Regression and/or Random Forest across joint/expression/splicing
    embeddings using shared StratifiedKFold splits, logs mean±std metrics, and records
    paired t-test p-values between spaces.

    Weighted f1:         ``'weighted'``:
            Calculate metrics for each label, and find their average weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
    """
    print(f"\n=== [CROSS-FOLD] Starting {split_name.upper()} cross-fold classification ===")
    spaces_order = ["joint", "expression", "splicing"]
    available_spaces = [s for s in spaces_order if s in latent_spaces]
    if len(available_spaces) == 0:
        print("[CROSS-FOLD] No latent spaces provided; skipping.")
        return

    metric_fns = {}
    for name in metrics:
        if name == "accuracy":
            metric_fns[name] = accuracy_score
        elif name == "f1_weighted":
            metric_fns[name] = lambda yt, yp: f1_score(
                yt, yp, average="weighted", zero_division=0
            )
        elif name == "precision_weighted":
            metric_fns[name] = lambda yt, yp: precision_score(
                yt, yp, average="weighted", zero_division=0
            )
        elif name == "recall_weighted":
            metric_fns[name] = lambda yt, yp: recall_score(
                yt, yp, average="weighted", zero_division=0
            )
        else:
            print(f"[CROSS-FOLD] Unknown metric '{name}' requested; skipping it.")
    if len(metric_fns) == 0:
        print("[CROSS-FOLD] No valid metrics provided; skipping cross-fold.")
        return

    def build_classifier(name: str):  # build either Random Forest or Logistic Regression
        if name == "logreg":
            lr_kwargs = dict(
                max_iter=2000,
                n_jobs=-1,
                class_weight="balanced",
                solver="lbfgs",
            )
            # Some older sklearn builds (or alternative backends) do not accept multi_class.
            try:
                logreg = LogisticRegression(multi_class="auto", **lr_kwargs)
            except TypeError:
                logreg = LogisticRegression(**lr_kwargs)
            return make_pipeline(StandardScaler(), logreg)
        if name == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced_subsample",
            )
        raise ValueError(f"Unsupported classifier '{name}'.")

    for target in targets:
        if target not in mdata.obs.columns:
            print(f"[CROSS-FOLD] Target '{target}' missing in obs; skipping.")
            continue

        labels_series_full = mdata.obs[target].astype("string").fillna("NA")
        total_n_samples = int(labels_series_full.size)
        labels_series = labels_series_full
        keep_indices = np.arange(total_n_samples)

        # Optionally drop singleton mice so StratifiedKFold has support.
        if target == "mouse.id":
            counts = labels_series_full.value_counts()
            singleton_labels = counts[counts == 1].index
            if len(singleton_labels) > 0:
                mask_keep = ~labels_series_full.isin(singleton_labels)
                removed = int((~mask_keep).sum())
                labels_series = labels_series_full[mask_keep]
                keep_indices = np.flatnonzero(mask_keep.to_numpy())
                print(
                    f"[CROSS-FOLD] Target '{target}' | filtering {removed} singleton mice for CV."
                )
            else:
                print(
                    f"[CROSS-FOLD] Target '{target}' | no singleton mice to filter for CV."
                )

        # Drop classes with fewer than k_folds samples (cross-fold only)
        class_counts = labels_series.value_counts()
        drop_labels = class_counts[class_counts < k_folds].index
        if len(drop_labels) > 0:
            mask_keep = ~labels_series.isin(drop_labels)
            labels_series = labels_series[mask_keep]
            keep_indices = keep_indices[mask_keep.to_numpy()]
            print(
                f"[CROSS-FOLD] Target '{target}' | removing {len(drop_labels)} classes with <{k_folds} samples: {list(drop_labels)}"
            )

        label_order = sorted(labels_series.unique())
        y = labels_series.to_numpy()
        n_samples = int(y.size)
        n_classes = int(labels_series.nunique())
        if n_classes < 2:
            print(
                f"[CROSS-FOLD] Target '{target}' has <2 classes ({n_classes}) after filtering; skipping."
            )
            continue

        min_count = int(labels_series.value_counts().min())
        if min_count < k_folds:
            print(
                f"[CROSS-FOLD] Target '{target}' still has a class with {min_count} samples (< k={k_folds}); skipping."
            )
            continue
        k_use = k_folds

        print(
            f"[CROSS-FOLD] Target '{target}' | classes={n_classes}, n={n_samples}, folds={k_use}"
        )

        # build stratified splits with the requested number of folds
        skf = StratifiedKFold(n_splits=k_use, shuffle=True, random_state=42)
        splits = list(skf.split(np.zeros(n_samples), y))  # stratified k-fold on target labels

        # accumulate scores keyed by (classifier, metric, latent_space)
        fold_scores: Dict[Tuple[str, str, str], List[float]] = {}

        for space_name in available_spaces:
            Z_full = latent_spaces[space_name]
            if Z_full.shape[0] != total_n_samples:
                print(
                    f"[CROSS-FOLD] Latent '{space_name}' has {Z_full.shape[0]} rows but expected {total_n_samples}; skipping this space."
                )
                continue
            Z = Z_full[keep_indices]

            for clf_name in classifiers:
                # loop over each fold's train/validation indices
                for fold_idx, (tr_idx, ev_idx) in enumerate(splits):
                    # build a fresh classifier instance per fold
                    clf_fit = build_classifier(clf_name)
                    # fit on the training fold features/labels
                    clf_fit.fit(Z[tr_idx], y[tr_idx])
                    # predict labels for the held-out fold
                    y_pred = clf_fit.predict(Z[ev_idx])
                    # true labels for the held-out fold
                    y_true = y[ev_idx]
                    for metric_name, metric_fn in metric_fns.items():
                        score = float(metric_fn(y_true, y_pred))
                        fold_scores.setdefault(
                            (clf_name, metric_name, space_name), []
                        ).append(score)

                        # Per-class metrics for this fold
                        if metric_name == "f1_weighted":
                            per_class_scores = f1_score(
                                y_true, y_pred, average=None, labels=label_order, zero_division=0
                            )
                        elif metric_name == "precision_weighted":
                            per_class_scores = precision_score(
                                y_true, y_pred, average=None, labels=label_order, zero_division=0
                            )
                        elif metric_name == "recall_weighted":
                            per_class_scores = recall_score(
                                y_true, y_pred, average=None, labels=label_order, zero_division=0
                            )
                        else:  # accuracy
                            per_class_scores = []
                            for lbl in label_order:
                                mask = y_true == lbl
                                if mask.any():
                                    per_class_scores.append(
                                        float((y_pred[mask] == lbl).mean())
                                    )
                                else:
                                    per_class_scores.append(np.nan)

                        for lbl, cls_score in zip(label_order, per_class_scores):
                            CROSS_FOLD_CLASS_RECORDS.append(
                                {
                                    "split": split_name,
                                    "target": target,
                                    "classifier": clf_name,
                                    "space": space_name,
                                    "metric": metric_name,
                                    "fold": int(fold_idx),
                                    "obs_category": lbl,
                                    "value": float(cls_score) if not np.isnan(cls_score) else np.nan,
                                    "n_eval_for_class": int((y_true == lbl).sum()),
                                }
                            )

                    # ── Label-permuted baseline ───────────────────────────────
                    if do_label_permute:
                        print(
                            f"[CROSS-FOLD] {split_name} | {target} | {clf_name}_label_perm | "
                            f"{space_name} | fold {fold_idx}: fitting on permuted train labels...",
                            flush=True,
                        )
                        rng_perm = np.random.default_rng(42 + fold_idx)
                        y_tr_perm = rng_perm.permutation(y[tr_idx])
                        clf_perm = build_classifier(clf_name)
                        clf_perm.fit(Z[tr_idx], y_tr_perm)
                        y_pred_perm = clf_perm.predict(Z[ev_idx])
                        # evaluate against REAL held-out labels — model should fail here
                        for perm_metric_name, perm_metric_fn in metric_fns.items():
                            score_perm = float(perm_metric_fn(y_true, y_pred_perm))
                            fold_scores.setdefault(
                                (f"{clf_name}_label_perm", perm_metric_name, space_name), []
                            ).append(score_perm)

                            # Per-class scores for the permuted run
                            if perm_metric_name == "f1_weighted":
                                pcs_perm = f1_score(
                                    y_true, y_pred_perm, average=None,
                                    labels=label_order, zero_division=0,
                                )
                            elif perm_metric_name == "precision_weighted":
                                pcs_perm = precision_score(
                                    y_true, y_pred_perm, average=None,
                                    labels=label_order, zero_division=0,
                                )
                            elif perm_metric_name == "recall_weighted":
                                pcs_perm = recall_score(
                                    y_true, y_pred_perm, average=None,
                                    labels=label_order, zero_division=0,
                                )
                            else:  # accuracy
                                pcs_perm = []
                                for lbl in label_order:
                                    m_perm = y_true == lbl
                                    if m_perm.any():
                                        pcs_perm.append(float((y_pred_perm[m_perm] == lbl).mean()))
                                    else:
                                        pcs_perm.append(np.nan)

                            for lbl, cls_score_perm in zip(label_order, pcs_perm):
                                CROSS_FOLD_CLASS_RECORDS.append(
                                    {
                                        "split": split_name,
                                        "target": target,
                                        "classifier": f"{clf_name}_label_perm",
                                        "space": space_name,
                                        "metric": perm_metric_name,
                                        "fold": int(fold_idx),
                                        "obs_category": lbl,
                                        "value": (
                                            float(cls_score_perm)
                                            if not np.isnan(cls_score_perm)
                                            else np.nan
                                        ),
                                        "n_eval_for_class": int((y_true == lbl).sum()),
                                    }
                                )

            # ── Dummy-classifier baseline (one pass per space, ignores features) ──
            if do_dummy:
                print(
                    f"\n[CROSS-FOLD] {split_name} | {target} | dummy | {space_name}: "
                    f"running DummyClassifier(strategy='prior') over {k_use} folds...",
                    flush=True,
                )
                for fold_idx, (tr_idx, ev_idx) in enumerate(splits):
                    dummy_clf = DummyClassifier(strategy="prior", random_state=42 + fold_idx)
                    dummy_clf.fit(Z[tr_idx], y[tr_idx])   # Z is ignored internally
                    y_pred_dummy = dummy_clf.predict(Z[ev_idx])
                    y_true_dummy = y[ev_idx]
                    print(
                        f"[CROSS-FOLD] {split_name} | {target} | dummy | "
                        f"{space_name} | fold {fold_idx}: done.",
                        flush=True,
                    )
                    for dummy_metric_name, dummy_metric_fn in metric_fns.items():
                        score_d = float(dummy_metric_fn(y_true_dummy, y_pred_dummy))
                        fold_scores.setdefault(
                            ("dummy", dummy_metric_name, space_name), []
                        ).append(score_d)

                        # Per-class scores for the dummy run
                        if dummy_metric_name == "f1_weighted":
                            pcs_d = f1_score(
                                y_true_dummy, y_pred_dummy, average=None,
                                labels=label_order, zero_division=0,
                            )
                        elif dummy_metric_name == "precision_weighted":
                            pcs_d = precision_score(
                                y_true_dummy, y_pred_dummy, average=None,
                                labels=label_order, zero_division=0,
                            )
                        elif dummy_metric_name == "recall_weighted":
                            pcs_d = recall_score(
                                y_true_dummy, y_pred_dummy, average=None,
                                labels=label_order, zero_division=0,
                            )
                        else:  # accuracy
                            pcs_d = []
                            for lbl in label_order:
                                m_d = y_true_dummy == lbl
                                if m_d.any():
                                    pcs_d.append(float((y_pred_dummy[m_d] == lbl).mean()))
                                else:
                                    pcs_d.append(np.nan)

                        for lbl, cls_score_d in zip(label_order, pcs_d):
                            CROSS_FOLD_CLASS_RECORDS.append(
                                {
                                    "split": split_name,
                                    "target": target,
                                    "classifier": "dummy",
                                    "space": space_name,
                                    "metric": dummy_metric_name,
                                    "fold": int(fold_idx),
                                    "obs_category": lbl,
                                    "value": (
                                        float(cls_score_d)
                                        if not np.isnan(cls_score_d)
                                        else np.nan
                                    ),
                                    "n_eval_for_class": int((y_true_dummy == lbl).sum()),
                                }
                            )

        # Summaries + logging
        for (clf_name, metric_name, space_name), scores in fold_scores.items():
            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            CROSS_FOLD_RECORDS.append(
                {
                    "split": split_name,
                    "target": target,
                    "classifier": clf_name,
                    "space": space_name,
                    "metric": metric_name,
                    "mean": mean_score,
                    "std": std_score,
                    "n_folds": len(scores),
                    "n_samples": n_samples,
                    "n_classes": n_classes,
                }
            )
            print(
                f"[CROSS-FOLD] {split_name} | {target} | {clf_name} | {space_name} | "
                f"{metric_name}: {mean_score:.4f} ± {std_score:.4f} (n={len(scores)})"
            )
            if wandb is not None:
                wandb.log(
                    {
                        f"crossfold/{split_name}/{target}/{clf_name}/{space_name}/{metric_name}_mean": mean_score,
                        f"crossfold/{split_name}/{target}/{clf_name}/{space_name}/{metric_name}_std": std_score,
                    }
                )
                score_id = 0
                for score in scores:
                    wandb.log({f"crossfold/{split_name}/{target}/{clf_name}/{space_name}/{metric_name}_fold{score_id}": score})
                    score_id+=1


            

        # Significance: paired t-tests between spaces for each classifier/metric
        for clf_name in classifiers:
            for metric_name in metric_fns.keys():
                for i in range(len(available_spaces)):
                    for j in range(i + 1, len(available_spaces)):
                        a = available_spaces[i]
                        b = available_spaces[j]
                        key_a = (clf_name, metric_name, a)
                        key_b = (clf_name, metric_name, b)
                        if key_a not in fold_scores or key_b not in fold_scores:
                            continue
                        scores_a = np.array(fold_scores[key_a], dtype=float)
                        scores_b = np.array(fold_scores[key_b], dtype=float)
                        if scores_a.size < 2 or scores_b.size < 2:
                            pval = np.nan
                            mean_diff = np.nan
                        else:
                            stat, pval = ttest_rel(scores_a, scores_b)
                            mean_diff = float(scores_a.mean() - scores_b.mean())

                        CROSS_FOLD_SIGNIFICANCE.append(
                            {
                                "split": split_name,
                                "target": target,
                                "classifier": clf_name,
                                "metric": metric_name,
                                "space_a": a,
                                "space_b": b,
                                "pvalue": float(pval) if pval is not None else np.nan,
                                "mean_diff_a_minus_b": mean_diff,
                                "n_folds": int(min(scores_a.size, scores_b.size)),
                            }
                        )
                        print(
                            f"[CROSS-FOLD] Significance {split_name} | {target} | {clf_name} | {metric_name} : "
                            f"{a} vs {b} p={pval:.4e} (diff={mean_diff:.4f})"
                        )


# ---------------------------------------------------------------------
# Subcluster split evaluation
# ---------------------------------------------------------------------
SUBCLUSTER_RECORDS = []


def run_subcluster_split_eval(
    mdata,
    model,
    obs_key: str,
    cell_type: str,
    k_values: List[int],
    splits: str,
    random_seed: int,
    embedding: str,
    metrics: List[str],
    fig_dir: str,
    wandb=None,
):
    """
    For a given cell type, split cells 50/50, then for each k:
      1. Fit KMeans(k) on train-half joint latent → train labels
      2. Embed train-half (umap/tsne) colored by cluster labels
      3. Predict labels on test-half → test labels
      4. Embed test-half colored by predicted labels
      5. Fit LogReg on (test latent, test labels)
      6. Evaluate LogReg on train latent → metrics vs train labels
      7. Log metrics + figures to W&B; append to SUBCLUSTER_RECORDS
    """
    print(f"\n=== [SUBCLUSTER] Starting subcluster_split_eval for '{cell_type}' ===")

    if obs_key not in mdata.obs.columns:
        print(f"[SUBCLUSTER] obs_key '{obs_key}' not found in mdata.obs; skipping.")
        return

    cell_mask = mdata.obs[obs_key] == cell_type
    n_cells = int(cell_mask.sum())
    print(f"[SUBCLUSTER] Found {n_cells} cells with {obs_key}=={cell_type!r}")
    if n_cells < 20:
        print(f"[SUBCLUSTER] Too few cells ({n_cells}); skipping.")
        return

    all_indices = np.flatnonzero(cell_mask.to_numpy())
    train_idx, test_idx = train_test_split(
        all_indices, test_size=0.5, random_state=random_seed
    )
    print(f"[SUBCLUSTER] Train half: {len(train_idx)} cells | Test half: {len(test_idx)} cells")

    print("[SUBCLUSTER] Computing joint latent representation...")
    Z_all = model.get_latent_representation(adata=mdata, modality="joint")
    Z_train = Z_all[train_idx] #1. Split into two equal  halves
    Z_test  = Z_all[test_idx]

    metric_fns = {}
    for name in metrics:
        if name == "accuracy":
            metric_fns[name] = accuracy_score
        elif name == "f1_weighted":
            metric_fns[name] = lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0)
        elif name == "precision_weighted":
            metric_fns[name] = lambda yt, yp: precision_score(yt, yp, average="weighted", zero_division=0)
        elif name == "recall_weighted":
            metric_fns[name] = lambda yt, yp: recall_score(yt, yp, average="weighted", zero_division=0)

    def _build_logreg():
        lr_kwargs = dict(max_iter=2000, n_jobs=-1, class_weight="balanced", solver="lbfgs")
        try:
            clf = LogisticRegression(multi_class="auto", **lr_kwargs)
        except TypeError:
            clf = LogisticRegression(**lr_kwargs)
        return make_pipeline(StandardScaler(), clf)

    def _embed(Z: np.ndarray, labels, split_name: str, k: int) -> Optional[str]:
        import anndata as ad
        tmp = ad.AnnData(X=Z)
        tmp.obs["cluster"] = [str(l) for l in labels]
        sc.pp.neighbors(tmp, use_rep="X", key_added="nn")
        if embedding == "umap":
            sc.tl.umap(tmp, min_dist=0.1, neighbors_key="nn")
            basis = "umap"
        else:
            sc.tl.tsne(tmp, use_rep="X")
            basis = "tsne"

        safe_ct = cell_type.replace(" ", "_").replace("/", "-")
        fname = f"subcluster_{safe_ct}_k{k}_{split_name}.png"
        out_path = os.path.join(fig_dir, fname)

        n_clusters = len(set(labels))
        palette = sns.color_palette("tab20", n_clusters)
        color_map = {str(i): palette[i % len(palette)] for i in range(n_clusters)}

        fig, ax = plt.subplots(figsize=(5, 5))
        coords = tmp.obsm[f"X_{basis}"]
        for lbl in sorted(color_map):
            mask_l = np.array(tmp.obs["cluster"] == lbl)
            ax.scatter(
                coords[mask_l, 0], coords[mask_l, 1],
                s=4, alpha=0.6, color=color_map[lbl], label=lbl, rasterized=True
            )
        ax.set_title(f"{cell_type} | k={k} | {split_name}", fontsize=9)
        ax.set_xlabel(basis.upper() + " 1")
        ax.set_ylabel(basis.upper() + " 2")
        ax.legend(markerscale=2, fontsize=6, loc="best", ncol=max(1, n_clusters // 10))
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[SUBCLUSTER] Saved {split_name} embedding → {out_path}")
        return out_path

    show_train = splits in {"train", "both"}
    show_test  = splits in {"test", "both"}

    for k in k_values:
        print(f"\n[SUBCLUSTER] k={k}")
        km = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        train_labels = km.fit_predict(Z_train) #2. fit k means on train
        test_labels  = km.predict(Z_test) #3. predict k labels on test

        train_fig_path = _embed(Z_train, train_labels, "train", k) if show_train else None
        test_fig_path  = _embed(Z_test,  test_labels,  "test",  k) if show_test  else None

        clf = _build_logreg()
        clf.fit(Z_test, test_labels) #4. fit log reg on test + k means pred labels
        pred_on_train = clf.predict(Z_train) #5. predict k means labels on train using trained log reg

        row = {
            "cell_type": cell_type,
            "obs_key": obs_key,
            "k": k,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        wandb_log = {}
        for metric_name, fn in metric_fns.items(): #6. evaluate against true k means labels
            score = float(fn(train_labels, pred_on_train))
            row[metric_name] = score
            wandb_log[f"subcluster/{cell_type}/k{k}/{metric_name}"] = score
            print(f"[SUBCLUSTER] k={k} | {metric_name} = {score:.4f}")

        SUBCLUSTER_RECORDS.append(row)

        if wandb is not None:
            if train_fig_path:
                wandb_log[f"subcluster/{cell_type}/k{k}/embed_train"] = wandb.Image(train_fig_path)
            if test_fig_path:
                wandb_log[f"subcluster/{cell_type}/k{k}/embed_test"] = wandb.Image(test_fig_path)
            wandb.log(wandb_log)

    if SUBCLUSTER_RECORDS:
        safe_ct = cell_type.replace(" ", "_").replace("/", "-")
        csv_path = os.path.join(fig_dir, f"subcluster_split_eval_{safe_ct}.csv")
        pd.DataFrame(SUBCLUSTER_RECORDS).to_csv(csv_path, index=False)
        print(f"[SUBCLUSTER] Records saved → {csv_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def build_argparser():
    parser = argparse.ArgumentParser(
        "eval_multivisplice_basic",
        description=(
            "Eval-only script for SPLICEVI: loads a trained model and MuData, "
            "then runs UMAPs, clustering, latent metrics, and masked imputation."
        ),
    )

    # Core paths
    parser.add_argument(
        "--train_mdata_path",
        type=str,
        required=True,
        help="Path to TRAIN MuData (.h5mu) used during training.",
    )
    parser.add_argument(
        "--test_mdata_path",
        type=str,
        required=True,
        help="Path to TEST MuData (.h5mu) for evaluation.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the trained SPLICEVI model.",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        required=True,
        help="Directory to save figures and CSV outputs.",
    )

    parser.add_argument(
        "--masked_test_mdata_paths",
        nargs="+",
        default=[],
        help="Optional: one or more masked TEST MuData .h5mu paths for ATSE imputation.",
    )

    parser.add_argument(
        "--impute_filter_boundary_psi",
        action="store_true",
        default=False,
        help=(
            "If set, exclude junctions with junc_ratio_original == 1.0 from the "
            "resampled imputation eval (in addition to the always-excluded == 0.0). "
            "Only applies when --masked_test_mdata_is_resampled is set."
        ),
    )

    parser.add_argument(
        "--min_atse_count",
        type=int,
        default=-1,
        help=(
            "Minimum cell_by_cluster_matrix_original value required to include a "
            "junction-cell entry in the resampled imputation eval. Entries whose "
            "original ATSE count is below this threshold are excluded. "
            "Set to -1 (default) to disable this filter. "
            "Only applies when --masked_test_mdata_is_resampled is set."
        ),
    )

    parser.add_argument(
        "--impute_top_n_hvj",
        type=int,
        default=-1,
        help=(
            "If set to a positive integer N, restrict all imputation evals "
            "(test_impute and every masked_impute file) to the top-N most highly "
            "variable junctions, ranked by per-junction PSI variance across observed "
            "cells in the unmasked test MuData. The same junction set is used across "
            "all eval conditions so results are directly comparable. "
            "Set to -1 (default) to disable this filter and evaluate all junctions."
        ),
    )

    parser.add_argument(
        "--impute_hvj_include_atse_buddies",
        action="store_true",
        default=False,
        help=(
            "If set, expand the top-N HVJ junction set to include all junctions that "
            "share an event_id with any selected HVJ junction. Requires "
            "--impute_top_n_hvj to be set. Applied to both masked_impute and test_impute."
        ),
    )

    parser.add_argument(
        "--impute_dataset_filter",
        type=str,
        default=None,
        help=(
            "If set, restrict masked_impute and test_impute to cells where "
            "obs['dataset'] equals this value "
            "(e.g. 'tabula_muris_senis' or 'allen_brain_exons'). "
            "Leave unset to evaluate all cells."
        ),
    )

    parser.add_argument(
        "--masked_test_mdata_is_resampled",
        action="store_true",
        default=False,
        help=(
            "If set, treat masked_test_mdata_paths as multinomial-resampled files. "
            "Evaluation uses junc_ratio_original as ground truth and evaluates only "
            "at positions where junc_ratio_original > 0. "
            "If not set (default), uses the legacy junc_ratio_masked_original / "
            "junc_ratio_masked_bin_mask layers."
        ),
    )

    parser.add_argument(
        "--mapping_csv",
        type=str,
        default=None,
        help="Optional: path to tissue/cell-type mapping CSV to overwrite obs fields.",
    )
    parser.add_argument(
        "--batch_key",
        type=str,
        default="None",
        help="Optional obs column used as batch_key in setup_mudata. Use 'None' to disable.",
    )

    # Imputation batch size: -1 means "no batching" (single batch of all cells)
    parser.add_argument(
        "--impute_batch_size",
        type=int,
        default=512,
        help=(
            "Batch size for masked imputation. "
            "If set to -1, process all cells in a single batch (no mini-batching)."
        ),
    )

    # Latent visualization settings
    parser.add_argument(
        "--latent_viz_splits",
        choices=["train", "test", "both"],
        default="train",
        help="Which data split(s) to run latent visualization on (train, test, or both).",
    )
    parser.add_argument(
        "--latent_viz_types",
        nargs="+",
        choices=["umap", "tsne"],
        default=["umap"],
        help="Which embedding type(s) to compute: umap, tsne, or both.",
    )
    parser.add_argument(
        "--umap_top_n_celltypes",
        type=int,
        default=None,
        help=(
            "Highlight up to N most frequent cell types when building UMAP palettes. "
            "If not set, uses all categories."
        ),
    )
    parser.add_argument(
        "--umap_obs_keys",
        nargs="+",
        default=None,
        help=(
            "List of .obs keys to color TRAIN UMAPs by. "
            "If not provided, defaults to ['broad_cell_type', 'medium_cell_type' (if present)]."
        ),
    )
    parser.add_argument(
        "--junction_color_ids",
        nargs="*",
        default=None,
        metavar="JUNCTION_ID",
        help=(
            "Junction IDs (from splicing var_names) to color embeddings by empirical PSI "
            "(junc_ratio layer). Unobserved cells (psi_mask==0) are shown in light gray."
        ),
    )

    # Which eval blocks to run
    parser.add_argument(
        "--mean_bayes_impute",
        action="store_true",
        default=False,
        help=(
            "If set, compute a mean-Bayes baseline alongside the model for imputation evals. "
            "For each junction the baseline predicts the mean junc_ratio across all cells "
            "that observed it (PSI > 0). Applied to both masked_impute and test_impute when "
            "those blocks are enabled."
        ),
    )

    parser.add_argument(
        "--mean_bayes_group_by",
        type=str,
        default="None",
        help=(
            "obs field to use as the group prior for MeanBayes imputation (e.g. 'broad_cell_type'). "
            "Set to 'None' (default) to use a single global prior."
        ),
    )

    parser.add_argument(
        "--cross_fold_dummy_classifier",
        action="store_true",
        default=False,
        help=(
            "If set, run a DummyClassifier(strategy='prior') baseline on the same cross-fold "
            "splits alongside the real classifiers. The dummy ignores all latent features and "
            "predicts purely from label frequencies, giving a random-chance F1 floor."
        ),
    )

    parser.add_argument(
        "--cross_fold_label_permute",
        action="store_true",
        default=False,
        help=(
            "If set, for each (classifier, fold, latent space) also fit the real classifier "
            "on randomly permuted training labels and evaluate on the real held-out labels. "
            "A model learning real signal should perform near chance here."
        ),
    )

    parser.add_argument(
        "--output_per_label_f1_csv",
        action="store_true",
        default=False,
        help=(
            "If set, write one CSV per cross-fold target with per-label F1 scores for every "
            "fold/space/classifier. Columns: label, f1, space, split, classifier, fold. "
            "File names: per_label_f1_<target>.csv in --fig_dir."
        ),
    )

    parser.add_argument(
        "--evals",
        nargs="+",
        default=[
            "latent_visualization",
            "clustering",
            "train_eval",
            "test_eval",
            "age_r2_heatmap",
            "masked_impute",
        ],
        help=(
            "Which eval blocks to run. Choices among: "
            "latent_visualization, clustering, train_eval, test_eval, age_r2_heatmap, "
            "masked_impute, cross_fold_classification, test_impute. "
            "'umap' is accepted as an alias for latent_visualization (legacy). "
            "test_impute runs the unmasked test mdata through the model and compares "
            "imputed PSI against junc_ratio (the perfect / upper-bound baseline). "
            "Uses the same boundary-PSI and min-ATSE-count filters as masked_impute."
        ),
    )

    # Cross-fold classification settings
    parser.add_argument(
        "--cross_fold_targets",
        nargs="+",
        default=["broad_cell_type", "batch"],
        help=(
            "List of .obs fields to classify in cross-fold evaluation (e.g., batch, broad_cell_type). "
            "Missing fields are skipped."
        ),
    )
    parser.add_argument(
        "--cross_fold_splits",
        choices=["train", "test", "both"],
        default="train",
        help="Run cross-fold classification on TRAIN, TEST, or both splits.",
    )
    parser.add_argument(
        "--cross_fold_k",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits for cross-fold classification.",
    )
    parser.add_argument(
        "--cross_fold_classifiers",
        nargs="+",
        choices=["logreg", "rf"],
        default=["logreg", "rf"],
        help="Classifiers to use for cross-fold evaluation (logreg=Logistic Regression, rf=Random Forest).",
    )
    parser.add_argument(
        "--cross_fold_metrics",
        nargs="+",
        choices=["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"],
        default=["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"],
        help="Metrics to report for cross-fold evaluation.",
    )

    # Subcluster split evaluation
    parser.add_argument(
        "--subcluster_obs_key",
        type=str,
        default="broad_cell_type",
        help="obs column used to select the cell type for subcluster_split_eval.",
    )
    parser.add_argument(
        "--subcluster_cell_type",
        nargs="*",
        default=None,
        help="One or more cell type values (in --subcluster_obs_key) to run subcluster_split_eval on. "
             "If omitted or empty, all cell types in the obs key are evaluated.",
    )
    parser.add_argument(
        "--subcluster_k_values",
        nargs="+",
        type=int,
        default=[2, 4, 8, 16],
        help="List of k values for KMeans in subcluster_split_eval.",
    )
    parser.add_argument(
        "--subcluster_splits",
        choices=["train", "test", "both"],
        default="both",
        help="Which 50/50 halves to generate embeddings for in subcluster_split_eval.",
    )
    parser.add_argument(
        "--subcluster_random_seed",
        type=int,
        default=42,
        help="Random seed for 50/50 split and KMeans in subcluster_split_eval.",
    )
    parser.add_argument(
        "--subcluster_embedding",
        choices=["umap", "tsne"],
        default="umap",
        help="Dimensionality reduction for subcluster_split_eval figures.",
    )

    # Optional W&B integration
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging for evaluation.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (required if --use_wandb).",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Optional W&B entity (team) name.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional W&B run name.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="Optional W&B group name.",
    )
    parser.add_argument(
        "--wandb_log_freq",
        type=int,
        default=1000,
        help="Logging frequency for wandb.watch (in training steps).",
    )

    return parser


# ---------------------------------------------------------------------
# Imputation diagnostic: 2-D density scatter (predicted vs observed PSI)
# ---------------------------------------------------------------------
def plot_psi_density_scatter(
    orig: np.ndarray,
    pred: np.ndarray,
    pearson_r: float,
    out_path: str,
    tag: str = "",
    l1_mean: float | None = None,
    y_label: str = "Predicted PSI",
    max_pts: int = 200_000,
    wandb=None,
    run=None,
):
    """
    Square 2-D density scatter: observed PSI (x) vs predicted PSI (y).

    Points are coloured by local density estimated with Gaussian KDE on a
    random subsample (max_pts) so the call stays fast even for millions of
    entries. A y=x diagonal reference line and a Pearson-r annotation are
    included.
    """
    from scipy.stats import gaussian_kde
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    rng = np.random.default_rng(42)
    n = len(orig)
    if n > max_pts:
        idx = rng.choice(n, size=max_pts, replace=False)
        x_plot = orig[idx].astype(np.float64)
        y_plot = pred[idx].astype(np.float64)
    else:
        x_plot = orig.astype(np.float64)
        y_plot = pred.astype(np.float64)

    # Density estimation on the (possibly subsampled) points
    xy = np.vstack([x_plot, y_plot])
    try:
        kde = gaussian_kde(xy, bw_method="scott")
        density = kde(xy)
    except np.linalg.LinAlgError:
        density = np.ones(len(x_plot))

    # Sort so densest points render on top
    order = np.argsort(density)
    x_plot, y_plot, density = x_plot[order], y_plot[order], density[order]

    fig, ax = plt.subplots(figsize=(4, 4))

    norm = Normalize(vmin=density.min(), vmax=density.max())
    sc_plot = ax.scatter(
        x_plot, y_plot,
        c=density,
        cmap="viridis",
        norm=norm,
        s=1,
        alpha=0.4,
        linewidths=0,
        rasterized=True,
    )

    # y = x diagonal reference line
    lims = [0.0, 1.0]
    ax.plot(lims, lims, color="crimson", linewidth=1.0, linestyle="--", zorder=3)

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed PSI", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)

    title = f"PSI Imputation{' — ' + tag if tag else ''}"
    ax.set_title(title, fontsize=11)

    n_label = f"n={n:,}" if n <= max_pts else f"n={max_pts:,} (subsampled from {n:,})"
    annotation = f"Pearson r = {pearson_r:.3f}"
    if l1_mean is not None:
        annotation += f"\nL1 mean = {l1_mean:.4f}"
    annotation += f"\n{n_label}"
    ax.text(
        0.04, 0.93,
        annotation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=0.75)
    cbar.set_label("Density", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[EVAL/IMPUTE] Saved density scatter to {out_path}")
    if run is not None and wandb is not None:
        wandb.log({f"impute/{tag}/psi_density_scatter": wandb.Image(out_path)})
    plt.close(fig)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _compute_imputation_metrics(orig_arr: np.ndarray, pred_arr: np.ndarray) -> dict:
    pearson_r = _safe_pearson(orig_arr, pred_arr)
    spearman_r = float(spearmanr(orig_arr, pred_arr, nan_policy="omit")[0])
    abs_diff = np.abs(orig_arr - pred_arr)
    l1_mean = float(np.mean(abs_diff))
    l1_median = float(np.median(abs_diff))
    l1_p90 = float(np.quantile(abs_diff, 0.90))
    pred_min = float(np.min(pred_arr))
    pred_max = float(np.max(pred_arr))
    smape = float(np.mean(2.0 * abs_diff / (np.abs(orig_arr) + np.abs(pred_arr) + 1e-8)))
    orig64 = orig_arr.astype(np.float64, copy=False)
    pred64 = pred_arr.astype(np.float64)
    denom = float(np.linalg.norm(orig64) * np.linalg.norm(pred64) + 1e-8)
    cosine_sim = float(np.dot(orig64, pred64) / denom)
    minmax_ratio = float(np.mean(
        np.minimum(np.abs(orig_arr), np.abs(pred_arr))
        / (np.maximum(np.abs(orig_arr), np.abs(pred_arr)) + 1e-8)
    ))
    rmse = float(np.sqrt(np.mean((orig_arr - pred_arr) ** 2)))
    return dict(
        pearson=pearson_r, spearman=spearman_r,
        l1_mean=l1_mean, l1_median=l1_median, l1_p90=l1_p90,
        pred_min=pred_min, pred_max=pred_max,
        smape=smape, cosine_sim=cosine_sim,
        minmax_ratio=minmax_ratio, rmse=rmse,
    )


def main():
    parser = build_argparser()
    args = parser.parse_args()
    # Normalize legacy "umap" alias → "latent_visualization"
    EVALS = {"latent_visualization" if e == "umap" else e for e in args.evals}
    cross_fold_targets = list(dict.fromkeys(args.cross_fold_targets))
    cross_fold_splits = args.cross_fold_splits
    cross_fold_classifiers = args.cross_fold_classifiers
    run_crossfold_train = cross_fold_splits in {"train", "both"}
    run_crossfold_test = cross_fold_splits in {"test", "both"}
    latent_viz_splits = args.latent_viz_splits
    run_viz_train = latent_viz_splits in {"train", "both"}
    run_viz_test = latent_viz_splits in {"test", "both"}
    latent_viz_types = list(dict.fromkeys(args.latent_viz_types))  # preserves order, dedupes
    batch_key = None if (args.batch_key is None or str(args.batch_key).lower() == "none") else args.batch_key
    mean_bayes_group_by = None if (args.mean_bayes_group_by is None or str(args.mean_bayes_group_by).lower() == "none") else args.mean_bayes_group_by

    os.makedirs(args.fig_dir, exist_ok=True)

    # W&B
    wandb = maybe_import_wandb()
    run = None

    # Basic keys used in several places
    umap_color_key_default = "broad_cell_type"
    cell_type_classification_key = (
        "medium_cell_type"
        if "medium_cell_type" in []  # placeholder, fixed after loading TRAIN
        else "broad_cell_type"
    )

    full_config = {
        "train_mdata_path": args.train_mdata_path,
        "test_mdata_path": args.test_mdata_path,
        "model_dir": args.model_dir,
        "fig_dir": args.fig_dir,
        "masked_test_mdata_paths": args.masked_test_mdata_paths,
        "mapping_csv": args.mapping_csv,
        "batch_key": batch_key,
        "impute_batch_size": args.impute_batch_size,
        "evals": list(EVALS),
        "latent_viz_splits": latent_viz_splits,
        "latent_viz_types": latent_viz_types,
        "umap_top_n_celltypes": args.umap_top_n_celltypes,
        "umap_obs_keys": args.umap_obs_keys,
        "cross_fold_targets": cross_fold_targets,
        "cross_fold_splits": cross_fold_splits,
        "cross_fold_k": args.cross_fold_k,
        "cross_fold_classifiers": cross_fold_classifiers,
    }

    if args.use_wandb:
        if wandb is None:
            raise ImportError(
                "[W&B] --use_wandb was set but wandb is not installed in this environment."
            )
        if args.wandb_project is None:
            raise ValueError("[W&B] --wandb_project is required when --use_wandb is set.")

        run_name = args.wandb_run_name
        if run_name is None:
            # default: base model dir name with EVAL prefix
            run_name = f"EVAL_{os.path.basename(os.path.normpath(args.model_dir))}"

        print("[W&B] Initializing Weights & Biases eval run...")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=args.wandb_group,
            config=full_config,
        )
    else:
        print("[W&B] W&B logging disabled for evaluation.")

    # -----------------------------------------------------------------
    # Load TRAIN MuData and model
    # -----------------------------------------------------------------
    print("=" * 80)
    print("[SETUP] EVAL-ONLY SPLICEVI")
    print(f"[SETUP] TRAIN MuData path : {args.train_mdata_path}")
    print(f"[SETUP] TEST  MuData path : {args.test_mdata_path}")
    print(f"[SETUP] Model directory   : {args.model_dir}")
    print(f"[SETUP] Figures directory : {args.fig_dir}")
    print(f"[SETUP] EVAL blocks       : {sorted(EVALS)}")
    print(f"[SETUP] batch_key         : {batch_key}")
    print("=" * 80)

    print(f"[DATA] Loading TRAIN MuData from {args.train_mdata_path} ...")
    mdata_train = mu.read_h5mu(args.train_mdata_path, backed="r")
    mdata_train.obs.rename(columns={"donor_id": "mouse.id"}, inplace=True)
    mdata_train.mod["rna"].obs.rename(
    columns={"donor_id": "mouse.id"},
    inplace=True)
    mdata_train.mod["splicing"].obs.rename(
    columns={"donor_id": "mouse.id"},
    inplace=True)


    print(f"[DATA] TRAIN MuData loaded with mods: {list(mdata_train.mod.keys())}")
    print(f"[DATA] TRAIN 'rna' n_obs: {mdata_train['rna'].n_obs}, n_vars: {mdata_train['rna'].n_vars}")

    if args.mapping_csv is not None:
        apply_obs_mapping_from_csv(mdata_train, args.mapping_csv)

    # Fixed layer names
    x_layer = "junc_ratio"
    junction_counts_layer = "cell_by_junction_matrix"
    cluster_counts_layer = "cell_by_cluster_matrix"
    mask_layer = "psi_mask"

    print("[DATA] TRAIN 'splicing' layers available:")
    print(f"       {list(mdata_train['splicing'].layers.keys())}")

    # Library size
    if "X_library_size" in mdata_train["rna"].obsm_keys():
        print("[DATA] Copying TRAIN RNA 'X_library_size' from .obsm to .obs...")
        mdata_train["rna"].obs["X_library_size"] = mdata_train["rna"].obsm["X_library_size"]

    print("[MODEL] Setting up SPLICEVI on TRAIN MuData ...")
    SPLICEVI.setup_mudata(
        mdata_train,
        batch_key=batch_key,
        size_factor_key="X_library_size",
        rna_layer="length_norm",
        junc_ratio_layer=x_layer,
        atse_counts_layer=cluster_counts_layer,
        junc_counts_layer=junction_counts_layer,
        psi_mask_layer=mask_layer,
        modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
    )

    print(f"[MODEL] Loading SPLICEVI model from {args.model_dir} ...")
    model = SPLICEVI.load(args.model_dir, adata=mdata_train)
    print("[MODEL] Model loaded. Showing anndata setup:")
    model.view_anndata_setup()

    if run is not None:
        wandb.watch(
            model.module,
            log="all",
            log_freq=args.wandb_log_freq,
            log_graph=False,
        )
        total_params = sum(p.numel() for p in model.module.parameters())
        print(f"[MODEL] Total model parameters: {total_params:,}")
        wandb.log({"total_parameters": total_params})

    # Decide classification and UMAP default keys now that TRAIN obs is available
    umap_color_key = "broad_cell_type" if "broad_cell_type" in mdata_train.obs.columns else "tissue"
    cell_type_classification_key = (
        "medium_cell_type"
        if "medium_cell_type" in mdata_train.obs.columns
        else umap_color_key
    )

    # Highlight top cell types (or all) for UMAP coloring
    highlight_key = (
        "medium_cell_type"
        if "medium_cell_type" in mdata_train["rna"].obs.columns
        else cell_type_classification_key
    )
    highlight_series = mdata_train["rna"].obs[highlight_key].astype(str)
    counts = highlight_series.value_counts()
    top_n = args.umap_top_n_celltypes
    top_groups = (
        counts.head(top_n).index.tolist()
        if top_n is not None
        else counts.index.tolist()
    )
    mdata_train["rna"].obs["group_highlighted"] = "Other"
    mdata_train["rna"].obs.loc[
        mdata_train["rna"].obs[highlight_key].isin(top_groups), "group_highlighted"
    ] = mdata_train["rna"].obs[highlight_key]

    cmap = cm.get_cmap("tab20", max(len(top_groups), 1))
    colors = [cmap(i) for i in range(len(top_groups))]
    color_dict = {group: colors[i] for i, group in enumerate(top_groups)}
    color_dict["Other"] = (0.9, 0.9, 0.9, 1.0)

    # UMAP obs keys list (always include highlighted groups first)
    if args.umap_obs_keys is not None:
        umap_obs_keys = list(dict.fromkeys(args.umap_obs_keys))
        if "group_highlighted" not in umap_obs_keys:
            umap_obs_keys.insert(0, "group_highlighted")
        print(f"[UMAP] Using user-provided UMAP obs keys: {umap_obs_keys}")
    else:
        umap_obs_keys = ["group_highlighted"]
        if cell_type_classification_key != umap_color_key:
            umap_obs_keys.extend([umap_color_key, cell_type_classification_key])
        else:
            umap_obs_keys.append(umap_color_key)
        print(f"[UMAP] UMAP obs keys not provided; using defaults: {umap_obs_keys}")

    # Latent spaces (TRAIN)
    # Required by: latent_visualization (train), clustering, train_eval, cross_fold_classification (train split)
    _need_train_latent = bool(
        EVALS & {"clustering", "train_eval"}
        or ("latent_visualization" in EVALS and run_viz_train)
        or ("cross_fold_classification" in EVALS and run_crossfold_train)
    )
    latent_spaces_train = {}
    if _need_train_latent:
        print("[MODEL] Computing latent representations on TRAIN...")
        latent_spaces_train = {
            "joint": model.get_latent_representation(),
            "expression": model.get_latent_representation(modality="expression"),
            "splicing": model.get_latent_representation(modality="splicing"),
        }
        for name, Z in latent_spaces_train.items():
            print(f"[MODEL] TRAIN latent '{name}' shape: {Z.shape}")
    else:
        print("[MODEL] Skipping TRAIN latent computation (not requested).")

    # -----------------------------------------------------------------
    # Latent visualization (UMAP / t-SNE, TRAIN and/or TEST)
    # -----------------------------------------------------------------
    def _build_palette(ad, obs_key, color_dict):
        """Return a palette dict for sc.pl.embedding, or None to let scanpy choose."""
        if obs_key == "group_highlighted":
            return color_dict
        obs_series = ad.obs[obs_key]
        obs_as_str = obs_series.astype(str)
        n_categories = obs_as_str.nunique()
        normalized_key = obs_key.lower().replace(".", "_")
        needs_large_palette = normalized_key in {"mouse_id", "mouseid"} or n_categories > 100
        if not needs_large_palette:
            return None
        # Preserve categorical ordering if present; otherwise sort for determinism
        if pd.api.types.is_categorical_dtype(obs_series):
            categories = list(obs_series.cat.categories.astype(str))
        else:
            categories = sorted(pd.Index(obs_as_str).unique())
        cmap = cm.get_cmap("hsv", max(len(categories), 1))
        base_colors = cmap(np.linspace(0, 1, len(categories), endpoint=False))
        rng_pal = np.random.default_rng(42)
        permuted = base_colors[rng_pal.permutation(len(categories))]
        return {cat: permuted[i] for i, cat in enumerate(categories)}

    junction_color_ids = args.junction_color_ids or []

    def _plot_junction_psi(
        mdata_split,
        junction_id: str,
        embed_coords: np.ndarray,
        embed_type: str,
        lat_name: str,
        split_label: str,
    ) -> None:
        """Plot embedding colored by empirical PSI for a single junction.

        Cells where psi_mask == 0 are drawn in light gray; observed cells get
        a viridis colorbar scaled 0–1.
        """
        ad_spl = mdata_split["splicing"]
        if "junction_id" not in ad_spl.var.columns:
            print(f"[EVAL/VIZ/PSI] WARNING: 'junction_id' column not found in splicing .var; skipping.")
            return
        junc_mask = ad_spl.var["junction_id"] == junction_id
        if not junc_mask.any():
            print(f"[EVAL/VIZ/PSI] WARNING: junction '{junction_id}' not found in splicing var['junction_id']; skipping.")
            return

        junc_idx = int(np.flatnonzero(junc_mask.values)[0])

        jr = ad_spl.layers["junc_ratio"]
        pm = ad_spl.layers["psi_mask"]

        psi_vals = np.asarray(jr[:, junc_idx].todense()).ravel() if sparse.issparse(jr) else np.asarray(jr[:, junc_idx]).ravel()
        mask_vals = np.asarray(pm[:, junc_idx].todense()).ravel() if sparse.issparse(pm) else np.asarray(pm[:, junc_idx]).ravel()

        observed = mask_vals != 0

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_box_aspect(1)
        ax.set_aspect(1)

        ax.scatter(
            embed_coords[~observed, 0], embed_coords[~observed, 1],
            c="#D3D3D3", s=5, linewidths=0, rasterized=True, label="unobserved",
        )
        sc_obs = ax.scatter(
            embed_coords[observed, 0], embed_coords[observed, 1],
            c=psi_vals[observed], cmap="viridis", vmin=0.0, vmax=1.0,
            s=5, linewidths=0, rasterized=True,
        )
        plt.colorbar(sc_obs, ax=ax, label="Empirical PSI", fraction=0.046, pad=0.04)

        axis1, axis2 = (("UMAP1", "UMAP2") if embed_type == "umap" else ("t-SNE 1", "t-SNE 2"))
        ax.set_xlabel(axis1)
        ax.set_ylabel(axis2)
        safe_jid = re.sub(r"[^A-Za-z0-9]+", "_", junction_id)
        embed_label = embed_type.upper()
        ax.set_title(f"SpliceVI $Z_{{{lat_name.capitalize()}}}$ {embed_label} ({split_label})\nPSI: {junction_id}")
        plt.tight_layout()

        out_path = os.path.join(
            args.fig_dir,
            f"{split_label}_{embed_type}_{lat_name}_psi_{safe_jid}.png",
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[EVAL/VIZ/PSI] Saved PSI plot to {out_path}")
        if run is not None:
            wandb.log({f"latent_viz/{split_label}_{embed_type}_{lat_name}_psi_{safe_jid}": wandb.Image(out_path)})
        plt.close(fig)

    def _run_latent_viz(split_label, mdata_split, latent_spaces, obs_keys):
        """Compute and save UMAP / t-SNE embeddings for one data split."""
        ad = mdata_split["rna"]
        print(f"[EVAL/VIZ] Starting latent visualization on {split_label.upper()}...")
        print(f"[EVAL/VIZ] Latent spaces : {list(latent_spaces.keys())}")
        print(f"[EVAL/VIZ] Viz types     : {latent_viz_types}")
        print(f"[EVAL/VIZ] Coloring by   : {obs_keys}")

        for lat_name, Z in tqdm(latent_spaces.items(), desc=f"[EVAL/VIZ/{split_label}] Latent spaces"):
            key_latent = f"X_latent_{lat_name}"
            key_nn = f"neighbors_{lat_name}"
            print(f"[EVAL/VIZ] Storing latent '{lat_name}' in .obsm['{key_latent}']...")
            ad.obsm[key_latent] = Z

            # Neighbors are required by UMAP; t-SNE can run without them but we compute
            # them anyway so UMAP and t-SNE use the same neighborhood graph when both are
            # requested, keeping results comparable.
            if "umap" in latent_viz_types or "tsne" in latent_viz_types:
                print(f"[EVAL/VIZ] Computing neighbors for '{lat_name}'...")
                sc.pp.neighbors(ad, use_rep=key_latent, key_added=key_nn)

            # --- UMAP ---
            if "umap" in latent_viz_types:
                key_embed = f"X_umap_{lat_name}"
                print(f"[EVAL/VIZ] Computing UMAP for '{lat_name}'...")
                sc.tl.umap(ad, min_dist=0.1, neighbors_key=key_nn)
                ad.obsm[key_embed] = ad.obsm["X_umap"]

                for obs_key in tqdm(obs_keys, desc=f"[EVAL/VIZ] UMAP plots for '{lat_name}'", leave=False):
                    if obs_key not in ad.obs.columns:
                        print(f"[EVAL/VIZ] WARNING: obs key '{obs_key}' not in {split_label} RNA. Skipping.")
                        continue
                    print(f"[EVAL/VIZ] Plotting {split_label} UMAP '{lat_name}' colored by '{obs_key}'...")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.set_box_aspect(1)
                    ax.set_aspect(1)
                    sc.pl.embedding(
                        ad, basis=key_embed, color=obs_key,
                        palette=_build_palette(ad, obs_key, color_dict),
                        show=False, frameon=True, legend_fontsize=10,
                        legend_loc="right margin", ax=ax,
                    )
                    ax.set_xlabel("UMAP1")
                    ax.set_ylabel("UMAP2")
                    plt.title(f"SpliceVI $Z_{{{lat_name.capitalize()}}}$ ({split_label})")
                    plt.tight_layout()
                    safe_obs = re.sub(r"[^A-Za-z0-9]+", "_", obs_key)
                    out_path = os.path.join(args.fig_dir, f"{split_label}_umap_{lat_name}_{safe_obs}.png")
                    plt.savefig(out_path, dpi=300, bbox_inches="tight")
                    print(f"[EVAL/VIZ] Saved UMAP to {out_path}")
                    if run is not None:
                        wandb.log({f"latent_viz/{split_label}_umap_{lat_name}_{safe_obs}": wandb.Image(out_path)})
                    plt.close(fig)

                # PSI coloring for UMAP
                for junc_id in junction_color_ids:
                    _plot_junction_psi(mdata_split, junc_id, ad.obsm[key_embed], "umap", lat_name, split_label)

            # --- t-SNE ---
            if "tsne" in latent_viz_types:
                key_embed = f"X_tsne_{lat_name}"
                print(f"[EVAL/VIZ] Computing t-SNE for '{lat_name}'...")
                sc.tl.tsne(ad, use_rep=key_latent)
                ad.obsm[key_embed] = ad.obsm["X_tsne"]

                for obs_key in tqdm(obs_keys, desc=f"[EVAL/VIZ] t-SNE plots for '{lat_name}'", leave=False):
                    if obs_key not in ad.obs.columns:
                        print(f"[EVAL/VIZ] WARNING: obs key '{obs_key}' not in {split_label} RNA. Skipping.")
                        continue
                    print(f"[EVAL/VIZ] Plotting {split_label} t-SNE '{lat_name}' colored by '{obs_key}'...")
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.set_box_aspect(1)
                    ax.set_aspect(1)
                    sc.pl.embedding(
                        ad, basis=key_embed, color=obs_key,
                        palette=_build_palette(ad, obs_key, color_dict),
                        show=False, frameon=True, legend_fontsize=10,
                        legend_loc="right margin", ax=ax,
                    )
                    ax.set_xlabel("t-SNE 1")
                    ax.set_ylabel("t-SNE 2")
                    plt.title(f"SpliceVI $Z_{{{lat_name.capitalize()}}}$ t-SNE ({split_label})")
                    plt.tight_layout()
                    safe_obs = re.sub(r"[^A-Za-z0-9]+", "_", obs_key)
                    out_path = os.path.join(args.fig_dir, f"{split_label}_tsne_{lat_name}_{safe_obs}.png")
                    plt.savefig(out_path, dpi=300, bbox_inches="tight")
                    print(f"[EVAL/VIZ] Saved t-SNE to {out_path}")
                    if run is not None:
                        wandb.log({f"latent_viz/{split_label}_tsne_{lat_name}_{safe_obs}": wandb.Image(out_path)})
                    plt.close(fig)

                # PSI coloring for t-SNE
                for junc_id in junction_color_ids:
                    _plot_junction_psi(mdata_split, junc_id, ad.obsm[key_embed], "tsne", lat_name, split_label)

        print(f"[EVAL/VIZ] All {split_label.upper()} latent visualizations complete.")

    if "latent_visualization" in EVALS:
        if run_viz_train:
            _run_latent_viz("train", mdata_train, latent_spaces_train, umap_obs_keys)
        elif not run_viz_test:
            print("[EVAL/VIZ] Latent visualization skipped by config.")
        # test split viz runs later, after mdata_test is loaded

    # -----------------------------------------------------------------
    # Clustering + consistency
    # -----------------------------------------------------------------
    LEIDEN_RESOLUTION = 1.0

    if "clustering" in EVALS:
        print("[EVAL/CLUSTER] Running Leiden clustering and consistency metrics...")
        cell_type_col = "broad_cell_type"
        if "medium_cell_type" in mdata_train["rna"].obs:
            cell_type_col = "medium_cell_type"

        def run_leiden_on_basis(ad, basis_key: str, neigh_key: str, leiden_key: str):
            sc.pp.neighbors(ad, use_rep=basis_key, key_added=neigh_key)
            sc.tl.leiden(
                ad,
                neighbors_key=neigh_key,
                key_added=leiden_key,
                resolution=LEIDEN_RESOLUTION,
            )

        excl_multi_records = []
        spaces_order = ["expression", "splicing", "joint"]
        leiden_keys = {}

        print("[EVAL/CLUSTER] Running Leiden clustering per latent space...")
        for name in ["joint", "expression", "splicing"]:
            basis_key = f"X_latent_{name}"
            neigh_key = f"neighbors_{name}_leiden"
            leiden_key = f"leiden_{name}"

            print(f"[EVAL/CLUSTER] Clustering in space '{name}'...")
            run_leiden_on_basis(mdata_train["rna"], basis_key, neigh_key, leiden_key)
            leiden_keys[name] = leiden_key

            n_cl = int(mdata_train["rna"].obs[leiden_key].nunique())
            print(f"[EVAL/CLUSTER] '{name}' produced {n_cl} clusters.")
            if run is not None:
                wandb.log({f"clustering/{name}_leiden_n_clusters": n_cl})

            cts_per_cluster = (
                mdata_train["rna"]
                .obs.groupby(leiden_key)[cell_type_col]
                .apply(lambda s: set(s.astype(str).values))
            )
            n_unique = sum(1 for s in cts_per_cluster.values if len(s) == 1)
            n_multi = sum(1 for s in cts_per_cluster.values if len(s) > 1)

            print(
                f"[EVAL/CLUSTER] '{name}': {n_unique} clusters map to a single cell type, {n_multi} span multiple types."
            )

            if run is not None:
                wandb.log(
                    {
                        f"clusters/{name}_n_unique_one_celltype": int(n_unique),
                        f"clusters/{name}_n_multi_celltypes": int(n_multi),
                    }
                )

            excl_multi_records.append(
                {
                    "space": name,
                    "category": "Unique to one cell type",
                    "count": int(n_unique),
                }
            )
            excl_multi_records.append(
                {
                    "space": name,
                    "category": "Multiple cell types",
                    "count": int(n_multi),
                }
            )

            # Plot joint UMAP colored by Leiden labels for each space
            plt.figure(figsize=(8, 6))
            sc.pl.embedding(
                mdata_train["rna"],
                basis="X_umap_joint",
                color=leiden_key,
                legend_loc=None,
                frameon=True,
                show=False,
            )
            plt.title(f"TRAIN joint UMAP colored by Leiden ({name})")
            plt.tight_layout()
            out_path = (
                f"{args.fig_dir}/train_umap_joint_colored_by_{name}_leiden.png"
            )
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[EVAL/CLUSTER] Saved cluster UMAP: {out_path}")
            if run is not None:
                wandb.log(
                    {
                        f"clustering/train_umap_joint_colored_by_{name}_leiden": wandb.Image(
                            out_path
                        )
                    }
                )
            plt.close()

        # Bar plot: subclusters per cell type
        print("[EVAL/CLUSTER] Building bar plot: subclusters per top-20 cell types...")
        cell_type_for_bars = cell_type_col
        obs = mdata_train["rna"].obs
        ct_counts = obs[cell_type_for_bars].value_counts()
        top20_cts = ct_counts.head(20).index.tolist()
        print(f"[EVAL/CLUSTER] Top 20 cell types: {top20_cts}")

        records_sub = []
        for space_name, leiden_key in leiden_keys.items():
            sub_df = (
                obs.loc[obs[cell_type_for_bars].isin(top20_cts), [cell_type_for_bars, leiden_key]]
                .groupby(cell_type_for_bars)[leiden_key]
                .nunique()
                .rename("n_subclusters")
                .reset_index()
            )
            sub_df["space"] = space_name
            records_sub.append(sub_df)

        sub_all = pd.concat(records_sub, ignore_index=True)
        sub_all["space"] = pd.Categorical(
            sub_all["space"], categories=spaces_order, ordered=True
        )
        sub_all[cell_type_for_bars] = pd.Categorical(
            sub_all[cell_type_for_bars], categories=top20_cts, ordered=True
        )

        plt.figure(figsize=(max(12, 0.6 * len(top20_cts)), 6))
        sns.barplot(
            data=sub_all.sort_values([cell_type_for_bars, "space"]),
            x=cell_type_for_bars,
            y="n_subclusters",
            hue="space",
        )
        plt.xticks(rotation=45, ha="right")
        plt.xlabel(cell_type_for_bars)
        plt.ylabel("Number of Leiden sub-clusters")
        plt.title(
            f"TRAIN sub-clusters per cell type (top 20, res={LEIDEN_RESOLUTION})"
        )
        plt.tight_layout()
        out_path = f"{args.fig_dir}/train_bar_subclusters_top20_{cell_type_for_bars}_leiden_res_{LEIDEN_RESOLUTION}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[EVAL/CLUSTER] Saved bar plot of subclusters: {out_path}")
        if run is not None:
            wandb.log({"clustering/train_bar_subclusters_top20": wandb.Image(out_path)})
        plt.close()

        # Cluster exclusivity plot
        ex_df = pd.DataFrame(excl_multi_records)
        ex_df["space"] = pd.Categorical(
            ex_df["space"], categories=spaces_order, ordered=True
        )
        ex_df["category"] = pd.Categorical(
            ex_df["category"],
            categories=["Unique to one cell type", "Multiple cell types"],
            ordered=True,
        )

        plt.figure(figsize=(8, 5))
        sns.barplot(data=ex_df, x="category", y="count", hue="space")
        plt.xlabel("")
        plt.ylabel("Number of Leiden clusters")
        plt.title("TRAIN cluster exclusivity across spaces")
        plt.tight_layout()
        out_path = (
            f"{args.fig_dir}/train_clusters_exclusive_vs_multi_by_space.png"
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[EVAL/CLUSTER] Saved exclusivity plot: {out_path}")
        if run is not None:
            wandb.log(
                {
                    "clustering/train_clusters_exclusive_vs_multi_by_space": wandb.Image(
                        out_path
                    )
                }
            )
        plt.close()

        # Pairwise same-cluster consistency
        print("[EVAL/CLUSTER] Computing pairwise same-cluster consistency...")
        pairs = [("expression", "joint"), ("splicing", "joint"), ("expression", "splicing")]

        n_cells = mdata_train["rna"].n_obs
        idx_all = np.arange(n_cells, dtype=np.int32)

        cluster_members = {}
        for name in ["joint", "expression", "splicing"]:
            labs = mdata_train["rna"].obs[leiden_keys[name]].values
            members = {}
            for cid, grp in pd.Series(idx_all).groupby(labs):
                members[cid] = grp.values.astype(np.int32, copy=False)
            cluster_members[name] = (labs, members)

        heat_records = []
        for a, b in pairs:
            print(f"[EVAL/CLUSTER] Computing consistency for {a} vs {b}...")
            labs_a, mem_a = cluster_members[a]
            labs_b, mem_b = cluster_members[b]

            overlap = np.empty(n_cells, dtype=np.float32)
            for i in range(n_cells):
                ca = labs_a[i]
                cb = labs_b[i]
                Sa = mem_a[ca]
                Sb = mem_b[cb]
                if Sa.size <= 1:
                    overlap[i] = np.nan
                    continue
                Sa_no_i = Sa[Sa != i]
                inter_sz = len(set(Sa_no_i).intersection(Sb))
                overlap[i] = inter_sz / float(Sa_no_i.size)

            key_cell = f"samecluster_overlap_{a}_vs_{b}"
            mdata_train["rna"].obs[key_cell] = overlap

            mean_ov = float(np.nanmean(overlap))
            median_ov = float(np.nanmedian(overlap))
            print(
                f"[EVAL/CLUSTER] {a} vs {b} mean overlap: {mean_ov:.4f}, median: {median_ov:.4f}"
            )
            if run is not None:
                wandb.log(
                    {
                        f"clustering/{a}_vs_{b}_samecluster_mean": mean_ov,
                        f"clustering/{a}_vs_{b}_samecluster_median": median_ov,
                    }
                )

            if "tissue" in mdata_train["rna"].obs:
                pair_label = (
                    mdata_train["rna"]
                    .obs["tissue"]
                    .astype("string")
                    .fillna("NA")
                    .str.cat(
                        mdata_train["rna"]
                        .obs[cell_type_col]
                        .astype("string")
                        .fillna("NA"),
                        sep=" | ",
                    )
                    .to_numpy()
                )
            else:
                pair_label = (
                    mdata_train["rna"]
                    .obs[cell_type_col]
                    .astype("string")
                    .fillna("NA")
                    .to_numpy()
                )

            df_tmp = (
                pd.DataFrame({"pair_label": pair_label, "overlap": overlap})
                .groupby("pair_label", as_index=False)["overlap"]
                .mean()
            )
            df_tmp["pct_consistent"] = df_tmp["overlap"].fillna(0.0) * 100.0
            df_tmp["pair"] = f"{a}_vs_{b}"
            heat_records.append(
                df_tmp[["pair_label", "pair", "pct_consistent"]]
            )

        heat_df = pd.concat(heat_records, ignore_index=True)
        heat_pivot = heat_df.pivot(
            index="pair_label", columns="pair", values="pct_consistent"
        ).fillna(0.0)

        print("[EVAL/CLUSTER] Plotting clustermap of percent consistent clusters...")
        plt.close("all")
        g = sns.clustermap(
            heat_pivot,
            cmap="viridis",
            vmin=0.0,
            vmax=100.0,
            metric="euclidean",
            method="average",
            figsize=(
                max(6, 0.25 * heat_pivot.shape[1] + 4),
                max(6, 0.30 * heat_pivot.shape[0] + 3),
            ),
            row_cluster=True,
            col_cluster=False,
            annot=False,
        )
        g.figure.suptitle(
            f"TRAIN percent consistent by tissue | cell type (Leiden, res={LEIDEN_RESOLUTION})",
            y=1.02,
            fontsize=12,
        )
        out_path = f"{args.fig_dir}/train_clustermap_pct_consistent_leiden_res_{LEIDEN_RESOLUTION}.png"
        g.figure.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"[EVAL/CLUSTER] Saved clustermap: {out_path}")
        if run is not None:
            wandb.log(
                {"clustering/train_clustermap_pct_consistent": wandb.Image(out_path)}
            )
        plt.close(g.figure)

        # AMI
        print("[EVAL/CLUSTER] Computing adjusted mutual information between clusterings...")
        for a, b in pairs:
            ami = adjusted_mutual_info_score(
                mdata_train["rna"].obs[leiden_keys[a]].values,
                mdata_train["rna"].obs[leiden_keys[b]].values,
            )
            print(f"[EVAL/CLUSTER] AMI {a} vs {b}: {ami:.4f}")
            if run is not None:
                wandb.log({f"clustering/{a}_vs_{b}_AMI": float(ami)})

        del heat_records, heat_pivot, heat_df
        gc.collect()
    else:
        print("[EVAL/CLUSTER] Clustering evaluation skipped by config.")

    # -----------------------------------------------------------------
    # Train / Test latent evaluation
    # -----------------------------------------------------------------
    if "train_eval" in EVALS:
        print("[EVAL/TRAIN] Starting train-split latent quality evaluation...")
        evaluate_split(
            "train",
            mdata_train,
            model,
            umap_color_key,
            cell_type_classification_key,
            Z_type="joint",
            wandb=wandb if run is not None else None,
            precomputed_Z=latent_spaces_train.get("joint"),
        )
        evaluate_split(
            "train",
            mdata_train,
            model,
            umap_color_key,
            cell_type_classification_key,
            Z_type="expression",
            wandb=wandb if run is not None else None,
            precomputed_Z=latent_spaces_train.get("expression"),
        )
        evaluate_split(
            "train",
            mdata_train,
            model,
            umap_color_key,
            cell_type_classification_key,
            Z_type="splicing",
            wandb=wandb if run is not None else None,
            precomputed_Z=latent_spaces_train.get("splicing"),
        )
    else:
        print("[EVAL/TRAIN] Train-split evaluation skipped by config.")

    # Cross-fold classification on TRAIN
    if "cross_fold_classification" in EVALS and run_crossfold_train:
        run_cross_fold_classification(
            "train",
            mdata_train,
            latent_spaces_train,
            cross_fold_targets,
            args.cross_fold_k,
            cross_fold_classifiers,
            args.cross_fold_metrics,
            args.fig_dir,
            wandb=wandb if run is not None else None,
            do_dummy=args.cross_fold_dummy_classifier,
            do_label_permute=args.cross_fold_label_permute,
        )
    elif "cross_fold_classification" in EVALS:
        print("[CROSS-FOLD] TRAIN split disabled by --cross_fold_splits.")

    # Subcluster split evaluation (runs on TRAIN mdata before it is freed)
    if "subcluster_split_eval" in EVALS:
        cell_types = args.subcluster_cell_type
        if not cell_types:
            cell_types = sorted(mdata_train.obs[args.subcluster_obs_key].dropna().unique().tolist())
            print(f"[SUBCLUSTER] --subcluster_cell_type not set; running all {len(cell_types)} labels.")
        for ct in cell_types:
            run_subcluster_split_eval(
                mdata=mdata_train,
                model=model,
                obs_key=args.subcluster_obs_key,
                cell_type=ct,
                k_values=args.subcluster_k_values,
                splits=args.subcluster_splits,
                random_seed=args.subcluster_random_seed,
                embedding=args.subcluster_embedding,
                metrics=args.cross_fold_metrics,
                fig_dir=args.fig_dir,
                wandb=wandb if run is not None else None,
            )
    else:
        print("[SUBCLUSTER] subcluster_split_eval skipped by config.")

    # Free TRAIN data
    print("[CLEANUP] Releasing TRAIN MuData from memory...")
    del mdata_train
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # TEST evaluation
    # -----------------------------------------------------------------
    print(f"[DATA] Loading TEST MuData from {args.test_mdata_path} ...")
    mdata_test = mu.read_h5mu(args.test_mdata_path, backed="r")
    mdata_test.obs.rename(columns={"donor_id": "mouse.id"}, inplace=True)
    mdata_test.mod["rna"].obs.rename(
    columns={"donor_id": "mouse.id"},
    inplace=True)
    mdata_test.mod["splicing"].obs.rename(
    columns={"donor_id": "mouse.id"},
    inplace=True)
    print(f"[DATA] TEST MuData loaded with mods: {list(mdata_test.mod.keys())}")
    print(f"[DATA] TEST 'rna' n_obs: {mdata_test['rna'].n_obs}, n_vars: {mdata_test['rna'].n_vars}")

    if args.mapping_csv is not None:
        apply_obs_mapping_from_csv(mdata_test, args.mapping_csv)

    if "X_library_size" in mdata_test["rna"].obsm_keys():
        print("[DATA] Copying TEST RNA 'X_library_size' from .obsm to .obs...")
        mdata_test["rna"].obs["X_library_size"] = mdata_test["rna"].obsm["X_library_size"]

    print("[MODEL] Setting up SPLICEVI on TEST MuData ...")
    SPLICEVI.setup_mudata(
        mdata_test,
        batch_key=batch_key,
        size_factor_key="X_library_size",
        rna_layer="length_norm",
        junc_ratio_layer=x_layer,
        atse_counts_layer=cluster_counts_layer,
        junc_counts_layer=junction_counts_layer,
        psi_mask_layer=mask_layer,
        modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
    )

    # -----------------------------------------------------------------
    # Compute HVJ column mask ONCE from unmasked test splicing data.
    # Applied consistently to every imputation eval (test_impute and all
    # masked_impute files) so results are directly comparable.
    # hvj_col_mask is a boolean array of length n_junc; None = no filter.
    # -----------------------------------------------------------------
    # True if any imputation eval block will run — only compute HVJ when needed.
    _do_any_impute = ("test_impute" in EVALS) or ("masked_impute" in EVALS)
    if args.impute_top_n_hvj != -1 and _do_any_impute:
        print(
            f"[HVJ] Computing top-{args.impute_top_n_hvj} most variable junctions "
            f"from unmasked test splicing data (junc_ratio layer)...",
            flush=True,
        )
        # Load the PSI matrix from the unmasked test splicing modality (cells x junctions).
        # Kept sparse throughout to avoid densifying a potentially huge matrix.
        _jr_hvj = mdata_test["splicing"].layers["junc_ratio"]
        if not sparse.isspmatrix_csr(_jr_hvj):
            _jr_hvj = sparse.csr_matrix(_jr_hvj)
        _n_junc_total = _jr_hvj.shape[1]

        # Count how many cells observed each junction (PSI > 0).
        # Sparse: only stored non-zeros satisfy > 0, so this never densifies.
        _n_obs_hvj = np.asarray((_jr_hvj > 0).sum(axis=0)).ravel().astype(np.float64)

        # Sum of PSI values per junction across observed cells — needed for the mean.
        _sum_hvj   = np.asarray(_jr_hvj.sum(axis=0)).ravel().astype(np.float64)

        # Sum of squared PSI values per junction — needed for E[X²] in the variance formula.
        # .power(2) squares only the stored values without densifying.
        _sumsq_hvj = np.asarray(_jr_hvj.power(2).sum(axis=0)).ravel().astype(np.float64)

        # Free the sparse PSI matrix; we only need the per-junction summary stats from here on.
        del _jr_hvj

        # Per-junction mean PSI over observed cells: E[X] = sum(PSI) / n_obs.
        # np.maximum guards against junctions with zero observations (divide-by-zero).
        _mean_hvj  = _sum_hvj / np.maximum(_n_obs_hvj, 1.0)

        # Per-junction variance using the computational formula: Var(X) = E[X²] - E[X]².
        # Both E[X²] and E[X] are computed only over cells where PSI > 0.
        _var_hvj   = _sumsq_hvj / np.maximum(_n_obs_hvj, 1.0) - _mean_hvj ** 2

        # Zero out variance for junctions seen in fewer than 2 cells — not estimable.
        _var_hvj[_n_obs_hvj < 2] = 0.0

        del _sum_hvj, _sumsq_hvj, _mean_hvj, _n_obs_hvj

        # Count junctions with positive variance (the pool we can meaningfully rank).
        _n_valid_hvj = int((_var_hvj > 0).sum())

        # Cap N at the number of junctions that actually have variance > 0.
        _actual_n_hvj = min(args.impute_top_n_hvj, _n_valid_hvj)

        # argpartition is O(n) and finds the top-N indices without a full sort.
        # The last _actual_n_hvj elements of the result are the top-N (unordered).
        _hvj_indices = np.argpartition(_var_hvj, -_actual_n_hvj)[-_actual_n_hvj:]
        del _var_hvj

        # Build a boolean column mask of length n_junc.
        # True = this junction is in the top-N HVJ set and will be included in eval.
        hvj_col_mask = np.zeros(_n_junc_total, dtype=bool)
        hvj_col_mask[_hvj_indices] = True

        if args.impute_hvj_include_atse_buddies:
            _var_event_ids = mdata_test["splicing"].var["event_id"].values
            _selected_event_ids = set(_var_event_ids[_hvj_indices])
            _buddy_mask = np.isin(_var_event_ids, list(_selected_event_ids))
            _n_before_buddies = int(hvj_col_mask.sum())
            hvj_col_mask |= _buddy_mask
            print(
                f"[HVJ] After including ATSE buddies: {int(hvj_col_mask.sum())} junctions "
                f"({int(hvj_col_mask.sum()) - _n_before_buddies} buddy junctions added "
                f"from {len(_selected_event_ids)} ATSEs).",
                flush=True,
            )

        del _hvj_indices
        print(
            f"[HVJ] Selected {_actual_n_hvj} seed junctions "
            f"(of {_n_junc_total} total, {_n_valid_hvj} with var > 0); "
            f"{int(hvj_col_mask.sum())} total after buddy expansion. "
            f"Same mask applied in all imputation evals.",
            flush=True,
        )
    else:
        # -1 means disabled: pass None so downstream filter blocks are skipped entirely.
        hvj_col_mask = None
        if _do_any_impute:
            print("[HVJ] impute_top_n_hvj=-1; no HVJ filter applied (all junctions used).")

    latent_spaces_test = {}
    if (
        ("test_eval" in EVALS)
        or ("cross_fold_classification" in EVALS and run_crossfold_test)
        or ("latent_visualization" in EVALS and run_viz_test)
    ):
        print("[MODEL] Computing latent representations on TEST for evaluation...")
        latent_spaces_test = {
            "joint": model.get_latent_representation(adata=mdata_test),
            "expression": model.get_latent_representation(
                adata=mdata_test, modality="expression"
            ),
            "splicing": model.get_latent_representation(
                adata=mdata_test, modality="splicing"
            ),
        }
        for name, Z in latent_spaces_test.items():
            print(f"[MODEL] TEST latent '{name}' shape: {Z.shape}")
    else:
        print("[MODEL] Skipping TEST latent computation (not requested).")

    if "latent_visualization" in EVALS and run_viz_test:
        _run_latent_viz("test", mdata_test, latent_spaces_test, umap_obs_keys)

    if "test_eval" in EVALS:
        print("[EVAL/TEST] Starting test-split latent quality evaluation...")
        evaluate_split(
            "test",
            mdata_test,
            model,
            umap_color_key,
            cell_type_classification_key,
            Z_type="joint",
            wandb=wandb if run is not None else None,
            precomputed_Z=latent_spaces_test.get("joint"),
        )
        evaluate_split(
            "test",
            mdata_test,
            model,
            umap_color_key,
            cell_type_classification_key,
            Z_type="expression",
            wandb=wandb if run is not None else None,
            precomputed_Z=latent_spaces_test.get("expression"),
        )
        evaluate_split(
            "test",
            mdata_test,
            model,
            umap_color_key,
            cell_type_classification_key,
            Z_type="splicing",
            wandb=wandb if run is not None else None,
            precomputed_Z=latent_spaces_test.get("splicing"),
        )
    else:
        print("[EVAL/TEST] Test-split evaluation skipped by config.")

    # Cross-fold classification on TEST
    if "cross_fold_classification" in EVALS and run_crossfold_test:
        run_cross_fold_classification(
            "test",
            mdata_test,
            latent_spaces_test,
            cross_fold_targets,
            args.cross_fold_k,
            cross_fold_classifiers,
            args.cross_fold_metrics,
            args.fig_dir,
            wandb=wandb if run is not None else None,
            do_dummy=args.cross_fold_dummy_classifier,
            do_label_permute=args.cross_fold_label_permute,
        )
    elif "cross_fold_classification" in EVALS:
        print("[CROSS-FOLD] TEST split disabled by --cross_fold_splits.")

    # Cross-fold CSV dump
    if "cross_fold_classification" in EVALS and len(CROSS_FOLD_RECORDS) > 0:
        cross_df = pd.DataFrame(CROSS_FOLD_RECORDS)
        cross_csv = os.path.join(args.fig_dir, "cross_fold_classification_results.csv")
        cross_df.to_csv(cross_csv, index=False)
        print(
            f"[CROSS-FOLD] Wrote aggregated cross-fold metrics to {cross_csv} ({cross_df.shape[0]} rows)."
        )
        if run is not None:
            wandb.log({"crossfold/results_csv_path": cross_csv})

        if len(CROSS_FOLD_CLASS_RECORDS) > 0:
            class_df = pd.DataFrame(CROSS_FOLD_CLASS_RECORDS)
            for tgt, tgt_df in class_df.groupby("target"):
                tgt_csv = os.path.join(
                    args.fig_dir, f"cross_fold_classification_per_class_{tgt}.csv"
                )
                tgt_df.to_csv(tgt_csv, index=False)
                print(
                    f"[CROSS-FOLD] Wrote per-class fold metrics for target '{tgt}' to {tgt_csv} "
                    f"({tgt_df.shape[0]} rows)."
                )
                if run is not None:
                    wandb.log({f"crossfold/per_class/{tgt}_csv_path": tgt_csv})

            if args.output_per_label_f1_csv:
                f1_df = class_df[class_df["metric"] == "f1_weighted"].copy()
                f1_df = f1_df.rename(columns={"obs_category": "label", "value": "f1"})
                for tgt, tgt_f1_df in f1_df.groupby("target"):
                    out_df = tgt_f1_df[["label", "f1", "space", "split", "classifier", "fold"]]
                    tgt_f1_csv = os.path.join(args.fig_dir, f"per_label_f1_{tgt}.csv")
                    out_df.to_csv(tgt_f1_csv, index=False)
                    print(
                        f"[CROSS-FOLD] Wrote per-label F1 CSV for target '{tgt}' to {tgt_f1_csv} "
                        f"({out_df.shape[0]} rows)."
                    )
                    if run is not None:
                        wandb.log({f"crossfold/per_label_f1/{tgt}_csv_path": tgt_f1_csv})

        if len(CROSS_FOLD_SIGNIFICANCE) > 0:
            sig_df = pd.DataFrame(CROSS_FOLD_SIGNIFICANCE)
            sig_csv = os.path.join(
                args.fig_dir, "cross_fold_classification_significance.csv"
            )
            sig_df.to_csv(sig_csv, index=False)
            print(
                f"[CROSS-FOLD] Wrote paired t-test results to {sig_csv} ({sig_df.shape[0]} rows)."
            )
            if run is not None:
                wandb.log({"crossfold/significance_csv_path": sig_csv})
    elif "cross_fold_classification" in EVALS:
        print("[CROSS-FOLD] No cross-fold records collected; no CSV written.")

    # Age R² CSV dump
    if "age_r2_heatmap" in EVALS:
        print("[EVAL/AGE] Writing age R² CSV if any records exist...")
        if len(AGE_R2_RECORDS) > 0:
            age_df = pd.DataFrame(AGE_R2_RECORDS)
            csv_path = f"{args.fig_dir}/age_r2_by_tissue_celltype_train_test.csv"
            age_df.to_csv(csv_path, index=False)
            print(
                f"[EVAL/AGE] Wrote age R² records to {csv_path} ({age_df.shape[0]} rows)."
            )
            if run is not None:
                wandb.log({"age_r2/records_csv_path": csv_path})
        else:
            print("[EVAL/AGE] No age R² pairing records collected; skipping CSV.")
    else:
        print("[EVAL/AGE] Age R² CSV skipped by config.")

    # -----------------------------------------------------------------
    # Test imputation eval (perfect / upper-bound baseline)
    # Runs the *unmasked* test mdata through the model and checks imputed
    # PSI against junc_ratio (the original, unmasked values).
    # -----------------------------------------------------------------
    if "test_impute" in EVALS:
        print("\n" + "=" * 60)
        print("[EVAL/TEST_IMPUTE] Starting test imputation eval (perfect baseline)...")
        print("[EVAL/TEST_IMPUTE] Ground truth = junc_ratio (unmasked test data)")
        ad_test_spl = mdata_test["splicing"]

        junc_ratio_test = ad_test_spl.layers["junc_ratio"]
        if not sparse.isspmatrix_csr(junc_ratio_test):
            junc_ratio_test = sparse.csr_matrix(junc_ratio_test)

        n_nonzero_ti = int(junc_ratio_test.nnz)
        n_eq_one_ti = int((junc_ratio_test.data == 1.0).sum())
        print(
            f"[EVAL/TEST_IMPUTE] junc_ratio: {n_nonzero_ti} stored non-zero entries "
            f"({n_eq_one_ti} with PSI == 1.0, "
            f"{n_nonzero_ti - n_eq_one_ti} with 0 < PSI < 1.0)."
        )

        if args.impute_filter_boundary_psi:
            mask_data_ti = (junc_ratio_test.data > 0.0) & (junc_ratio_test.data < 1.0)
        else:
            mask_data_ti = junc_ratio_test.data > 0.0

        bin_mask_ti = sparse.csr_matrix(
            (mask_data_ti.astype(np.float32),
             junc_ratio_test.indices.copy(),
             junc_ratio_test.indptr.copy()),
            shape=junc_ratio_test.shape,
        )
        bin_mask_ti.eliminate_zeros()

        if args.impute_filter_boundary_psi:
            print(
                f"[EVAL/TEST_IMPUTE] After PSI boundary filter (exclude PSI == 1.0): "
                f"{bin_mask_ti.nnz} entries remain "
                f"({n_nonzero_ti - bin_mask_ti.nnz} removed)."
            )

        if args.min_atse_count != -1:
            cc_ti = ad_test_spl.layers["cell_by_cluster_matrix"]
            if not sparse.isspmatrix_csr(cc_ti):
                cc_ti = sparse.csr_matrix(cc_ti)
            count_ok_ti_data = cc_ti.data >= args.min_atse_count
            count_ok_ti = sparse.csr_matrix(
                (count_ok_ti_data.astype(np.float32), cc_ti.indices, cc_ti.indptr),
                shape=cc_ti.shape,
            )
            before_count_ti = bin_mask_ti.nnz
            bin_mask_ti = bin_mask_ti.multiply(count_ok_ti)
            if not sparse.isspmatrix_csr(bin_mask_ti):
                bin_mask_ti = sparse.csr_matrix(bin_mask_ti)
            print(
                f"[EVAL/TEST_IMPUTE] After ATSE count filter (>= {args.min_atse_count}): "
                f"{bin_mask_ti.nnz} entries remain "
                f"({before_count_ti - bin_mask_ti.nnz} removed)."
            )

        print(f"[EVAL/TEST_IMPUTE] Final eval set: {bin_mask_ti.nnz} entries.")

        bs_ti = args.impute_batch_size if args.impute_batch_size != -1 else 512
        n_cells_ti = bin_mask_ti.shape[0]
        print(
            f"[EVAL/TEST_IMPUTE] Running get_normalized_splicing_DM over {n_cells_ti} cells "
            f"(batch_size={bs_ti})...",
            flush=True,
        )
        model.module.eval()
        with torch.inference_mode():
            all_preds_ti = model.get_normalized_splicing_DM(
                adata=mdata_test,
                return_numpy=True,
                batch_size=bs_ti,
            )

        print(f"[EVAL/TEST_IMPUTE] Extracting eval pairs from mask...", flush=True)
        eval_rows_ti, eval_cols_ti = bin_mask_ti.nonzero()
        if hvj_col_mask is not None:
            _before_hvj_ti = eval_rows_ti.size
            _keep_hvj_ti = hvj_col_mask[eval_cols_ti]
            eval_rows_ti = eval_rows_ti[_keep_hvj_ti]
            eval_cols_ti = eval_cols_ti[_keep_hvj_ti]
            print(
                f"[EVAL/TEST_IMPUTE] After HVJ filter (top-{args.impute_top_n_hvj} junctions): "
                f"{eval_rows_ti.size} entries remain "
                f"({_before_hvj_ti - eval_rows_ti.size} removed).",
                flush=True,
            )
        if args.impute_dataset_filter:
            _ds_vals_ti = mdata_test.obs["dataset"].values
            _ds_keep_ti = _ds_vals_ti[eval_rows_ti] == args.impute_dataset_filter
            _before_ds_ti = eval_rows_ti.size
            eval_rows_ti = eval_rows_ti[_ds_keep_ti]
            eval_cols_ti = eval_cols_ti[_ds_keep_ti]
            print(
                f"[EVAL/TEST_IMPUTE] After dataset filter (dataset=={args.impute_dataset_filter}): "
                f"{eval_rows_ti.size} entries remain "
                f"({_before_ds_ti - eval_rows_ti.size} removed).",
                flush=True,
            )
        pairs_total_ti = eval_rows_ti.size

        if pairs_total_ti == 0:
            print("[EVAL/TEST_IMPUTE] No eval entries found; skipping correlation.")
        else:
            if sparse.issparse(junc_ratio_test):
                orig_all_ti = np.asarray(
                    junc_ratio_test[eval_rows_ti, eval_cols_ti]
                ).ravel().astype(np.float32)
            else:
                orig_all_ti = junc_ratio_test[eval_rows_ti, eval_cols_ti].astype(np.float32)
            pred_all_ti = all_preds_ti[eval_rows_ti, eval_cols_ti].astype(np.float32)
            del all_preds_ti
            torch.cuda.empty_cache()

            pearson_ti = float(np.corrcoef(orig_all_ti, pred_all_ti)[0, 1])
            spearman_ti = float(spearmanr(orig_all_ti, pred_all_ti, nan_policy="omit")[0])
            abs_diff_ti = np.abs(orig_all_ti - pred_all_ti)
            l1_mean_ti = float(np.mean(abs_diff_ti))
            l1_median_ti = float(np.median(abs_diff_ti))
            l1_p90_ti = float(np.quantile(abs_diff_ti, 0.90))
            pred_min_ti = float(np.min(pred_all_ti))
            pred_max_ti = float(np.max(pred_all_ti))
            smape_ti = float(
                np.mean(2.0 * abs_diff_ti / (np.abs(orig_all_ti) + np.abs(pred_all_ti) + 1e-8))
            )
            orig64_ti = orig_all_ti.astype(np.float64, copy=False)
            pred64_ti = pred_all_ti.astype(np.float64, copy=False)
            denom_ti = float(np.linalg.norm(orig64_ti) * np.linalg.norm(pred64_ti) + 1e-8)
            cosine_sim_ti = float(np.dot(orig64_ti, pred64_ti) / denom_ti)
            minmax_ratio_ti = float(
                np.mean(
                    np.minimum(np.abs(orig_all_ti), np.abs(pred_all_ti))
                    / (np.maximum(np.abs(orig_all_ti), np.abs(pred_all_ti)) + 1e-8)
                )
            )
            rmse_ti = float(np.sqrt(np.mean((orig_all_ti - pred_all_ti) ** 2)))

            print(
                f"[EVAL/TEST_IMPUTE] PSI corr — "
                f"Pearson: {pearson_ti:.4f}, Spearman: {spearman_ti:.4f}  (n={pairs_total_ti})"
            )
            print(
                f"[EVAL/TEST_IMPUTE] PSI L1 — "
                f"mean: {l1_mean_ti:.4e}, median: {l1_median_ti:.4e}, p90: {l1_p90_ti:.4e}"
            )
            print(
                f"[EVAL/TEST_IMPUTE] PSI range — min: {pred_min_ti:.4e}, max: {pred_max_ti:.4e}"
            )
            print(
                f"[EVAL/TEST_IMPUTE] PSI SMAPE: {smape_ti:.4e}, "
                f"cosine: {cosine_sim_ti:.4f}, minmax_ratio: {minmax_ratio_ti:.4f}, "
                f"RMSE: {rmse_ti:.4e}"
            )

            if run is not None:
                wandb.log(
                    {
                        "impute-test/unmasked/psi_pearson_corr": pearson_ti,
                        "impute-test/unmasked/psi_spearman_corr": spearman_ti,
                        "impute-test/unmasked/psi_l1_mean": l1_mean_ti,
                        "impute-test/unmasked/psi_l1_median": l1_median_ti,
                        "impute-test/unmasked/psi_l1_p90": l1_p90_ti,
                        "impute-test/unmasked/psi_pred_min": pred_min_ti,
                        "impute-test/unmasked/psi_pred_max": pred_max_ti,
                        "impute-test/unmasked/psi_smape": smape_ti,
                        "impute-test/unmasked/psi_cosine_sim": cosine_sim_ti,
                        "impute-test/unmasked/psi_minmax_ratio": minmax_ratio_ti,
                        "impute-test/unmasked/psi_rmse": rmse_ti,
                        "impute-test/unmasked/n_eval_entries": int(pairs_total_ti),
                    }
                )

        print("[EVAL/TEST_IMPUTE] Test imputation eval complete.")
        print("=" * 60)
    else:
        print("[EVAL/TEST_IMPUTE] Test imputation eval skipped by config.")

    # -----------------------------------------------------------------
    # Masked-ATSE imputation on TEST
    # -----------------------------------------------------------------
    print("[CLEANUP] Releasing TEST MuData from memory before masked imputation...")
    del mdata_test
    torch.cuda.empty_cache()

    if "masked_impute" in EVALS:
        if not args.masked_test_mdata_paths:
            print("[EVAL/IMPUTE] No masked_test_mdata_paths provided. Skipping.")
        else:
            print("[EVAL/IMPUTE] Starting masked imputation on provided TEST files...")
            for masked_path in tqdm(
                args.masked_test_mdata_paths,
                desc="[EVAL/IMPUTE] Masked TEST files",
            ):
                fname = os.path.basename(masked_path)
                m = re.search(
                    r"(\d+)\s*%|RESAMPLED[_-](\d+)[_-]PERCENT|MASKED[_-]?(\d+)",
                    fname, flags=re.IGNORECASE
                )
                if m:
                    pct = m.group(1) or m.group(2) or m.group(3)
                    tag = f"{pct}pct"
                else:
                    tag = re.sub(
                        r"[^A-Za-z0-9]+", "_", os.path.splitext(fname)[0]
                    )[:40]

                print(
                    f"\n[EVAL/IMPUTE] Masked-ATSE imputation on TEST using {masked_path} (tag={tag})"
                )
                mdata_masked = mu.read_h5mu(masked_path, backed="r")
                mdata_masked.obs.rename(columns={"donor_id": "mouse.id"}, inplace=True)
                mdata_masked.mod["rna"].obs.rename(
                columns={"donor_id": "mouse.id"},
                inplace=True)
                mdata_masked.mod["splicing"].obs.rename(
                columns={"donor_id": "mouse.id"},
                inplace=True)
                print(
                    f"[EVAL/IMPUTE/{tag}] Masked MuData loaded. 'rna' n_obs: {mdata_masked['rna'].n_obs}"
                )
                if args.mapping_csv is not None:
                    apply_obs_mapping_from_csv(mdata_masked, args.mapping_csv)

                ad_masked = mdata_masked["splicing"]

                if "X_library_size" in mdata_masked["rna"].obsm_keys():
                    print(
                        f"[EVAL/IMPUTE/{tag}] Copying RNA 'X_library_size' from .obsm to .obs..."
                    )
                    mdata_masked["rna"].obs["X_library_size"] = mdata_masked["rna"].obsm[
                        "X_library_size"
                    ]

                print(f"[EVAL/IMPUTE/{tag}] Setting up SPLICEVI on masked MuData...")
                SPLICEVI.setup_mudata(
                    mdata_masked,
                    batch_key=batch_key,
                    size_factor_key="X_library_size",
                    rna_layer="length_norm",
                    junc_ratio_layer=x_layer,
                    atse_counts_layer=cluster_counts_layer,
                    junc_counts_layer=junction_counts_layer,
                    psi_mask_layer=mask_layer,
                    modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
                )

                print(f"[EVAL/IMPUTE/{tag}] Running PSI imputation and computing metrics...")
                print(f"[EVAL/IMPUTE/{tag}] Mode: {'resampled' if args.masked_test_mdata_is_resampled else 'legacy masked'}")
                model.module.eval()

                if args.masked_test_mdata_is_resampled:
                    # ── Resampled mode ────────────────────────────────────────
                    # Ground truth = junc_ratio_original (pre-resampling values).
                    # Split eval into three cases:
                    #   "dropped"  – originally observed but dropped by downsampling
                    #   "observed" – still observed after downsampling (noisy PSI)
                    #   "all"      – union of the two (current / legacy behaviour)
                    junc_ratio_orig = ad_masked.layers["junc_ratio_original"]
                    if not sparse.isspmatrix_csr(junc_ratio_orig):
                        junc_ratio_orig = sparse.csr_matrix(junc_ratio_orig)

                    n_nonzero = int(junc_ratio_orig.nnz)
                    n_eq_one  = int((junc_ratio_orig.data == 1.0).sum())
                    print(
                        f"[EVAL/IMPUTE/{tag}] junc_ratio_original: "
                        f"{n_nonzero} stored non-zero entries "
                        f"({n_eq_one} with PSI == 1.0, "
                        f"{n_nonzero - n_eq_one} with 0 < PSI < 1.0)."
                    )

                    # ── Build bin_mask ("all"): junc_ratio_orig > 0 + optional filters ──
                    if args.impute_filter_boundary_psi:
                        mask_data = (junc_ratio_orig.data > 0.0) & (junc_ratio_orig.data < 1.0)
                    else:
                        mask_data = junc_ratio_orig.data > 0.0
                    bin_mask = sparse.csr_matrix(
                        (mask_data.astype(np.float32),
                         junc_ratio_orig.indices.copy(),
                         junc_ratio_orig.indptr.copy()),
                        shape=junc_ratio_orig.shape,
                    )
                    del mask_data
                    bin_mask.eliminate_zeros()

                    if args.impute_filter_boundary_psi:
                        print(
                            f"[EVAL/IMPUTE/{tag}] After PSI boundary filter (exclude PSI == 1.0): "
                            f"{bin_mask.nnz} entries remain "
                            f"({n_nonzero - bin_mask.nnz} removed)."
                        )

                    if args.min_atse_count != -1:
                        cc_orig = ad_masked.layers["cell_by_cluster_matrix_original"]
                        if not sparse.isspmatrix_csr(cc_orig):
                            cc_orig = sparse.csr_matrix(cc_orig)
                        count_ok_data = cc_orig.data >= args.min_atse_count
                        count_ok = sparse.csr_matrix(
                            (count_ok_data.astype(np.float32), cc_orig.indices, cc_orig.indptr),
                            shape=cc_orig.shape,
                        )
                        del count_ok_data, cc_orig
                        before_count_filter = bin_mask.nnz
                        bin_mask = bin_mask.multiply(count_ok)
                        if not sparse.isspmatrix_csr(bin_mask):
                            bin_mask = sparse.csr_matrix(bin_mask)
                        bin_mask.eliminate_zeros()
                        del count_ok
                        print(
                            f"[EVAL/IMPUTE/{tag}] After ATSE count filter (>= {args.min_atse_count}): "
                            f"{bin_mask.nnz} entries remain "
                            f"({before_count_filter - bin_mask.nnz} removed)."
                        )

                    # ── Split bin_mask by psi_mask into "observed" and "dropped" ──
                    psi_mask_sp = ad_masked.layers[mask_layer]
                    if not sparse.isspmatrix_csr(psi_mask_sp):
                        psi_mask_sp = sparse.csr_matrix(psi_mask_sp)
                    psi_mask_sp.eliminate_zeros()

                    # "observed": bin_mask positions still observed after downsampling
                    still_obs_mask = bin_mask.multiply(psi_mask_sp)
                    if not sparse.isspmatrix_csr(still_obs_mask):
                        still_obs_mask = sparse.csr_matrix(still_obs_mask)
                    still_obs_mask.eliminate_zeros()

                    # "dropped": bin_mask positions that became unobserved (psi_mask==0)
                    newly_unobs_mask = bin_mask - still_obs_mask
                    if not sparse.isspmatrix_csr(newly_unobs_mask):
                        newly_unobs_mask = sparse.csr_matrix(newly_unobs_mask)
                    newly_unobs_mask.eliminate_zeros()
                    del psi_mask_sp

                    print(
                        f"[EVAL/IMPUTE/{tag}] Case splits — "
                        f"dropped: {newly_unobs_mask.nnz}, "
                        f"observed: {still_obs_mask.nnz}, "
                        f"all: {bin_mask.nnz}"
                    )

                    # ── Load downsampled PSI for noise scatters ─────────────
                    junc_ratio_ds = ad_masked.layers["junc_ratio"]
                    if not sparse.isspmatrix_csr(junc_ratio_ds):
                        junc_ratio_ds = sparse.csr_matrix(junc_ratio_ds)

                    # ── Run model ─────────────────────────────────────────────
                    bs = args.impute_batch_size if args.impute_batch_size != -1 else 512
                    n_cells_r = bin_mask.shape[0]
                    print(
                        f"[EVAL/IMPUTE/{tag}] Running get_normalized_splicing_DM_DM over "
                        f"{n_cells_r} cells (batch_size={bs})...",
                        flush=True,
                    )
                    with torch.inference_mode():
                        all_preds = model.get_normalized_splicing_DM(
                            adata=mdata_masked,
                            return_numpy=True,
                            batch_size=bs,
                        )

                    # ── Helper: apply HVJ filter to (rows, cols) ───────────
                    def _apply_hvj(rows, cols, case_label):
                        if hvj_col_mask is None:
                            return rows, cols
                        if hvj_col_mask.shape[0] != bin_mask.shape[1]:
                            print(
                                f"[EVAL/IMPUTE/{tag}/{case_label}] WARNING: hvj_col_mask "
                                f"length ({hvj_col_mask.shape[0]}) != n_junc "
                                f"({bin_mask.shape[1]}); skipping HVJ filter.",
                                flush=True,
                            )
                            return rows, cols
                        before = rows.size
                        keep = hvj_col_mask[cols]
                        rows, cols = rows[keep], cols[keep]
                        print(
                            f"[EVAL/IMPUTE/{tag}/{case_label}] After HVJ filter "
                            f"(top-{args.impute_top_n_hvj}): {rows.size} entries "
                            f"({before - rows.size} removed).",
                            flush=True,
                        )
                        return rows, cols

                    # ── Helper: apply dataset row filter to (rows, cols) ────
                    def _apply_dataset_filter(rows, cols, case_label):
                        if not args.impute_dataset_filter:
                            return rows, cols
                        _ds_vals = mdata_masked.obs["dataset"].values
                        _keep_ds = _ds_vals[rows] == args.impute_dataset_filter
                        before = rows.size
                        rows, cols = rows[_keep_ds], cols[_keep_ds]
                        print(
                            f"[EVAL/IMPUTE/{tag}/{case_label}] After dataset filter "
                            f"(dataset=={args.impute_dataset_filter}): {rows.size} entries "
                            f"({before - rows.size} removed).",
                            flush=True,
                        )
                        return rows, cols

                    # ── Evaluate each case ────────────────────────────────────
                    eval_cases = [
                        ("dropped",  newly_unobs_mask),
                        ("observed", still_obs_mask),
                        ("all",      bin_mask),
                    ]
                    case_eval_data: dict = {}  # case_name -> (rows, cols, orig, pred)

                    for case_name, case_mask in eval_cases:
                        eval_rows_c, eval_cols_c = case_mask.nonzero()
                        eval_rows_c, eval_cols_c = _apply_hvj(
                            eval_rows_c, eval_cols_c, case_name
                        )
                        eval_rows_c, eval_cols_c = _apply_dataset_filter(
                            eval_rows_c, eval_cols_c, case_name
                        )
                        n_eval_c = eval_rows_c.size
                        print(
                            f"[EVAL/IMPUTE/{tag}/{case_name}] n_eval = {n_eval_c}",
                            flush=True,
                        )

                        if n_eval_c == 0:
                            case_eval_data[case_name] = (eval_rows_c, eval_cols_c, None, None)
                            print(
                                f"[EVAL/IMPUTE/{tag}/{case_name}] No entries; skipping metrics."
                            )
                            continue

                        orig_c = np.asarray(
                            junc_ratio_orig[eval_rows_c, eval_cols_c]
                        ).ravel().astype(np.float32)
                        pred_c = all_preds[eval_rows_c, eval_cols_c].astype(np.float32)
                        case_eval_data[case_name] = (eval_rows_c, eval_cols_c, orig_c, pred_c)

                        m = _compute_imputation_metrics(orig_c, pred_c)
                        print(
                            f"[EVAL/IMPUTE/{tag}/{case_name}] SpliceVI PSI corr — "
                            f"Pearson: {m['pearson']:.4f}, Spearman: {m['spearman']:.4f}  "
                            f"(n={n_eval_c})"
                        )
                        print(
                            f"[EVAL/IMPUTE/{tag}/{case_name}] SpliceVI PSI L1 — "
                            f"mean: {m['l1_mean']:.4e}, median: {m['l1_median']:.4e}, "
                            f"p90: {m['l1_p90']:.4e}"
                        )
                        print(
                            f"[EVAL/IMPUTE/{tag}/{case_name}] SpliceVI PSI range — "
                            f"min: {m['pred_min']:.4e}, max: {m['pred_max']:.4e}"
                        )
                        print(
                            f"[EVAL/IMPUTE/{tag}/{case_name}] SpliceVI PSI SMAPE: "
                            f"{m['smape']:.4e}, cosine: {m['cosine_sim']:.4f}, "
                            f"minmax_ratio: {m['minmax_ratio']:.4f}, "
                            f"RMSE: {m['rmse']:.4e}"
                        )
                        if run is not None:
                            wandb.log({
                                f"impute-test/{tag}/{case_name}/splicevi/psi_pearson_corr":  m["pearson"],
                                f"impute-test/{tag}/{case_name}/splicevi/psi_spearman_corr": m["spearman"],
                                f"impute-test/{tag}/{case_name}/splicevi/psi_l1_mean":       m["l1_mean"],
                                f"impute-test/{tag}/{case_name}/splicevi/psi_l1_median":     m["l1_median"],
                                f"impute-test/{tag}/{case_name}/splicevi/psi_l1_p90":        m["l1_p90"],
                                f"impute-test/{tag}/{case_name}/splicevi/psi_pred_min":      m["pred_min"],
                                f"impute-test/{tag}/{case_name}/splicevi/psi_pred_max":      m["pred_max"],
                                f"impute-test/{tag}/{case_name}/splicevi/psi_smape":         m["smape"],
                                f"impute-test/{tag}/{case_name}/splicevi/psi_cosine_sim":    m["cosine_sim"],
                                f"impute-test/{tag}/{case_name}/splicevi/psi_minmax_ratio":  m["minmax_ratio"],
                                f"impute-test/{tag}/{case_name}/splicevi/psi_rmse":          m["rmse"],
                                f"impute-test/{tag}/{case_name}/splicevi/n_eval_entries":    n_eval_c,
                                f"impute-test/{tag}/impute_batch_size":                      bs,
                                f"impute-test/{tag}/masked_file":                            masked_path,
                            })

                        # Noise scatter: downsampled PSI vs original PSI
                        ds_c = np.asarray(
                            junc_ratio_ds[eval_rows_c, eval_cols_c]
                        ).ravel().astype(np.float32)
                        noise_pearson = _safe_pearson(orig_c, ds_c)
                        noise_l1 = float(np.mean(np.abs(orig_c - ds_c)))
                        noise_scatter_path = os.path.join(
                            args.fig_dir,
                            f"impute_{tag}_{case_name}_noise_scatter_orig_vs_downsampled.png",
                        )
                        plot_psi_density_scatter(
                            orig_c, ds_c,
                            pearson_r=noise_pearson,
                            l1_mean=noise_l1,
                            y_label="Downsampled PSI",
                            out_path=noise_scatter_path,
                            tag=f"{tag} — {case_name} entries (downsampled vs original)",
                            wandb=wandb if run is not None else None,
                            run=run,
                        )
                        del ds_c

                        # SpliceVI scatter: SpliceVI PSI vs original PSI
                        splicevi_scatter_path = os.path.join(
                            args.fig_dir,
                            f"impute_{tag}_{case_name}_splicevi_psi_density_scatter.png",
                        )
                        plot_psi_density_scatter(
                            orig_c, pred_c,
                            pearson_r=m["pearson"],
                            l1_mean=m["l1_mean"],
                            out_path=splicevi_scatter_path,
                            tag=f"{tag} — {case_name} entries (SpliceVI)",
                            wandb=wandb if run is not None else None,
                            run=run,
                        )

                    del all_preds, junc_ratio_ds
                    torch.cuda.empty_cache()

                    # ── MeanBayes baseline — evaluated on "dropped" entries only ──
                    if args.mean_bayes_impute:
                        dropped_rows, dropped_cols, orig_dropped, _ = case_eval_data.get(
                            "dropped", (None, None, None, None)
                        )
                        if orig_dropped is not None and orig_dropped.size > 0:
                            n_dropped = orig_dropped.size
                            print(
                                f"[EVAL/IMPUTE/{tag}/dropped] Running MeanBayes "
                                f"(group_by={mean_bayes_group_by}, "
                                f"n_eval={n_dropped})...",
                                flush=True,
                            )
                            mb = MeanBayes(
                                ad_masked,
                                junc_ratio_layer="junc_ratio",
                                psi_mask_layer=mask_layer,
                                group_by=mean_bayes_group_by,
                            )
                            mb_imputed = mb.get_imputed_splicing(return_numpy=True)
                            del mb
                            pred_mb = mb_imputed[dropped_rows, dropped_cols].astype(
                                np.float32
                            )
                            del mb_imputed
                            gc.collect()

                            m_mb = _compute_imputation_metrics(orig_dropped, pred_mb)
                            print(
                                f"[EVAL/IMPUTE/{tag}/dropped] MeanBayes n_eval = "
                                f"{n_dropped} (same positions as SpliceVI dropped)"
                            )
                            print(
                                f"[EVAL/IMPUTE/{tag}/dropped] MeanBayes PSI corr — "
                                f"Pearson: {m_mb['pearson']:.4f}, "
                                f"Spearman: {m_mb['spearman']:.4f}  (n={n_dropped})"
                            )
                            print(
                                f"[EVAL/IMPUTE/{tag}/dropped] MeanBayes PSI L1 — "
                                f"mean: {m_mb['l1_mean']:.4e}, "
                                f"median: {m_mb['l1_median']:.4e}, "
                                f"p90: {m_mb['l1_p90']:.4e}"
                            )
                            print(
                                f"[EVAL/IMPUTE/{tag}/dropped] MeanBayes PSI range — "
                                f"min: {m_mb['pred_min']:.4e}, max: {m_mb['pred_max']:.4e}"
                            )
                            print(
                                f"[EVAL/IMPUTE/{tag}/dropped] MeanBayes PSI SMAPE: "
                                f"{m_mb['smape']:.4e}, cosine: {m_mb['cosine_sim']:.4f}, "
                                f"minmax_ratio: {m_mb['minmax_ratio']:.4f}, "
                                f"RMSE: {m_mb['rmse']:.4e}"
                            )
                            if run is not None:
                                wandb.log({
                                    f"impute-test/{tag}/dropped/mean_bayes/psi_pearson_corr":  m_mb["pearson"],
                                    f"impute-test/{tag}/dropped/mean_bayes/psi_spearman_corr": m_mb["spearman"],
                                    f"impute-test/{tag}/dropped/mean_bayes/psi_l1_mean":       m_mb["l1_mean"],
                                    f"impute-test/{tag}/dropped/mean_bayes/psi_l1_median":     m_mb["l1_median"],
                                    f"impute-test/{tag}/dropped/mean_bayes/psi_l1_p90":        m_mb["l1_p90"],
                                    f"impute-test/{tag}/dropped/mean_bayes/psi_pred_min":      m_mb["pred_min"],
                                    f"impute-test/{tag}/dropped/mean_bayes/psi_pred_max":      m_mb["pred_max"],
                                    f"impute-test/{tag}/dropped/mean_bayes/psi_smape":         m_mb["smape"],
                                    f"impute-test/{tag}/dropped/mean_bayes/psi_cosine_sim":    m_mb["cosine_sim"],
                                    f"impute-test/{tag}/dropped/mean_bayes/psi_minmax_ratio":  m_mb["minmax_ratio"],
                                    f"impute-test/{tag}/dropped/mean_bayes/psi_rmse":          m_mb["rmse"],
                                    f"impute-test/{tag}/dropped/mean_bayes/n_eval_entries":    n_dropped,
                                })
                            mb_scatter_path = os.path.join(
                                args.fig_dir,
                                f"impute_{tag}_dropped_mean_bayes_psi_density_scatter.png",
                            )
                            plot_psi_density_scatter(
                                orig_dropped, pred_mb,
                                pearson_r=m_mb["pearson"],
                                l1_mean=m_mb["l1_mean"],
                                out_path=mb_scatter_path,
                                tag=f"{tag} — dropped entries (MeanBayes)",
                                wandb=wandb if run is not None else None,
                                run=run,
                            )
                            del pred_mb
                        else:
                            print(
                                f"[EVAL/IMPUTE/{tag}/dropped] No dropped entries; "
                                f"skipping MeanBayes eval."
                            )

                    # ── Residuals CSV for "all" case ──────────────────────────
                    eval_rows_all, eval_cols_all, orig_all_case, pred_all_case = (
                        case_eval_data.get("all", (None, None, None, None))
                    )
                    if orig_all_case is not None:
                        junc_names = np.asarray(ad_masked.var_names)
                        residuals_df = pd.DataFrame({
                            "cell_idx":       eval_rows_all,
                            "junction_idx":   eval_cols_all,
                            "junction_name":  junc_names[eval_cols_all],
                            "observed_psi":   orig_all_case,
                            "model_pred_psi": pred_all_case,
                            "model_residual": pred_all_case - orig_all_case,
                        })
                        residuals_csv_path = os.path.join(
                            args.fig_dir, f"impute_{tag}_residuals.csv"
                        )
                        residuals_df.to_csv(residuals_csv_path, index=False)
                        del residuals_df
                        print(
                            f"[EVAL/IMPUTE/{tag}] Residuals CSV (all case) saved to: "
                            f"{residuals_csv_path}"
                        )
                        if run is not None:
                            wandb.log(
                                {f"impute-test/{tag}/residuals_csv": residuals_csv_path}
                            )

                    del (
                        bin_mask, newly_unobs_mask, still_obs_mask,
                        junc_ratio_orig, case_eval_data,
                    )

                else:
                    # ── Legacy masked mode ────────────────────────────────────
                    masked_orig = ad_masked.layers["junc_ratio_masked_original"]
                    if not sparse.isspmatrix_csr(masked_orig):
                        masked_orig = sparse.csr_matrix(masked_orig)
                    bin_mask = ad_masked.layers["junc_ratio_masked_bin_mask"]
                    if not sparse.isspmatrix_csr(bin_mask):
                        bin_mask = sparse.csr_matrix(bin_mask)

                    bs = args.impute_batch_size if args.impute_batch_size != -1 else 512
                    n_cells = bin_mask.shape[0]
                    print(
                        f"[EVAL/IMPUTE/{tag}] Running get_normalized_splicing_DM over "
                        f"{n_cells} cells (batch_size={bs})...",
                        flush=True,
                    )
                    with torch.inference_mode():
                        all_preds = model.get_normalized_splicing_DM(
                            adata=mdata_masked,
                            return_numpy=True,
                            batch_size=bs,
                        )

                    print(
                        f"[EVAL/IMPUTE/{tag}] Extracting eval pairs from mask...",
                        flush=True,
                    )
                    eval_rows, eval_cols = bin_mask.nonzero()
                    if hvj_col_mask is not None:
                        if hvj_col_mask.shape[0] != bin_mask.shape[1]:
                            print(
                                f"[EVAL/IMPUTE/{tag}] WARNING: hvj_col_mask length "
                                f"({hvj_col_mask.shape[0]}) != n_junc "
                                f"({bin_mask.shape[1]}); skipping HVJ filter.",
                                flush=True,
                            )
                        else:
                            _before_hvj = eval_rows.size
                            _keep_hvj = hvj_col_mask[eval_cols]
                            eval_rows = eval_rows[_keep_hvj]
                            eval_cols = eval_cols[_keep_hvj]
                            print(
                                f"[EVAL/IMPUTE/{tag}] After HVJ filter "
                                f"(top-{args.impute_top_n_hvj} junctions): "
                                f"{eval_rows.size} entries remain "
                                f"({_before_hvj - eval_rows.size} removed).",
                                flush=True,
                            )
                    pairs_total = eval_rows.size

                    if pairs_total == 0:
                        print(f"[EVAL/IMPUTE/{tag}] No eval entries found; skipping.")
                    else:
                        if sparse.issparse(masked_orig):
                            orig_all = np.asarray(
                                masked_orig[eval_rows, eval_cols]
                            ).ravel().astype(np.float32)
                        else:
                            orig_all = masked_orig[eval_rows, eval_cols].astype(np.float32)
                        pred_all = all_preds[eval_rows, eval_cols].astype(np.float32)
                        del all_preds
                        torch.cuda.empty_cache()

                        m = _compute_imputation_metrics(orig_all, pred_all)
                        print(
                            f"[EVAL/IMPUTE/{tag}] PSI corr — "
                            f"Pearson: {m['pearson']:.4f}, "
                            f"Spearman: {m['spearman']:.4f}  (n={pairs_total})"
                        )
                        print(
                            f"[EVAL/IMPUTE/{tag}] PSI L1 — "
                            f"mean: {m['l1_mean']:.4e}, "
                            f"median: {m['l1_median']:.4e}, p90: {m['l1_p90']:.4e}"
                        )
                        print(
                            f"[EVAL/IMPUTE/{tag}] PSI range — "
                            f"min: {m['pred_min']:.4e}, max: {m['pred_max']:.4e}"
                        )
                        print(
                            f"[EVAL/IMPUTE/{tag}] PSI SMAPE: {m['smape']:.4e}, "
                            f"cosine: {m['cosine_sim']:.4f}, "
                            f"minmax_ratio: {m['minmax_ratio']:.4f}, "
                            f"RMSE: {m['rmse']:.4e}"
                        )
                        if run is not None:
                            wandb.log({
                                f"impute-test/{tag}/psi_pearson_corr_masked_atse":  m["pearson"],
                                f"impute-test/{tag}/psi_spearman_corr_masked_atse": m["spearman"],
                                f"impute-test/{tag}/psi_l1_mean_masked_atse":       m["l1_mean"],
                                f"impute-test/{tag}/psi_l1_median_masked_atse":     m["l1_median"],
                                f"impute-test/{tag}/psi_l1_p90_masked_atse":        m["l1_p90"],
                                f"impute-test/{tag}/psi_pred_min_masked_atse":      m["pred_min"],
                                f"impute-test/{tag}/psi_pred_max_masked_atse":      m["pred_max"],
                                f"impute-test/{tag}/psi_smape_masked_atse":         m["smape"],
                                f"impute-test/{tag}/psi_cosine_sim_masked_atse":    m["cosine_sim"],
                                f"impute-test/{tag}/psi_minmax_ratio_masked_atse":  m["minmax_ratio"],
                                f"impute-test/{tag}/psi_rmse_masked_atse":          m["rmse"],
                                f"impute-test/{tag}/n_masked_entries":              int(pairs_total),
                                f"impute-test/{tag}/impute_batch_size":             bs,
                                f"impute-test/{tag}/masked_file":                   masked_path,
                            })

                        junc_names = np.asarray(ad_masked.var_names)
                        residuals_df = pd.DataFrame({
                            "cell_idx":       eval_rows,
                            "junction_idx":   eval_cols,
                            "junction_name":  junc_names[eval_cols],
                            "observed_psi":   orig_all,
                            "model_pred_psi": pred_all,
                            "model_residual": pred_all - orig_all,
                        })
                        residuals_csv_path = os.path.join(
                            args.fig_dir, f"impute_{tag}_residuals.csv"
                        )
                        residuals_df.to_csv(residuals_csv_path, index=False)
                        del residuals_df
                        print(
                            f"[EVAL/IMPUTE/{tag}] Residuals CSV saved to: "
                            f"{residuals_csv_path}"
                        )
                        if run is not None:
                            wandb.log(
                                {f"impute-test/{tag}/residuals_csv": residuals_csv_path}
                            )
                        del orig_all, pred_all

                    del masked_orig, bin_mask, eval_rows, eval_cols

                print(f"[EVAL/IMPUTE/{tag}] Cleaning up masked MuData from memory...")
                del mdata_masked, ad_masked
                gc.collect()
                torch.cuda.empty_cache()
    else:
        print("[EVAL/IMPUTE] Masked imputation eval skipped by config.")

    # -----------------------------------------------------------------
    # Finish
    # -----------------------------------------------------------------
    print("[CLEANUP] Releasing SPLICEVI model and GPU memory...")
    del model
    torch.cuda.empty_cache()

    if run is not None:
        print("[W&B] Finishing W&B eval run.")
        run.finish()

    print("[DONE] Evaluation pipeline complete.")


if __name__ == "__main__":
    main()
