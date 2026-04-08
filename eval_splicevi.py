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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeCV
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

    # UMAP settings
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

    # Which eval blocks to run
    parser.add_argument(
        "--evals",
        nargs="+",
        default=[
            "umap",
            "clustering",
            "train_eval",
            "test_eval",
            "age_r2_heatmap",
            "masked_impute",
        ],
        help=(
            "Which eval blocks to run. Choices among: "
            "umap, clustering, train_eval, test_eval, age_r2_heatmap, masked_impute, cross_fold_classification"
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


def main():
    parser = build_argparser()
    args = parser.parse_args()
    EVALS = set(args.evals)
    cross_fold_targets = list(dict.fromkeys(args.cross_fold_targets))
    cross_fold_splits = args.cross_fold_splits
    cross_fold_classifiers = args.cross_fold_classifiers
    run_crossfold_train = cross_fold_splits in {"train", "both"}
    run_crossfold_test = cross_fold_splits in {"test", "both"}
    batch_key = None if (args.batch_key is None or str(args.batch_key).lower() == "none") else args.batch_key

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
    # Required by: umap, clustering, train_eval, cross_fold_classification (train split)
    _need_train_latent = bool(
        EVALS & {"umap", "clustering", "train_eval"}
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
    # UMAP evaluation (TRAIN)
    # -----------------------------------------------------------------
    if "umap" in EVALS:
        print("[EVAL/UMAP] Starting UMAP evaluation on TRAIN...")
        print(f"[EVAL/UMAP] Will compute UMAPs for latent spaces: {list(latent_spaces_train.keys())}")
        print(f"[EVAL/UMAP] Will color UMAPs by obs keys: {umap_obs_keys}")

        for name, Z in tqdm(
            latent_spaces_train.items(),
            desc="[EVAL/UMAP] Latent spaces",
        ):
            key_latent = f"X_latent_{name}"
            key_nn = f"neighbors_{name}"
            key_umap = f"X_umap_{name}"

            print(f"[EVAL/UMAP] Working on latent space '{name}'...")
            print(f"[EVAL/UMAP] Storing latent in .obsm['{key_latent}']...")
            mdata_train["rna"].obsm[key_latent] = Z

            print(f"[EVAL/UMAP] Computing neighbors for '{name}'...")
            sc.pp.neighbors(
                mdata_train["rna"],
                use_rep=key_latent,
                key_added=key_nn,
            )

            print(f"[EVAL/UMAP] Computing UMAP embedding for '{name}'...")
            sc.tl.umap(mdata_train["rna"], min_dist=0.1, neighbors_key=key_nn)
            mdata_train["rna"].obsm[key_umap] = mdata_train["rna"].obsm["X_umap"]

            # Plot UMAPs for all requested obs keys
            for obs_key in tqdm(
                umap_obs_keys,
                desc=f"[EVAL/UMAP] Plotting colors for '{name}'",
                leave=False,
            ):
                if obs_key not in mdata_train["rna"].obs.columns:
                    print(
                        f"[EVAL/UMAP] WARNING: obs key '{obs_key}' not found in TRAIN RNA. Skipping."
                    )
                    continue

                print(
                    f"[EVAL/UMAP] Plotting TRAIN UMAP for latent '{name}' colored by '{obs_key}'..."
                )

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.set_box_aspect(1)
                ax.set_aspect(1)

                palette = color_dict if obs_key == "group_highlighted" else None

                # Avoid Scanpy defaulting to a single gray color when there are many categories
                # (e.g., lots of mouse IDs). Force a large distinct palette for high-cardinality keys.
                if palette is None:
                    obs_series = mdata_train["rna"].obs[obs_key]
                    obs_as_str = obs_series.astype(str)
                    n_categories = obs_as_str.nunique()
                    normalized_key = obs_key.lower().replace(".", "_")
                    needs_large_palette = normalized_key in {"mouse_id", "mouseid"} or n_categories > 100
                    if needs_large_palette:
                        # Preserve categorical ordering if present; otherwise sort for determinism
                        if pd.api.types.is_categorical_dtype(obs_series):
                            categories = list(obs_series.cat.categories.astype(str))
                        else:
                            categories = sorted(pd.Index(obs_as_str).unique())

                        # Draw well-separated colors and shuffle them so adjacent labels differ
                        cmap = cm.get_cmap("hsv", max(len(categories), 1))
                        base_colors = cmap(np.linspace(0, 1, len(categories), endpoint=False))
                        rng = np.random.default_rng(42)
                        permuted = base_colors[rng.permutation(len(categories))]
                        palette = {cat: permuted[i] for i, cat in enumerate(categories)}

                sc.pl.embedding(
                    mdata_train["rna"],
                    basis=key_umap,
                    color=obs_key,
                    palette=palette,
                    show=False,
                    frameon=True,
                    legend_fontsize=10,
                    legend_loc="right margin",
                    ax=ax,
                )

                ax.set_xlabel("UMAP1")
                ax.set_ylabel("UMAP2")
                plt.title(f"SpliceVI $Z_{{{name.capitalize()}}}$")
                plt.tight_layout()

                safe_obs = re.sub(r"[^A-Za-z0-9]+", "_", obs_key)
                out_path = os.path.join(
                    args.fig_dir, f"train_umap_{name}_{safe_obs}.png"
                )
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"[EVAL/UMAP] Saved UMAP to {out_path}")
                if run is not None:
                    wandb.log({f"umap/train_{name}_{safe_obs}": wandb.Image(out_path)})

                plt.close(fig)

        print("[EVAL/UMAP] All TRAIN UMAPs complete.")
    else:
        print("[EVAL/UMAP] UMAP evaluation skipped by config.")

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
        )
    elif "cross_fold_classification" in EVALS:
        print("[CROSS-FOLD] TRAIN split disabled by --cross_fold_splits.")

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

    latent_spaces_test = {}
    if ("test_eval" in EVALS) or (
        "cross_fold_classification" in EVALS and run_crossfold_test
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
                    # Ground truth = junc_ratio_original (pre-resampling values)
                    # Evaluate only where junc_ratio_original > 0
                    junc_ratio_orig = ad_masked.layers["junc_ratio_original"]
                    if not sparse.isspmatrix_csr(junc_ratio_orig):
                        junc_ratio_orig = sparse.csr_matrix(junc_ratio_orig)
                    # Build binary eval mask: positions where 0 < junc_ratio_original < 1
                    # (exclude unobserved = 0, and trivially certain = 1)
                    # Build eval mask directly from .data to avoid densification.
                    # junc_ratio_orig is CSR; .data contains only stored non-zeros.
                    # gt_zero (> 0) is always sparse-safe; lt_one (< 1) is NOT
                    # (implicit zeros also satisfy < 1, forcing a dense result).
                    n_nonzero = int(junc_ratio_orig.nnz)
                    n_eq_one  = int((junc_ratio_orig.data == 1.0).sum())
                    print(
                        f"[EVAL/IMPUTE/{tag}] junc_ratio_original: "
                        f"{n_nonzero} stored non-zero entries "
                        f"({n_eq_one} with PSI == 1.0, "
                        f"{n_nonzero - n_eq_one} with 0 < PSI < 1.0)."
                    )

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
                    if not sparse.isspmatrix_csr(bin_mask):
                        bin_mask = sparse.csr_matrix(bin_mask)
                    bin_mask.eliminate_zeros()
                    masked_orig = junc_ratio_orig
                    

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
                            (count_ok_data.astype(np.float32),
                             cc_orig.indices,
                             cc_orig.indptr),
                            shape=cc_orig.shape,
                        )
                        before_count_filter = bin_mask.nnz
                        bin_mask = bin_mask.multiply(count_ok)
                        if not sparse.isspmatrix_csr(bin_mask):
                            bin_mask = sparse.csr_matrix(bin_mask)
                        print(
                            f"[EVAL/IMPUTE/{tag}] After ATSE count filter (>= {args.min_atse_count}): "
                            f"{bin_mask.nnz} entries remain "
                            f"({before_count_filter - bin_mask.nnz} removed)."
                        )

                    print(
                        f"[EVAL/IMPUTE/{tag}] Final eval set: {bin_mask.nnz} entries."
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
                print(f"[EVAL/IMPUTE/{tag}] Running get_normalized_splicing over all {n_cells} cells (internal batch_size={bs})...", flush=True)

                with torch.inference_mode():
                    all_preds = model.get_normalized_splicing(
                        adata=mdata_masked,
                        return_numpy=True,
                        batch_size=bs,
                    )  # (n_cells, n_junc)

                print(f"[EVAL/IMPUTE/{tag}] Extracting eval pairs from mask...", flush=True)
                eval_rows, eval_cols = bin_mask.nonzero()
                pairs_total = eval_rows.size

                if pairs_total == 0:
                    print(
                        f"[EVAL/IMPUTE/{tag}] No eval entries found; skipping correlation."
                    )
                else:
                    # Extract values at eval positions without densifying
                    if sparse.issparse(masked_orig):
                        orig_all = np.asarray(masked_orig[eval_rows, eval_cols]).ravel().astype(np.float32)
                    else:
                        orig_all = masked_orig[eval_rows, eval_cols].astype(np.float32)
                    pred_all = all_preds[eval_rows, eval_cols].astype(np.float32)
                    del all_preds
                    torch.cuda.empty_cache()

                    pearson_m = float(np.corrcoef(orig_all, pred_all)[0, 1])
                    spearman_m = float(
                        spearmanr(orig_all, pred_all, nan_policy="omit")[0]
                    )

                    abs_diff = np.abs(orig_all - pred_all)
                    l1_mean = float(np.mean(abs_diff))
                    l1_median = float(np.median(abs_diff))
                    l1_p90 = float(np.quantile(abs_diff, 0.90))

                    pred_min = float(np.min(pred_all))
                    pred_max = float(np.max(pred_all))
                    smape = float(
                        np.mean(
                            2.0
                            * abs_diff
                            / (np.abs(orig_all) + np.abs(pred_all) + 1e-8)
                        )
                    )
                    orig64 = orig_all.astype(np.float64, copy=False)
                    pred64 = pred_all.astype(np.float64, copy=False)
                    denom = float(
                        np.linalg.norm(orig64) * np.linalg.norm(pred64) + 1e-8
                    )
                    cosine_sim = float(np.dot(orig64, pred64) / denom)
                    minmax_ratio = float(
                        np.mean(
                            np.minimum(np.abs(orig_all), np.abs(pred_all))
                            / (np.maximum(np.abs(orig_all), np.abs(pred_all)) + 1e-8)
                        )
                    )
                    rmse = float(np.sqrt(np.mean((orig_all - pred_all) ** 2)))

                    print(
                        f"[EVAL/IMPUTE/{tag}] PSI corr — "
                        f"Pearson: {pearson_m:.4f}, Spearman: {spearman_m:.4f}  (n={pairs_total})"
                    )
                    print(
                        f"[EVAL/IMPUTE/{tag}] PSI L1 — "
                        f"mean: {l1_mean:.4e}, median: {l1_median:.4e}, p90: {l1_p90:.4e}"
                    )
                    print(
                        f"[EVAL/IMPUTE/{tag}] PSI range — min: {pred_min:.4e}, max: {pred_max:.4e}"
                    )
                    print(
                        f"[EVAL/IMPUTE/{tag}] PSI SMAPE: {smape:.4e}, "
                        f"cosine: {cosine_sim:.4f}, minmax_ratio: {minmax_ratio:.4f}, "
                        f"RMSE: {rmse:.4e}"
                    )

                    if run is not None:
                        wandb.log(
                            {
                                f"impute-test/{tag}/psi_pearson_corr_masked_atse": pearson_m,
                                f"impute-test/{tag}/psi_spearman_corr_masked_atse": spearman_m,
                                f"impute-test/{tag}/psi_l1_mean_masked_atse": l1_mean,
                                f"impute-test/{tag}/psi_l1_median_masked_atse": l1_median,
                                f"impute-test/{tag}/psi_l1_p90_masked_atse": l1_p90,
                                f"impute-test/{tag}/psi_pred_min_masked_atse": pred_min,
                                f"impute-test/{tag}/psi_pred_max_masked_atse": pred_max,
                                f"impute-test/{tag}/psi_smape_masked_atse": smape,
                                f"impute-test/{tag}/psi_cosine_sim_masked_atse": cosine_sim,
                                f"impute-test/{tag}/psi_minmax_ratio_masked_atse": minmax_ratio,
                                f"impute-test/{tag}/psi_rmse_masked_atse": rmse,
                                f"impute-test/{tag}/n_masked_entries": int(pairs_total),
                                f"impute-test/{tag}/impute_batch_size": bs,
                                f"impute-test/{tag}/masked_file": masked_path,
                            }
                        )

                print(f"[EVAL/IMPUTE/{tag}] Cleaning up masked MuData from memory...")
                del mdata_masked, ad_masked, masked_orig, bin_mask, eval_rows, eval_cols
                # these only exist when pairs_total > 0
                try: del orig_all
                except NameError: pass
                try: del pred_all
                except NameError: pass
                try: del orig64
                except NameError: pass
                try: del pred64
                except NameError: pass
                try: del abs_diff
                except NameError: pass
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
