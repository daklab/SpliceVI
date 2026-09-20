#!/usr/bin/env python
"""
train_multivisplice_basic.py

Training pipeline for SPLICEVI:

1. Load MuData from disk
2. Set up SPLICEVI with standard layer configuration
3. Initialize the model with configurable hyperparameters
4. Train the model
5. Save the trained model to a user-specified output directory

This script is designed to be:
- Safe to run on an HPC cluster or locally
- Driven by a shell wrapper (e.g., a Slurm job script)
"""

import os
import json
import inspect
import argparse
from datetime import datetime, timezone

import scanpy as sc
from splicevi import SPLICEVI
import mudata as mu
import numpy as np
import torch


def maybe_import_wandb():
    """Import wandb + WandbLogger if available; otherwise return (None, None)."""
    try:
        import wandb
        from pytorch_lightning.loggers import WandbLogger
    except ImportError:
        wandb = None
        WandbLogger = None
    return wandb, WandbLogger


def str2bool(v):
    """Parse common string/int representations into bools for argparse."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, str):
        v_lower = v.lower()
        if v_lower in {"yes", "true", "t", "y", "1"}:
            return True
        if v_lower in {"no", "false", "f", "n", "0", "none", ""}:
            return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def write_experiment_fragment(experiment_dir, model_dir, full_config, run):
    """Write a train_<model_basename>.json fragment for the experiment-tracking
    leaderboard (see SpliceVI-utils/script_outputs/experiments/tally_experiments.py).
    Best-effort: a failure here must never fail the training job itself.
    """
    try:
        os.makedirs(experiment_dir, exist_ok=True)
        model_basename = os.path.basename(os.path.normpath(model_dir))
        fragment = {
            "model_dir": model_dir,
            "model_kind": "splicevi",
            "config": full_config,
            "wandb_run_url": getattr(run, "url", None) if run is not None else None,
            "wandb_run_id": getattr(run, "id", None) if run is not None else None,
            "wandb_project": getattr(run, "project", None) if run is not None else None,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        out_path = os.path.join(experiment_dir, f"train_{model_basename}.json")
        with open(out_path, "w") as f:
            json.dump(fragment, f, indent=2, default=str)
        print(f"[EXPERIMENT] Wrote {out_path}")
    except Exception as e:  # noqa: BLE001
        print(f"[EXPERIMENT] WARNING: failed to write experiment fragment: {e}")


def build_argparser(init_defaults, train_defaults):
    parser = argparse.ArgumentParser(
        "train_multivisplice_basic",
        description=(
            "Minimal trainer for SPLICEVI: "
            "loads a MuData file, sets up the model, trains, and saves the model."
        ),
    )

    # Core paths
    parser.add_argument(
        "--train_mdata_path",
        type=str,
        required=True,
        help="Path to training MuData (.h5mu).",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Output directory to save the trained model.",
    )
    parser.add_argument(
        "--batch_key",
        type=str,
        default="None",
        help="Optional obs column used as batch_key in setup_mudata. Use 'None' to disable.",
    )

    # Optional: W&B integration
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
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
        help="How often (in training steps) to log model parameters/gradients.",
    )

    # Optional: experiment-tracking leaderboard hook (see SpliceVI-utils/script_outputs/experiments/)
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help=(
            "If set, write a train_<model_basename>.json fragment into this directory "
            "with the resolved config and W&B run link, for the experiment-tracking leaderboard."
        ),
    )

    # Optional: override SPLICEVI __init__ arguments
    for name, default in init_defaults.items():
        if isinstance(default, bool):
            arg_type = str2bool
        elif default is not None:
            arg_type = type(default)
        else:
            arg_type = float
        parser.add_argument(
            f"--{name}",
            type=arg_type,
            default=None,
            help=f"(model init) {name} (default = {default!r})",
        )

    # Optional: override SPLICEVI.train arguments
    for name, default in train_defaults.items():
        if isinstance(default, bool):
            arg_type = str2bool
        elif default is not None:
            arg_type = type(default)
        else:
            arg_type = float
        parser.add_argument(
            f"--{name}",
            type=arg_type,
            default=None,
            help=f"(training) {name} (default = {default!r})",
        )

    return parser


def main():
    # ------------------------------
    # 1. Introspect SPLICEVI signatures
    # ------------------------------
    train_sig = inspect.signature(SPLICEVI.train)
    train_defaults = {
        name: param.default
        for name, param in train_sig.parameters.items()
        if name != "self" and param.default is not inspect._empty
    }

    init_sig = inspect.signature(SPLICEVI.__init__)
    init_defaults = {
        name: param.default
        for name, param in init_sig.parameters.items()
        if name not in ("self", "adata") and param.default is not inspect._empty
    }

    # ------------------------------
    # 2. Parse CLI
    # ------------------------------
    parser = build_argparser(init_defaults, train_defaults)
    args = parser.parse_args()

    # Ensure output dir exists
    os.makedirs(args.model_dir, exist_ok=True)

    print("=" * 80)
    print("[SETUP] SPLICEVI basic training")
    print(f"[SETUP] Training MuData path : {args.train_mdata_path}")
    print(f"[SETUP] Model output path    : {args.model_dir}")
    print(f"[SETUP] W&B enabled          : {args.use_wandb}")
    print("=" * 80)

    # ------------------------------
    # 3. Load MuData
    # ------------------------------
    print("[DATA] Loading MuData from disk...")
    mdata = mu.read_h5mu(args.train_mdata_path, backed="r")
    mdata.obs.rename(columns={"donor_id": "mouse.id"}, inplace=True)
    mdata.mod["rna"].obs.rename(
    columns={"donor_id": "mouse.id"},
    inplace=True)
    mdata.mod["splicing"].obs.rename(
    columns={"donor_id": "mouse.id"},
    inplace=True)
    if "rna" not in mdata.mod:
        raise ValueError("[DATA] Expected 'rna' modality in MuData.")
    if "splicing" not in mdata.mod:
        raise ValueError("[DATA] Expected 'splicing' modality in MuData.")
    print("[DATA] MuData loaded successfully.")
    print(f"[DATA] # cells (rna)      : {mdata['rna'].n_obs}")
    print(f"[DATA] # features (rna)   : {mdata['rna'].n_vars}")
    print(f"[DATA] # features (splice): {mdata['splicing'].n_vars}")

    # Fixed layer configuration (same as your main script)
    x_layer = "junc_ratio"
    junction_counts_layer = "cell_by_junction_matrix"
    cluster_counts_layer = "cell_by_cluster_matrix"
    mask_layer = "psi_mask"

    print("[DATA] Checking expected layers in 'splicing' modality...")
    print(f"[DATA] Available layers: {list(mdata['splicing'].layers.keys())}")

    for layer in [x_layer, junction_counts_layer, cluster_counts_layer, mask_layer]:
        if layer not in mdata["splicing"].layers:
            raise ValueError(f"[DATA] Required layer '{layer}' not found in mdata['splicing'].layers")

    # ------------------------------
    # 4. Setup SPLICEVI
    # ------------------------------
    batch_key = None if (args.batch_key is None or str(args.batch_key).lower() == "none") else args.batch_key
    print(f"[MODEL] Using batch_key      : {batch_key}")
    print("[MODEL] Setting up SPLICEVI with standard MuData configuration...")
    SPLICEVI.setup_mudata(
        mdata,
        batch_key=batch_key,
        size_factor_key="X_library_size",
        rna_layer="length_norm",
        junc_ratio_layer=x_layer,
        atse_counts_layer=cluster_counts_layer,
        junc_counts_layer=junction_counts_layer,
        psi_mask_layer=mask_layer,
        modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
    )
    print("[MODEL] AnnData setup complete.")

    # ------------------------------
    # 5. Initialize model
    # ------------------------------
    # Params typed ``int | None`` (e.g. n_latent, n_hidden) default to None, so the CLI parses them
    # as float; cast whole-number values back to int (nn.Linear rejects 128.0).
    for name in ("n_latent", "n_hidden"):
        value = getattr(args, name, None)
        if value is not None:
            if float(value) != int(value):
                raise ValueError(f"--{name} must be a whole number, got {value}")
            setattr(args, name, int(value))

    model_kwargs = {
        name: getattr(args, name)
        for name in init_defaults
        if getattr(args, name) is not None
    }

    # Derive counts for n_genes / n_junctions from var["modality"] fields
    n_genes = int((mdata["rna"].var["modality"] == "Gene_Expression").sum())
    n_junctions = int((mdata["splicing"].var["modality"] == "Splicing").sum())

    print("[MODEL] Initializing SPLICEVI with:")
    print(f"        n_genes     = {n_genes}")
    print(f"        n_junctions = {n_junctions}")
    if model_kwargs:
        print("[MODEL] Additional init kwargs:")
        for k, v in model_kwargs.items():
            print(f"        {k}: {v}")
    else:
        print("[MODEL] Using all default init parameters.")

    model = SPLICEVI(
        mdata,
        n_genes=n_genes,
        n_junctions=n_junctions,
        **model_kwargs,
    )
    model.view_anndata_setup()

    # ------------------------------
    # 6. Optional W&B setup
    # ------------------------------
    wandb, WandbLogger = maybe_import_wandb()
    wandb_logger = None

    # Build unified config for potential logging
    full_config = {**init_defaults, **train_defaults}
    # Overwrite with CLI overrides
    for key in list(full_config):
        if hasattr(args, key):
            val = getattr(args, key)
            if val is not None:
                full_config[key] = val
    full_config.update(
        {
            "train_mdata_path": args.train_mdata_path,
            "model_dir": args.model_dir,
        }
    )

    if args.use_wandb:
        if wandb is None or WandbLogger is None:
            raise ImportError(
                "[W&B] --use_wandb was set but wandb / pytorch-lightning not installed in this environment."
            )
        if args.wandb_project is None:
            raise ValueError("[W&B] --wandb_project is required when --use_wandb is set.")

        print("[W&B] Initializing Weights & Biases run...")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            config=full_config,
        )
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            group=args.wandb_group,
            config=full_config,
        )

        # Watch model
        wandb.watch(
            model.module,
            log="all",
            log_freq=args.wandb_log_freq,
            log_graph=False,
        )

        # Log parameter count
        total_params = sum(p.numel() for p in model.module.parameters())
        print(f"[MODEL] Total model parameters: {total_params:,}")
        wandb.log({"total_parameters": total_params})
    else:
        run = None
        print("[W&B] Disabled. No W&B logging will be performed.")

    # ------------------------------
    # 7. Train
    # ------------------------------
    train_kwargs = {
        name: getattr(args, name)
        for name in train_defaults
        if getattr(args, name) is not None
    }

    print("[TRAIN] Starting training with the following overrides:")
    if train_kwargs:
        for k, v in train_kwargs.items():
            print(f"        {k}: {v}")
    else:
        print("        (no overrides; using all training defaults)")

    if wandb_logger is not None:
        model.train(logger=wandb_logger, **train_kwargs)
    else:
        model.train(**train_kwargs)

    # ------------------------------
    # 8. Save model
    # ------------------------------
    print(f"[SAVE] Saving trained model to: {args.model_dir}")
    model.save(args.model_dir, overwrite=True)
    print("[SAVE] Model saved successfully.")

    if args.experiment_dir:
        write_experiment_fragment(args.experiment_dir, args.model_dir, full_config, run)

    if run is not None:
        print("[W&B] Finishing W&B run.")
        run.finish()

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    print("[DONE] Training pipeline complete.")


if __name__ == "__main__":
    main()
