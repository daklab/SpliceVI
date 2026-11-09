"""Named recipes for SpliceVI: the settings behind the paper's models, kept apart from the class defaults, which do not change.

    cfg = SPLICEVI.preset("paper", atlas="allen")
    SPLICEVI.setup_mudata(mdata, batch_key=cfg["setup"]["batch_key"], ...)
    model = SPLICEVI(mdata, n_genes=mdata["rna"].n_vars, n_junctions=mdata["splicing"].n_vars, **cfg["init"]); model.train(**cfg["train"])
    ds = filter_read_support(model.differential_splicing(adata=mdata, groupby=..., group1=..., group2=..., **cfg["ds"]), **cfg["ds_filter"])
    de = model.differential_expression(adata=mdata, groupby=..., group1=..., group2=..., **cfg["de"])

"paper" is the full-depth recipe. "sparse" is the same recipe for EasySci-like depth; it differs only when joint=True.
Both presets use psi_prior_pseudocount 0 (decision 2026-11-06: the psi prior stays in the code as an option, not in a preset).
Values come from the stored records of the rep-1 checkpoints; tests/test_presets.py checks them against those records.
"""
import copy

_INIT = {'modality_weights': 'equal',
 'modality_penalty': 'None',
 'variance_mixing': 'squared',
 'mask_cells_without_splicing': False,
 'n_hidden': None,
 'n_latent': 30,
 'n_layers_encoder': 2,
 'n_layers_decoder': 2,
 'dropout_rate': 0.01,
 'use_batch_norm': 'none',
 'use_layer_norm': 'both',
 'latent_distribution': 'normal',
 'deeply_inject_covariates': False,
 'encode_covariates': False,
 'gene_likelihood': 'zinb',
 'dispersion': 'gene',
 'splicing_loss_type': 'dirichlet_multinomial',
 'dm_concentration': 'atse',
 'splicing_loss_weight': 1.0,
 'lambda_prior': 0.01,
 'phi_floor': 0.0,
 'phi_prior': 'l2_log',
 'phi_prior_shape': 2.0,
 'phi_prior_rate': 0.1,
 'phi_init': 'constant',
 'splicing_encoder_architecture': 'partial',
 'splicing_decoder_architecture': 'vanilla',
 'expression_architecture': 'vanilla',
 'encoder_hidden_dim': 128,
 'code_dim': 64,
 'h_hidden_dim': 128,
 'pool_mode': 'mean',
 'psi_input': 'deviation',
 'decoder_intercept_init': 'population',
 'decoder_depth_covariates': True,
 'max_nobs': -1,
 'initialize_embeddings_from_pca': True,
 'fully_paired': False,
 'psi_prior_pseudocount': 0.0}

_TRAIN = {'max_epochs': 300,
 'lr': 0.0001,
 'accelerator': 'auto',
 'devices': 'auto',
 'train_size': None,
 'validation_size': None,
 'shuffle_set_split': True,
 'batch_size': 256,
 'weight_decay': 0.0001,
 'eps': 1e-08,
 'early_stopping': True,
 'early_stopping_patience': 50,
 'check_val_every_n_epoch': None,
 'n_steps_kl_warmup': None,
 'n_epochs_kl_warmup': 200,
 'lr_scheduler_type': 'step',
 'reduce_lr_on_plateau': False,
 'lr_factor': 0.1,
 'lr_patience': 30,
 'lr_threshold': 0.0,
 'lr_scheduler_metric': 'elbo_validation',
 'step_size': 400,
 'gradient_clipping': True,
 'gradient_clipping_max_norm': 5.0,
 'disable_cross_gate': False,
 'datasplitter_kwargs': None,
 'plan_kwargs': None}

_DS = {"norm_splicing_function": "dm_posterior_sample", "delta": 0.1, "fdr_target": 0.05}
_DS_FILTER = {"min_units": 3}
_DE = {"norm_expression_function": "zinb_expected", "fdr_target": 0.05}


def preset(name, atlas=None, joint=False):
    """Return {"setup", "init", "train", "ds", "ds_filter", "de"} for a named recipe.

    name: "paper" (full depth) or "sparse" (EasySci-like depth). Both psi_prior_pseudocount 0; set it on the model to use the psi prior.
    atlas ("paper" only): None or "allen" gives the Allen recipe (no batch key, no adversarial classifier). "tms" gives the TMS recipe
        (batch key mouse.id, adversarial classifier on z). The two checkpoints share every other setting.
    joint ("sparse" only): True for a reference + sparse model on one object: batch key technology, dispersion gene-batch.
    """
    if name not in ("paper", "sparse"):
        raise ValueError(f"unknown preset {name!r}; use 'paper' or 'sparse'")
    if atlas not in (None, "allen", "tms") or (atlas is not None and name != "paper"):
        raise ValueError("atlas applies to the 'paper' preset and must be None, 'allen' or 'tms'")
    if joint and name != "sparse":
        raise ValueError("joint applies to the 'sparse' preset")
    init, train = copy.deepcopy(_INIT), copy.deepcopy(_TRAIN)
    train["adversarial_mixing"] = atlas == "tms"
    batch_key = "mouse.id" if atlas == "tms" else None
    if name == "sparse":
        if joint:
            init["dispersion"], batch_key = "gene-batch", "technology"
    return {"setup": {"batch_key": batch_key, "size_factor_key": "X_library_size"}, "init": init, "train": train, "ds": dict(_DS),
            "ds_filter": dict(_DS_FILTER), "de": dict(_DE)}


def filter_read_support(ds, min_units=3):
    """Keep junctions observed in at least min_units units of each group. Applied after calling, so the FDR rule itself is unchanged."""
    return ds[(ds["n_obs_group1"] >= min_units) & (ds["n_obs_group2"] >= min_units)]
