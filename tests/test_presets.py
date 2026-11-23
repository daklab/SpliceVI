"""Tests for splicevi.presets: the presets equal the recorded settings of the paper models, and the class defaults equal
the "paper" preset. Run: pytest tests/test_presets.py

The recorded settings (tests/fixtures/paper_model_settings_{tms,allen}.json) are the init parameters stored in the first
TMS and Allen replicate checkpoints and the full training configuration of those runs. The tests that read the
checkpoints themselves run only when SPLICEVI_MODELS points to a folder holding them, and are skipped otherwise.
"""
import inspect
import json
import os

import pandas as pd
import pytest
import torch

from splicevi import SPLICEVI, filter_read_support
from splicevi.splicevae import SPLICEVAE

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("SPLICEVI_MODELS")   # folder with the paper checkpoints (optional)

RECORDED = {atlas: json.load(open(os.path.join(HERE, "fixtures", f"paper_model_settings_{atlas}.json")))
            for atlas in ("tms", "allen")}

# Recorded training options that SPLICEVI no longer has. Each was set to the value that switches it off, so leaving it
# out changes nothing.
REMOVED_OPTIONS_AT_OFF_VALUE = {
    "adversary_input": "z", "aux_ratio_weight": 0.0, "distill_only": False, "distill_weight": 0.0,
    "early_stopping_monitor": "reconstruction_loss_validation", "embedding_init_permute": False, "event_dropout": 0.0,
    "free_bits": 0.0, "keep_best": "", "n_splicing_private": 0, "phi_lr_mult": 1.0, "precision_weight": "atse_total",
    "restore_best": "", "splicing_decoder_batch": True, "splicing_mix_weight": None, "splicing_refine_lr": 0.05,
    "splicing_refine_steps": 0, "stop_gradient_spl_to_expr": False,
}

# Recorded entries that describe the run (paths, seed, data sizes, code version), not model or training settings.
RUN_DESCRIPTION_KEYS = {"train_mdata_path", "model_dir", "seed", "n_genes", "n_junctions", "git_sha", "git_branch",
                        "git_dirty", "splicevi_commit", "splicevi_file", "experiment_dir", "help"}

# Folder names of the paper checkpoints read by the optional tests.
TMS_CHECKPOINT = "pipeline_20260927_231518_s5-std-devp_rep1"
ALLEN_CHECKPOINT = "pipeline_20261001_012113_s5-ab-devp_rep1"
DOWNSAMPLED_ALLEN_CHECKPOINT = "allen_thinned_v2in80_devp_rep1"


def _parameter_names(function):
    return set(inspect.signature(function).parameters)


def _defaults(function):
    return {name: parameter.default for name, parameter in inspect.signature(function).parameters.items()}


# ----------------------------------------------------------------------------------------------------------------------
# Presets against the recorded settings.
# ----------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("atlas", ["tms", "allen"])
def test_paper_preset_equals_record(atlas):
    settings, recorded = SPLICEVI.preset("paper", atlas=atlas), RECORDED[atlas]
    init_names, train_names = _parameter_names(SPLICEVI.__init__), _parameter_names(SPLICEVI.train)

    for name, value in settings["init"].items():
        assert name in init_names, name
        if name != "psi_prior_pseudocount":   # not recorded: these models predate the option, which was off
            assert recorded["checkpoint_init"][name] == value, name
    assert settings["init"]["psi_prior_pseudocount"] == 0.0

    for name, value in settings["train"].items():
        assert name in train_names and recorded["train_config"][name] == value, name
    assert settings["setup"]["batch_key"] == recorded["setup_args"]["batch_key"]


@pytest.mark.parametrize("atlas", ["tms", "allen"])
def test_every_recorded_option_is_accounted_for(atlas):
    settings = SPLICEVI.preset("paper", atlas=atlas)
    covered = set(settings["init"]) | set(settings["train"]) | set(settings["setup"])
    not_covered = {name: value for name, value in RECORDED[atlas]["train_config"].items()
                   if name not in covered and not name.startswith("wandb") and name not in RUN_DESCRIPTION_KEYS}
    for name, value in not_covered.items():
        assert name in REMOVED_OPTIONS_AT_OFF_VALUE and str(REMOVED_OPTIONS_AT_OFF_VALUE[name]) == str(value), (name, value)


# ----------------------------------------------------------------------------------------------------------------------
# Relations between the presets.
# ----------------------------------------------------------------------------------------------------------------------

def test_default_atlas_is_allen():
    assert SPLICEVI.preset("paper") == SPLICEVI.preset("paper", atlas="allen")


def test_tms_and_allen_differ_only_in_batch_and_adversary():
    tms, allen = SPLICEVI.preset("paper", atlas="tms"), SPLICEVI.preset("paper", atlas="allen")
    assert tms["init"] == allen["init"]
    assert tms["setup"]["batch_key"] == "mouse.id" and allen["setup"]["batch_key"] is None
    assert {name for name in tms["train"] if tms["train"][name] != allen["train"][name]} == {"adversarial_mixing"}


def test_sparse_equals_paper_without_joint():
    """Neither preset uses the PSI prior (psi_prior_pseudocount = 0)."""
    paper, sparse = SPLICEVI.preset("paper"), SPLICEVI.preset("sparse")
    assert paper["init"] == sparse["init"] and sparse["init"]["psi_prior_pseudocount"] == 0.0
    assert paper["train"] == sparse["train"] and paper["setup"] == sparse["setup"]

    joint = SPLICEVI.preset("sparse", joint=True)
    assert joint["init"]["dispersion"] == "gene-batch" and joint["setup"]["batch_key"] == "technology"


def test_analysis_settings():
    settings = SPLICEVI.preset("paper")
    assert settings["ds"]["norm_splicing_function"] == "dm_posterior_sample"
    assert settings["ds_filter"]["min_units"] == 3
    assert settings["de"]["norm_expression_function"] == "zinb_expected"

    table = pd.DataFrame(dict(n_obs_group1=[3, 2, 9], n_obs_group2=[3, 9, 2]))
    assert len(filter_read_support(table)) == 1


# ----------------------------------------------------------------------------------------------------------------------
# Class defaults.
# ----------------------------------------------------------------------------------------------------------------------

def test_class_defaults_are_the_paper_settings():
    """Every model, training, DS and DE default equals the "paper" preset.

    The one exception is adversarial_mixing: the default is True, which scvi-tools switches off when the data has a single
    batch (the Allen setting, False in the preset) and keeps on with a batch key (the TMS setting, True)."""
    settings = SPLICEVI.preset("paper")
    model_defaults, train_defaults = _defaults(SPLICEVI.__init__), _defaults(SPLICEVI.train)

    for name, value in settings["init"].items():
        assert model_defaults[name] == value, name
    for name, value in settings["train"].items():
        if name != "adversarial_mixing":
            assert train_defaults[name] == value, name
    assert train_defaults["adversarial_mixing"] is True
    assert SPLICEVI.preset("paper", atlas="tms")["train"]["adversarial_mixing"] is True

    ds_defaults, de_defaults = _defaults(SPLICEVI.differential_splicing), _defaults(SPLICEVI.differential_expression)
    for name, value in settings["ds"].items():
        assert ds_defaults[name] == value, name
    for name, value in settings["de"].items():
        assert de_defaults[name] == value, name


def test_module_defaults_match_the_model_defaults():
    """SPLICEVAE's defaults equal SPLICEVI's for every setting the two share (the module is built with all of them passed)."""
    model_defaults, module_defaults = _defaults(SPLICEVI.__init__), _defaults(SPLICEVAE.__init__)
    renamed = {"dispersion": "gene_dispersion"}
    for name, value in model_defaults.items():
        module_name = renamed.get(name, name)
        if module_name in module_defaults and name not in ("n_genes", "n_junctions"):
            assert module_defaults[module_name] == value, name


# ----------------------------------------------------------------------------------------------------------------------
# Optional: the presets against the paper checkpoints themselves (needs SPLICEVI_MODELS).
# ----------------------------------------------------------------------------------------------------------------------

def _stored_checkpoint(folder_name):
    """The attribute dictionary saved in a checkpoint's model.pt; skips the test when the checkpoint is not available."""
    path = os.path.join(MODELS, folder_name, "model.pt") if MODELS else None
    if path is None or not os.path.exists(path):
        pytest.skip("checkpoint not available (set SPLICEVI_MODELS to the folder with the paper checkpoints)")
    return torch.load(path, map_location="cpu", weights_only=False)["attr_dict"]


@pytest.mark.parametrize("atlas,folder_name", [("tms", TMS_CHECKPOINT), ("allen", ALLEN_CHECKPOINT)])
def test_paper_preset_equals_checkpoint(atlas, folder_name):
    stored = _stored_checkpoint(folder_name)
    stored_init = stored["init_params_"]["non_kwargs"]
    settings = SPLICEVI.preset("paper", atlas=atlas)
    for name, value in settings["init"].items():
        # psi_prior_pseudocount is not stored by these models (it was added later, off by default)
        stored_value = stored_init.get(name, 0.0 if name == "psi_prior_pseudocount" else object())
        assert stored_value == value, name
    assert stored["registry_"]["setup_args"]["batch_key"] == settings["setup"]["batch_key"]


def test_sparse_preset_equals_downsampled_allen_checkpoint():
    """The model trained on Allen data downsampled to EasySci-like sparsity."""
    stored_init = _stored_checkpoint(DOWNSAMPLED_ALLEN_CHECKPOINT)["init_params_"]["non_kwargs"]
    for name, value in SPLICEVI.preset("sparse")["init"].items():
        assert stored_init[name] == value, name
