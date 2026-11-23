"""Tests for splicevi.presets: the presets equal the stored records of the paper models, and the class defaults equal the
"paper" preset. Run: pytest tests/test_presets.py"""
import inspect
import json
import os

import pandas as pd
import pytest
import torch

from splicevi import SPLICEVI, filter_read_support

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("SPLICEVI_MODELS", "/gpfs/commons/home/svaidyanathan/repos/SpliceVI_Repositories/SpliceVI/models")
FIX = {a: json.load(open(f"{HERE}/fixtures/iprime_{a}_rep1.json")) for a in ("tms", "allen")}
# recorded options that are absent from main; each has the value that means "off", so leaving them out changes nothing on main
OFF = {"adversary_input": "z", "aux_ratio_weight": 0.0, "distill_only": False, "distill_weight": 0.0, "early_stopping_monitor": "reconstruction_loss_validation",
       "embedding_init_permute": False, "event_dropout": 0.0, "free_bits": 0.0, "keep_best": "", "n_splicing_private": 0, "phi_lr_mult": 1.0, "precision_weight": "atse_total",
       "restore_best": "", "splicing_decoder_batch": True, "splicing_mix_weight": None, "splicing_refine_lr": 0.05, "splicing_refine_steps": 0, "stop_gradient_spl_to_expr": False}
NOT_RECIPE = {"train_mdata_path", "model_dir", "seed", "n_genes", "n_junctions", "git_sha", "git_branch", "git_dirty", "splicevi_commit", "splicevi_file", "experiment_dir", "help"}
INIT, TRAIN = [set(inspect.signature(f).parameters) for f in (SPLICEVI.__init__, SPLICEVI.train)]


@pytest.mark.parametrize("atlas", ["tms", "allen"])
def test_paper_preset_equals_record(atlas):
    cfg, fx = SPLICEVI.preset("paper", atlas=atlas), FIX[atlas]
    for k, v in cfg["init"].items():
        assert k in INIT, k
        if k != "psi_prior_pseudocount":
            assert fx["checkpoint_init"][k] == v, k
    assert cfg["init"]["psi_prior_pseudocount"] == 0.0
    for k, v in cfg["train"].items():
        assert k in TRAIN and fx["train_config"][k] == v, k
    assert cfg["setup"]["batch_key"] == fx["setup_args"]["batch_key"]


@pytest.mark.parametrize("atlas", ["tms", "allen"])
def test_every_recorded_option_is_accounted_for(atlas):
    cfg = SPLICEVI.preset("paper", atlas=atlas); covered = set(cfg["init"]) | set(cfg["train"]) | set(cfg["setup"])
    left = {k: v for k, v in FIX[atlas]["train_config"].items() if k not in covered and not k.startswith("wandb") and k not in NOT_RECIPE}
    for k, v in left.items():
        assert k in OFF and str(OFF[k]) == str(v), (k, v)


def test_default_is_the_allen_recipe():
    assert SPLICEVI.preset("paper") == SPLICEVI.preset("paper", atlas="allen")


def test_tms_and_allen_differ_only_in_batch_and_adversary():
    t, a = SPLICEVI.preset("paper", atlas="tms"), SPLICEVI.preset("paper", atlas="allen")
    assert t["init"] == a["init"] and t["setup"]["batch_key"] == "mouse.id" and a["setup"]["batch_key"] is None
    assert {k for k in t["train"] if t["train"][k] != a["train"][k]} == {"adversarial_mixing"}


def test_sparse_equals_paper_without_joint():
    """Decision 2026-11-06: no psi prior in either preset."""
    p, s = SPLICEVI.preset("paper"), SPLICEVI.preset("sparse")
    assert p["init"] == s["init"] and s["init"]["psi_prior_pseudocount"] == 0.0
    assert p["train"] == s["train"] and p["setup"] == s["setup"]
    j = SPLICEVI.preset("sparse", joint=True); assert j["init"]["dispersion"] == "gene-batch" and j["setup"]["batch_key"] == "technology"


def test_analysis_settings():
    c = SPLICEVI.preset("paper")
    assert c["ds"]["norm_splicing_function"] == "dm_posterior_sample" and c["ds_filter"]["min_units"] == 3 and c["de"]["norm_expression_function"] == "zinb_expected"
    d = pd.DataFrame(dict(n_obs_group1=[3, 2, 9], n_obs_group2=[3, 9, 2])); assert len(filter_read_support(d)) == 1


def _defaults(function):
    return {name: parameter.default for name, parameter in inspect.signature(function).parameters.items()}


def test_class_defaults_are_the_paper_settings():
    """Every model, training, DS and DE default equals the "paper" preset.

    The one exception is adversarial_mixing: the default is True, which scvi-tools switches off when the data has a single
    batch (the Allen setting, False in the preset) and keeps on with a batch key (the TMS setting, True)."""
    cfg = SPLICEVI.preset("paper")
    model_defaults, train_defaults = _defaults(SPLICEVI.__init__), _defaults(SPLICEVI.train)
    for name, value in cfg["init"].items():
        assert model_defaults[name] == value, name
    for name, value in cfg["train"].items():
        if name != "adversarial_mixing":
            assert train_defaults[name] == value, name
    assert train_defaults["adversarial_mixing"] is True and SPLICEVI.preset("paper", atlas="tms")["train"]["adversarial_mixing"] is True
    ds_defaults, de_defaults = _defaults(SPLICEVI.differential_splicing), _defaults(SPLICEVI.differential_expression)
    for name, value in cfg["ds"].items():
        assert ds_defaults[name] == value, name
    for name, value in cfg["de"].items():
        assert de_defaults[name] == value, name


def test_module_defaults_match_the_model_defaults():
    """SPLICEVAE's defaults equal SPLICEVI's for every setting the two share (the module is built with all of them passed)."""
    from splicevi.splicevae import SPLICEVAE
    model_defaults, module_defaults = _defaults(SPLICEVI.__init__), _defaults(SPLICEVAE.__init__)
    renamed = {"dispersion": "gene_dispersion"}
    for name, value in model_defaults.items():
        module_name = renamed.get(name, name)
        if module_name in module_defaults and name not in ("n_genes", "n_junctions"):
            assert module_defaults[module_name] == value, name


@pytest.mark.parametrize("atlas,name,kappa", [("tms", "pipeline_20260927_231518_s5-std-devp_rep1", 0.0), ("allen", "pipeline_20261001_012113_s5-ab-devp_rep1", 0.0)])
def test_paper_preset_equals_live_checkpoint(atlas, name, kappa):
    p = f"{MODELS}/{name}/model.pt"
    if not os.path.exists(p):
        pytest.skip("checkpoint not available")
    a = torch.load(p, map_location="cpu", weights_only=False)["attr_dict"]; ck = a["init_params_"]["non_kwargs"]; cfg = SPLICEVI.preset("paper", atlas=atlas)
    for k, v in cfg["init"].items():
        assert ck.get(k, kappa if k == "psi_prior_pseudocount" else object()) == v, k
    assert a["registry_"]["setup_args"]["batch_key"] == cfg["setup"]["batch_key"]


def test_sparse_preset_equals_thinned_checkpoint():
    """The in80 thinning model (kappa 0), Smriti's v2in80 run."""
    p = f"{MODELS}/allen_thinned_v2in80_devp_rep1/model.pt"
    if not os.path.exists(p):
        pytest.skip("checkpoint not available")
    ck = torch.load(p, map_location="cpu", weights_only=False)["attr_dict"]["init_params_"]["non_kwargs"]
    for k, v in SPLICEVI.preset("sparse")["init"].items():
        assert ck[k] == v, k
