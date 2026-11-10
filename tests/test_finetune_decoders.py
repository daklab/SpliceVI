"""Tests for decoder covariate inputs and SPLICEVI.finetune_decoders, on the paper's Allen base model and a small slice
of the Allen replay set (skipped when those files are not available). Run: pytest tests/test_finetune_decoders.py

Checked: adding the covariate inputs leaves the model's output unchanged (zero weights); fine-tuning trains only the
decoders and the covariate weights; it fixes the depth inputs and keeps them through save / load; a saved fine-tuned
model reloads with identical decoder output; set_decoder_covariate / decoder_covariate switch query vs reference output;
models without covariate inputs load as before.
"""
import os

import numpy as np
import pytest
import torch

MODELS = os.environ.get("SPLICEVI_MODELS", "/gpfs/commons/home/svaidyanathan/repos/SpliceVI_Repositories/SpliceVI/models")
BASE = os.path.join(MODELS, "pipeline_20261001_012113_s5-ab-devp_rep1")   # the Allen deviation-input model (v9's base)
# 256 cells of the 20,000-cell Allen replay set (v9's reference replay), written once by build_test_subset() below
SUBSET = os.environ.get("SPLICEVI_FINETUNE_TEST_MDATA", "/gpfs/commons/home/svaidyanathan/repos/SpliceVI_Repositories/script_outputs/scratch_tests/finetune_test_subset_256.h5mu")
REPLAY = "/gpfs/commons/home/kisaev/SpliceVI-utils/manuscript/age_analysis/audit_2026_09/screen_evals/s134/allen_train_20k.h5mu"
N_QUERY = 96   # first 96 cells = "query", the other 160 = "reference"

pytestmark = pytest.mark.skipif(not (os.path.exists(os.path.join(BASE, "model.pt")) and os.path.exists(SUBSET)),
                                reason="base model or test subset not available")


def build_test_subset(out=SUBSET, n=256, seed=0):
    """Write n random cells of the Allen replay set to out (python -c "import test_finetune_decoders as t; t.build_test_subset()")."""
    import mudata as mu
    full = mu.read_h5mu(REPLAY)
    idx = np.sort(np.random.default_rng(seed).choice(full.n_obs, n, replace=False))
    full[idx].copy().write(out)


def _prep(mdata):
    """As finetune_v9.prep: obs copied to the rna modality, library size mirrored to obs."""
    for c in mdata.obs.columns:
        if c not in mdata["rna"].obs.columns:
            mdata["rna"].obs[c] = mdata.obs[c].values
    if "X_library_size" in mdata["rna"].obsm:
        mdata["rna"].obs["X_library_size"] = mdata["rna"].obsm["X_library_size"]
    return mdata


@pytest.fixture(scope="module")
def data():
    import mudata as mu
    sub = mu.read_h5mu(SUBSET)
    return _prep(sub[:N_QUERY].copy()), _prep(sub[N_QUERY:].copy())


def _load(adata):
    from splicevi import SPLICEVI
    model = SPLICEVI.load(BASE, adata=adata)
    model.module.eval()
    return model


def _decoded(model, adata, n=32):
    """Decoder output (splicing p and expression scale) at the latent mean for the first n cells."""
    module = model.module
    dl = model._make_data_loader(adata=adata, indices=np.arange(n), batch_size=n, shuffle=False)
    t = next(iter(dl))
    with torch.no_grad():
        inf = module.inference(**module._get_inference_input(t))
        gin = module._get_generative_input(t, inf)
        gen = module.generative(**gin, use_z_mean=True)
    return torch.cat([gen["p"].reshape(n, -1), gen["px_scale"].reshape(n, -1)], 1).cpu()


def test_adding_covariates_keeps_output(data):
    query, _ = data
    model = _load(query)
    before = _decoded(model, query)
    model.add_decoder_covariates(1)
    assert model.init_params_["non_kwargs"]["n_decoder_covariates"] == 1
    assert torch.equal(before, _decoded(model, query))   # zero weights: identical
    with pytest.raises(ValueError):
        model.add_decoder_covariates(1)


def test_finetune_trains_only_decoders_and_reloads(data, tmp_path):
    from splicevi import SPLICEVI
    query, reference = data
    model = _load(query)
    frozen_before = {k: v.detach().clone() for k, v in model.module.state_dict().items()
                     if not k.startswith(("z_decoder_expression.", "z_decoder_splicing.", "decoder_covariate_", "depth_cov_std", "cross_gate"))}
    decoders_before = {k: v.detach().clone() for k, v in model.module.state_dict().items() if k.startswith("z_decoder_")}
    history = model.finetune_decoders(query, reference, n_query_per_step=16, n_reference_per_step=32, max_epochs=2,
                                      early_stopping_patience=5)
    state = model.module.state_dict()
    assert list(history.epoch) == [-1, 0, 1]
    assert all(torch.equal(state[k], v) for k, v in frozen_before.items())        # encoders, phi, dispersions, pop psi ...
    assert any(not torch.equal(state[k], v) for k, v in decoders_before.items())  # decoders trained
    assert model.module.decoder_covariate_weight_splicing.abs().sum() > 0         # technology weights trained
    assert float(model.module.decoder_covariate_value[0]) == 1.0                  # left decoding as query data
    if getattr(model.module, "decoder_depth_covariates", False):
        assert torch.isinf(model.module.depth_cov_std).all()
    assert all(p.requires_grad for p in model.module.parameters())                # flags restored (base: all True)
    out = _decoded(model, query)
    model.save(str(tmp_path / "ft"), overwrite=True)
    reloaded = SPLICEVI.load(str(tmp_path / "ft"), adata=query)
    reloaded.module.eval()
    assert reloaded.module.n_decoder_covariates == 1
    assert torch.allclose(out, _decoded(reloaded, query), atol=1e-6)               # same weights, buffers, depth inputs
    assert reloaded.finetune_params_["n_query_per_step"] == 16
    with reloaded.decoder_covariate(0.0):                                          # decode as reference data
        as_reference = _decoded(reloaded, query)
    assert not torch.allclose(out, as_reference)
    assert torch.allclose(out, _decoded(reloaded, query), atol=1e-6)               # restored after the block


def test_base_model_loads_unchanged(data):
    query, _ = data
    model = _load(query)
    assert model.module.n_decoder_covariates == 0
    assert not hasattr(model.module, "decoder_covariate_weight_splicing")
    with pytest.raises(ValueError):
        model.set_decoder_covariate(1.0)
