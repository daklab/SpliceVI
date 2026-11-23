"""Tests for decoder covariate inputs and SPLICEVI.finetune_decoders, on a small model trained here with the default
settings (CPU, under a minute). Run: pytest tests/test_finetune_decoders.py

Checked: adding the covariate inputs leaves the model's output unchanged (zero weights); fine-tuning trains only the
decoders and the covariate weights; it fixes the depth inputs and keeps them through save / load; a saved fine-tuned
model reloads with identical decoder output; set_decoder_covariate / decoder_covariate switch query vs reference output;
models without covariate inputs load as before.
"""
import numpy as np
import pytest
import torch

from splicevi import SPLICEVI
from small_data import small_mudata

N_REFERENCE_CELLS = 200   # cells the base model is trained on; they also serve as the reference (replay) set
N_QUERY_CELLS = 96        # cells of a second random data set with the same genes and junctions, used as the query


@pytest.fixture(scope="module")
def base_model_dir(tmp_path_factory):
    """Directory of a base model trained with the default settings for two epochs on the reference cells."""
    reference = small_mudata(n_cells=N_REFERENCE_CELLS, seed=0)
    SPLICEVI.setup_mudata(reference)
    model = SPLICEVI(reference)
    model.train(max_epochs=2, accelerator="cpu", batch_size=64)
    model_dir = tmp_path_factory.mktemp("base_model")
    model.save(str(model_dir), overwrite=True)
    return str(model_dir)


@pytest.fixture(scope="module")
def data():
    """(query, reference) MuData objects, not yet registered with any model."""
    return small_mudata(n_cells=N_QUERY_CELLS, seed=1), small_mudata(n_cells=N_REFERENCE_CELLS, seed=0)


def _load(model_dir, adata):
    model = SPLICEVI.load(model_dir, adata=adata)
    model.module.eval()
    return model


def _decoded(model, adata, n=32):
    """Decoder output (splicing p and expression scale) at the latent mean for the first n cells."""
    module = model.module
    data_loader = model._make_data_loader(adata=adata, indices=np.arange(n), batch_size=n, shuffle=False)
    tensors = next(iter(data_loader))
    with torch.no_grad():
        inference_outputs = module.inference(**module._get_inference_input(tensors))
        generative_inputs = module._get_generative_input(tensors, inference_outputs)
        generative_outputs = module.generative(**generative_inputs, use_z_mean=True)
    return torch.cat([generative_outputs["p"].reshape(n, -1), generative_outputs["px_scale"].reshape(n, -1)], 1).cpu()


def test_adding_covariates_keeps_output(base_model_dir, data):
    query, _ = data
    model = _load(base_model_dir, query)
    before = _decoded(model, query)
    model.add_decoder_covariates(1)
    assert model.init_params_["non_kwargs"]["n_decoder_covariates"] == 1
    assert torch.equal(before, _decoded(model, query))   # zero weights: identical
    with pytest.raises(ValueError):
        model.add_decoder_covariates(1)


def test_finetune_trains_only_decoders_and_reloads(base_model_dir, data, tmp_path):
    query, reference = data
    model = _load(base_model_dir, query)
    trained_prefixes = ("z_decoder_expression.", "z_decoder_splicing.", "decoder_covariate_", "depth_cov_std", "cross_gate")
    frozen_before = {name: value.detach().clone() for name, value in model.module.state_dict().items()
                     if not name.startswith(trained_prefixes)}
    decoders_before = {name: value.detach().clone() for name, value in model.module.state_dict().items()
                       if name.startswith("z_decoder_")}

    history = model.finetune_decoders(query, reference, n_query_per_step=16, n_reference_per_step=32, max_epochs=2,
                                      early_stopping_patience=5)

    # Only the decoders and the covariate weights changed; the model is left decoding as query data.
    state = model.module.state_dict()
    assert list(history.epoch) == [-1, 0, 1]
    assert all(torch.equal(state[name], value) for name, value in frozen_before.items())        # encoders, phi, dispersions ...
    assert any(not torch.equal(state[name], value) for name, value in decoders_before.items())  # decoders trained
    assert model.module.decoder_covariate_weight_splicing.abs().sum() > 0                       # technology weights trained
    assert float(model.module.decoder_covariate_value[0]) == 1.0
    if getattr(model.module, "decoder_depth_covariates", False):
        assert torch.isinf(model.module.depth_cov_std).all()
    assert all(parameter.requires_grad for parameter in model.module.parameters())              # flags restored

    # Save and reload: same output, fine-tuning settings recorded, covariate value switchable.
    out = _decoded(model, query)
    model.save(str(tmp_path / "finetuned"), overwrite=True)
    reloaded = _load(str(tmp_path / "finetuned"), query)
    assert reloaded.module.n_decoder_covariates == 1
    assert torch.allclose(out, _decoded(reloaded, query), atol=1e-6)
    assert reloaded.finetune_params_["n_query_per_step"] == 16
    with reloaded.decoder_covariate(0.0):                                          # decode as reference data
        as_reference = _decoded(reloaded, query)
    assert not torch.allclose(out, as_reference)
    assert torch.allclose(out, _decoded(reloaded, query), atol=1e-6)               # restored after the block


def test_base_model_loads_unchanged(base_model_dir, data):
    query, _ = data
    model = _load(base_model_dir, query)
    assert model.module.n_decoder_covariates == 0
    assert not hasattr(model.module, "decoder_covariate_weight_splicing")
    with pytest.raises(ValueError):
        model.set_decoder_covariate(1.0)
