"""Unit tests for the standard-config options: decoder_intercept_init="population" (STAGE5 68.6) and
decoder_depth_covariates=True (STAGE5 68.4a). Toy dimensions, CPU. Run: pytest tests/test_std_config.py"""
import numpy as np
import pytest
import torch
from scvi import REGISTRY_KEYS

from splicevi.partialvae import group_logsumexp, subtract_group_logsumexp
from splicevi.splicevae import SPLICEVAE

N, G, J, A, LAT = 16, 40, 60, 20, 6
CODES = np.repeat(np.arange(A), 3)
KW = dict(n_input_genes=G, n_input_junctions=J, n_batch=2, n_obs=N, splicing_encoder_architecture="partial",
          splicing_decoder_architecture="vanilla", splicing_loss_type="dirichlet_multinomial", dm_concentration="atse",
          variance_mixing="squared", encoder_hidden_dim=16, code_dim=8, h_hidden_dim=16, n_latent=LAT, n_hidden=16,
          dropout_rate=0.0, phi_init="constant")


def _j2a():
    return torch.sparse_coo_tensor(torch.stack([torch.arange(J), torch.tensor(CODES)]), torch.ones(J), (J, A)).coalesce()


def _toy():
    rng = np.random.default_rng(1)
    tot = rng.integers(0, 40, size=(N, A)); k = np.zeros((N, J)); n_rep = tot[:, CODES].astype(float)
    for i in range(N):
        for a in range(A):
            if tot[i, a] > 0:
                k[i, CODES == a] = rng.multinomial(tot[i, a], rng.dirichlet(np.ones(3)))
    psi = np.divide(k, n_rep, out=np.zeros_like(k), where=n_rep > 0); mask = (n_rep > 0).astype(float)
    x_expr = rng.poisson(2.0, size=(N, G)).astype(float)
    T = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return {REGISTRY_KEYS.X_KEY: T(x_expr), "junc_ratio": T(psi), "psi_observed_mask": T(mask), "atse_counts_key": T(n_rep),
            "junc_counts_key": T(k), REGISTRY_KEYS.BATCH_KEY: torch.randint(0, 2, (N, 1)),
            REGISTRY_KEYS.LABELS_KEY: torch.zeros(N, 1, dtype=torch.long),
            REGISTRY_KEYS.INDICES_KEY: torch.arange(N).float().unsqueeze(1),
            REGISTRY_KEYS.SIZE_FACTOR_KEY: T(x_expr.sum(1, keepdims=True))}


def _module(**extra):
    torch.manual_seed(0); m = SPLICEVAE(**dict(KW, **extra)); m.junc2atse = _j2a(); m.set_cross_gate(1.0); m.eval(); return m


def _run(m, t):
    torch.manual_seed(7)
    inf = m.inference(**m._get_inference_input(t)); gen = m.generative(**m._get_generative_input(t, inf))
    return inf, gen, m.loss(t, inf, gen)


def test_defaults_off():
    """Default construction: no depth buffers, decoders see the latent only, no depth_cov in the forward pass."""
    m = _module(); inf, _, out = _run(m, _toy())
    assert not hasattr(m, "depth_cov_mean") and inf["depth_cov"] is None
    assert m.z_decoder_expression.px_decoder.fc_layers[0][0].in_features == LAT
    assert torch.isfinite(out.loss)


def test_depth_covariates_decoders_only():
    """68.4a: three standardised covariates reach both decoders (+3 inputs); the encoders are unchanged."""
    t = _toy(); m0 = _module(); m4 = _module(decoder_depth_covariates=True)
    inf, _, out = _run(m4, t)
    assert tuple(inf["depth_cov"].shape) == (N, 3)
    assert m4.z_decoder_expression.px_decoder.fc_layers[0][0].in_features == LAT + 3
    assert m4.z_decoder_splicing.ps_hidden.fc_layers[0][0].in_features == m0.z_decoder_splicing.ps_hidden.fc_layers[0][0].in_features + 3
    assert m4.z_encoder_splicing.h_layer[0].in_features == m0.z_encoder_splicing.h_layer[0].in_features
    assert m4.z_encoder_expression.encoder.fc_layers[0][0].in_features == m0.z_encoder_expression.encoder.fc_layers[0][0].in_features
    assert torch.isfinite(out.loss)
    # covariates = [log1p detected genes, log1p observed junctions, log1p library size], standardised by the stored buffers
    x = t[REGISTRY_KEYS.X_KEY]; raw = torch.stack([torch.log1p((x > 0).sum(1).float()), torch.log1p(t["psi_observed_mask"].sum(1)),
                                                   torch.log1p(t[REGISTRY_KEYS.SIZE_FACTOR_KEY][:, 0])], 1)
    assert torch.allclose(inf["depth_cov"], (raw - m4.depth_cov_mean) / m4.depth_cov_std)


def test_depth_covariates_linear_decoder_rejected():
    with pytest.raises(ValueError):
        SPLICEVAE(**dict(KW, splicing_decoder_architecture="linear", decoder_depth_covariates=True))


def test_population_intercept_reproduces_population_psi():
    """68.6: with bias = log(clip(p_pop, 1e-4, 1)), the decoder's sigmoid -> clamp -> logit -> within-event softmax returns p_pop."""
    j2a = _j2a(); p_pop = np.concatenate([np.random.default_rng(1).dirichlet(np.ones(3)) for _ in range(A)])
    p_pop[:3] = [0.9999, 1e-5, 0.0]   # an event with a zero-read junction: clipped, still sums to 1
    bias = torch.tensor(np.log(np.clip(p_pop, 1e-4, 1.0)), dtype=torch.float32).unsqueeze(0)
    ps = torch.sigmoid(bias).clamp(1e-6, 1 - 1e-6); lg = torch.log(ps) - torch.log1p(-ps)
    p_dec = torch.exp(subtract_group_logsumexp(j2a, lg, group_logsumexp(j2a, lg))).squeeze(0).numpy()
    assert np.abs(p_dec[3:] - p_pop[3:]).max() < 1e-5
    assert abs(p_dec[:3].sum() - 1) < 1e-5
