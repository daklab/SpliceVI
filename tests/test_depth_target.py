"""STAGE5 §116: decoder_depth_covariates_target ("both" = 68.4a behaviour, default; "splicing" = depth covariates into the splicing
decoder only). Toy dimensions, CPU. Run: pytest tests/test_depth_target.py"""
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



def test_default_target_is_both_and_identical():
    """Default target = "both": same decoder widths and the same forward pass / loss as passing "both" explicitly."""
    t = _toy(); m_def = _module(decoder_depth_covariates=True); m_both = _module(decoder_depth_covariates=True, decoder_depth_covariates_target="both")
    assert m_def.decoder_depth_covariates_target == "both"
    assert m_def.z_decoder_expression.px_decoder.fc_layers[0][0].in_features == LAT + 3
    sd_a, sd_b = m_def.state_dict(), m_both.state_dict(); assert sd_a.keys() == sd_b.keys()
    assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)
    _, ga, oa = _run(m_def, t); _, gb, ob = _run(m_both, t)
    assert torch.equal(oa.loss, ob.loss) and torch.equal(ga["p"], gb["p"])


def test_splicing_target_widths():
    """"splicing": expression decoder sees the latent only (+ no depth), splicing decoder sees latent + 3 depth covariates."""
    m0 = _module(); ms = _module(decoder_depth_covariates=True, decoder_depth_covariates_target="splicing")
    assert ms.z_decoder_expression.px_decoder.fc_layers[0][0].in_features == LAT
    assert ms.z_decoder_splicing.ps_hidden.fc_layers[0][0].in_features == m0.z_decoder_splicing.ps_hidden.fc_layers[0][0].in_features + 3
    inf, _, out = _run(ms, _toy()); assert tuple(inf["depth_cov"].shape) == (N, 3) and torch.isfinite(out.loss)


def test_splicing_target_expression_ignores_depth():
    """"splicing": changing the depth covariates changes the splicing p but not the expression scale."""
    t = _toy(); ms = _module(decoder_depth_covariates=True, decoder_depth_covariates_target="splicing")
    inf = ms.inference(**ms._get_inference_input(t)); gi = ms._get_generative_input(t, inf)
    g1 = ms.generative(**gi, use_z_mean=True); gi2 = dict(gi); gi2["depth_cov"] = gi["depth_cov"] + 1.0; g2 = ms.generative(**gi2, use_z_mean=True)
    assert torch.equal(g1["px_scale"], g2["px_scale"]) if "px_scale" in g1 else torch.equal(g1["px"].scale, g2["px"].scale)
    assert not torch.allclose(g1["p"], g2["p"])


def test_bad_target_rejected():
    with pytest.raises(ValueError):
        SPLICEVAE(**dict(KW, decoder_depth_covariates=True, decoder_depth_covariates_target="expression"))
