"""Unit tests for the splicing-encoder input option psi_input ("raw" default, "centred", "deviation"). Toy dimensions, CPU.
Run: pytest tests/test_psi_input.py"""
import numpy as np
import pytest
import torch

from splicevi.splicevae import SPLICEVAE
from test_std_config import KW, CODES, N, J, _toy, _module, _run


def _enc(m, x, mask):
    with torch.no_grad():
        return m.z_encoder_splicing(x, mask)


def _with_pop(m, seed=3):
    rng = np.random.default_rng(seed); pop = np.concatenate([rng.dirichlet(np.ones(3)) for _ in range(J // 3)])
    with torch.no_grad():
        m.z_encoder_splicing.pop_psi.copy_(torch.tensor(pop, dtype=torch.float32))
        if hasattr(m.z_encoder_splicing, "pool_divisor"):
            m.z_encoder_splicing.pool_divisor.fill_(10.0)
    return torch.tensor(pop, dtype=torch.float32)


def test_raw_default_has_no_buffers():
    m = _module()
    assert m.z_encoder_splicing.psi_input == "raw"
    assert not hasattr(m.z_encoder_splicing, "pop_psi") and not hasattr(m.z_encoder_splicing, "pool_divisor")


def test_raw_explicit_matches_default():
    t = _toy(); a = _module(); b = _module(psi_input="raw")
    x, mask = t["junc_ratio"], t["psi_observed_mask"]
    for u, v in zip(_enc(a, x, mask), _enc(b, x, mask)):
        assert torch.equal(u, v)


def test_deviation_zero_input_gives_identical_latent_for_every_cell():
    """Every observed psi at its population value: each junction contributes exactly 0, so the latent no longer depends on which
    junctions are observed (no presence signal)."""
    t = _toy(); m = _module(psi_input="deviation"); pop = _with_pop(m)
    mask = t["psi_observed_mask"]; x = pop.unsqueeze(0).expand(N, J) * mask
    mu, _ = _enc(m, x, mask)
    assert torch.equal(mu, mu[:1].expand_as(mu))


def test_deviation_adding_an_event_at_its_expected_ratio_changes_nothing():
    t = _toy(); m = _module(psi_input="deviation"); pop = _with_pop(m)
    x, mask = t["junc_ratio"].clone(), t["psi_observed_mask"].clone()
    i = int(np.flatnonzero((mask.numpy() == 0).any(1))[0]); unobs = np.flatnonzero(mask[i].numpy() == 0); a = CODES[unobs[0]]; js = np.flatnonzero(CODES == a)
    mu0, _ = _enc(m, x, mask)
    mask[i, js] = 1.0; x[i, js] = pop[js]
    mu1, _ = _enc(m, x, mask)
    assert torch.allclose(mu0, mu1, atol=1e-6)   # float rounding only (~1e-7): the added junctions contribute exactly 0, but N_obs changes the GEMM batch


def test_deviation_ratio_change_moves_the_latent():
    t = _toy(); m = _module(psi_input="deviation"); _with_pop(m)
    x, mask = t["junc_ratio"].clone(), t["psi_observed_mask"]
    mu0, _ = _enc(m, x, mask)
    j = int(np.flatnonzero(mask[0].numpy() > 0)[0]); x[0, j] = (x[0, j] + 0.3) % 1.0
    mu1, _ = _enc(m, x, mask)
    assert not torch.equal(mu0[0], mu1[0]) and torch.equal(mu0[1:], mu1[1:])


def test_centred_subtracts_population_psi():
    t = _toy(); m = _module(psi_input="centred"); pop = _with_pop(m)
    r = _module(); r.z_encoder_splicing.load_state_dict({k: v for k, v in m.z_encoder_splicing.state_dict().items() if k != "pop_psi"})
    x, mask = t["junc_ratio"], t["psi_observed_mask"]
    for u, v in zip(_enc(m, x, mask), _enc(r, (x - pop.unsqueeze(0)) * mask, mask)):
        assert torch.allclose(u, v, atol=1e-6)


def test_deviation_chunked_matches_unchunked():
    t = _toy(); a = _module(psi_input="deviation"); _with_pop(a); b = _module(psi_input="deviation", max_nobs=7); _with_pop(b)
    x, mask = t["junc_ratio"], t["psi_observed_mask"]
    for u, v in zip(_enc(a, x, mask), _enc(b, x, mask)):
        assert torch.allclose(u, v, atol=1e-5)


def test_deviation_requires_mean_pool():
    with pytest.raises(ValueError):
        SPLICEVAE(**dict(KW, psi_input="deviation", pool_mode="sum"))


def test_deviation_full_forward_and_loss_finite():
    m = _module(psi_input="deviation"); _with_pop(m); _, _, out = _run(m, _toy())
    assert torch.isfinite(out.loss)
