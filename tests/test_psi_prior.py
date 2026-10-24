"""STAGE5 §171: Smriti's psi_prior_pseudocount (f7b1062). Toy dimensions, CPU. Run: pytest tests/test_psi_prior.py (or python tests/test_psi_prior.py)"""
import numpy as np
import torch

from test_depth_target import N, _module, _run, _toy


def test_default_is_off_and_identical():
    """Default 0: no psi_prior_pop buffer; same parameters, decoder output and loss as passing 0 explicitly."""
    t = _toy(); a = _module(decoder_depth_covariates=True); b = _module(decoder_depth_covariates=True, psi_prior_pseudocount=0.0)
    assert a.psi_prior_pseudocount == 0.0 and not hasattr(a, "psi_prior_pop") and "psi_prior_pop" not in a.state_dict()
    sa, sb = a.state_dict(), b.state_dict(); assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    _, ga, oa = _run(a, t); _, gb, ob = _run(b, t)
    assert torch.equal(oa.loss, ob.loss) and torch.equal(ga["p"], gb["p"])


def test_prior_adds_kappa_cross_entropy():
    """kappa > 0: per-cell splicing reconstruction loss = off + kappa * (-sum_j pi_j log p_j); same weights otherwise."""
    t = _toy(); off = _module(); on = _module(psi_prior_pseudocount=0.1)
    pop = torch.rand(on.psi_prior_pop.shape[0]); on.psi_prior_pop.copy_(pop)
    on.load_state_dict(dict(off.state_dict(), psi_prior_pop=pop))
    _, g_off, o_off = _run(off, t); _, g_on, o_on = _run(on, t)
    assert torch.equal(g_off["p"], g_on["p"])
    extra = -0.1 * (pop * torch.log(g_on["p"].clamp_min(1e-8))).sum(-1)
    d = o_on.reconstruction_loss["reconstruction_loss_splicing"] - o_off.reconstruction_loss["reconstruction_loss_splicing"]
    assert torch.allclose(d, extra * on.splicing_loss_weight, rtol=1e-5, atol=1e-5)


def test_invalid():
    for kw in (dict(psi_prior_pseudocount=-1.0), dict(psi_prior_pseudocount=0.1, splicing_loss_type="binomial", dm_concentration="scalar")):
        try:
            _module(**kw); assert False, kw
        except ValueError:
            pass


if __name__ == "__main__":
    for f in (test_default_is_off_and_identical, test_prior_adds_kappa_cross_entropy, test_invalid):
        f(); print("PASS", f.__name__)
