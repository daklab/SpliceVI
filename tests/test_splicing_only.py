"""STAGE5 §200: splicing_only (splicing-only SpliceVI: no cell counts as having expression). Toy dimensions, CPU.
Run: PYTHONPATH=src pytest tests/test_splicing_only.py"""
import pytest
import torch

from test_depth_target import _module, _run, _toy

EXPR_PARTS = ("z_encoder_expression", "l_encoder_expression", "z_decoder_expression")


def _grads(m, t):
    m.zero_grad(set_to_none=True); _, _, out = _run(m, t); out.loss.backward()
    return {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in m.named_parameters()}


def test_default_off_identical():
    """Default False: same parameters, forward pass, loss and gradients as passing False explicitly."""
    t = _toy(); a = _module(decoder_depth_covariates=True); b = _module(decoder_depth_covariates=True, splicing_only=False)
    assert a.splicing_only is False
    sa, sb = a.state_dict(), b.state_dict(); assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    ia, ga, oa = _run(a, t); ib, gb, ob = _run(b, t)
    assert torch.equal(oa.loss, ob.loss) and torch.equal(ia["qz_m"], ib["qz_m"]) and torch.equal(ga["p"], gb["p"])
    da, db = _grads(a, t), _grads(b, t)
    assert all((da[k] is None and db[k] is None) or torch.equal(da[k], db[k]) for k in da)


@pytest.mark.parametrize("gate", [1.0, 0.0])   # after warmup and during warmup (the routing coin must not pick expression)
def test_joint_latent_is_splicing_posterior(gate):
    m = _module(decoder_depth_covariates=True, splicing_only=True); m.set_cross_gate(gate)
    for _ in range(5):   # several draws of the warmup coin
        inf, _, out = _run(m, _toy())
        assert torch.equal(inf["qz_m"], inf["qzm_spl"]) and torch.equal(inf["qz_v"], inf["qzv_spl"]) and torch.isfinite(out.loss)


def test_no_expression_loss_and_no_gradient_to_expression_parts():
    t = _toy(); m = _module(decoder_depth_covariates=True, splicing_only=True); m.train()
    _, _, out = _run(m, t); assert float(out.reconstruction_loss["reconstruction_loss_expression"].abs().sum()) == 0.0
    assert float(out.reconstruction_loss["reconstruction_loss_splicing"].abs().sum()) > 0
    g = _grads(m, t)
    expr = {k: v for k, v in g.items() if k.startswith(EXPR_PARTS)}; spl = {k: v for k, v in g.items() if k.startswith("z_encoder_splicing")}
    assert expr and all(v is None or float(v.abs().max()) == 0.0 for v in expr.values()), "expression parts received gradient"
    assert spl and any(v is not None and float(v.abs().max()) > 0 for v in spl.values()), "splicing encoder received no gradient"


def test_expression_input_does_not_reach_latent():
    """On: changing the expression counts changes nothing in the joint latent; off: it does."""
    t = _toy(); t2 = dict(t); t2[next(k for k in t if k == "X")] = torch.poisson(torch.full_like(t["X"], 5.0))
    on = _module(decoder_depth_covariates=False, splicing_only=True); off = _module(decoder_depth_covariates=False)
    assert torch.equal(_run(on, t)[0]["qz_m"], _run(on, t2)[0]["qz_m"])
    assert not torch.allclose(_run(off, t)[0]["qz_m"], _run(off, t2)[0]["qz_m"])


def test_rejects_incompatible_settings():
    with pytest.raises(ValueError):
        _module(splicing_only=True, splicing_encoder_off=True)
    with pytest.raises(ValueError):
        _module(splicing_only=True, modality_weights="concatenate")
