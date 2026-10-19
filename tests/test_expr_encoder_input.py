"""STAGE5 §163: expr_encoder_input ("as_is" default = original behaviour; "log1p_cp10k" = log1p(x / rowsum * 1e4) into the expression encoder only).
Toy dimensions, CPU. Run: pytest tests/test_expr_encoder_input.py"""
import torch
from scvi import REGISTRY_KEYS

from test_depth_target import _module, _run, _toy


def test_default_is_as_is_and_identical():
    """Default: same parameters, latent, decoder output and loss as passing "as_is" explicitly."""
    t = _toy(); torch.manual_seed(0); a = _module(decoder_depth_covariates=True); torch.manual_seed(0); b = _module(decoder_depth_covariates=True, expr_encoder_input="as_is")
    assert a.expr_encoder_input == "as_is"
    sa, sb = a.state_dict(), b.state_dict(); assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    ia, ga, oa = _run(a, t); ib, gb, ob = _run(b, t)
    assert torch.equal(oa.loss, ob.loss) and torch.equal(ia["qz_m"], ib["qz_m"]) and torch.equal(ga["p"], gb["p"])


def test_log1p_cp10k_is_the_encoder_view():
    """On: the expression posterior equals the encoder applied to log1p(x / rowsum * 1e4); the decoder covariates (raw counts) are unchanged."""
    t = _toy(); torch.manual_seed(0); off = _module(decoder_depth_covariates=True); torch.manual_seed(0); on = _module(decoder_depth_covariates=True, expr_encoder_input="log1p_cp10k")
    on.load_state_dict(off.state_dict()); on.eval(); off.eval()
    x = t[REGISTRY_KEYS.X_KEY]; xl = torch.log1p(x / x.sum(1, keepdim=True).clamp_min(1e-8) * 1e4)
    t2 = dict(t); t2[REGISTRY_KEYS.X_KEY] = xl
    i_on = _run(on, t)[0]; i_ref = _run(off, t2)[0]; i_off = _run(off, t)[0]
    assert torch.allclose(i_on["qzm_expr"], i_ref["qzm_expr"], atol=1e-6)
    assert not torch.allclose(i_on["qzm_expr"], i_off["qzm_expr"])
    assert torch.equal(i_on["depth_cov"], i_off["depth_cov"])


def test_invalid_value():
    try:
        _module(expr_encoder_input="log"); assert False
    except ValueError:
        pass
