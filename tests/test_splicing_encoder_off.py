"""STAGE5 §128: splicing_encoder_off (expression-only ablation with the SpliceVI architecture). Toy dimensions, CPU.
Run: pytest tests/test_splicing_encoder_off.py"""
import torch

from test_depth_target import _module, _run, _toy


def test_default_off_identical():
    """Default False: same parameters and the same forward pass / loss as passing False explicitly."""
    t = _toy(); a = _module(decoder_depth_covariates=True); b = _module(decoder_depth_covariates=True, splicing_encoder_off=False)
    assert a.splicing_encoder_off is False
    sa, sb = a.state_dict(), b.state_dict(); assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    ia, ga, oa = _run(a, t); ib, gb, ob = _run(b, t)
    assert torch.equal(oa.loss, ob.loss) and torch.equal(ia["qz_m"], ib["qz_m"]) and torch.equal(ga["p"], gb["p"])


def test_joint_latent_is_expression_posterior():
    """On: the joint latent mean equals the expression posterior mean for every cell."""
    m = _module(decoder_depth_covariates=True, splicing_encoder_off=True); inf, _, out = _run(m, _toy())
    assert torch.allclose(inf["qz_m"], inf["qzm_expr"], atol=1e-6) and torch.isfinite(out.loss)


def test_depth_covariates_keep_observed_mask():
    """On: the decoder depth covariates (incl. log1p observed junctions) equal those of the default model on the same batch."""
    t = _toy(); a = _module(decoder_depth_covariates=True); b = _module(decoder_depth_covariates=True, splicing_encoder_off=True)
    ia, _, _ = _run(a, t); ib, _, _ = _run(b, t)
    assert torch.equal(ia["depth_cov"], ib["depth_cov"]) and float(ib["depth_cov"][:, 1].std()) > 0


def test_splicing_input_does_not_reach_latent():
    """On: changing the junction ratios changes nothing in the latent; off: it does."""
    t = _toy(); t2 = dict(t); t2["junc_ratio"] = torch.rand_like(t["junc_ratio"]) * t["psi_observed_mask"]
    on = _module(decoder_depth_covariates=True, splicing_encoder_off=True); off = _module(decoder_depth_covariates=True)
    assert torch.equal(_run(on, t)[0]["qz_m"], _run(on, t2)[0]["qz_m"])
    assert not torch.allclose(_run(off, t)[0]["qz_m"], _run(off, t2)[0]["qz_m"])
