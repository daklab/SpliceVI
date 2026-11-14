"""STAGE5 §200: flag off on kisaev/splicing-only vs the branch base (kisaev/experiments). Run once per code tree, then compare:
PYTHONPATH=<tree>/src:tests python tests/check_splicing_only_base_identity.py <out.pt>; python tests/check_splicing_only_base_identity.py --compare a.pt b.pt"""
import sys, torch
if sys.argv[1] == "--compare":
    a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3]); assert a.keys() == b.keys(), "different tensor sets"
    bad = [k for k in a if not torch.equal(a[k], b[k])]; print(f"{len(a) - len(bad)}/{len(a)} tensors identical; differing: {bad[:10]}"); sys.exit(1 if bad else 0)
from test_depth_target import _module, _run, _toy   # noqa: E402
import splicevi; print("splicevi from", splicevi.__file__)
t = _toy(); out = {}
for tag, kw in (("depthcov", dict(decoder_depth_covariates=True)), ("plain", {})):
    m = _module(**kw); m.train(); inf, gen, lo = _run(m, t); lo.loss.backward()
    out[f"{tag}/loss"] = lo.loss.detach(); out[f"{tag}/qz_m"] = inf["qz_m"].detach(); out[f"{tag}/p"] = gen["p"].detach()
    out.update({f"{tag}/param/{n}": p.detach().clone() for n, p in m.named_parameters()})
    out.update({f"{tag}/grad/{n}": p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None})
torch.save(out, sys.argv[1])
