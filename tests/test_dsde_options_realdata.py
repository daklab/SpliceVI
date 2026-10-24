"""STAGE5 §167: checks of the ported DE/DS options on real data (EasySci in80 metacells, (i') rep 1). Run as a script (no pytest in the env):
python tests/test_dsde_options_realdata.py <out.npz> [check]   — without 'check': write the default-path outputs (run on old and new code);
with 'check' (new code only): sampling and closed-form tests, and comparison with <out.npz> written by the old code."""
import sys, numpy as np, torch, mudata as mu, scvi
from splicevi import SPLICEVI
MD = "/gpfs/commons/home/svaidyanathan/repos/SpliceVI_Repositories/SpliceVI/models/pipeline_20261001_012113_s5-ab-devp_rep1"
MDATA = "/gpfs/commons/home/svaidyanathan/repos/SpliceVI_Repositories/script_outputs/easysci_transfer/data_v20261008/easysci_mdata_1000_2026-10-08.h5mu"   # = our primary all-nuclei file (md5 37ab19c1)
OUT, CHECK = sys.argv[1], len(sys.argv) > 2
md = mu.read_h5mu(MDATA)
for c in md.obs.columns:
    if c not in md["rna"].obs.columns: md["rna"].obs[c] = md.obs[c].values
md["rna"].obs["X_library_size"] = md["rna"].obsm["X_library_size"]
md["rna"].obs["mouse.id"] = "easysci"; md.obs["mouse.id"] = "easysci"   # as stage5_s146_encode.prep (our models were set up with batch_key None and mouse.id in obs)
sys.path.insert(0, "/gpfs/commons/home/kisaev/SpliceVI-utils-evalsuite/eval_suite/stages"); from stage5_lib70 import load_model   # the loader all STAGE5 scripts use (sets the MuData up first)
m = load_model(MD, md)
idx = np.arange(200); i1, i2 = np.arange(0, 100), np.arange(100, 200)
def default_outputs():
    scvi.settings.seed = 0; dm = m.get_normalized_splicing_DM(adata=md, indices=idx, return_numpy=True)
    scvi.settings.seed = 0; ex = m.get_normalized_expression(adata=md, indices=idx, return_numpy=True)
    scvi.settings.seed = 0; ds = m.differential_splicing(adata=md, idx1=i1, idx2=i2, silent=True)
    scvi.settings.seed = 0; de = m.differential_expression(adata=md, idx1=i1, idx2=i2, silent=True)
    return dict(dm=np.asarray(dm), ex=np.asarray(ex), ds=ds.sort_index()[["proba_ds", "effect_size"]].values.astype(float), de=de.sort_index()[["proba_de", "lfc_mean"]].values.astype(float))
o = default_outputs()
if not CHECK:
    np.savez(OUT, **o); print("written", OUT); sys.exit(0)
ref = np.load(OUT); ok = True
for k in o:
    same = np.array_equal(o[k], ref[k]); ok &= same; print(f"[1] default path {k}: identical to the pre-port code = {same}")
# [2] DM posterior sample: mean of 400 draws vs posterior mean
draws = np.stack([np.asarray(m.get_normalized_splicing_DM(adata=md, indices=idx[:50], return_numpy=True, sample=True)) for _ in range(400)])
pm = np.asarray(m.get_normalized_splicing_DM(adata=md, indices=idx[:50], return_numpy=True))
z = np.abs(draws.mean(0) - pm) / (draws.std(0) / np.sqrt(400) + 1e-9); f2 = float(np.mean(z < 4)); print(f"[2] DM sample mean vs posterior mean: share within 4 SE = {f2:.4f}"); ok &= f2 > 0.995
# [3] ZINB posterior mean vs the closed form from the generative outputs
zm = np.asarray(m.get_normalized_expression_ZINB(adata=md, indices=idx[:50], return_numpy=True))
dl = m._make_data_loader(adata=md, indices=idx[:50], batch_size=50); t = next(iter(dl))
with torch.no_grad(): _, g = m.module.forward(tensors=t, inference_kwargs={"n_samples": 1}, generative_kwargs={"use_z_mean": True}, compute_loss=False)
mu_, sc, th, pi = (g[k].double().numpy() for k in ("px_rate", "px_scale", "px_r", "px_dropout")); pi = 1 / (1 + np.exp(-pi)); th = np.broadcast_to(th, mu_.shape)
y = t["X"][:, : m.n_genes].double().numpy(); pos = mu_ * (th + y) / (th + mu_); p0 = np.exp(th * (np.log(th) - np.log(th + mu_))); w = pi / (pi + (1 - pi) * p0)
post = np.where(y > 0, pos, w * mu_ + (1 - w) * th * mu_ / (th + mu_)); ref3 = post / (mu_.sum(1, keepdims=True) / sc.sum(1, keepdims=True))
rel = float(np.max(np.abs(zm - ref3) / (np.abs(ref3) + 1e-12))); print(f"[3] ZINB posterior mean vs closed form (float64 numpy): max relative difference = {rel:.2e}"); ok &= rel < 1e-3
# [4] ZINB posterior sample: mean of 400 draws vs posterior mean
zd = np.stack([np.asarray(m.get_normalized_expression_ZINB(adata=md, indices=idx[:50], return_numpy=True, sample=True)) for _ in range(400)])
z4 = np.abs(zd.mean(0) - zm) / (zd.std(0) / np.sqrt(400) + 1e-12); f4 = float(np.mean(z4 < 4)); print(f"[4] ZINB sample mean vs posterior mean: share within 4 SE = {f4:.4f}"); ok &= f4 > 0.995
# [5] §171 zinb_expected (Smriti f7b1062): (1 - sigmoid(px_dropout)) * px_scale, and DE runs with it
if True:
    ze = np.asarray(m.get_normalized_expression(adata=md, indices=idx[:50], return_numpy=True, expected=True))
    pi5 = 1 / (1 + np.exp(-g["px_dropout"].double().numpy())); ref5 = sc * (1 - pi5)
    rel5 = float(np.max(np.abs(ze - ref5) / (np.abs(ref5) + 1e-12))); print(f"[5] zinb_expected vs (1 - dropout) * px_scale: max relative difference = {rel5:.2e}"); ok &= rel5 < 1e-5
    scvi.settings.seed = 0; de5 = m.differential_expression(adata=md, idx1=i1, idx2=i2, silent=True, norm_expression_function="zinb_expected")
    f5 = bool(np.isfinite(de5.proba_de).all() and len(de5) == len(o["de"])); print(f"[5] DE with zinb_expected: {len(de5)} genes, finite = {f5}; |lfc| median {de5.lfc_mean.abs().median():.3f} vs decoder {np.median(np.abs(o['de'][:, 1])):.3f}"); ok &= f5
print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")
