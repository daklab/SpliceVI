#!/usr/bin/env bash
# STAGE5 §200: train_splicevi.py smoke test of --splicing_only (CLI auto-generated from SPLICEVI.__init__). Tiny 400-cell Allen subset, CPU, 2 epochs.
# no flag vs --splicing_only false must be bit-identical; --splicing_only true trains, saves, reloads, and its splicing latent equals its joint latent.
set -euo pipefail
source /gpfs/commons/home/svaidyanathan/miniconda3/etc/profile.d/conda.sh; conda activate splicevi-env
TREE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$TREE/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; cd "${FLAGTEST_DIR:?set FLAGTEST_DIR to a folder holding tiny400.h5mu (400 Allen TRAIN cells)}"
python -c "import splicevi; print('splicevi from', splicevi.__file__)"
COMMON="--train_mdata_path tiny400.h5mu --batch_key None --max_epochs 2 --batch_size 128 --n_epochs_kl_warmup 200 --n_latent 30 --dropout_rate 0.01 --splicing_loss_type dirichlet_multinomial --dm_concentration atse --gene_likelihood zinb --phi_init constant --psi_input deviation --decoder_intercept_init population --decoder_depth_covariates true --variance_mixing squared --use_layer_norm both --use_batch_norm none --initialize_embeddings_from_pca true --seed 20260926"
for tag in none off on; do
  extra=""; [ $tag = off ] && extra="--splicing_only false"; [ $tag = on ] && extra="--splicing_only true"
  rm -rf m_$tag; python $TREE/train_splicevi.py $COMMON --model_dir m_$tag $extra > log_$tag.txt 2>&1 || { echo "FAILED $tag"; tail -20 log_$tag.txt; exit 1; }
done
python - <<'PY'
import torch, numpy as np, mudata as md
from splicevi import SPLICEVI
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, p + k + "."))
        elif torch.is_tensor(v): out[p + k] = v
    return out
def sd(t): return torch.load(f"m_{t}/model.pt", map_location="cpu", weights_only=False)
A, B, C = (sd(t) for t in ("none", "off", "on")); a, b, c = flat(A["model_state_dict"]), flat(B["model_state_dict"]), flat(C["model_state_dict"])
same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
print(f"[flag] none vs --splicing_only false: {len(a)} tensors, bit-identical = {same}"); assert same
print(f"[flag] on: init splicing_only = {C['attr_dict']['init_params_']['non_kwargs'].get('splicing_only')}; tensors differing from none = {sum(not torch.equal(a[k], c[k]) for k in a if k in c)} of {len(a)}")
m = SPLICEVI.load("m_on", adata=md.read_h5mu("tiny400.h5mu")); assert m.module.splicing_only
zj, zs = (np.asarray(m.get_latent_representation(modality=x)) for x in ("joint", "splicing"))
print(f"[flag] on: splicing latent == joint latent: {np.array_equal(zj, zs)} (max |diff| {np.abs(zj - zs).max():.2e})"); assert np.allclose(zj, zs, atol=1e-6)
h = [l for l in open("log_on.txt") if "train_loss" in l or "Epoch" in l][-2:]; print("[flag] on: last log lines:", *[x.strip()[:160] for x in h], sep="\n  ")
PY
