#!/usr/bin/env bash
# STAGE5 §171 item 6 (Karin 2026-10-24): train_splicevi.py with no --psi_prior_pseudocount vs "0" (must be bit-identical) and "0.1" (arm B path smoke test). Tiny 400-cell Allen subset, CPU, 2 epochs.
set -euo pipefail
source /gpfs/commons/home/svaidyanathan/miniconda3/etc/profile.d/conda.sh; conda activate splicevi-env
export PYTHONPATH=/gpfs/commons/home/kisaev/SpliceVI/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; cd "${FLAGTEST_DIR:?set FLAGTEST_DIR to a folder holding tiny400.h5mu (400 Allen TRAIN cells)}"
python -c "import splicevi; print('splicevi from', splicevi.__file__)"
COMMON="--train_mdata_path tiny400.h5mu --batch_key None --max_epochs 2 --batch_size 128 --n_epochs_kl_warmup 200 --n_latent 30 --dropout_rate 0.01 --splicing_loss_type dirichlet_multinomial --dm_concentration atse --gene_likelihood zinb --phi_init constant --psi_input deviation --decoder_intercept_init population --decoder_depth_covariates true --variance_mixing squared --use_layer_norm both --use_batch_norm none --initialize_embeddings_from_pca true --seed 20260926"
for tag in none k0 k01; do
  extra=""; [ $tag = k0 ] && extra="--psi_prior_pseudocount 0"; [ $tag = k01 ] && extra="--psi_prior_pseudocount 0.1"
  rm -rf m_$tag; python /gpfs/commons/home/kisaev/SpliceVI/train_splicevi.py $COMMON --model_dir m_$tag $extra > log_$tag.txt 2>&1 || { echo "FAILED $tag"; tail -20 log_$tag.txt; }
  grep -h "psi prior\|Additional init\|psi_prior_pseudocount" log_$tag.txt | head -3
done
python - <<'PY'
import torch
def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict): out.update(flat(v, p + k + "."))
        elif torch.is_tensor(v): out[p + k] = v
    return out
def sd(t): return flat(torch.load(f"m_{t}/model.pt", map_location="cpu", weights_only=False)["model_state_dict"])
a, b, c = sd("none"), sd("k0"), sd("k01")
same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
print(f"[flag] none vs --psi_prior_pseudocount 0: {len(a)} tensors, bit-identical = {same}")
print(f"[flag] k=0.1: psi_prior_pop saved = {any('psi_prior_pop' in k for k in c)}; tensors differing from none = {sum(not torch.equal(a[k], c[k]) for k in a if k in c)} of {len(a)}")
assert same
PY
