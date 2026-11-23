#!/usr/bin/env bash
#SBATCH --job-name=splicevi_train
#SBATCH --output=logs/splicevi_train_%j.out
#SBATCH --error=logs/splicevi_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#
# Train one SpliceVI model with train_splicevi.py on a SLURM cluster.
#
# Usage:
#   1. Set TRAIN_MDATA_PATH (a model-ready MuData file, see docs/data-format.md), MODEL_DIR_BASE and BATCH_KEY below,
#      and the environment activation lines for your system.
#   2. Change the #SBATCH lines above to your cluster's partition and limits.
#   3. mkdir -p logs && sbatch train_splicevi.sh
#
# Every model and training setting not passed below keeps its default, and the defaults are the paper's settings.
# To change one, add its flag to EXTRA_ARGS (for example EXTRA_ARGS="--n_latent 20 --max_epochs 100");
# `python train_splicevi.py --help` lists all flags with their defaults. Without SLURM, run the same command directly.

set -euo pipefail

#######################################
# Settings
#######################################

TRAIN_MDATA_PATH="path/to/train_model_ready.h5mu"
MODEL_DIR_BASE="models"
BATCH_KEY="None"        # obs column used as batch (the paper's TMS models use "mouse.id"); "None" = no batch correction
SEED=1

# Flags passed on to train_splicevi.py; any setting left out keeps its paper default. Each flag has the name of a
# SPLICEVI(...) or train() argument. Examples (combine as many as needed, in one string):
#   Train for fewer epochs:              EXTRA_ARGS="--max_epochs 100"                      (default 300)
#   Change the learning rate:            EXTRA_ARGS="--lr 5e-4"                             (default 1e-4)
#   Expression likelihood:               EXTRA_ARGS="--gene_likelihood nb"                  (zinb | nb | poisson; default zinb)
#   Splicing likelihood:                 EXTRA_ARGS="--splicing_loss_type beta_binomial"    (binomial | beta_binomial |
#                                                                                            dirichlet_multinomial; default
#                                                                                            dirichlet_multinomial)
#   Several at once:                     EXTRA_ARGS="--max_epochs 100 --lr 5e-4 --gene_likelihood nb --n_latent 20"
EXTRA_ARGS=""

# Weights & Biases logging (optional; needs `pip install wandb`)
USE_WANDB=false
WANDB_PROJECT="splicevi"

#######################################
# Environment
#######################################

# Activate the environment created from environment.yml (adapt to your system).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate splicevi-env

RUN_NAME="splicevi_$(date +"%Y%m%d_%H%M%S")"
MODEL_DIR="${MODEL_DIR_BASE}/${RUN_NAME}"
mkdir -p "${MODEL_DIR}"

echo "[JOB] SLURM job ID   : ${SLURM_JOB_ID:-none}"
echo "[JOB] Training data  : ${TRAIN_MDATA_PATH}"
echo "[JOB] Model directory: ${MODEL_DIR}"

WANDB_ARGS=""
if [ "${USE_WANDB}" = true ]; then
  WANDB_ARGS="--use_wandb --wandb_project ${WANDB_PROJECT} --wandb_run_name ${RUN_NAME}"
fi

#######################################
# Train
#######################################

python train_splicevi.py \
  --train_mdata_path "${TRAIN_MDATA_PATH}" \
  --model_dir "${MODEL_DIR}" \
  --batch_key "${BATCH_KEY}" \
  --seed "${SEED}" \
  ${WANDB_ARGS} \
  ${EXTRA_ARGS}

echo "[DONE] Model saved to ${MODEL_DIR}"
