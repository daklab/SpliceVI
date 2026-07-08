#!/usr/bin/env bash
#SBATCH --job-name=splicevi_train_basic
#SBATCH --output=logs/splicevi_train_%j.out
#SBATCH --error=logs/splicevi_train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --exclude=ne1dg7-001,ne1dg7-002,ne1dg7-003,ne1dg7-004,ne1dg7-005,ne1dg7-006,ne1dg7-007,ne1dg7-008,ne1dg7-009,ne1dg7-010
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
-u svai

set -euo pipefail
# train_splicevi.sh
#
# Minimal Slurm job script to:
#   1. Load a MuData file
#   2. Set up a SPLICEVI model using chosen hyperparameters
#   3. Train the model
#   4. Save the trained model to a specified directory
#
# This script is intentionally focused on TRAINING ONLY.
# No evaluation, UMAP plots, clustering, or imputation.
#
# Submit with:
#   sbatch train_splicevi.sh

#######################################
# USER CONFIGURATION
#######################################

# 1) Paths
TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu"
# TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS_SUBSAMPLED_N_50000_SUBSAMPLED_N_25000_SUBSAMPLED_N_10000.h5mu"
# TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS_SUBSAMPLED_N_50000_SUBSAMPLED_N_25000.h5mu"
# TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS_SUBSAMPLED_N_50000.h5mu"
# TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_25_PERCENT_train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"
# TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_25_PERCENT_train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"
# TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_50_PERCENT_train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"
# TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_75_PERCENT_train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406.h5mu"
# TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS_SUBSAMPLED_N_50000_SUBSAMPLED_N_25000_SUBSAMPLED_N_10000_SUBSAMPLED_N_1000.h5mu"
# TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_75_PERCENT_train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS_SUBSAMPLED_N_50000_SUBSAMPLED_N_25000_SUBSAMPLED_N_10000_SUBSAMPLED_N_1000.h5mu"
# Base directory where trained models will be stored
MODEL_DIR_BASE="models"

# Location of the training Python script (in your repo)
SCRIPT_PATH="train_splicevi.py"

# Batch column in obs; set to "None" to disable batch correction
BATCH_KEY="mouse.id"

# 2) Conda / environment
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="splicevi-env"

# 3) Core hyperparameters
MAX_EPOCHS=800          # Total training epochs
LR=1e-4                 # Learning rate
BATCH_SIZE=256          # Minibatch size
N_EPOCHS_KL_WARMUP=200  # KL warmup epochs
N_LATENT=30             # Latent dimensionality
DROPOUT_RATE=0.01       # Model dropout rate
SPLICING_LOSS_TYPE="dirichlet_multinomial"  # binomial | beta_binomial | dirichlet_multinomial

# 4) Optional architecture knobs (SPLICEVI __init__ parameters)
MODALITY_WEIGHTS="per_dimension_weighted_average"             # equal | cell | universal | concatenate | per_dimension_weighted_average
MODALITY_PENALTY="Jeffreys"          # Jeffreys | MMD | None
N_LAYERS_ENCODER=2
N_LAYERS_DECODER=2                   # not used if linear
USE_BATCH_NORM="none"                # encoder | decoder | none | both
USE_LAYER_NORM="both"                # encoder | decoder | none | both
LATENT_DISTRIBUTION="normal"         # normal | ln
DEEPLY_INJECT_COVARIATES=false
ENCODE_COVARIATES=false
GENE_LIKELIHOOD="zinb"               # zinb | nb | poisson
DISPERSION="gene"                    # gene | gene-batch | gene-label | gene-cell
DM_CONCENTRATION="atse"              # atse | scalar
SPLICING_ENCODER_ARCHITECTURE="partial"  # vanilla | partial
SPLICING_DECODER_ARCHITECTURE="vanilla"   # vanilla | linear
EXPRESSION_ARCHITECTURE="vanilla"         # vanilla | linear
ENCODER_HIDDEN_DIM=128
CODE_DIM=64
H_HIDDEN_DIM=128
POOL_MODE="mean"                     # mean | sum
MAX_NOBS=-1                          # cap for scatter chunking (-1 disables)
INITIALIZE_EMBEDDINGS_FROM_PCA=true
FULLY_PAIRED=false

WEIGHT_DECAY=1e-4
EARLY_STOPPING_PATIENCE=50
LR_SCHEDULER_TYPE="step"          # plateau | step
LR_FACTOR=0.1
LR_PATIENCE=30
STEP_SIZE=650
GRADIENT_CLIPPING=true
GRADIENT_CLIPPING_MAX_NORM=5.0

# 5) Optional: W&B configuration
USE_WANDB=true                       # Set to "false" to disable W&B logging
WANDB_PROJECT="MLCB_SUBMISSION"      # Required if USE_WANDB=true
WANDB_ENTITY=""                      # Optional: W&B entity (team). Leave empty if not used.
WANDB_GROUP="splicevi_basic"    # Optional group label
WANDB_RUN_NAME_PREFIX="basic_train"  # Prefix for W&B run names
WANDB_LOG_FREQ=1000                  # Log frequency for wandb.watch


#######################################
# DERIVED SETTINGS
#######################################

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="splicevi_basic_${TIMESTAMP}"

MODEL_DIR="${MODEL_DIR_BASE}/${RUN_NAME}"
mkdir -p "${MODEL_DIR}"

echo "=================================================================="
echo "[JOB] SPLICEVI basic training job"
echo "[JOB] Slurm job ID          : ${SLURM_JOB_ID:-N/A}"
echo "[JOB] Run name              : ${RUN_NAME}"
echo "[JOB] Training MuData path  : ${TRAIN_MDATA_PATH}"
echo "[JOB] Model output directory: ${MODEL_DIR}"
echo "=================================================================="


#######################################
# ENVIRONMENT SETUP
#######################################

echo "[ENV] Activating conda environment '${ENV_NAME}'..."
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
echo "[ENV] Python executable: $(which python)"
echo "[ENV] Python version  : $(python -V)"
echo


#######################################
# BUILD OPTIONAL W&B ARGUMENT STRING
#######################################

WANDB_ARGS=""
if [ "${USE_WANDB}" = true ]; then
  echo "[W&B] Enabling Weights & Biases logging."
  WANDB_ARGS+=" --use_wandb"
  if [ -n "${WANDB_PROJECT}" ]; then
    WANDB_ARGS+=" --wandb_project ${WANDB_PROJECT}"
  else
    echo "[W&B] ERROR: USE_WANDB=true but WANDB_PROJECT is empty."
    exit 1
  fi
  if [ -n "${WANDB_ENTITY}" ]; then
    WANDB_ARGS+=" --wandb_entity ${WANDB_ENTITY}"
  fi
  if [ -n "${WANDB_GROUP}" ]; then
    WANDB_ARGS+=" --wandb_group ${WANDB_GROUP}"
  fi
  WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX}_${RUN_NAME}"
  WANDB_ARGS+=" --wandb_run_name ${WANDB_RUN_NAME}"
  WANDB_ARGS+=" --wandb_log_freq ${WANDB_LOG_FREQ}"
else
  echo "[W&B] W&B logging disabled for this run."
fi
echo


#######################################
# LAUNCH TRAINING
#######################################

echo "[RUN] Launching SPLICEVI training script..."
set -x

python "${SCRIPT_PATH}" \
  --train_mdata_path "${TRAIN_MDATA_PATH}" \
  --model_dir "${MODEL_DIR}" \
  --batch_key "${BATCH_KEY}" \
  --max_epochs "${MAX_EPOCHS}" \
  --lr "${LR}" \
  --batch_size "${BATCH_SIZE}" \
  --n_epochs_kl_warmup "${N_EPOCHS_KL_WARMUP}" \
  --n_latent "${N_LATENT}" \
  --dropout_rate "${DROPOUT_RATE}" \
  --splicing_loss_type "${SPLICING_LOSS_TYPE}" \
  --modality_weights "${MODALITY_WEIGHTS}" \
  --modality_penalty "${MODALITY_PENALTY}" \
  --n_layers_encoder "${N_LAYERS_ENCODER}" \
  --n_layers_decoder "${N_LAYERS_DECODER}" \
  --use_batch_norm "${USE_BATCH_NORM}" \
  --use_layer_norm "${USE_LAYER_NORM}" \
  --latent_distribution "${LATENT_DISTRIBUTION}" \
  --deeply_inject_covariates "${DEEPLY_INJECT_COVARIATES}" \
  --encode_covariates "${ENCODE_COVARIATES}" \
  --gene_likelihood "${GENE_LIKELIHOOD}" \
  --dispersion "${DISPERSION}" \
  --dm_concentration "${DM_CONCENTRATION}" \
  --encoder_hidden_dim "${ENCODER_HIDDEN_DIM}" \
  --code_dim "${CODE_DIM}" \
  --h_hidden_dim "${H_HIDDEN_DIM}" \
  --pool_mode "${POOL_MODE}" \
  --max_nobs "${MAX_NOBS}" \
  --splicing_encoder_architecture "${SPLICING_ENCODER_ARCHITECTURE}" \
  --splicing_decoder_architecture "${SPLICING_DECODER_ARCHITECTURE}" \
  --expression_architecture "${EXPRESSION_ARCHITECTURE}" \
  --initialize_embeddings_from_pca "${INITIALIZE_EMBEDDINGS_FROM_PCA}" \
  --fully_paired "${FULLY_PAIRED}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
  --lr_factor "${LR_FACTOR}" \
  --lr_patience "${LR_PATIENCE}" \
  --step_size "${STEP_SIZE}" \
  --gradient_clipping "${GRADIENT_CLIPPING}" \
  --gradient_clipping_max_norm "${GRADIENT_CLIPPING_MAX_NORM}" \
  ${WANDB_ARGS}

set +x

echo
echo "[DONE] SPLICEVI basic training job finished."
echo "[DONE] Trained model saved to: ${MODEL_DIR}"
