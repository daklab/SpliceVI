#!/usr/bin/env bash
#SBATCH --job-name=splicevi_eval
#SBATCH --output=logs/splicevi_eval_%j.out
#SBATCH --error=logs/splicevi_eval_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --exclude=ne1dg7-001,ne1dg7-002,ne1dg7-003,ne1dg7-004,ne1dg7-005,ne1dg7-006,ne1dg7-007,ne1dg7-008,ne1dg7-009,ne1dg7-010
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00

set -euo pipefail

# eval_splicevi.sh
#
# Minimal Slurm job script to:
#   1. Load TRAIN and TEST MuData
#   2. Load a trained SPLICEVI model from MODEL_DIR
#   3. Run selected evaluation blocks (UMAP, clustering, metrics, imputation)
#   4. Save evaluation figures/CSVs under a per-run directory
#
# Submit with:
#   sbatch eval_splicevi.sh

#######################################
# USER CONFIGURATION
#######################################

# 1) Data paths
TRAIN_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/train_70_30_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu"
TEST_MDATA_PATH="/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu"

# Optional mapping CSV (set to empty to skip)
# MAPPING_CSV="/gpfs/commons/home/svaidyanathan/repos/multivi_tools_splicing/multivi_splice_utils/runfiles/tissue_celltype_mapping.csv"
: "${MAPPING_CSV:=}"

# Optional masked TEST MuData paths for imputation
# MASKED_TEST_MDATA_PATHS="\
# /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_25_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu \
# /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_50_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu \
# /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/MOUSE_SPLICING_FOUNDATION/MODEL_INPUT/102025/MASKED_75_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu"


MASKED_TEST_MDATA_PATHS="\
/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/resampled/RESAMPLED_10_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu \
/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/resampled/RESAMPLED_50_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu \
/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/resampled/RESAMPLED_75_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu \
/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/resampled/RESAMPLED_90_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu \
/gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/resampled_1/RESAMPLED_99_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu"
# MASKED_TEST_MDATA_PATHS="\
# /gpfs/commons/groups/knowles_lab/Karin/Leaflet-analysis-WD/EasySci2024/LeafletFA/resampled_1/RESAMPLED_99_PERCENT_test_30_70_model_ready_combined_gene_expression_aligned_splicing_20251009_024406_UPDATEDOBS.h5mu"
# Set to "true" if MASKED_TEST_MDATA_PATHS are multinomial-resampled files
# (produced by multinomial_resampling_masking.py). Evaluation will use
# junc_ratio_original as ground truth and only score originally non-zero entries.
# Set to "false" for legacy hard-masked files that carry junc_ratio_masked_original
# and junc_ratio_masked_bin_mask layers.
MASKED_TEST_MDATA_IS_RESAMPLED="true"

# If "true", also exclude junctions where junc_ratio_original == 1.0 from eval
# (on top of always-excluded == 0.0). Only applies when IS_RESAMPLED=true.
IMPUTE_FILTER_BOUNDARY_PSI="true"

# Minimum original ATSE count (cell_by_cluster_matrix_original) required to
# include a junction-cell entry in the imputation eval.
# Set to -1 to disable this filter.
MIN_ATSE_COUNT=30

# 2) Single model directory to evaluate

# MODEL_DIR="models/splicevi_basic_20260420_115901"  #batch key 1
# MODEL_DIR="models/splicevi_basic_20260506_123318"  #batch key 2
# MODEL_DIR="models/splicevi_basic_20260506_111916"  #batch key 3

# MODEL_DIR="models/splicevi_basic_20260417_142643"  #no batch key 1
# MODEL_DIR="models/splicevi_basic_20260420_101631"  #no batch key 2
# MODEL_DIR="models/splicevi_basic_20260420_101634"  #no batch key 3

# MODEL_DIR="models/splicevi_basic_20260508_155550"
# MODEL_DIR="models/splicevi_basic_20260508_230756"
# MODEL_DIR="models/splicevi_basic_20260509_112050"
# MODEL_DIR="models/splicevi_basic_20260509_112142"

MODEL_DIR="models/splicevi_basic_20260511_111304" #1e-5 8X splicing only
# MODEL_DIR="models/splicevi_basic_20260511_111722" #1e-5 1X flip
# MODEL_DIR="models/splicevi_basic_20260511_111749" #1e-4 1X flip

# MODEL_DIR="models/splicevi_basic_20260515_145208" #LeakyRELU activation and then linear 
MODEL_DIR="models/splicevi_basic_20260515_145049" #ReLU aciation

#basic_train_splicevi_basic_20260406_162253 - linear yes batch

MODEL_DIR="models/splicevi_basic_20260525_172744"
MODEL_DIR="models/splicevi_basic_20260526_011702"



# 2.5) Batch key used during training (set to "None" to disable)
BATCH_KEY="mouse.id"
# 3) Evaluation blocks to run
# These must match argparse in eval_splicevi.py (nargs="+")
# test_impute = perfect/upper-bound baseline: runs unmasked test mdata through
#               the model and compares imputed PSI against junc_ratio.
EVALS=(
  latent_visualization
  # # clustering
  # # train_eval
  test_eval
  cross_fold_classification
  # # # age_r2_heatmap
  # # subcluster_split_eval
  # masked_impute
  # test_impute
)

# Which split(s) to visualize when latent_visualization is enabled
# Options: "train" | "test" | "both"
LATENT_VIZ_SPLITS="test"

# Which embedding type(s) to compute for latent_visualization
# Options: "umap" | "tsne" | "umap tsne"
LATENT_VIZ_TYPES=(
  # "umap"
  "tsne"
)

# If "true", compute mean-Bayes baseline for imputation evals (masked_impute only,
# evaluated on the "dropped" entries). Set MEAN_BAYES_GROUP_BY to an obs field
# (e.g. "broad_cell_type") to use per-group priors, or "None" for a global prior.
DO_MEAN_BAYES_IMPUTE="true"
MEAN_BAYES_GROUP_BY="None"

# Restrict imputation evals to the top-N most highly variable junctions.
# HVJ are computed once from the unmasked test MuData (junc_ratio layer) and
# the same junction set is applied across test_impute and all masked_impute files
# so results are directly comparable. Set to -1 to disable (use all junctions).
IMPUTE_TOP_N_HVJ=10000

# Cross-fold classification baselines:
# If "true", run DummyClassifier(strategy='prior') on every fold alongside
# the real classifiers — gives the random-chance F1 floor.
DO_CROSS_FOLD_DUMMY="true"

# If "true", fit real classifiers on randomly permuted training labels
# (evaluated on real held-out labels) — model should fail here if learning real signal.
DO_CROSS_FOLD_LABEL_PERMUTE="true"

# If "true", write per_label_f1_<target>.csv files with per-class F1 scores
# for every fold/space/classifier combination (f1 only, not precision/recall/accuracy).
OUTPUT_PER_LABEL_F1_CSV="true"

# 4) UMAP and imputation config
UMAP_TOP_N_CELLTYPES=15       # kept for compatibility (currently unused in plotting logic)
IMPUTE_BATCH_SIZE=512         # set to -1 to disable batching (one big batch per masked file)

# List of obs keys to use for coloring TRAIN UMAPs for each latent space
UMAP_OBS_KEYS=(
  "broad_cell_type"
  # "medium_cell_type"
  # "mouse.id"
  # "tissue_celltype"
  # "tissue"
  # "tissue_celltype"
)

# 4.4) Subcluster split evaluation config
SUBCLUSTER_OBS_KEY="broad_cell_type"
SUBCLUSTER_CELL_TYPES=()
# Leave array empty () to run all cell types in SUBCLUSTER_OBS_KEY
SUBCLUSTER_K_VALUES=(8)
SUBCLUSTER_SPLITS="test"          # train | test | both
SUBCLUSTER_RANDOM_SEED=42
SUBCLUSTER_EMBEDDING="tsne"       # umap | tsne

# 4.6) Junction PSI coloring for latent viz embeddings
# Each ID must match a var_name in the splicing modality.
# Leave array empty () to skip PSI coloring.
# Cells where psi_mask == 0 are shown in gray; observed cells get a 0–1 viridis colorbar.
JUNCTION_COLOR_IDS=("chr17_23821690_23822412_+" "chr17_23821690_23821920_+")
# Example:
# JUNCTION_COLOR_IDS=("chr1:12345:12500:clu_1_+" "chr2:9000:9100:clu_2_-")

# 4.5) Cross-fold classification config
CROSS_FOLD_SPLITS="train"  # train | test | both
CROSS_FOLD_TARGETS=(
  # "broad_cell_type"
  # "mouse.id"
  "tissue_celltype"
  # "tissue"
)
CROSS_FOLD_K=5
CROSS_FOLD_CLASSIFIERS=(
  "logreg"
  # "rf"
)
CROSS_FOLD_METRICS=(
  # "accuracy"
  "f1_weighted"
  # "precision_weighted"
  # "recall_weighted"
)

# 5) Conda / script locations
CONDA_BASE="/gpfs/commons/home/svaidyanathan/miniconda3"
ENV_NAME="splicevi-env"
SCRIPT_PATH="eval_splicevi.py"   # relative to repo root

# 6) W&B configuration (optional)
USE_WANDB=true                        # set to "false" to disable W&B logging
WANDB_PROJECT="MLCB_SUBMISSION"       # required if USE_WANDB=true
WANDB_ENTITY=""                       # optional W&B entity (team)
WANDB_GROUP="splicevi_eval"           # optional W&B group name
WANDB_RUN_NAME_PREFIX="basic_eval"    # prefix for run names
WANDB_LOG_FREQ=1000                   # how often to log from wandb.watch

# 7) Output directory for eval run
BASE_RUN_DIR="/gpfs/commons/home/svaidyanathan/repos/SpliceVI/logs"

#######################################
# DERIVED SETTINGS
#######################################

TS=$(date +"%Y%m%d_%H%M%S")
MODEL_BASENAME=$(basename "${MODEL_DIR}")
RUN_NAME="eval_${MODEL_BASENAME}_${TS}"

RUN_DIR="${BASE_RUN_DIR}/${RUN_NAME}"
FIG_DIR="${RUN_DIR}/figures"
mkdir -p "${FIG_DIR}"

echo "=================================================================="
echo "[JOB] SPLICEVI basic eval job"
echo "[JOB] Slurm job ID           : ${SLURM_JOB_ID:-N/A}"
echo "[JOB] Run name               : ${RUN_NAME}"
echo "[JOB] MODEL_DIR              : ${MODEL_DIR}"
echo "[JOB] BATCH_KEY              : ${BATCH_KEY}"
echo "[JOB] TRAIN_MDATA_PATH       : ${TRAIN_MDATA_PATH}"
echo "[JOB] TEST_MDATA_PATH        : ${TEST_MDATA_PATH}"
echo "[JOB] Mapping CSV            : ${MAPPING_CSV:-"(none)"}"
echo "[JOB] Eval output directory  : ${RUN_DIR}"
echo "[JOB] Figures directory      : ${FIG_DIR}"
echo "=================================================================="
echo "[JOB] MIN_ATSE_COUNT              : ${MIN_ATSE_COUNT}"
echo "[JOB] IMPUTE_TOP_N_HVJ            : ${IMPUTE_TOP_N_HVJ}"
echo "[JOB] DO_MEAN_BAYES_IMPUTE        : ${DO_MEAN_BAYES_IMPUTE}"
echo "[JOB] MEAN_BAYES_GROUP_BY         : ${MEAN_BAYES_GROUP_BY}"
echo "[JOB] DO_CROSS_FOLD_DUMMY         : ${DO_CROSS_FOLD_DUMMY}"
echo "[JOB] DO_CROSS_FOLD_LABEL_PERMUTE : ${DO_CROSS_FOLD_LABEL_PERMUTE}"
echo "[JOB] OUTPUT_PER_LABEL_F1_CSV     : ${OUTPUT_PER_LABEL_F1_CSV}"
echo "[JOB] MASKED_TEST_MDATA_PATHS:"
for p in ${MASKED_TEST_MDATA_PATHS}; do
  echo "         - ${p}"
done
echo "[JOB] UMAP_OBS_KEYS:"
for k in "${UMAP_OBS_KEYS[@]}"; do
  echo "         - ${k}"
done
echo "[JOB] LATENT_VIZ_SPLITS          : ${LATENT_VIZ_SPLITS}"
echo "[JOB] LATENT_VIZ_TYPES           : ${LATENT_VIZ_TYPES[*]}"
echo "[JOB] CROSS_FOLD_SPLITS : ${CROSS_FOLD_SPLITS}"
echo "[JOB] CROSS_FOLD_TARGETS:"
for t in "${CROSS_FOLD_TARGETS[@]}"; do
  echo "         - ${t}"
done
echo "[JOB] CROSS_FOLD_CLASSIFIERS:"
for c in "${CROSS_FOLD_CLASSIFIERS[@]}"; do
  echo "         - ${c}"
done
echo "[JOB] CROSS_FOLD_METRICS:"
for m in "${CROSS_FOLD_METRICS[@]}"; do
  echo "         - ${m}"
done
echo "[JOB] CROSS_FOLD_K      : ${CROSS_FOLD_K}"
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
  WANDB_RUN_NAME="${WANDB_RUN_NAME_PREFIX}_${MODEL_BASENAME}"
  WANDB_ARGS+=" --wandb_run_name ${WANDB_RUN_NAME}"
  WANDB_ARGS+=" --wandb_log_freq ${WANDB_LOG_FREQ}"
else
  echo "[W&B] W&B logging disabled for this run."
fi
echo

#######################################
# PREPARE MULTI-VALUE ARGUMENTS
#######################################

EVALS_JOINED="${EVALS[*]}"
UMAP_OBS_KEYS_JOINED="${UMAP_OBS_KEYS[*]}"
CROSS_FOLD_TARGETS_JOINED="${CROSS_FOLD_TARGETS[*]}"
CROSS_FOLD_CLASSIFIERS_JOINED="${CROSS_FOLD_CLASSIFIERS[*]}"
CROSS_FOLD_METRICS_JOINED="${CROSS_FOLD_METRICS[*]}"
LATENT_VIZ_TYPES_JOINED="${LATENT_VIZ_TYPES[*]}"
SUBCLUSTER_K_VALUES_JOINED="${SUBCLUSTER_K_VALUES[*]}"
JUNCTION_COLOR_IDS_JOINED="${JUNCTION_COLOR_IDS[*]}"

#######################################
# LAUNCH EVALUATION
#######################################

echo "[RUN] Launching SPLICEVI eval script..."
set -x

python "${SCRIPT_PATH}" \
  --train_mdata_path "${TRAIN_MDATA_PATH}" \
  --test_mdata_path "${TEST_MDATA_PATH}" \
  --model_dir "${MODEL_DIR}" \
  --batch_key "${BATCH_KEY}" \
  --fig_dir "${FIG_DIR}" \
  ${MAPPING_CSV:+--mapping_csv "${MAPPING_CSV}"} \
  --impute_batch_size "${IMPUTE_BATCH_SIZE}" \
  --latent_viz_splits "${LATENT_VIZ_SPLITS}" \
  --latent_viz_types ${LATENT_VIZ_TYPES_JOINED} \
  --umap_top_n_celltypes "${UMAP_TOP_N_CELLTYPES}" \
  --umap_obs_keys ${UMAP_OBS_KEYS_JOINED} \
  --cross_fold_splits "${CROSS_FOLD_SPLITS}" \
  --cross_fold_targets ${CROSS_FOLD_TARGETS_JOINED} \
  --cross_fold_k "${CROSS_FOLD_K}" \
  --cross_fold_classifiers ${CROSS_FOLD_CLASSIFIERS_JOINED} \
  --cross_fold_metrics ${CROSS_FOLD_METRICS_JOINED} \
  --evals ${EVALS_JOINED} \
  ${MASKED_TEST_MDATA_PATHS:+--masked_test_mdata_paths ${MASKED_TEST_MDATA_PATHS}} \
  ${MASKED_TEST_MDATA_IS_RESAMPLED:+$([ "${MASKED_TEST_MDATA_IS_RESAMPLED}" = "true" ] && echo "--masked_test_mdata_is_resampled")} \
  ${IMPUTE_FILTER_BOUNDARY_PSI:+$([ "${IMPUTE_FILTER_BOUNDARY_PSI}" = "true" ] && echo "--impute_filter_boundary_psi")} \
  --min_atse_count "${MIN_ATSE_COUNT}" \
  --impute_top_n_hvj "${IMPUTE_TOP_N_HVJ}" \
  $([ "${DO_MEAN_BAYES_IMPUTE}" = "true" ] && echo "--mean_bayes_impute") \
  --mean_bayes_group_by "${MEAN_BAYES_GROUP_BY}" \
  $([ "${DO_CROSS_FOLD_DUMMY}" = "true" ] && echo "--cross_fold_dummy_classifier") \
  $([ "${DO_CROSS_FOLD_LABEL_PERMUTE}" = "true" ] && echo "--cross_fold_label_permute") \
  $([ "${OUTPUT_PER_LABEL_F1_CSV}" = "true" ] && echo "--output_per_label_f1_csv") \
  --subcluster_obs_key "${SUBCLUSTER_OBS_KEY}" \
  --subcluster_k_values ${SUBCLUSTER_K_VALUES_JOINED} \
  --subcluster_splits "${SUBCLUSTER_SPLITS}" \
  --subcluster_random_seed "${SUBCLUSTER_RANDOM_SEED}" \
  --subcluster_embedding "${SUBCLUSTER_EMBEDDING}" \
  ${SUBCLUSTER_CELL_TYPES[@]:+--subcluster_cell_type "${SUBCLUSTER_CELL_TYPES[@]}"} \
  ${JUNCTION_COLOR_IDS[@]:+--junction_color_ids "${JUNCTION_COLOR_IDS[@]}"} \
  ${WANDB_ARGS}

set +x

echo
echo "[DONE] SPLICEVI basic eval job finished."
echo "[DONE] Eval outputs in: ${RUN_DIR}"
