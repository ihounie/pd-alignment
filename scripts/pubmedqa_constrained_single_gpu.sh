#!/usr/bin/env bash
#
# Helper script to launch PubMedQA constrained supervised fine-tuning on a *single* GPU.
# Allows overriding of key hyper-parameters via CLI flags.
#
# Example usages
# --------------
#   sh scripts/pubmedqa_constrained_single_gpu.sh                             # run with defaults
#   sh scripts/pubmedqa_constrained_single_gpu.sh --epochs 3 --lr 1e-5        # override flags
#   sh scripts/pubmedqa_constrained_single_gpu.sh \
#       --ce_threshold_yes 0.8 --ce_threshold_no 0.8 --dual_step_size 0.05
#
# Notes
# -----
#   • DeepSpeed is still used even on a single GPU for consistency with multi-GPU launches.
#   • A timestamped output directory is created under `output/` and ignored by Git.
# -----------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-INFO}"

# ---------------------------------------------------------------------------
# Default hyper-parameters (can be overridden via script flags)
# ---------------------------------------------------------------------------
MODEL_NAME_OR_PATH="meta-llama/Llama-3.2-3B"
TRAIN_DATASETS="qiaojin/PubMedQA:pqa_labeled:train"
EVAL_DATASETS="qiaojin/PubMedQA:pqa_labeled:test"
CACHE_DIR="cache/pubmedqa"
EPOCHS=5
PER_DEVICE_TRAIN_BATCH_SIZE=16
PER_DEVICE_EVAL_BATCH_SIZE=16
GRADIENT_ACC_STEPS=4
LEARNING_RATE=1e-3
MAX_LENGTH=512
SEED=42
ZERO_STAGE=0
OFFLOAD="none"
LOG_TYPE="wandb"
LOG_PROJECT="pubmedqa"
# Constraint defaults
CE_THRESHOLD_YES=0.15
CE_THRESHOLD_NO=0.15
DUAL_STEP_SIZE=0.5 # 0.5
LAMBDA_YES_INIT=0.0
LAMBDA_NO_INIT=0.0
DUAL_WARMUP_EPOCHS=3 # 3
DUAL_UPDATE_INTERVAL=0
DUAL_WEIGHT_DECAY=0.1
KL_COEF=0.01

EXTRA_ARGS=()
# ---------------------------------------------------------------------------
# Simple argument parsing (override defaults)
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  key="$1"; shift
  case "${key}" in
    --model_name_or_path)          MODEL_NAME_OR_PATH="$1"; shift ;;
    --train_datasets)              TRAIN_DATASETS="$1"; shift ;;
    --eval_datasets)               EVAL_DATASETS="$1"; shift ;;
    --epochs)                      EPOCHS="$1"; shift ;;
    --per_device_train_batch_size) PER_DEVICE_TRAIN_BATCH_SIZE="$1"; shift ;;
    --per_device_eval_batch_size)  PER_DEVICE_EVAL_BATCH_SIZE="$1"; shift ;;
    --gradient_accumulation_steps) GRADIENT_ACC_STEPS="$1"; shift ;;
    --lr|--learning_rate)          LEARNING_RATE="$1"; shift ;;
    --max_length)                  MAX_LENGTH="$1"; shift ;;
    --seed)                        SEED="$1"; shift ;;
    --zero_stage)                  ZERO_STAGE="$1"; shift ;;
    --offload)                     OFFLOAD="$1"; shift ;;
    --log_type)                    LOG_TYPE="$1"; shift ;;
    --log_project)                 LOG_PROJECT="$1"; shift ;;
    --ce_threshold_yes)            CE_THRESHOLD_YES="$1"; shift ;;
    --ce_threshold_no)             CE_THRESHOLD_NO="$1"; shift ;;
    --dual_step_size)              DUAL_STEP_SIZE="$1"; shift ;;
    --lambda_yes_init)             LAMBDA_YES_INIT="$1"; shift ;;
    --lambda_no_init)              LAMBDA_NO_INIT="$1"; shift ;;
    --dual_warmup_epochs)          DUAL_WARMUP_EPOCHS="$1"; shift ;;
    --dual_update_interval)        DUAL_UPDATE_INTERVAL="$1"; shift ;;
    --dual_weight_decay)           DUAL_WEIGHT_DECAY="$1"; shift ;;
    --kl_coef)                     KL_COEF="$1"; shift ;;
    *)                             EXTRA_ARGS+=("${key}") ;;
  esac
done

# ---------------------------------------------------------------------------
# Output directory (timestamped)
# ---------------------------------------------------------------------------
timestamp="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${ROOT_DIR}/output/pubmedqa_constrained-${timestamp}"
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
  echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

# Keep a copy of this launch script
cp -f "$0" "${OUTPUT_DIR}/launch.sh"

# ---------------------------------------------------------------------------
# Launch DeepSpeed
# ---------------------------------------------------------------------------
deepspeed --master_port 29502 --module safe_rlhf.algorithms.pubmedqa_constrained \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --train_datasets "${TRAIN_DATASETS}" \
  --eval_datasets "${EVAL_DATASETS}" \
  --epochs "${EPOCHS}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACC_STEPS}" \
  --max_length "${MAX_LENGTH}" \
  --lr "${LEARNING_RATE}" \
  --seed "${SEED}" \
  --zero_stage "${ZERO_STAGE}" \
  --offload "${OFFLOAD}" \
  --need_eval True \
  --bf16 True \
  --output_dir "${OUTPUT_DIR}" \
  --log_type "${LOG_TYPE}" \
  --log_project "${LOG_PROJECT}" \
  --ce_threshold_yes "${CE_THRESHOLD_YES}" \
  --ce_threshold_no "${CE_THRESHOLD_NO}" \
  --dual_step_size "${DUAL_STEP_SIZE}" \
  --lambda_yes_init "${LAMBDA_YES_INIT}" \
  --lambda_no_init "${LAMBDA_NO_INIT}" \
  --dual_warmup_epochs "${DUAL_WARMUP_EPOCHS}" \
  --dual_update_interval "${DUAL_UPDATE_INTERVAL}" \
  --dual_weight_decay "${DUAL_WEIGHT_DECAY}" \
  --kl_coef "${KL_COEF}" \
  ${EXTRA_ARGS[@]:-} 