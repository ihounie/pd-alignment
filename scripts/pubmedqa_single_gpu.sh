#!/usr/bin/env bash
#
# Simple helper script to launch the PubMedQA supervised fine-tuning trainer on a
# **single** GPU with reasonable default hyper-parameters. You can override any
# of the defaults via command-line flags (see examples below).
#
# Example usages
# --------------
#   sh scripts/pubmedqa_single_gpu.sh                             # run with defaults
#   sh scripts/pubmedqa_single_gpu.sh --epochs 3 --lr 1e-5        # override flags
#   sh scripts/pubmedqa_single_gpu.sh --model_name_or_path "mistralai/Mistral-7B-v0.1"
#
# Notes
# -----
#   • DeepSpeed is still used even for a single GPU so that its zero-overhead
#     optimizer & logging utilities remain available. DeepSpeed will inject
#     `--local_rank` automatically; our trainer CLI already supports it.
#   • The script creates a timestamped `output/` directory to store checkpoints
#     and logs. A `.gitignore` is placed inside so that everything within is
#     ignored by Git.

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
KL_COEF=0.01
EXTRA_ARGS=()

# ---------------------------------------------------------------------------
# Parse user-provided overrides (very light argument parsing)
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  key="$1"; shift
  case "${key}" in
    --model_name_or_path)        MODEL_NAME_OR_PATH="$1"; shift ;;
    --train_datasets)            TRAIN_DATASETS="$1"; shift ;;
    --eval_datasets)             EVAL_DATASETS="$1"; shift ;;
    --epochs)                    EPOCHS="$1"; shift ;;
    --per_device_train_batch_size) PER_DEVICE_TRAIN_BATCH_SIZE="$1"; shift ;;
    --per_device_eval_batch_size) PER_DEVICE_EVAL_BATCH_SIZE="$1"; shift ;;
    --gradient_accumulation_steps) GRADIENT_ACC_STEPS="$1"; shift ;;
    --lr|--learning_rate)        LEARNING_RATE="$1"; shift ;;
    --max_length)                MAX_LENGTH="$1"; shift ;;
    --seed)                      SEED="$1"; shift ;;
    --zero_stage)                ZERO_STAGE="$1"; shift ;;
    --offload)                   OFFLOAD="$1"; shift ;;
    --log_type)                  LOG_TYPE="$1"; shift ;;
    --log_project)               LOG_PROJECT="$1"; shift ;;
    --kl_coef)                   KL_COEF="$1"; shift ;;
    *)                           # Forward any unrecognised flag directly to trainer
       EXTRA_ARGS+=("${key}") ;;
  esac
done

# ---------------------------------------------------------------------------
# Output directory (timestamped)
# ---------------------------------------------------------------------------
timestamp="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${ROOT_DIR}/output/pubmedqa-${timestamp}"
mkdir -p "${OUTPUT_DIR}"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
  echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

# Keep a copy of the launch script for reproducibility
cp -f "$0" "${OUTPUT_DIR}/launch.sh"

# ---------------------------------------------------------------------------
# Launch DeepSpeed (single GPU)
# ---------------------------------------------------------------------------
deepspeed --module safe_rlhf.algorithms.pubmedqa \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --reference_model_name_or_path "${MODEL_NAME_OR_PATH}" \
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
  --kl_coef "${KL_COEF}" \
  ${EXTRA_ARGS[@]:-} 