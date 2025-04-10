#!/usr/bin/env bash
#
# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

if [ -z "${BASH_VERSION}" ]; then
	echo "Please use bash to run this script." >&2
	exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export LOGLEVEL="${LOGLEVEL:-WARNING}"

export WANDB_ENTITY="alelab"

MODEL_NAME_OR_PATH="PKU-Alignment/alpaca-7b-reproduced"
COST_MODEL_NAME_OR_PATH="PKU-Alignment/beaver-7b-v1.0-cost"
REWARD_MODEL_NAME_OR_PATH="PKU-Alignment/beaver-7b-v1.0-reward"
timestamp="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${ROOT_DIR}/output/pd_alignment-${timestamp}"
unset HOSTFILE
ZERO_STAGE=0
OFFLOAD="none"
SAFETY_RATIO_TOL=0.1
RESILIENT_COEFF=1.0
SCALE_COEFF=0.1
while [[ "$#" -gt 0 ]]; do
	arg="$1"
	shift
	case "${arg}" in
		--model_name_or_path)
			MODEL_NAME_OR_PATH="$1"
			shift
			;;
		--model_name_or_path=*)
			MODEL_NAME_OR_PATH="${arg#*=}"
			;;
		--output_dir)
			OUTPUT_DIR="$1"
			shift
			;;
		--output_dir=*)
			OUTPUT_DIR="${arg#*=}"
			;;
		--hostfile)
			HOSTFILE="$1"
			shift
			;;
		--hostfile=*)
			HOSTFILE="${arg#*=}"
			;;
		--zero_stage)
			ZERO_STAGE="$1"
			shift
			;;
		--zero_stage=*)
			ZERO_STAGE="${arg#*=}"
			;;
		--offload)
			OFFLOAD="$1"
			shift
			;;
		--offload=*)
			OFFLOAD="${arg#*=}"
			;;
		--safety_ratio_tol)
			SAFETY_RATIO_TOL="$1"
			shift
			;;
		--safety_ratio_tol=*)
			SAFETY_RATIO_TOL="${arg#*=}"
			;;
		--resilient_coeff)
			RESILIENT_COEFF="$1"
			shift
			;;
		--resilient_coeff=*)
			RESILIENT_COEFF="${arg#*=}"
			;;
		--scale_coeff)
			SCALE_COEFF="$1"
			shift
			;;
		--scale_coeff=*)
			SCALE_COEFF="${arg#*=}"
			;;
		*)
			echo "Unknown parameter passed: '${arg}'" >&2
			exit 1
			;;
	esac
done

mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" &>/dev/null && pwd)"
if [[ ! -f "${OUTPUT_DIR}/.gitignore" ]]; then
	echo '*' >"${OUTPUT_DIR}/.gitignore"
fi

cp -f "$0" "${OUTPUT_DIR}/script.sh"

MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
	comm -23 \
		<(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
		<(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
		shuf | head -n 1
)"

DEEPSPEED_ARGS=()
if [[ -n "${HOSTFILE+x}" ]]; then
	DEEPSPEED_ARGS+=("--hostfile" "${HOSTFILE}")
fi
DEEPSPEED_ARGS+=("--master_port" "${MASTER_PORT}")

exec 1> >(tee "${OUTPUT_DIR}/stdout.log" >&1) 2> >(tee "${OUTPUT_DIR}/stderr.log" >&2)


CUDA_VISIBLE_DEVICES=0,1 deepspeed "${DEEPSPEED_ARGS[@]}" \
	--module safe_rlhf.algorithms.pd_alignment \
	--train_datasets PKU-SafeRLHF/train \
	--eval_datasets PKU-SafeRLHF/test \
	--model_name_or_path "${MODEL_NAME_OR_PATH}" \
	--max_length 512 \
	--trust_remote_code True \
	--epochs 1 \
	--per_device_train_batch_size 1 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps 16 \
	--gradient_checkpointing \
	--learning_rate 1e-6 \
	--lr_scheduler_type cosine \
	--lr_warmup_ratio 0.03 \
	--weight_decay 0.05 \
	--seed 42 \
	--need_eval \
	--eval_strategy epoch \
	--resilient_coeff "${RESILIENT_COEFF}" \
	--dual_step_size 0.01 \
	--scale_coeff "${SCALE_COEFF}" \
	--safety_ratio_tol "${SAFETY_RATIO_TOL}" \
	--output_dir "${OUTPUT_DIR}" \
	--log_type wandb \
	--log_project Safe-RLHF-PDA \
	--zero_stage "${ZERO_STAGE}" \
	--offload "${OFFLOAD}" \
	--bf16 False \
	--tf32 True \
	--cost_model_name_or_path "${COST_MODEL_NAME_OR_PATH}" \
	--reward_model_name_or_path "${REWARD_MODEL_NAME_OR_PATH}"
