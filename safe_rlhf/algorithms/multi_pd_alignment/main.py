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
"""The main training script to run the DPO algorithm."""

import argparse

import deepspeed
import torch
import torch.distributed as dist
from transformers import SchedulerType
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available

from safe_rlhf.algorithms.multi_pd_alignment.trainer import MultiPdAlignementTrainer
from safe_rlhf.configs import get_deepspeed_eval_config, get_deepspeed_train_config
from safe_rlhf.datasets import parse_dataset
from safe_rlhf.logger import set_logger_level
from safe_rlhf.utils import seed_everything, str2bool


def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='deepspeed --module safe_rlhf.algorithms.multi_pd_alignment',
        description='Train language model with the DPO algorithm.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument(
        '--debug',
        type=str2bool,
        default=False,
        help='Whether to run in debug mode.',
    )
    model_parser.add_argument(
        '--model_name_or_path',
        type=str,
        help='Path to the model checkpoint or its name.',
        required=True,
    )
    model_parser.add_argument(
        '--cost_model_name_or_path',
        type=str,
        help='Path to the model checkpoint or its name.',
        required=True,
    )
    model_parser.add_argument(
        '--reward_model_name_or_path',
        type=str,
        help='Path to the model checkpoint or its name.',
        required=True,
    )
    model_parser.add_argument(
        '--normalize_cost',
        type=str2bool,
        default=True,
        help='Whether to normalize the cost.',
    )
    model_parser.add_argument(
        '--normalize_reward',
        type=str2bool,
        default=True,
        help='Whether to normalize the reward.',
    )

    model_parser.add_argument(
        "--recompute_costs",
        action="store_true",
        help="Force recomputation of costs even if cache exists",
    )
    model_parser.add_argument(
        "--recompute_rewards",
        action="store_true",
        help="Force recomputation of rewards even if cache exists",
    )
    model_parser.add_argument(
        "--recompute_baseline",
        action="store_true",
        help="Force recomputation of baseline even if cache exists",
    )

    model_parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="Directory to store cached computations",
    )

    model_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
    )
    model_parser.add_argument(
        '--trust_remote_code',
        type=str2bool,
        default=False,
        help='Whether to trust the remote code.',
    )
    # Lora args
    model_parser.add_argument(
        '--lora_r',
        type=int,
        default=8,
        help='The rank of the LoRA matrices.',
    )
    model_parser.add_argument(
        '--lora_alpha',
        type=float,
        default=1.0,
        help='The alpha of the LoRA matrices.',
    )
    model_parser.add_argument(
        '--lora_dropout',
        type=float,
        default=0.0,
        help='The dropout of the LoRA matrices.',
    )
    model_parser.add_argument(
        '--lora_target_modules',
        type=str,
        default=None,
        help='The target modules of the model.',
    )
    # Dataset
    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument(
        '--train_datasets',
        type=str,
        required=True,
    )
    dataset_parser.add_argument(
        '--eval_datasets',
        type=str,
        required=True,
    )
    dataset_parser.add_argument(
        '--num_classes',
        type=int,
        default=4,
        help='The number of classes in the dataset.',
    )  # TODO: infer from dataset

    # Training
    training_parser = parser.add_argument_group('training')
    training_parser.add_argument(
        '--scale_coeff',
        type=float,
        default=0.02,
        help='The coefficient for the KL divergence between the reference and actor policy.',
    )
    # Dual args
    training_parser.add_argument(
        '--resilient_coeff',
        type=float,
        default=1e-2,
        help='Dual weight decay to improve stability.',
    )
    training_parser.add_argument(
        '--dual_init',
        type=str,
        default='0.333,0.333,0.333',
        help='Initial values for dual variables as comma-separated string. Default is equal weights.',
    )
    training_parser.add_argument(
        '--run_closed_form_dual',
        type=str2bool,
        default=False,
        help='Whether to run the closed form dual.',
    )
    training_parser.add_argument(
        '--num_batches_dual',
        type=int,
        default=100,
        help='The number of batches to use for the closed form dual.',
    )
    training_parser.add_argument(
        '--sample_responses_for_dual',
        type=str2bool,
        default=False,
        help='Whether to sample responses for the dual.',
    )
    training_parser.add_argument(
        '--num_responses_for_dual',
        type=int,
        default=10,
        help='The number of responses to use for the dual.',
    )
    training_parser.add_argument(
        '--dual_solver_use_both_only',
        type=str2bool,
        default=True,
        help='Whether to use only the responses that are both safe and helpful.',
    )

    training_parser.add_argument(
        '--dual_step_size',
        type=float,
        default=0.01,
        help='The step size for the dual step.',
    )

    training_parser.add_argument(
        '--dual_weight_decay',
        type=float,
        default=0.0,
        help='The weight decay for the dual step.',
    )

    training_parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='Total number of training epochs to perform.',
    )
    training_parser.add_argument(
        '--per_device_train_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the training dataloader.',
    )
    training_parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=16,
        help='Batch size (per device) for the evaluation dataloader.',
    )
    training_parser.add_argument(
        '--eval_batch_size',
        type=int,
        default=0,
        help='Batch size to use during evaluation, overriding per_device_eval_batch_size. If 0, uses per_device_eval_batch_size.',
    )
    training_parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=1,
        help='Number of updates steps to accumulate before performing a backward/update pass.',
    )
    training_parser.add_argument(
        '--gradient_checkpointing',
        action='store_true',
        help='Enable HF gradient checkpointing for actor model.',
    )
    training_parser.add_argument(
        '--lr',
        '--learning_rate',
        type=float,
        default=2e-5,
        help='Initial learning rate (after the potential warmup period) to use.',
    )
    training_parser.add_argument(
        '--lr_scheduler_type',
        type=SchedulerType,
        default='cosine',
        help='The scheduler type to use.',
        choices=[
            'linear',
            'cosine',
            'cosine_with_restarts',
            'polynomial',
            'constant',
            'constant_with_warmup',
        ],
    )
    training_parser.add_argument(
        '--lr_warmup_ratio',
        type=float,
        default=0.0,
        help='Ratio of warm steps over total training steps for the lr scheduler.',
    )
    training_parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay to for the model training.',
    )
    training_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='A seed for reproducible training.',
    )
    training_parser.add_argument(
        '--fp16',
        type=str2bool,
        default=False,
        help='Whether to use float16 precision.',
    )
    training_parser.add_argument(
        '--bf16',
        type=str2bool,
        default=False,
        help='Whether to use bfloat16 precision.',
    )
    training_parser.add_argument(
        '--tf32',
        type=str2bool,
        default=None,
        help='Whether to use tf32 mix precision.',
    )
    training_parser.add_argument(
        '--safety_threshold',
        type=float,
        default=0.1,
        help='The safety threshold for the model.',
    )
    training_parser.add_argument(
        '--scale_costs',
        type=float,
        default=10.0,
        help='The scale factor for the costs.',
    )


    training_parser.add_argument(
        '--train_batches_on_eval',
        type=int,
        default=0,
        help='The number of training batches to evaluate on.',
    )

    # Evaluation
    evaluation_parser = parser.add_argument_group('evaluation')
    evaluation_parser.add_argument(
        '--eval_strategy',
        type=str,
        default='epoch',
        help='The evaluation strategy to adopt.',
        choices=['epoch', 'steps'],
    )
    evaluation_parser.add_argument(
        '--eval_interval',
        type=int,
        default=1,
        help='The interval to evaluate the model.',
    )
    evaluation_parser.add_argument(
        '--need_eval',
        default=False,
        help='Whether to evaluate the model during training.',
        action='store_true',
    )
    evaluation_parser.add_argument(
        '--eval_split_ratio',
        type=float,
        default=None,
        help='The split ratio of the evaluation dataset.',
    )
    evaluation_parser.add_argument(
        '--eval_at_init',
        type=str2bool,
        default=False,
        help='Whether to evaluate the model at the initialization.',
    )
    evaluation_parser.add_argument(
        '--num_responses_eval',
        type=int,
        default=4,
        help='The number of responses to use for the evaluation at the initialization.',
    )
    evaluation_parser.add_argument(
        '--compute_kl_eval',
        type=str2bool,
        default=False,
        help='Whether to compute the KL divergence between the reference and actor policy.',
    )
    evaluation_parser.add_argument(
        '--compute_costs_eval',
        type=str2bool,
        default=False,
        help='Whether to compute the costs of the model.',
    )

    # Logging
    logging_parser = parser.add_argument_group('logging')
    logging_parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Where to store the model.',
    )
    logging_parser.add_argument(
        '--log_type',
        type=str,
        help='The type of logging.',
        default='wandb',
        choices=['wandb', 'tensorboard'],
    )
    logging_parser.add_argument(
        '--log_dir',
        type=str,
        help='The directory to store the logs.',
        default=None,
    )
    logging_parser.add_argument(
        '--log_project',
        type=str,
        help='The project name for logging.',
        default=None,
    )
    logging_parser.add_argument(
        '--log_run_name',
        type=str,
        help='The run name for logging.',
        default=None,
    )
    logging_parser.add_argument(
        '--save_16bit',
        action='store_true',
        help='Whether to save the model in 16-bit precision.',
    )
    logging_parser.add_argument(
        '--save_interval',
        type=int,
        default=1000000,
        help='The interval to save the model.',
    )

    # DeepSpeed
    deepspeed_parser = parser.add_argument_group('deepspeed')
    deepspeed_parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training on GPUs',
    )
    deepspeed_parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='ZeRO optimization stage for models.',
    )
    deepspeed_parser.add_argument(
        '--offload',
        type=str,
        default='none',
        choices=['none', 'parameter', 'optimizer', 'all'],
        help='Offload parameters and/or optimizer states to CPU.',
    )
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    if args.local_rank == -1:
        parser.error('`local_rank` not set, please use DeepSpeed launcher to run this script.')
    if args.fp16 and args.bf16:
        parser.error('Cannot use both bf16 and fp16 precision.')
    if args.bf16 and not is_torch_bf16_gpu_available():
        parser.error(
            'bf16 precision is not supported on this GPU. '
            'Please disable `--bf16` flag or use another precision flag (e.g., `--fp16`).',
        )
    if args.tf32 is not None and is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = args.tf32

    return args


def main() -> None:
    """Main training routine."""
    args = parse_arguments()
    args.dual_init = [float(x) for x in args.dual_init.split(',')]

    deepspeed.init_distributed()

    args.global_rank = dist.get_rank()
    args.device = torch.device('cuda', args.local_rank)
    torch.cuda.set_device(args.device)
    seed_everything(args.seed)
    set_logger_level()

    dist.barrier()

    ds_train_config = get_deepspeed_train_config(
        micro_batch_size_per_gpu=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        stage=args.zero_stage,
        offload=args.offload,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    ds_eval_config = get_deepspeed_eval_config(
        stage=args.zero_stage,
        offload=args.offload,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    trainer = MultiPdAlignementTrainer(args, ds_train_config, ds_eval_config)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    main()
