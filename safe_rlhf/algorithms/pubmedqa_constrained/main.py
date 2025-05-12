from __future__ import annotations

import argparse
import sys

import deepspeed
import torch
import torch.distributed as dist
from transformers import SchedulerType
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available

from safe_rlhf.algorithms.pubmedqa_constrained.trainer import PubMedQAConstrainedTrainer
from safe_rlhf.configs import get_deepspeed_train_config
from safe_rlhf.logger import set_logger_level
from safe_rlhf.utils import seed_everything, str2bool

# -----------------------------------------------------------------------------


def parse_arguments() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(
        prog='deepspeed --module safe_rlhf.algorithms.pubmedqa_constrained',
        description='Supervised fine-tuning on PubMedQA with CE constraints (yes/no).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model -------------------------------------------------------------------
    model_group = parser.add_argument_group('model')
    model_group.add_argument('--model_name_or_path', type=str, required=True)
    model_group.add_argument('--max_length', type=int, default=512)
    model_group.add_argument('--trust_remote_code', type=str2bool, default=False)

    # Dataset -----------------------------------------------------------------
    data_group = parser.add_argument_group('dataset')
    data_group.add_argument('--train_datasets', type=str, required=True,
                            help='HF dataset identifier, e.g. "qiaojin/PubMedQA:pqa_artificial:train"')
    data_group.add_argument('--eval_datasets', type=str, required=True,
                            help='HF dataset identifier used for evaluation, e.g. "qiaojin/PubMedQA:pqa_labeled:train"')

    # Training args -----------------------------------------------------------
    train_group = parser.add_argument_group('training')
    train_group.add_argument('--epochs', type=int, default=1)
    train_group.add_argument('--per_device_train_batch_size', type=int, default=8)
    train_group.add_argument('--per_device_eval_batch_size', type=int, default=8)
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=1)
    train_group.add_argument('--gradient_checkpointing', action='store_true')

    train_group.add_argument('--lr', '--learning_rate', type=float, default=2e-5, dest='lr')
    train_group.add_argument('--lr_scheduler_type', type=SchedulerType, default='cosine')
    train_group.add_argument('--lr_warmup_ratio', type=float, default=0.0)
    train_group.add_argument('--weight_decay', type=float, default=0.0)

    train_group.add_argument('--dataloader_workers', type=int, default=4,
                            help='Number of CPU workers for the dataloader. Increase to improve GPU utilization.')

    # Constraint args ---------------------------------------------------------
    constraint_group = parser.add_argument_group('constraints')
    constraint_group.add_argument('--ce_threshold_yes', type=float, required=True,
                                  help='Maximum allowed average CE loss for positive (yes) class.')
    constraint_group.add_argument('--ce_threshold_no', type=float, required=True,
                                  help='Maximum allowed average CE loss for negative (no) class.')
    constraint_group.add_argument('--dual_step_size', type=float, default=0.1,
                                  help='Step size (learning rate) for dual ascent on multipliers.')
    constraint_group.add_argument('--lambda_yes_init', type=float, default=0.0,
                                  help='Initial value of Lagrange multiplier for positive class constraint.')
    constraint_group.add_argument('--lambda_no_init', type=float, default=0.0,
                                  help='Initial value of Lagrange multiplier for negative class constraint.')
    constraint_group.add_argument('--dual_warmup_epochs', type=int, default=0,
                                  help='Number of epochs to linearly warm up the dual step size from 0 to its target value.')
    constraint_group.add_argument('--dual_update_interval', type=int, default=0,
                                  help='Number of training batches between dual ascent updates. 0 = once per epoch.')
    constraint_group.add_argument('--dual_weight_decay', type=float, default=0.1,
                                  help='Weight decay (multiplicative factor) applied to Lagrange multipliers at each update.')
    constraint_group.add_argument('--kl_coef', type=float, default=0.01,
                                  help='KL divergence coefficient for regularizing predictions against the base model.')

    # Misc --------------------------------------------------------------------
    misc_group = parser.add_argument_group('misc')
    misc_group.add_argument('--seed', type=int, default=42)
    misc_group.add_argument('--need_eval', type=str2bool, default=True)
    misc_group.add_argument('--eval_split_ratio', type=float, default=None)

    misc_group.add_argument('--fp16', type=str2bool, default=False)
    misc_group.add_argument('--bf16', type=str2bool, default=False)
    misc_group.add_argument('--tf32', type=str2bool, default=None)

    misc_group.add_argument('--lora_r', type=int, default=8)
    misc_group.add_argument('--lora_alpha', type=int, default=16)
    misc_group.add_argument('--lora_dropout', type=float, default=0.05)
    misc_group.add_argument('--lora_target_modules', type=str, default=None)

    misc_group.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (added by DeepSpeed).')

    # Logging ------------------------------------------------------------------
    logging_group = parser.add_argument_group('logging')
    logging_group.add_argument('--output_dir', type=str, default=None,
                               help='Where to store checkpoints and logs.')
    logging_group.add_argument('--log_type', type=str, default='tensorboard',
                               choices=['wandb', 'tensorboard', 'none'],
                               help='Backend used for experiment logging.')
    logging_group.add_argument('--log_dir', type=str, default=None,
                               help='Explicit directory for logs (defaults to output_dir).')
    logging_group.add_argument('--log_project', type=str, default=None,
                               help='Project name for WandB logging.')
    logging_group.add_argument('--log_run_name', type=str, default=None,
                               help='Run name for WandB / TensorBoard subdirectory.')

    logging_group.add_argument('--save_16bit', action='store_true',
                               help='Save model in 16bit weights when checkpointing.')
    logging_group.add_argument('--save_interval', type=int, default=1000000,
                               help='Interval (in steps) to save checkpoints.')

    # Evaluation ----------------------------------------------------------------
    eval_group = parser.add_argument_group('evaluation')
    eval_group.add_argument('--eval_strategy', type=str, default='epoch', choices=['epoch', 'steps'],
                            help='Whether to run evaluation at each epoch or every N steps.')
    eval_group.add_argument('--eval_interval', type=int, default=1,
                            help='The interval (epochs or steps) for evaluation depending on strategy.')

    # DeepSpeed -----------------------------------------------------------------
    ds_group = parser.add_argument_group('deepspeed')
    ds_group.add_argument('--zero_stage', type=int, default=0, choices=[0,1,2,3],
                          help='ZeRO optimization stage (0 disables ZeRO).')
    ds_group.add_argument('--offload', type=str, default='none', choices=['none','parameter','optimizer','all'],
                          help='Offload parameters and/or optimizer states to CPU/NVMe.')

    args = parser.parse_args()

    # Sanity for precision args
    if args.bf16 and not is_torch_bf16_gpu_available():
        parser.error('`--bf16` is specified but bf16 is not supported on this GPU.')
    if args.tf32 is None:
        args.tf32 = is_torch_tf32_available()

    return args


# -----------------------------------------------------------------------------


def main() -> None:  # noqa: D401
    args = parse_arguments()

    # Seed & logger -----------------------------------------------------------
    seed_everything(args.seed)
    set_logger_level()

    # --------------------------------------
    # Distributed init (DeepSpeed)
    # --------------------------------------
    if args.local_rank == -1:
        raise RuntimeError('`local_rank` not set, please launch via the `deepspeed` launcher.')

    # Initialize distributed
    deepspeed.init_distributed()

    args.global_rank = dist.get_rank()
    args.device = torch.device('cuda', args.local_rank)
    torch.cuda.set_device(args.device)

    dist.barrier()

    ds_config = get_deepspeed_train_config(
        micro_batch_size_per_gpu=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        stage=args.zero_stage,
        offload=args.offload,
    )

    trainer = PubMedQAConstrainedTrainer(args, ds_config)
    trainer.train()


if __name__ == '__main__':
    sys.exit(main())