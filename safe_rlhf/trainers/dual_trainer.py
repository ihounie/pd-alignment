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
"""Trainer base class for supervised training."""

from __future__ import annotations

import abc
import argparse
import os
from typing import Any, ClassVar

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, get_scheduler
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from safe_rlhf.configs import ADAM_BETAS
from safe_rlhf.datasets import TokenizedDataset
from safe_rlhf.models import AutoModelForScore, load_pretrained_models
from safe_rlhf.trainers.base import TrainerBase
from safe_rlhf.utils import get_optimizer_grouped_parameters, is_main_process, to_device


class DualTrainer(TrainerBase):
    """Trainer base class for supervised training.

    Abstract methods:
        loss: Compute supervised training loss.
        train_step: Perform a single training step.
        dual_step: Perfrom a single dual step.
    """

    TRAINING_TYPE: ClassVar[str] = 'dual'
    DATASET_TYPE: ClassVar[type[TokenizedDataset]]
    MODEL_TYPE = AutoModelForCausalLM

    model: deepspeed.DeepSpeedEngine
    ds_config: dict[str, Any]

    extra_model_kwargs: dict[str, Any] | None = None
    extra_tokenizer_kwargs: dict[str, Any] | None = None

    def __init__(self, args: argparse.Namespace, ds_config: dict[str, Any]) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_config = ds_config
        self.global_step = 0

        print("initializing models ...")
        self.init_models()
        dist.barrier()
        print("initializing datasets ...")
        self.init_datasets()
        dist.barrier()
        print("calculating baseline ...")
        self.init_baseline()
        dist.barrier()
        print("calculating rewards ...")
        self.init_rewards()
        dist.barrier()
        print("calculating costs ...")
        self.init_costs()
        dist.barrier()
        print("initializing engines ...")
        self.init_engines()
        dist.barrier()
        print("initializing logger ...")
        self.init_logger()
        dist.barrier()
        print("initializing multipliers ...")
        self.init_multipliers()
        print("initialization done")

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_config is not None and self.ds_config['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_config)

        self.model, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='right',
            auto_model_type=self.MODEL_TYPE,
            trust_remote_code=self.args.trust_remote_code,
            auto_model_kwargs=self.extra_model_kwargs,
            auto_tokenizer_kwargs=self.extra_tokenizer_kwargs,
        )

    def init_multipliers(self) -> None:
        self.multipliers = torch.zeros(
            (len(self.train_dataloader.dataset), self.train_dataloader.dataset.num_respones)
        )
        self.multipliers = to_device(self.multipliers, self.args.device)
        return

    def init_baseline(self) -> None:
        """Initialize baseline log probabilities with caching functionality."""
        # Create cache directory if it doesn't exist
        os.makedirs(self.args.cache_dir, exist_ok=True)
        baseline_cache_path = os.path.join(self.args.cache_dir, "cached_baseline_logprobs.pt")

        # Load cached baseline if available and not recomputing
        if os.path.exists(baseline_cache_path) and not self.args.recompute_baseline:
            print(f"Loading cached baseline logprobs from {baseline_cache_path}")
            self.baseline_logprobs = torch.load(baseline_cache_path, map_location=self.args.device)
            print("Loaded cached baseline logprobs successfully")
            return

        # If we need to compute baseline logprobs
        print("Computing baseline logprobs...")

        # Initialize baseline tensor
        self.baseline_logprobs = torch.zeros(
            (len(self.train_dataloader.dataset), self.train_dataloader.dataset.num_respones),
            dtype=self.model.dtype,
        )
        self.baseline_logprobs = to_device(self.baseline_logprobs, self.args.device)

        # Load and setup reference model
        reference_model, _ = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )
        reference_model.requires_grad_(False)
        reference_model.eval()
        reference_model.to(self.args.device)

        # Compute logprobs for each batch
        for batch in tqdm(self.train_dataloader, desc='Computing baseline logprobs'):
            batch = to_device(batch, self.args.device)
            # Compute logprobs for better and worse responses
            better_logprobs = (
                self.compute_log_probs(
                    reference_model,
                    batch["better_input_ids"],
                    batch["better_attention_mask"],
                )
                * batch["better_attention_mask"][:, 1:]
                * batch["response_masks"][:, 1:]
            )
            worse_logprobs = (
                self.compute_log_probs(
                    reference_model,
                    batch["worse_input_ids"],
                    batch["worse_attention_mask"],
                )
                * batch["worse_attention_mask"][:, 1:]
                * batch["response_masks"][:, 1:]
            )

            self.baseline_logprobs[batch['index'], 0] = better_logprobs.sum(dim=1)
            self.baseline_logprobs[batch['index'], 1] = worse_logprobs.sum(dim=1)

        # Save computed baseline logprobs
        print(f"Saving computed baseline logprobs to {baseline_cache_path}")
        torch.save(self.baseline_logprobs, baseline_cache_path)
        print("Saved baseline logprobs successfully")

        # Free up memory
        del reference_model
        torch.cuda.empty_cache()
        return

    def init_costs(self) -> None:
        """Initialize costs with caching functionality."""
        # Create cache directory if it doesn't exist
        os.makedirs(self.args.cache_dir, exist_ok=True)
        costs_cache_path = os.path.join(self.args.cache_dir, "cached_costs.pt")

        # Load cached costs if available and not recomputing
        if os.path.exists(costs_cache_path) and not self.args.recompute_costs:
            print(f"Loading cached costs from {costs_cache_path}")
            self.costs = torch.load(costs_cache_path, map_location=self.args.device)
            print("Loaded cached costs successfully")
            return

        # If we need to compute costs
        # Initialize costs tensor
        self.costs = torch.zeros(
            (len(self.train_dataloader.dataset), self.train_dataloader.dataset.num_respones)
        )
        self.costs = to_device(self.costs, self.args.device)
        print("Computing costs...")
        if self.args.cost_model_name_or_path == "indicator":
            for batch in tqdm(self.train_dataloader, desc='Computing indicator costs'):
                self.costs[batch['index'], 0] = batch['better_safe']
                self.costs[batch['index'], 1] = batch['worse_safe']
        else:
            cost_model, _ = load_pretrained_models(
                self.args.cost_model_name_or_path,
                model_max_length=self.args.max_length,
                auto_model_type=AutoModelForScore,
                padding_side='right',
                trust_remote_code=self.args.trust_remote_code,
                auto_model_kwargs={
                    'score_type': 'cost',
                    'do_normalize': self.args.normalize_cost,
                },
            )

            cost_model.set_normalize(self.args.normalize_cost)
            cost_model.requires_grad_(False)
            cost_model.eval()
            cost_model.to(self.args.device)

            for batch in tqdm(self.train_dataloader, desc='Computing model costs'):
                batch = to_device(batch, self.args.device)
                self.costs[batch['index'], 0] = cost_model(
                    batch["better_input_ids"],
                    attention_mask=batch["better_attention_mask"],
                ).end_scores.squeeze(dim=-1)
                self.costs[batch['index'], 1] = cost_model(
                    batch["worse_input_ids"],
                    attention_mask=batch["worse_attention_mask"],
                ).end_scores.squeeze(dim=-1)

        # Save computed costs
        print(f"Saving computed costs to {costs_cache_path}")
        torch.save(self.costs, costs_cache_path)
        print("Saved costs successfully")
        # Free up memory
        del cost_model
        torch.cuda.empty_cache()
        return

    def init_rewards(self) -> None:
        """Initialize rewards with caching functionality."""

        # Create cache directory if it doesn't exist
        os.makedirs(self.args.cache_dir, exist_ok=True)
        rewards_cache_path = os.path.join(self.args.cache_dir, "cached_rewards.pt")

        # Load cached rewards if available and not recomputing
        if os.path.exists(rewards_cache_path) and not self.args.recompute_rewards:
            print(f"Loading cached rewards from {rewards_cache_path}")
            self.rewards = torch.load(rewards_cache_path, map_location=self.args.device)
            print("Loaded cached rewards successfully")
            return

        # If we need to compute rewards

        # Initialize rewards tensor
        self.rewards = torch.zeros(
            (len(self.train_dataloader.dataset), self.train_dataloader.dataset.num_respones)
        )
        self.rewards = to_device(self.rewards, self.args.device)

        print("Computing rewards...")
        reward_model, _ = load_pretrained_models(
            self.args.reward_model_name_or_path,
            model_max_length=self.args.max_length,
            auto_model_type=AutoModelForScore,
            padding_side='right',
            trust_remote_code=self.args.trust_remote_code,
            auto_model_kwargs={
                'score_type': 'reward',
                'do_normalize': self.args.normalize_reward,
            },
        )

        reward_model.set_normalize(self.args.normalize_reward)
        reward_model.requires_grad_(False)
        reward_model.eval()
        reward_model.to(self.args.device)

        for batch in tqdm(self.train_dataloader, desc='Computing rewards'):
            batch = to_device(batch, self.args.device)
            self.rewards[batch['index'], 0] = reward_model(
                batch["better_input_ids"],
                attention_mask=batch["better_attention_mask"],
            ).end_scores.squeeze(dim=-1)
            self.rewards[batch['index'], 1] = reward_model(
                batch["worse_input_ids"],
                attention_mask=batch["worse_attention_mask"],
            ).end_scores.squeeze(dim=-1)

        # Save computed rewards
        print(f"Saving computed rewards to {rewards_cache_path}")
        torch.save(self.rewards, rewards_cache_path)
        print("Saved rewards successfully")
        # Free up memory
        del reward_model
        torch.cuda.empty_cache()
        return

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        print(self.DATASET_TYPE)
        train_dataset = self.DATASET_TYPE(
            self.args.train_datasets,
            tokenizer=self.tokenizer,
            lazy_tokenization=False,
            seed=42,
        )
        if self.args.need_eval:
            if self.args.eval_datasets is None and self.args.eval_split_ratio is not None:
                train_dataset, eval_dataset = train_dataset.split_train_test(
                    split_ratio=self.args.eval_split_ratio,
                )
            elif self.args.eval_datasets is not None and self.args.eval_split_ratio is None:
                eval_dataset = self.DATASET_TYPE(
                    self.args.eval_datasets,
                    tokenizer=self.tokenizer,
                )
            else:
                raise ValueError('Either `eval_datasets` or `eval_split_ratio` should be provided.')

            self.eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=eval_dataset.get_collator(),
                sampler=DistributedSampler(eval_dataset, shuffle=True),
                batch_size=self.args.per_device_eval_batch_size,
            )
        else:
            self.eval_dataloader = None

        self.train_dataloader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.get_collator(),
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=self.args.per_device_train_batch_size,
        )

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.args.num_update_steps_per_epoch = (
            len(self.train_dataloader) + self.args.gradient_accumulation_steps - 1
        ) // self.args.gradient_accumulation_steps
        self.args.total_training_steps = self.args.epochs * self.args.num_update_steps_per_epoch

        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            self.model,
            self.args.weight_decay,
        )
        if (
            self.ds_config['zero_optimization'].get('offload_optimizer', {}).get('device', 'none')
            != 'none'
        ):
            optimizer = DeepSpeedCPUAdam(
                optimizer_grouped_parameters,
                lr=self.args.lr,
                betas=ADAM_BETAS,
            )
        else:
            optimizer = FusedAdam(
                optimizer_grouped_parameters,
                lr=self.args.lr,
                betas=ADAM_BETAS,
            )

        num_warmup_steps = int(self.args.lr_warmup_ratio * self.args.total_training_steps)
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.args.total_training_steps,
        )

        self.model, *_ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            args=self.args,
            config=self.ds_config,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
        )

        # if self.args.gradient_checkpointing:
        #    self.model.gradient_checkpointing_enable()

    @abc.abstractmethod
    def loss(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Compute supervised training loss."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform a single training step."""
        raise NotImplementedError

    @abc.abstractmethod
    def dual_step(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Perform a single dual step."""
        raise NotImplementedError

    def eval(self) -> dict[str, Any]:
        """Evaluate the model."""
        # log multiplier stats
        multipliers = self.multipliers.detach().cpu().numpy()
        # zeros, max, min, mean, median, std
        multiplier_stats = {
            'zeros': (multipliers == 0).mean(),
            'max': multipliers.max(),
            'mean': multipliers.mean(),
            'median': np.median(multipliers),
            'std': multipliers.std(),
        }
        return multiplier_stats

    def train(self) -> None:
        """Train the model."""
        self.logger.print('***** Running training *****')

        progress_bar = tqdm(
            total=self.args.epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.args.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )

        if self.args.need_eval:
            self.logger.print('\n***** Evaluating at the beginning *****')
            self.logger.log(self.eval(), step=0)
        for epoch in range(self.args.epochs):
            self.model.train()

            for batch in self.train_dataloader:

                info = self.train_step(**to_device(batch, self.args.device))

                torch.cuda.empty_cache()

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.args.epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )
                progress_bar.update(1)

                info['train/epoch'] = self.global_step / len(self.train_dataloader)
                self.logger.log(info, step=self.global_step)

                if self.global_step % self.args.save_interval == 0:
                    self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                    self.model.save_checkpoint(self.args.output_dir, tag=self.global_step)
                    self.logger.print('Checkpoint saved.')

                if (
                    self.args.need_eval
                    and self.args.eval_strategy == 'steps'
                    and self.global_step % self.args.eval_interval == 0
                ):
                    self.logger.print(f'\n***** Evaluating at step {self.global_step} *****')
                    self.logger.log(self.eval(), step=self.global_step)

            if self.args.need_eval and self.args.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.args.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)

            self.model.tput_timer.update_epoch_count()

    def set_train(self, mode: bool = True) -> None:
        """Set training mode for model."""
        if mode:
            self.model.train()
            if self.args.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
        else:
            self.model.eval()
            if self.args.gradient_checkpointing:
                self.model.gradient_checkpointing_disable()
