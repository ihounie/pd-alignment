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
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, get_scheduler
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from safe_rlhf.configs import ADAM_BETAS
from safe_rlhf.datasets import TokenizedDataset
from safe_rlhf.models import AutoModelForScore, load_pretrained_models
from safe_rlhf.trainers.base import TrainerBase
from safe_rlhf.utils import get_optimizer_grouped_parameters, is_main_process, to_device


class PrefMultiDualTrainer(TrainerBase):
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

        print("initializing tokenizer ...")
        self.init_models()
        dist.barrier()

        print("initializing datasets ...")
        self.init_datasets()
        dist.barrier()

        print("calculating costs ...")
        self.init_costs()
        dist.barrier()

        print("calculating rewards ...")
        self.init_rewards()
        dist.barrier()

        print("calculating baseline ...")
        self.init_baseline()
        dist.barrier()

        print("initializing multipliers ...")
        self.init_multipliers()
        dist.barrier()

        print("initializing engines ...")
        self.init_engines()
        dist.barrier()

        print("initializing logger ...")
        self.init_logger()
        dist.barrier()

        print("initialization done")

    def init_tokenizer(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_config is not None and self.ds_config['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_config)

        _, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='right',
            auto_model_type=self.MODEL_TYPE,
            trust_remote_code=self.args.trust_remote_code,
            auto_model_kwargs=self.extra_model_kwargs,
            auto_tokenizer_kwargs=self.extra_tokenizer_kwargs,
        )

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
        print(f"Initializing multipliers with {self.args.dual_init}")
        self.multipliers = torch.tensor(self.args.dual_init)
        # clamp to 0
        self.multipliers = torch.clamp(self.multipliers, min=0)
        self.multipliers = to_device(self.multipliers, self.args.device)
        return

    def init_baseline(self) -> None:
        """Initialize baseline log probabilities with caching functionality."""
        # Create cache directory if it doesn't exist
        os.makedirs(self.args.cache_dir, exist_ok=True)
        baseline_cache_path = os.path.join(self.args.cache_dir, "cached_baseline_logprobs_pref.pt")

        if self.args.compute_kl_eval:
            reference_model, _ = load_pretrained_models(
                self.args.model_name_or_path,
                model_max_length=self.args.max_length,
                padding_side='left',
                auto_model_type=AutoModelForCausalLM,
                trust_remote_code=self.args.trust_remote_code,
            )
            reference_model.requires_grad_(False)
            reference_model.eval()
            self.reference_model = reference_model

        # Load cached baseline if available and not recomputing
        if os.path.exists(baseline_cache_path) and not self.args.recompute_baseline:
            print(f"Loading cached baseline logprobs from {baseline_cache_path}")
            self.baseline_logprobs = torch.load(baseline_cache_path, map_location=self.args.device)
            print("Loaded cached baseline logprobs successfully")
            return

        # If we need to compute baseline logprobs
        print("Computing baseline logprobs...")
        if self.train_dataloader.dataset.num_respones != 1:
            raise NotImplementedError(
                "Baseline logprob computation is not implemented for multi-response datasets"
            )
        # Initialize baseline tensor
        self.baseline_logprobs = torch.zeros(
            (len(self.train_dataloader.dataset)),
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

    def init_datasets(self, tokenizer=None) -> None:
        """Initialize training and evaluation datasets."""
        if tokenizer is None:
            tokenizer = self.tokenizer
        train_dataset = self.DATASET_TYPE(
            self.args.train_datasets,
            tokenizer=tokenizer,
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
                    self.args.eval_datasets, tokenizer=tokenizer, lazy_tokenization=False, seed=42
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

        # Add gradient clipping to DeepSpeed configuration
        if hasattr(self.args, 'gradient_clipping') and self.args.gradient_clipping > 0:
            self.ds_config['gradient_clipping'] = self.args.gradient_clipping
        elif not self.ds_config.get('gradient_clipping', None):
            # Default value if not specified
            self.ds_config['gradient_clipping'] = 1.0

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
        eval_dict = self.run_eval(self.eval_dataloader, prefix="eval/test")
        if self.args.train_batches_on_eval > 0:
            train_dict = self.run_eval(
                self.train_dataloader,
                prefix="eval/train",
                num_batches=self.args.train_batches_on_eval,
            )
            eval_dict = {**eval_dict, **train_dict}
        return eval_dict

    def run_eval(
        self, eval_dataloader: DataLoader, prefix: str = "eval", num_batches: int = None
    ) -> dict[str, Any]:
        """Evaluate the model."""
        # sample one response from the model over the whole eval dataset
        # evaluate the cost model on the sampled responses
        all_costs = []
        all_kl_divs = []
        with torch.no_grad():
            ref_model = to_device(self.reference_model, self.args.device)
            cost_model = to_device(self.cost_model, self.args.device)
            batch_count = 0
            for batch in tqdm(eval_dataloader, desc='Evaluating model'):
                batch = to_device(batch, self.args.device)
                if batch["input_ids"].shape[0] != 1:
                    raise NotImplementedError("Batch size should be 1 for evaluation for now")
                response_start = (
                    (batch["response_mask"][0] == 1).nonzero(as_tuple=True)[0][0].item()
                )
                prompt = batch["input_ids"][:, :response_start]
                generated_ids = self.model.generate(
                    prompt,
                    attention_mask=batch["attention_mask"],
                    max_length=self.args.max_length,
                    do_sample=True,
                    num_return_sequences=1,
                )
                # evaluate sequence logprobs of the generated responses
                logprobs = self.compute_log_probs(
                    self.model, generated_ids, batch["attention_mask"]
                )
                sequence_logprobs = logprobs.sum(dim=1)
                # evaluate sequence logprobs of the generated responses for the reference model
                reference_logprobs = self.compute_log_probs(
                    ref_model, generated_ids, batch["attention_mask"]
                )
                reference_sequence_logprobs = reference_logprobs.sum(dim=1)
                # evaluate KL divergence between the sequence logprobs
                kl_div = (sequence_logprobs - reference_sequence_logprobs).mean()
                # torch.nn.functional.kl_div(sequence_logprobs, reference_sequence_logprobs, reduction='none', log_target=True)
                all_kl_divs.append(kl_div.cpu().unsqueeze(0))
                # evaluate the cost model on the sampled responses
                answer = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                answer = [a.replace("BEGINNING OF CONVERSATION: USER: ", "") for a in answer]
                answer = [a.replace("ASSISTANT:", "") for a in answer]
                tokenized_answer = self.cost_tokenizer(
                    answer, return_tensors="pt", padding=True, truncation=True
                ).to(self.args.device)
                outputs = cost_model(
                    tokenized_answer["input_ids"], attention_mask=tokenized_answer["attention_mask"]
                )
                probs = torch.softmax(outputs.logits, dim=-1)
                all_costs.append(probs.cpu())
                batch_count += 1
                if num_batches is not None and batch_count >= num_batches:
                    break
        all_costs = torch.cat(all_costs, dim=0)
        all_kl_divs = torch.cat(all_kl_divs, dim=0)
        # average costs over responses
        all_costs = all_costs.mean(dim=0)
        all_kl_divs = all_kl_divs.mean()
        # get slacks
        slacks = all_costs - self.args.safety_threshold
        cost_dict = {f'{prefix}/cost[{i}]': cost.item() for i, cost in enumerate(all_costs)}
        slack_dict = {f'{prefix}/slack[{i}]': slack.item() for i, slack in enumerate(slacks)}
        multiplier_dict = {
            f'{prefix}/multiplier[{i}]': m.item() for i, m in enumerate(self.multipliers)
        }
        kl_dict = {f'{prefix}/kl_div': all_kl_divs.item()}

        return {**cost_dict, **slack_dict, **multiplier_dict, **kl_dict}

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

        if self.args.need_eval and self.args.eval_at_init:
            self.logger.print('\n***** Evaluating at the beginning *****')
            # only eval on the main process
            if is_main_process():
                self.logger.log(self.eval(), step=0)
        for epoch in range(self.args.epochs):
            self.model.train()

            for batch in self.train_dataloader:

                info = self.train_step(**to_device(batch, self.args.device))

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.args.epochs} epoch '
                    + f'(loss {info["train/loss"]:.4f})'
                    + f'dkl_loss {info["train/dkl_loss"]:.4f}'
                    + f'safety_loss {info["train/safety_loss"]:.4f}'
                    + f'importance_weights {info["train/importance_weights"]:.4f}',
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
                # only eval on the main process
                if is_main_process():
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
