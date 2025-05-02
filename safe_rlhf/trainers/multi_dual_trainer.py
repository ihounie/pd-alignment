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
from scipy.special import softmax
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


class MultiDualTrainer(TrainerBase):
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

        print("initializing engines ...")
        self.init_engines()
        dist.barrier()

        print("initializing multipliers ...")
        self.init_multipliers()
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
        if self.args.run_closed_form_dual:
            print("Computing costs to solve closed form dual...")
            if self.args.sample_responses_for_dual:
                # Try loading cached costs for dual initialization from disk first
                dual_costs_cache_path = os.path.join(self.args.cache_dir, "cached_dual_costs.pt")

                if os.path.exists(dual_costs_cache_path) and not self.args.recompute_costs:
                    print(f"Loading cached dual costs from {dual_costs_cache_path}")
                    costs = torch.load(dual_costs_cache_path, map_location=self.args.device)[
                        :, :, :-1
                    ]
                    print("Loaded cached dual costs successfully")
                else:
                    # Compute costs and cache them for future runs
                    os.makedirs(self.args.cache_dir, exist_ok=True)
                    temp_train_dataloader = DataLoader(
                        self.train_dataloader.dataset,
                        collate_fn=self.train_dataloader.dataset.get_collator(),
                        sampler=DistributedSampler(self.train_dataloader.dataset, shuffle=True),
                        batch_size=self.args.eval_batch_size,
                    )
                    costs = self.run_eval(
                        temp_train_dataloader,
                        return_costs=True,
                        num_batches=self.args.num_batches_dual,
                        num_responses=self.args.num_responses_for_dual,
                    )["costs"].cpu()

                    # Only the main process saves to disk to avoid race conditions
                    if is_main_process():
                        print(f"Saving computed dual costs to {dual_costs_cache_path}")
                        torch.save(costs, dual_costs_cache_path)
                        print("Saved dual costs successfully")
                    # Ensure every rank waits until the file is written
                    if dist.is_available() and dist.is_initialized():
                        dist.barrier()
                    costs = costs[:, :, :-1]
            else:
                costs = self.costs
            # ------------------------------------------------------------------
            # Solve the closed-form dual only on the main process and broadcast
            # the resulting multipliers to all other ranks so that every worker
            # has a consistent view while avoiding redundant computation.
            # ------------------------------------------------------------------
            if is_main_process():
                print("Solving closed form dual...")
                print("Mean costs: ", costs.mean(dim=(0, 1)))
                print("threshold: ", self.args.safety_threshold)
                lam_np = self.solve_closed_form_dual(costs, self.rewards)
                if lam_np is None:
                    raise ValueError("Failed to solve closed form dual")
                multipliers_tensor = torch.tensor(lam_np, device=self.args.device)
                print(f"Multipliers set to closed form solution: {multipliers_tensor}")
            else:
                # create an empty tensor with correct shape for broadcast
                multipliers_tensor = torch.empty(self.args.num_classes - 1, device=self.args.device)

            # Broadcast multipliers from rank 0 to all other ranks
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(multipliers_tensor, src=0)

            self.multipliers = multipliers_tensor
        else:
            self.multipliers = torch.tensor(self.args.dual_init)
        # clamp to 0
        self.multipliers = torch.clamp(self.multipliers, min=0)
        self.multipliers = to_device(self.multipliers, self.args.device)
        return

    def init_baseline(self) -> None:
        """Initialize baseline log probabilities with caching functionality."""
        # Create cache directory if it doesn't exist
        os.makedirs(self.args.cache_dir, exist_ok=True)
        baseline_cache_path = os.path.join(self.args.cache_dir, "cached_baseline_logprobs.pt")

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
        for batch in tqdm(self.train_dataloader, desc='Computing baseline logprobs'):
            batch = to_device(batch, self.args.device)
            # Compute logprobs
            logprobs = (
                self.compute_log_probs(
                    reference_model,
                    batch["input_ids"],
                    batch["attention_mask"],
                )
                * batch["response_mask"][:, 1:]
                * batch["attention_mask"][:, 1:]
            )

            self.baseline_logprobs[batch['index']] = logprobs.sum(dim=1)

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

        if self.args.compute_costs_eval:
            cost_model, cost_tokenizer = load_pretrained_models(
                self.args.cost_model_name_or_path,
                model_max_length=self.args.max_length,
                auto_model_type=AutoModelForSequenceClassification,
                padding_side='right',
                trust_remote_code=self.args.trust_remote_code,
            )
            cost_model.requires_grad_(False)
            cost_model.eval()
            self.cost_model = cost_model
            self.cost_tokenizer = cost_tokenizer

        # Load cached costs if available and not recomputing
        if os.path.exists(costs_cache_path) and not self.args.recompute_costs:
            print(f"Loading cached costs from {costs_cache_path}")
            self.costs = torch.load(costs_cache_path, map_location=self.args.device)
            self.costs = self.costs[:, :-1]
            print("Loaded cached costs successfully")
            return

        # If we need to compute costs
        # Initialize costs tensor
        self.costs = torch.zeros(len(self.train_dataloader.dataset), self.args.num_classes)
        self.costs = to_device(self.costs, self.args.device)
        print("Computing costs...")
        if self.args.cost_model_name_or_path == "indicator":
            for batch in tqdm(self.train_dataloader, desc='Computing indicator costs'):
                self.costs[batch['index'], :] = batch['labels']
        else:
            print("Loading cost model...")
            print(self.args.cost_model_name_or_path)
            cost_model, cost_tokenizer = load_pretrained_models(
                self.args.cost_model_name_or_path,
                model_max_length=self.args.max_length,
                auto_model_type=AutoModelForSequenceClassification,
                padding_side='right',
                trust_remote_code=self.args.trust_remote_code,
            )
            cost_model.requires_grad_(False)
            cost_model.eval()
            cost_model.to(self.args.device)
            self.init_datasets(cost_tokenizer)
            self.costs = torch.zeros(
                len(self.train_dataloader.dataset), self.args.num_classes, dtype=cost_model.dtype
            )
            self.costs = to_device(self.costs, self.args.device)

            for batch in tqdm(self.train_dataloader, desc='Computing model costs'):
                batch = to_device(batch, self.args.device)
                outputs = cost_model(batch["input_ids"], attention_mask=batch["attention_mask"])
                probs = torch.softmax(outputs.logits, dim=-1)
                self.costs[batch['index']] = probs
        # Save computed costs
        print(f"Saving computed costs to {costs_cache_path}")
        torch.save(self.costs, costs_cache_path)
        print("Saved costs successfully")
        # Free up memory
        del cost_model
        torch.cuda.empty_cache()
        self.costs = self.costs[:, :-1]
        # self.costs = 1 - self.costs
        # reinitialize datasets
        self.init_datasets(self.tokenizer)
        return

    def init_rewards(self) -> None:
        """Initialize rewards with caching functionality."""
        if self.args.reward_model_name_or_path == "none":
            self.rewards = None
            return
        elif self.args.reward_model_name_or_path == "safety_prob":
            costs_cache_path = os.path.join(self.args.cache_dir, "cached_costs.pt")
            if os.path.exists(costs_cache_path) and not self.args.recompute_costs:
                print(f"Loading cached probabilities as rewards from {costs_cache_path}")
                self.rewards = torch.load(costs_cache_path, map_location=self.args.device)[:, :-1]
                print("Loaded cached rewards successfully")
                return
            else:
                raise NotImplementedError("Init costs should be called before init rewards")
        else:
            raise NotImplementedError("Reward model is not implemented for multiclass safety")

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
        # Create temporary eval dataloader with the eval_batch_size if specified
        if hasattr(self.args, 'eval_batch_size') and self.args.eval_batch_size > 0:
            temp_eval_dataloader = DataLoader(
                self.eval_dataloader.dataset,
                collate_fn=self.eval_dataloader.dataset.get_collator(),
                sampler=DistributedSampler(self.eval_dataloader.dataset, shuffle=True),
                batch_size=self.args.eval_batch_size,
            )
            eval_dict = self.run_eval(temp_eval_dataloader, prefix="eval/test")
        else:
            eval_dict = self.run_eval(self.eval_dataloader, prefix="eval/test")

        if self.args.train_batches_on_eval > 0:
            # Create temporary train dataloader with the eval_batch_size if specified
            if hasattr(self.args, 'eval_batch_size') and self.args.eval_batch_size > 0:
                temp_train_dataloader = DataLoader(
                    self.train_dataloader.dataset,
                    collate_fn=self.train_dataloader.dataset.get_collator(),
                    sampler=DistributedSampler(self.train_dataloader.dataset, shuffle=True),
                    batch_size=self.args.eval_batch_size,
                )
                train_dict = self.run_eval(
                    temp_train_dataloader,
                    prefix="eval/train",
                    num_batches=self.args.train_batches_on_eval,
                    num_responses=self.args.num_responses_eval,
                )
            else:
                train_dict = self.run_eval(
                    self.train_dataloader,
                    prefix="eval/train",
                    num_batches=self.args.train_batches_on_eval,
                    num_responses=self.args.num_responses_eval,
                )
            eval_dict = {**eval_dict, **train_dict}
        return eval_dict

    def run_eval(
        self,
        eval_dataloader: DataLoader,
        prefix: str = "eval",
        num_batches: int = None,
        return_costs: bool = False,
        num_responses: int = 1,
    ) -> dict[str, Any]:
        """Evaluate the model."""
        # Accumulate costs and KL divergences across all evaluated samples
        #   all_costs: list[tensor] where each tensor has shape (B, R, C)
        #   all_kl_divs: list[tensor] where each tensor has shape (B, R)
        #   Here B = batch_size, R = num_responses, C = num_classes
        all_costs = []
        all_kl_divs = []

        with torch.no_grad():
            # Move reference & cost models to the correct device (no-op if already there)
            ref_model = to_device(self.reference_model, self.args.device)
            cost_model = to_device(self.cost_model, self.args.device)

            batch_count = 0
            for batch in tqdm(eval_dataloader, desc='Evaluating model'):
                batch = to_device(batch, self.args.device)

                batch_size = batch["input_ids"].size(0)

                # ----------------------------------------------------------------------------------
                # 1) Prepare prompts (remove response tokens) and generate new responses
                # ----------------------------------------------------------------------------------
                prompts = []
                response_starts = []
                for i in range(batch_size):
                    # Find the first index where response_mask == 1 for sample i
                    response_start_i = (
                        (batch["response_mask"][i] == 1).nonzero(as_tuple=True)[0][0].item()
                    )
                    response_starts.append(response_start_i)
                    prompt_i = batch["input_ids"][i, :response_start_i]
                    prompts.append(prompt_i)
                # Pad prompts to the same length so we can perform batched generation
                prompts_padded = torch.nn.utils.rnn.pad_sequence(
                    prompts,
                    batch_first=True,
                    padding_value=self.tokenizer.pad_token_id,
                )
                prompts_attention_mask = prompts_padded.ne(self.tokenizer.pad_token_id)
                # breakpoint()
                # Generate one response per prompt (keeping the prompts as prefix)
                generated_ids = self.model.generate(
                    prompts_padded,
                    attention_mask=prompts_attention_mask,
                    max_length=self.args.max_length,
                    do_sample=True,
                    num_return_sequences=num_responses,
                )
                # breakpoint()

                # ----------------------------------------------------------------------------------
                # 2) Compute KL divergence between policy and reference model for the generated samples
                # ----------------------------------------------------------------------------------
                generated_attention_mask = generated_ids.ne(self.tokenizer.pad_token_id)

                logprobs = self.compute_log_probs(
                    self.model, generated_ids, generated_attention_mask
                )
                sequence_logprobs = logprobs.sum(dim=1)  # shape = (B,)

                reference_logprobs = self.compute_log_probs(
                    ref_model, generated_ids, generated_attention_mask
                )
                reference_sequence_logprobs = reference_logprobs.sum(dim=1)  # shape = (B,)

                kl_div_batch = sequence_logprobs - reference_sequence_logprobs  # shape = (B,)

                # Reshape KL divergence to (batch_size, num_responses) so we can later
                # concatenate on the batch dimension while preserving the num_responses axis
                kl_div_batch = kl_div_batch.view(batch_size, num_responses)  # (B, R)
                all_kl_divs.append(kl_div_batch.cpu())

                # ----------------------------------------------------------------------------------
                # 3) Evaluate cost model on generated responses
                # ----------------------------------------------------------------------------------
                answers = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                # Remove hard-coded conversation markers if they exist
                answers = [a.replace("BEGINNING OF CONVERSATION: USER: ", "") for a in answers]
                answers = [a.replace("ASSISTANT:", "") for a in answers]

                tokenized_answer = self.cost_tokenizer(
                    answers,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.args.device)

                outputs = cost_model(
                    tokenized_answer["input_ids"],
                    attention_mask=tokenized_answer["attention_mask"],
                )
                probs = torch.softmax(outputs.logits, dim=-1)  # shape = (B, C)

                # Reshape probs to (batch_size, num_responses, C)
                probs = probs.view(batch_size, num_responses, -1)  # (B, R, C)

                all_costs.append(probs.cpu())

                batch_count += 1
                if num_batches is not None and batch_count >= num_batches:
                    break

        # Concatenate collected tensors along the batch dimension
        #   Costs: list[(B, R, C)] -> (N_prompt, R, C)
        #   KL   : list[(B, R)]    -> (N_prompt, R)
        all_costs = torch.cat(all_costs, dim=0)  # (N, R, C)
        all_kl_divs = torch.cat(all_kl_divs, dim=0)  # (N, R)

        # Treat each response as an independent sample when averaging -> collapse first two dims
        all_costs_flat = all_costs.view(-1, all_costs.size(-1))  # (N*R, C)
        all_kl_divs_flat = all_kl_divs.view(-1)  # (N*R,)

        # ----------------------------------------------------------------------
        # Aggregate metrics across GPUs (if distributed) by computing global sums
        # and counts, then deriving the means. This avoids the need for
        # concatenating all individual samples which could be memory-intensive
        # when operating with many GPUs.
        # ----------------------------------------------------------------------

        # Local (per-GPU) sums and counts
        local_sum_costs = all_costs_flat.sum(dim=0).to(self.args.device)  # (C,)
        local_sum_kl = all_kl_divs_flat.sum().to(self.args.device)
        local_count = torch.tensor(
            all_costs_flat.size(0), device=self.args.device, dtype=local_sum_kl.dtype
        )

        # Reduce across all processes if using distributed training
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sum_costs, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_sum_kl, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

        # Global means
        all_costs_mean = local_sum_costs / local_count  # (C,)
        all_kl_divs_mean = local_sum_kl / local_count

        # get slacks
        slacks = all_costs_mean - self.args.safety_threshold
        cost_dict = {
            f'{prefix}/cost[{i}]': cost.item() for i, cost in enumerate(all_costs_mean.cpu())
        }
        slack_dict = {f'{prefix}/slack[{i}]': slack.item() for i, slack in enumerate(slacks)}
        kl_dict = {f'{prefix}/kl_div': all_kl_divs_mean.item()}
        if return_costs:
            # Optionally gather the per-sample costs across all GPUs so that the
            # caller can access the complete tensor on the main process. We use
            # `all_gather_object` to support tensors with different lengths.
            if dist.is_available() and dist.is_initialized():
                gathered_costs: list[torch.Tensor] = [None for _ in range(dist.get_world_size())]  # type: ignore[arg-type]
                dist.all_gather_object(gathered_costs, all_costs.cpu())
                if is_main_process():
                    all_costs_full = torch.cat(gathered_costs, dim=0)
                else:
                    all_costs_full = torch.empty(0, dtype=all_costs.dtype)
            else:
                all_costs_full = all_costs

            return {**cost_dict, **slack_dict, **kl_dict, "costs": all_costs_full}
        else:
            return {**slack_dict, **kl_dict}

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

    def solve_closed_form_dual(self, costs: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Solve the dual problem."""
        if rewards is None:
            rewards = torch.zeros((costs.shape[0], costs.shape[1]))

        dual_optimizer = DualOptimizer(
            costs.float().cpu().numpy(),
            rewards.cpu().numpy(),
            self.args.safety_threshold,
            self.args.scale_coeff,
            self.args.dual_weight_decay,
            use_both_only=self.args.dual_solver_use_both_only,
        )
        print("Solving dual ...")
        lam = dual_optimizer.solve(optimizer='GD', set_optimum=True)
        return lam


class DualOptimizer:
    def __init__(
        self,
        safety_scores,
        reward,
        thresholds,  # E_{pi}[safety] >= thresholds
        kl_coeff,
        weight_decay=0.0,
        use_both_only=False,
        **kwargs,
    ):
        if reward is None:
            self.helpfulness_scores = np.zeros((safety_scores.shape[0], 1))
        else:
            self.helpfulness_scores = reward
        if use_both_only:
            worst_score_per_prompt = safety_scores.max(axis=(2))
            both = (worst_score_per_prompt.max(axis=1) > thresholds) * (
                worst_score_per_prompt.min(axis=1) < thresholds
            )
            self.safety_scores = safety_scores[both]
            self.helpfulness_scores = self.helpfulness_scores[both]
        else:
            self.safety_scores = safety_scores
        self.thresholds = thresholds
        self.kl_coeff = kl_coeff
        self.kwargs = kwargs
        self.weight_decay = weight_decay
        print(
            f"Dual solver initialized with {self.safety_scores.shape[0]} samples and {self.safety_scores.shape[1]} responses"
        )

    def log_mean_Z_values_old(self, logits):
        return np.log(np.mean(np.exp(logits - logits.max(axis=-1)), axis=1)) + logits.max(axis=-1)

    def log_mean_Z_values(self, logits):
        # Get max along last dimension for each batch and response
        # logits shape: (2000,10)
        max_logits = logits.max(axis=-1, keepdims=True)  # Shape: (2000,1)

        # Subtract max and exp (stable computation)
        exp_logits = np.exp(logits - max_logits)  # Shape: (2000,10)

        # Mean along last two dimensions (over the 10 responses)
        mean_exp = np.mean(exp_logits, axis=-1)  # Shape: (2000)

        # Log and add back the max values
        return np.log(mean_exp) + max_logits.squeeze(-1)  # Shape: (2000)

    def solve(
        self, optimizer='GD', set_optimum=False, beta=0.1, verbose=False, max_loops=100, **kwargs
    ):
        lam_init = 1 if 'lam_init' not in kwargs.keys() else kwargs['lam_init']
        lr = 5 if 'lr' not in kwargs.keys() else 2 * kwargs['lr']
        max_iters = 1000 if 'num_iters' not in kwargs.keys() else kwargs['num_iters']
        err = 1e-3 if 'err' not in kwargs.keys() else kwargs['err']
        if optimizer == 'scipy':
            raise NotImplementedError

        if optimizer == 'GD':
            is_converge = False
            num_loops = 0
            momentum = 0
            lam = lam_init * np.ones(self.safety_scores.shape[-1])
            lam_trajectory = []
            objective_trajectory = []
            constraint_trajectory = []
            helpfulness_trajectory = []
            safety_trajectory = []
            for num_loops in tqdm(range(max_loops)):
                lr = lr / (1.05 ** (num_loops))  # learning rate decay
                for idx_iter in range(max_iters):
                    logits = (
                        self.helpfulness_scores
                        - ((self.safety_scores - self.thresholds) * lam[None, None, :]).sum(axis=-1)
                    ) / self.kl_coeff
                    sm_probs = softmax(logits, axis=-1)
                    gradient = (
                        np.sum(
                            (sm_probs)[:, :, None] * (self.safety_scores - self.thresholds), axis=1
                        ).mean(axis=0)
                        + self.weight_decay * lam
                    )
                    # Nesterov momentum update
                    momentum = beta * momentum + lr * gradient
                    lam = np.maximum(lam + momentum, 0)
                    lam_trajectory.append(lam)
                    objective_trajectory.append(
                        self.kl_coeff * np.mean(self.log_mean_Z_values(logits))
                        - (lam * gradient).sum()
                    )
                    constraint_trajectory.append(gradient)
                    helpfulness_trajectory.append(
                        np.sum(sm_probs * self.helpfulness_scores, axis=1).mean()
                    )
                    safety_trajectory.append(
                        np.mean(sm_probs[:, :, None] * self.safety_scores, axis=(0, 1))
                    )
                    if (
                        idx_iter >= 3
                        and np.abs(lam - np.array(lam_trajectory[-1:-3:-1])).max() < err
                    ):  # the maximal difference, compared to the last 2 iterations, are smaller than 1e-5
                        is_converge = True
                        if verbose:
                            print(f'The optimization converges to a finite maximizer!')
                        break
                if is_converge:
                    print("Converged!")
                    if set_optimum:
                        self.lam_star = lam
                    break
                else:
                    print("Current Lambda")
                    print(lam)
                    print(f"Didn't converge in loop {num_loops+1}")
                num_loops += 1
                if is_converge:
                    break
            if not is_converge:
                print(
                    f'The dual problem (threshold={self.thresholds}, sample_shape={self.helpfulness_scores.shape}) may not have a finite maximizer!'
                )
                if set_optimum:
                    self.lam_star = None

            return lam_trajectory[-1] if is_converge else None
