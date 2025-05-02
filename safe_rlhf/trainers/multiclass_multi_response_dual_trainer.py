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
    num_classes: int

    def __init__(self, args: argparse.Namespace, ds_config: dict[str, Any]) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_config = ds_config
        self.global_step = 0
        self.num_classes = args.num_classes

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

    @abc.abstractmethod
    def compute_log_probs(
        self, model: AutoModelForCausalLM, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probabilities for given sequences."""
        # This needs to be implemented in the subclass inheriting MultiDualTrainer
        # Example implementation might involve model forward pass and gathering logprobs
        raise NotImplementedError("compute_log_probs must be implemented in subclass")

    def init_baseline(self) -> None:
        """Initialize baseline log probabilities with caching functionality for multi-response data."""
        os.makedirs(self.args.cache_dir, exist_ok=True)
        baseline_cache_path = os.path.join(
            self.args.cache_dir, "cached_baseline_logprobs_multiresponse.pt"
        )  # Adjusted cache name

        # Ensure dataset object is available
        if not hasattr(self, 'train_dataloader') or self.train_dataloader is None:
            raise RuntimeError("train_dataloader not initialized. Call init_datasets first.")
        train_dataset = self.train_dataloader.dataset
        num_prompts = len(train_dataset)
        num_responses = train_dataset.num_responses

        # Load reference model for KL computation during evaluation if needed
        # Moved reference model loading to run_eval to avoid keeping it in memory
        # if self.args.compute_kl_eval: ...

        # Load cached baseline if available and not recomputing
        if os.path.exists(baseline_cache_path) and not self.args.recompute_baseline:
            print(f"Loading cached baseline logprobs from {baseline_cache_path}")
            cached_data = torch.load(baseline_cache_path, map_location=self.args.device)
            # Validate shape: (num_prompts, num_responses)
            expected_shape = (num_prompts, num_responses)
            if cached_data.shape != expected_shape:
                print(
                    f"Warning: Cached baseline shape mismatch! Expected {expected_shape}, got {cached_data.shape}. Recomputing..."
                )
            else:
                self.baseline_logprobs = cached_data
                print(
                    f"Loaded cached baseline logprobs successfully with shape {self.baseline_logprobs.shape}."
                )
                return

        # If cache miss or recompute needed:
        print("Computing baseline logprobs for multi-response data...")

        # Initialize baseline tensor: (num_prompts, num_responses)
        self.baseline_logprobs = torch.zeros(
            (num_prompts, num_responses),
            # Use policy model dtype (assuming model is initialized)
            dtype=self.model.dtype if hasattr(self, 'model') else torch.float32,
            device=self.args.device,  # Initialize directly on target device
        )

        # Load reference model specifically for baseline computation
        print("Loading reference model for baseline computation...")
        baseline_reference_model, _ = load_pretrained_models(
            self.args.model_name_or_path,  # Use the same base model as policy
            model_max_length=self.args.max_length,
            padding_side='right',  # IMPORTANT: Must match the padding used in the dataset/collator for logprob calculation
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )
        baseline_reference_model.requires_grad_(False)
        baseline_reference_model.eval()
        baseline_reference_model.to(self.args.device)

        # Compute logprobs for each batch
        for batch in tqdm(self.train_dataloader, desc='Computing baseline logprobs'):
            batch = to_device(batch, self.args.device)
            b_size, n_responses, seq_len = batch['input_ids'].shape

            # Reshape inputs for compute_log_probs: (B, N, L) -> (B * N, L)
            input_ids_flat = batch['input_ids'].reshape(b_size * n_responses, seq_len)
            attention_mask_flat = batch['attention_mask'].reshape(b_size * n_responses, seq_len)
            response_mask_flat = batch['response_mask'].reshape(b_size * n_responses, seq_len)

            # Compute logprobs using the reference model
            # Assumes compute_log_probs takes (B*N, L) input and returns (B*N, L-1)
            with torch.no_grad():
                logprobs_flat = self.compute_log_probs(
                    baseline_reference_model,
                    input_ids_flat,
                    attention_mask_flat,  # Pass attention mask
                )

            # Mask and sum logprobs for each sequence
            # Apply response mask (shifted) and attention mask (shifted)
            # Make sure masks align with logprobs shape (L-1)
            if logprobs_flat.shape[1] == seq_len - 1:
                mask = response_mask_flat[:, 1:] & attention_mask_flat[:, 1:]
            elif logprobs_flat.shape[1] == seq_len:
                # Some compute_log_probs might return L, handle slicing
                mask = response_mask_flat & attention_mask_flat
                # We need to mask the first token's logprob if it exists
                # This depends on how compute_log_probs handles the first token
                # Assuming we sum logprobs from the first *response* token onwards
                # Let's stick to the L-1 assumption based on typical CausalLM output
                print(
                    "Warning: compute_log_probs returned shape L, expected L-1. Adjust masking logic if needed."
                )
                mask = mask[:, 1:]  # Attempt to align
            else:
                raise ValueError(
                    f"Unexpected logprobs shape {logprobs_flat.shape} from compute_log_probs for input seq_len {seq_len}"
                )

            # Ensure mask shape matches logprobs shape
            if mask.shape != logprobs_flat.shape:
                raise ValueError(
                    f"Mask shape {mask.shape} does not match logprobs shape {logprobs_flat.shape}"
                )

            sequence_logprobs_flat = (logprobs_flat * mask).sum(dim=-1)

            # Reshape summed logprobs back: (B * N,) -> (B, N)
            sequence_logprobs = sequence_logprobs_flat.view(b_size, n_responses)

            # Store in the main tensor using prompt indices
            prompt_indices = batch['index']  # Shape: (B,)
            self.baseline_logprobs[prompt_indices] = sequence_logprobs.to(
                self.baseline_logprobs.dtype
            )

        # Save computed baseline logprobs
        print(
            f"Saving computed baseline logprobs (shape {self.baseline_logprobs.shape}) to {baseline_cache_path}"
        )
        torch.save(self.baseline_logprobs, baseline_cache_path)
        print("Saved baseline logprobs successfully")

        # Free up memory
        del baseline_reference_model
        torch.cuda.empty_cache()
        return

    def init_costs(self) -> None:
        """Initialize costs with caching functionality for multi-response data."""
        os.makedirs(self.args.cache_dir, exist_ok=True)
        costs_cache_path = os.path.join(
            self.args.cache_dir, "cached_costs_multiresponse.pt"
        )  # Adjusted cache name

        # Ensure dataset object is available
        if not hasattr(self, 'train_dataloader') or self.train_dataloader is None:
            raise RuntimeError("train_dataloader not initialized. Call init_datasets first.")
        # Access the dataset instance to get num_prompts and num_responses
        train_dataset = self.train_dataloader.dataset
        num_prompts = len(train_dataset)
        num_responses = train_dataset.num_responses

        # Load cost model for evaluation if needed (Moved loading to run_eval)
        # if self.args.compute_costs_eval: ...

        # Load cached costs if available and not recomputing
        if os.path.exists(costs_cache_path) and not self.args.recompute_costs:
            print(f"Loading cached costs from {costs_cache_path}")
            cached_data = torch.load(costs_cache_path, map_location=self.args.device)
            # Shape check: (num_prompts, num_responses, num_classes)
            expected_shape = (num_prompts, num_responses, self.args.num_classes)
            if cached_data.shape != expected_shape:
                print(
                    f"Warning: Cached costs shape mismatch! Expected {expected_shape}, got {cached_data.shape}. Recomputing..."
                )
            else:
                self.costs = cached_data
                print(f"Loaded cached costs successfully with shape {self.costs.shape}.")
                # Apply final processing (e.g., removing last class dimension)
                self.costs = self.costs[:, :, :-1]  # Shape: (num_prompts, num_responses, C-1)
                print(f"Processed cached costs shape: {self.costs.shape}")
                return

        # If cache miss or recompute needed:
        print("Computing costs for multi-response data...")

        # Initialize costs tensor: (num_prompts, num_responses, num_classes)
        # Initialize directly on target device, dtype determined later
        temp_costs = torch.zeros(
            (num_prompts, num_responses, self.args.num_classes), device=self.args.device
        )

        if self.args.cost_model_name_or_path == "indicator":
            print("Using indicator costs from dataset labels.")
            # Labels should already be on the correct device from the dataloader
            for batch in tqdm(self.train_dataloader, desc='Computing indicator costs'):
                batch = to_device(batch, self.args.device)  # Ensure device
                labels = batch['labels']  # Shape: (B, N, C)
                prompt_indices = batch['index']  # Shape: (B,)
                if labels.shape[-1] != self.args.num_classes:
                    raise ValueError(
                        f"Indicator label dimension ({labels.shape[-1]}) does not match num_classes ({self.args.num_classes})"
                    )
                # Ensure dtype consistency
                temp_costs[prompt_indices] = labels.to(dtype=temp_costs.dtype)
            self.costs = temp_costs  # Assign computed costs

        else:
            print("Loading cost model for cost computation...")
            cost_model, cost_tokenizer = load_pretrained_models(
                self.args.cost_model_name_or_path,
                model_max_length=self.args.max_length,
                auto_model_type=AutoModelForSequenceClassification,
                padding_side='right',  # Cost model typically uses right padding
                trust_remote_code=self.args.trust_remote_code,
            )
            cost_model.requires_grad_(False)
            cost_model.eval()
            cost_model.to(self.args.device)

            # Ensure temp_costs has the correct dtype from the cost model
            temp_costs = temp_costs.to(dtype=cost_model.dtype)

            for batch in tqdm(self.train_dataloader, desc='Computing model costs'):
                batch = to_device(batch, self.args.device)
                b_size, n_responses, seq_len = batch['input_ids'].shape
                prompt_indices = batch['index']  # Shape: (B,)

                # Reshape inputs: (B, N, L) -> (B * N, L)
                input_ids_flat = batch['input_ids'].reshape(b_size * n_responses, seq_len)
                attention_mask_flat = batch['attention_mask'].reshape(b_size * n_responses, seq_len)

                # Compute costs using the cost model
                with torch.no_grad():
                    outputs = cost_model(input_ids_flat, attention_mask=attention_mask_flat)
                    # Probs shape: (B * N, C)
                    probs_flat = torch.softmax(outputs.logits, dim=-1)

                # Reshape probs back: (B * N, C) -> (B, N, C)
                probs = probs_flat.view(b_size, n_responses, self.args.num_classes)

                # Store in the main tensor using prompt indices
                temp_costs[prompt_indices] = probs

            self.costs = temp_costs  # Assign computed costs
            # Free up memory
            del cost_model, cost_tokenizer
            torch.cuda.empty_cache()

        # Save computed costs (before final processing)
        print(f"Saving computed costs (shape {self.costs.shape}) to {costs_cache_path}")
        torch.save(self.costs, costs_cache_path)
        print("Saved costs successfully")

        # Apply final processing (e.g., removing last class)
        self.costs = self.costs[:, :, :-1]  # Shape: (num_prompts, num_responses, C-1)
        print(f"Processed final costs shape: {self.costs.shape}")

        # No need to reinitialize dataset, tokenizer handling is in dataset class
        return

    def init_rewards(self) -> None:
        """Initialize rewards with caching functionality."""
        if self.args.reward_model_name_or_path == "none":
            self.rewards = None
            print("No reward model specified, self.rewards set to None.")
            return
        elif self.args.reward_model_name_or_path == "safety_prob":
            # Use the same cache path as init_costs (before final processing)
            costs_cache_path = os.path.join(self.args.cache_dir, "cached_costs_multiresponse.pt")

            # Ensure dataset object is available to know expected shape
            if not hasattr(self, 'train_dataloader') or self.train_dataloader is None:
                raise RuntimeError("train_dataloader not initialized. Call init_datasets first.")
            train_dataset = self.train_dataloader.dataset
            num_prompts = len(train_dataset)
            num_responses = train_dataset.num_responses

            if os.path.exists(costs_cache_path) and not self.args.recompute_costs:
                print(f"Loading cached cost probabilities as rewards from {costs_cache_path}")
                # Load the raw costs (num_prompts, num_responses, num_classes)
                loaded_rewards = torch.load(costs_cache_path, map_location=self.args.device)

                # Validate shape before processing
                expected_shape = (num_prompts, num_responses, self.args.num_classes)
                if loaded_rewards.shape != expected_shape:
                    print(
                        f"Warning: Cached rewards (costs) shape mismatch! Expected {expected_shape}, got {loaded_rewards.shape}. Cannot use cache."
                    )
                    # Proceed to raise error below as costs are needed but cache is invalid
                else:
                    # Process similarly to costs: remove last class
                    self.rewards = loaded_rewards[
                        :, :, :-1
                    ]  # Shape: (num_prompts, num_responses, C-1)
                    print(
                        f"Loaded and processed cached rewards successfully. Shape: {self.rewards.shape}"
                    )
                    return

            # If cache is missing or invalid, costs must be computed first
            raise RuntimeError(
                "Cannot initialize rewards from safety_prob because cached costs are missing, invalid, or recompute_costs is True. "
                "Ensure init_costs runs successfully before init_rewards when using safety_prob."
            )
        else:
            # If implementing other reward models, ensure they handle the multi-response structure
            # For now, assuming rewards would have shape (num_prompts, num_responses) or (num_prompts, num_responses, 1)
            # This part needs specific implementation based on the reward model type
            raise NotImplementedError(
                f"Reward model type '{self.args.reward_model_name_or_path}' is not implemented for multi-response multiclass safety."
                + " If implementing, ensure correct shape handling (e.g., (num_prompts, num_responses, reward_dim))."
            )

    def init_datasets(self, tokenizer=None) -> None:
        """Initialize training and evaluation datasets."""
        if tokenizer is None:
            tokenizer = self.tokenizer

        # Pass self.num_classes to dataset constructor
        train_dataset = self.DATASET_TYPE(
            self.args.train_datasets,
            tokenizer=tokenizer,
            lazy_tokenization=False,
            num_classes=self.num_classes,
            # seed=42, # Removed seed argument
        )

        if self.args.need_eval:
            if self.args.eval_datasets is None and self.args.eval_split_ratio is not None:
                # Assuming split_train_test method doesn't require seed, or uses internal seed
                # split_train_test might need adjustment if it re-initializes the dataset
                # For now, assume it works or handle split differently if needed.
                train_dataset, eval_dataset = train_dataset.split_train_test(
                    split_ratio=self.args.eval_split_ratio,
                )
            elif self.args.eval_datasets is not None and self.args.eval_split_ratio is None:
                # Pass self.num_classes to eval dataset constructor
                eval_dataset = self.DATASET_TYPE(
                    self.args.eval_datasets,
                    tokenizer=tokenizer,
                    lazy_tokenization=False,
                    num_classes=self.num_classes,
                    # seed=42 # Removed seed argument
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
        """Evaluate the model on the provided responses in the dataloader."""
        self.set_train(mode=False)  # Ensure model is in eval mode

        # Load necessary models if not already loaded (consider moving outside loop if memory allows)
        # Reference model for KL
        if self.args.compute_kl_eval:
            print("Loading reference model for KL evaluation...")
            reference_model, _ = load_pretrained_models(
                self.args.model_name_or_path,
                model_max_length=self.args.max_length,
                padding_side='right',  # Match dataset padding
                auto_model_type=AutoModelForCausalLM,
                trust_remote_code=self.args.trust_remote_code,
            )
            reference_model.requires_grad_(False)
            reference_model.eval()
            ref_model = to_device(reference_model, self.args.device)
        else:
            ref_model = None
            print("Skipping KL computation as compute_kl_eval is False.")

        # Cost model
        if self.args.compute_costs_eval:
            print("Loading cost model for evaluation...")
            cost_model, cost_tokenizer = load_pretrained_models(
                self.args.cost_model_name_or_path,
                model_max_length=self.args.max_length,
                auto_model_type=AutoModelForSequenceClassification,
                padding_side='right',  # Match cost model needs
                trust_remote_code=self.args.trust_remote_code,
            )
            cost_model.requires_grad_(False)
            cost_model.eval()
            cost_model = to_device(cost_model, self.args.device)
        else:
            cost_model = None
            print("Skipping cost computation as compute_costs_eval is False.")

        all_costs_per_class = []  # List to store mean costs per class for each batch
        all_kl_divs = []  # List to store mean KL divergence for each batch

        with torch.no_grad():
            batch_count = 0
            for batch in tqdm(eval_dataloader, desc=f'Evaluating ({prefix})'):
                batch = to_device(batch, self.args.device)
                b_size, n_responses, seq_len = batch['input_ids'].shape

                # Flatten inputs for processing: (B, N, L) -> (B * N, L)
                input_ids_flat = batch['input_ids'].reshape(b_size * n_responses, seq_len)
                attention_mask_flat = batch['attention_mask'].reshape(b_size * n_responses, seq_len)
                response_mask_flat = batch['response_mask'].reshape(b_size * n_responses, seq_len)

                # 1. Compute KL Divergence (if enabled)
                if ref_model is not None:
                    # Policy logprobs
                    policy_logprobs_flat = self.compute_log_probs(
                        self.model.module,  # Use underlying module for compute_log_probs
                        input_ids_flat,
                        attention_mask_flat,
                    )
                    # Reference logprobs
                    ref_logprobs_flat = self.compute_log_probs(
                        ref_model,
                        input_ids_flat,
                        attention_mask_flat,
                    )

                    # Mask and sum logprobs for each sequence
                    # Use the same mask logic as in init_baseline
                    if policy_logprobs_flat.shape[1] == seq_len - 1:
                        mask = response_mask_flat[:, 1:] & attention_mask_flat[:, 1:]
                    else:  # Assuming shape L, adjust if needed
                        print(
                            "Warning (eval KL): logprobs shape is L, expected L-1. Adjust mask if needed."
                        )
                        mask = response_mask_flat[:, 1:] & attention_mask_flat[:, 1:]
                        if mask.shape[1] != policy_logprobs_flat.shape[1]:
                            # If policy returns L, ref likely does too
                            mask = response_mask_flat & attention_mask_flat
                            policy_logprobs_flat = policy_logprobs_flat * mask
                            ref_logprobs_flat = ref_logprobs_flat * mask
                            # Sum over L, but need to decide how to handle KL calculation with differing masks
                            # Safest: assume L-1 output or ensure compute_log_probs is consistent
                            raise ValueError(
                                "Inconsistent logprob shapes or masking logic needed for KL eval"
                            )

                    policy_seq_logprobs_flat = (policy_logprobs_flat * mask).sum(dim=-1)
                    ref_seq_logprobs_flat = (ref_logprobs_flat * mask).sum(dim=-1)

                    # KL divergence per sequence: (B * N,)
                    kl_div_flat = policy_seq_logprobs_flat - ref_seq_logprobs_flat
                    # Average KL across all responses and prompts in the batch
                    batch_mean_kl = kl_div_flat.mean()
                    all_kl_divs.append(batch_mean_kl.cpu())  # Store mean KL for the batch

                # 2. Compute Costs (if enabled)
                if cost_model is not None:
                    # Compute costs using the cost model
                    outputs = cost_model(input_ids_flat, attention_mask=attention_mask_flat)
                    # Probs shape: (B * N, C)
                    probs_flat = torch.softmax(outputs.logits, dim=-1)

                    # Process costs: remove last class -> (B * N, C-1)
                    costs_flat = probs_flat[:, :-1]

                    # Average costs per class across all responses and prompts in the batch
                    batch_mean_costs_per_class = costs_flat.mean(dim=0)  # Shape: (C-1,)
                    all_costs_per_class.append(
                        batch_mean_costs_per_class.cpu()
                    )  # Store mean costs for the batch

                batch_count += 1
                if num_batches is not None and batch_count >= num_batches:
                    break

        # Aggregate results across batches
        final_metrics = {}

        # Aggregate KL
        if all_kl_divs:
            mean_kl = torch.stack(all_kl_divs).mean().item()
            final_metrics[f'{prefix}/kl_div'] = mean_kl
            print(f"[{prefix}] Mean KL Divergence: {mean_kl:.4f}")
        else:
            final_metrics[f'{prefix}/kl_div'] = np.nan  # Or some indicator that it wasn't computed

        # Aggregate Costs and Slacks
        if all_costs_per_class:
            # Average the batch means to get overall mean costs per class
            mean_costs = torch.stack(all_costs_per_class).mean(dim=0)  # Shape: (C-1,)
            # Compute slacks based on overall mean costs
            slacks = mean_costs - torch.tensor(self.args.safety_threshold, device=mean_costs.device)

            cost_dict = {f'{prefix}/cost[{i}]': cost.item() for i, cost in enumerate(mean_costs)}
            slack_dict = {f'{prefix}/slack[{i}]': slack.item() for i, slack in enumerate(slacks)}
            final_metrics.update(cost_dict)
            final_metrics.update(slack_dict)
            print(f"[{prefix}] Mean Costs per class: {mean_costs.cpu().numpy()}")
            print(f"[{prefix}] Mean Slacks per class: {slacks.cpu().numpy()}")
        else:
            num_cost_classes = self.args.num_classes - 1
            cost_dict = {f'{prefix}/cost[{i}]': np.nan for i in range(num_cost_classes)}
            slack_dict = {f'{prefix}/slack[{i}]': np.nan for i in range(num_cost_classes)}
            final_metrics.update(cost_dict)
            final_metrics.update(slack_dict)

        # Add multipliers (these don't change during eval)
        multiplier_dict = {
            f'{prefix}/multiplier[{i}]': m.item() for i, m in enumerate(self.multipliers)
        }
        final_metrics.update(multiplier_dict)

        # Clean up models loaded specifically for eval if needed
        if ref_model is not None:
            del ref_model
        if cost_model is not None:
            del cost_model, cost_tokenizer
        torch.cuda.empty_cache()

        self.set_train(mode=True)  # Set model back to train mode
        return final_metrics

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
