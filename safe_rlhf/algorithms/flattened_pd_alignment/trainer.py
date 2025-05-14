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

from __future__ import annotations

import argparse
from typing import Any

import deepspeed
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig

# Inherit from the trainer that uses FlattenedPreferenceDataset
from safe_rlhf.trainers.flattened_preference_dual_trainer import FlattenedPreferenceDualTrainer
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.utils import gather_log_probabilities, get_all_reduce_mean


class FlattenedPdAlignmentTrainer(FlattenedPreferenceDualTrainer):
    TRAINING_TYPE = 'flattened_pd_alignment'
    # DATASET_TYPE is already set to FlattenedPreferenceDataset in the parent class

    model: deepspeed.DeepSpeedEngine

    ds_train_config: dict[str, Any]
    ds_eval_config: dict[str, Any]

    def __init__(
        self,
        args: argparse.Namespace,
        ds_train_config: dict[str, Any],
        ds_eval_config: dict[str, Any],
    ) -> None:
        """Initialize trainer."""
        self.args = args
        self.ds_train_config = ds_train_config
        self.ds_eval_config = ds_eval_config
        self.scale_coeff = args.scale_coeff
        # Initialize the parent class (FlattenedPreferenceDualTrainer -> MultiDualTrainer)
        super().__init__(args, ds_train_config)

    def init_models(self) -> None:
        """Initialize model and tokenizer, applying LoRA."""
        if (
            self.ds_train_config is not None
            and self.ds_train_config['zero_optimization']['stage'] == 3
        ):
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_config)

        if (
            self.ds_eval_config is not None
            and self.ds_eval_config['zero_optimization']['stage'] == 3
        ):
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_config)

        self.model, self.tokenizer = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left', # Ensure padding side is correct for generation/causal LM
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )
        # Apply LoRA configuration
        if self.args.lora_r > 0:
            print(f"Applying LoRA with r={self.args.lora_r}, alpha={self.args.lora_alpha}")
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=self.args.lora_r,
                    lora_alpha=self.args.lora_alpha,
                    lora_dropout=self.args.lora_dropout,
                    target_modules=self.args.lora_target_modules,
                    # Ensure bias is handled appropriately if needed
                    # bias="none", # or "all" or "lora_only"
                ),
            )
            print("LoRA applied.")
        else:
            print("LoRA not applied (lora_r <= 0).")


    @staticmethod
    def compute_log_probs(
        model: AutoModelForCausalLM, # Should be the underlying model, not DeepSpeedEngine
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        # Ensure the model is accessed correctly (might be model.module with DeepSpeed)
        unwrapped_model = model.module if hasattr(model, 'module') else model
        logits = unwrapped_model(input_ids, attention_mask=attention_mask, use_cache=False).logits
        # Logits shape: (B, L, V), input_ids shape: (B, L)
        # We need logprobs for input_ids[:, 1:], computed from logits[:, :-1]
        log_probs = gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])
        return log_probs # Shape: (B, L-1)

    def loss(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        multipliers: torch.FloatTensor,
        costs: torch.FloatTensor, # Derived from batch['labels']
        rewards: torch.FloatTensor, # Could be None or derived
        ref_sequence_log_probs: torch.FloatTensor, # Precomputed baseline
        response_mask: torch.BoolTensor,  # Provided by dataloader
        debug: bool = False,
        **kwargs, # To catch extra args like original_index, is_safe_response
    ) -> dict[str, torch.Tensor]:
        """Loss function adapted for flattened data."""
        # Compute log probs for the current policy model
        # Note: self.model is the DeepSpeedEngine, access module for underlying model
        sequence_log_probs = self.compute_log_probs(
            self.model, # Pass the engine, static method handles unwrapping if needed
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Apply response mask: We only care about the log probs of the response part
        # sequence_log_probs has shape (B, L-1), response_mask[:, 1:] has shape (B, L-1)
        masked_sequence_log_probs = sequence_log_probs * response_mask[:, 1:]
        summed_sequence_log_probs = masked_sequence_log_probs.sum(dim=1) # Shape: (B,)

        # Calculate log ratio relative to the precomputed baseline
        log_ratio = summed_sequence_log_probs - ref_sequence_log_probs # Shape: (B,)

        with torch.no_grad():
            importance_weights = torch.exp(log_ratio)
            # Clip importance weights for stability
            importance_weights = torch.clamp(importance_weights, min=1e-6, max=10)

        if debug:
            breakpoint()

        # DKL loss component (regularization towards baseline)
        # Note: The formula 0.5 * (1 + log_ratio)^2 might be specific; adjust if needed.
        # This term penalizes large deviations from the baseline log prob ratio.
        dkl_loss = self.scale_coeff * importance_weights * 0.5 * (1 + log_ratio) ** 2

        # Safety loss component (incorporating costs, rewards, thresholds, multipliers)
        # Ensure rewards are handled correctly (might be 0 if not provided)
        if rewards is None:
            effective_rewards = 0.0
        else:
            effective_rewards = rewards # Assumes rewards shape is compatible (e.g., (B, C))

        # Cost term: (costs - threshold) * multipliers
        # Costs shape: (B, C), threshold shape: (C,), multipliers shape: (C,)
        # Result shape: (B, C)
        cost_term = (costs - self.args.safety_threshold) * multipliers[None, :]

        # Combine rewards and cost term -> shape (B, C)
        # Sum over the cost dimension (C) -> shape (B,)
        safety_objective = (-effective_rewards + cost_term).sum(dim=-1)

        # Weight safety objective by importance weights and log probs
        # This seems unusual - typically safety loss is weighted only by importance weights
        # Original: importance_weights * sequence_log_probs * safety_objective.sum()
        # Let's assume it should be weighted by importance weights only:
        safety_loss = importance_weights * safety_objective

        # Total loss per sample
        losses = safety_loss + dkl_loss # Shape: (B,)
        loss = losses.mean() # Average loss over the batch

        # Detach tensors for logging to prevent gradient tracking
        dkl_loss_d = dkl_loss.mean().detach()
        safety_loss_d = safety_loss.mean().detach()
        importance_weights_d = importance_weights.mean().detach()

        with torch.no_grad():
            # Calculate cost and reward metrics weighted by importance weights
            weighted_costs = (costs * importance_weights[:, None]).sum(dim=0) / importance_weights.sum()
            if rewards is not None:
                 weighted_rewards = (rewards * importance_weights[:, None]).sum(dim=0) / importance_weights.sum()
            else:
                weighted_rewards = torch.zeros_like(costs[0]) # Placeholder if no rewards

            # Feasibility calculation based on costs vs threshold
            infeasible_mask = (costs > self.args.safety_threshold).any(dim=-1).float()
            feasible_mask = 1.0 - infeasible_mask
            importance_infeasible = (importance_weights * infeasible_mask).sum() / infeasible_mask.sum().clamp(min=1)
            importance_feasible = (importance_weights * feasible_mask).sum() / feasible_mask.sum().clamp(min=1)
            feasible_frac = feasible_mask.mean()


        return {
            'loss': loss,
            'dkl_loss': dkl_loss_d,
            'safety_loss': safety_loss_d,
            'importance_weights': importance_weights_d,
            'costs': weighted_costs, # Log average weighted cost per category
            'rewards': weighted_rewards, # Log average weighted reward per category
            'importance_infeasible': importance_infeasible.detach(),
            'importance_feasible': importance_feasible.detach(),
            'feasible': feasible_frac.detach(),
        }

    def dual_step(
        self,
        slacks: torch.Tensor,
        multipliers: torch.Tensor
    ):
        """Dual variable update step based on slacks."""
        # The dual update rule from MultiPdAlignementTrainer
        # Note: Requires resilient_coeff argument
        # Update: slacks = costs - threshold (averaged over eval)
        gradient = slacks # Gradient is just the average slack
        if self.args.resilient_coeff > 0:
             gradient -= 1 / (2 * self.args.resilient_coeff) * multipliers

        multipliers = multipliers + self.args.dual_step_size * gradient

        # Clamp multipliers to be non-negative
        multipliers = torch.clamp(multipliers, min=0)
        return multipliers

    def train_step(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
        index: torch.LongTensor,  # size = (B,) - Indices within the *current* dataset view
        labels: torch.FloatTensor, # size = (B, C) - These are the costs from flattened data
        response_mask: torch.BoolTensor,  # size = (B, L)
        original_index: torch.LongTensor | None = None, # Index from original HF dataset
        is_safe_response: torch.BoolTensor | None = None, # Flag from flattening
        **kwargs: Any, # Catch-all for other potential args
    ) -> dict[str, Any]:
        """Perform a single training step using flattened preference data."""
        # Fetch pre-computed costs using the batch indices.
        # self.costs is already shaped to (num_samples, num_classes-1) in init_costs.
        # 'index' from the batch refers to the sample's position in the (shuffled) dataset.
        batch_costs = self.costs[index]

        # Rewards might be loaded separately or be None
        # If self.rewards exists (e.g., loaded from cache), index it using `original_index`
        # This assumes self.rewards maps original indices to rewards
        if self.rewards is not None and original_index is not None:
            # Ensure original_index is valid and self.rewards is populated correctly
            batch_rewards = self.rewards[original_index]
        elif self.rewards is not None: # Fallback if original_index not available but self.rewards is
            batch_rewards = self.rewards[index]
        else:
            # Handle case where rewards are not used or not available
            batch_rewards = None # Or torch.zeros_like(batch_costs)

        # Baseline logprobs are indexed using the `original_index` if available,
        # assuming self.baseline_logprobs maps original indices to logprobs.
        # If original_index isn't passed, assumes `index` refers to baseline correctly.
        current_baseline_idx = original_index if original_index is not None else index
        batch_ref_sequence_log_probs = self.baseline_logprobs[current_baseline_idx]

        # Calculate loss using the policy model
        loss_dict = self.loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            multipliers=self.multipliers, # Current dual variables
            costs=batch_costs,
            rewards=batch_rewards,
            ref_sequence_log_probs=batch_ref_sequence_log_probs,
            response_mask=response_mask,
            debug=self.args.debug, # Pass debug flag
            # Pass other relevant fields if needed by loss or metrics
            original_index=original_index,
            is_safe_response=is_safe_response,
        )

        loss = loss_dict['loss']

        # Backpropagation and optimizer step via DeepSpeed engine
        self.model.backward(loss)
        self.model.step()

        # Post-step checks and logging (similar to MultiPdAlignementTrainer)
        if self.args.debug:
            # Check gradients
            for name, param in self.model.module.named_parameters():
                if param.requires_grad:
                    print(f"{name} Grad norm: {param.grad.norm().item() if param.grad is not None else 'None'}")
            # Recompute loss for debugging if needed
            loss_dict = self.loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                multipliers=self.multipliers,
                costs=batch_costs,
                rewards=batch_rewards,
                ref_sequence_log_probs=batch_ref_sequence_log_probs,
                response_mask=response_mask,
                debug=True,
                original_index=original_index,
                is_safe_response=is_safe_response,
            )

        # Aggregate metrics for logging (using get_all_reduce_mean for distributed training)
        with torch.no_grad():
            dkl_loss = get_all_reduce_mean(loss_dict['dkl_loss'])
            safety_loss = get_all_reduce_mean(loss_dict['safety_loss'])
            importance_weights = get_all_reduce_mean(loss_dict['importance_weights'])
            feasible = get_all_reduce_mean(loss_dict['feasible'])
            importance_infeasible = get_all_reduce_mean(loss_dict['importance_infeasible'])
            importance_feasible = get_all_reduce_mean(loss_dict['importance_feasible'])
            loss = get_all_reduce_mean(loss) # Average the final loss as well

            # Aggregate costs and rewards (which are per-category tensors)
            # Keep on device for all_reduce
            avg_costs = loss_dict['costs'].detach() 
            avg_rewards = loss_dict['rewards'].detach()
            
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(avg_costs, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(avg_rewards, op=torch.distributed.ReduceOp.AVG)

            # Now move to CPU for logging
            avg_costs_cpu = avg_costs.cpu()
            avg_rewards_cpu = avg_rewards.cpu()

        log_data = {
            'train/loss': loss.item(),
            'train/dkl_loss': dkl_loss.item(),
            'train/safety_loss': safety_loss.item(),
            'train/importance_weights': importance_weights.item(),
            'train/importance_infeasible': importance_infeasible.item(),
            'train/importance_feasible': importance_feasible.item(),
            'train/feasible_fraction': feasible.item(),
        }
        # Add per-category costs and rewards to log
        for i, cost_val in enumerate(avg_costs_cpu): # Use CPU tensors for logging
            log_data[f'train/costs_{i}'] = cost_val.item()
        for i, reward_val in enumerate(avg_rewards_cpu): # Use CPU tensors for logging
             log_data[f'train/rewards_{i}'] = reward_val.item()

        # Log multipliers
        for i, mult_val in enumerate(self.multipliers):
             log_data[f'multipliers/{i}'] = mult_val.item()

        return log_data 