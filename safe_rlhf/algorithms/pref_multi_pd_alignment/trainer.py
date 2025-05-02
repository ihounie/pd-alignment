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
import os
from typing import Any

import deepspeed
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from safe_rlhf.datasets import PreferenceCostDataset
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.trainers import PrefMultiDualTrainer
from safe_rlhf.utils import gather_log_probabilities, get_all_reduce_mean


class PrefMultiPdAlignmentTrainer(PrefMultiDualTrainer):
    TRAINING_TYPE = 'pref_multi_pd_alignment'
    DATASET_TYPE = PreferenceCostDataset

    model: deepspeed.DeepSpeedEngine
    reference_model: deepspeed.DeepSpeedEngine

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
        super().__init__(args, ds_train_config)

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
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
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )
        self.model = get_peft_model(
            self.model,
            LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                target_modules=self.args.lora_target_modules,
            ),
        )

    def init_costs(self) -> None:
        """Costs are provided in the dataset, no need to initialize separately."""
        pass

    def init_rewards(self) -> None:
        """Rewards are not needed as costs are provided in the dataset."""
        self.rewards = None
        pass

    @staticmethod
    def compute_log_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(input_ids, attention_mask=attention_mask).logits
        return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])

    def loss(  # pylint: disable=too-many-locals
        self,
        better_input_ids: torch.LongTensor,
        better_attention_mask: torch.BoolTensor,
        worse_input_ids: torch.LongTensor,
        worse_attention_mask: torch.BoolTensor,
        better_cost: torch.FloatTensor,
        worse_cost: torch.FloatTensor,
        multipliers: torch.FloatTensor,
        better_reward: torch.FloatTensor = None,
        worse_reward: torch.FloatTensor = None,
        debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Loss function for the pdalignment algorithm with preference data.

        Args:
            better_input_ids (torch.LongTensor): The input ids of the better answer.
            better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
            worse_input_ids (torch.LongTensor): The input ids of the worse answer.
            worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.
            better_cost (torch.FloatTensor): The cost of the better answer.
            worse_cost (torch.FloatTensor): The cost of the worse answer.
            multipliers (torch.FloatTensor): The dual multipliers.

        Returns:
            dict[str, torch.Tensor]: loss and other metrics
        """
        # Create response masks for better and worse samples
        better_response_index = (
            (better_attention_mask[:, 1:] == 1).nonzero(as_tuple=True)[1][0].item()
        )
        worse_response_index = (
            (worse_attention_mask[:, 1:] == 1).nonzero(as_tuple=True)[1][0].item()
        )

        better_response_mask = (
            torch.arange(better_input_ids.size(1), device=better_input_ids.device)[None, :]
            >= better_response_index
        )
        worse_response_mask = (
            torch.arange(worse_input_ids.size(1), device=worse_input_ids.device)[None, :]
            >= worse_response_index
        )

        # Compute log probabilities for better and worse sequences
        better_sequence_log_probs = self.compute_log_probs(
            self.model.module,
            better_input_ids,
            better_attention_mask,
        )
        better_sequence_log_probs = (
            better_sequence_log_probs * better_response_mask[:, 1:] * better_attention_mask[:, 1:]
        )
        better_sequence_log_probs = better_sequence_log_probs.sum(dim=1)

        worse_sequence_log_probs = self.compute_log_probs(
            self.model.module,
            worse_input_ids,
            worse_attention_mask,
        )
        worse_sequence_log_probs = (
            worse_sequence_log_probs * worse_response_mask[:, 1:] * worse_attention_mask[:, 1:]
        )
        worse_sequence_log_probs = worse_sequence_log_probs.sum(dim=1)

        # Get reference log probabilities
        with torch.no_grad():
            better_ref_log_probs = self.compute_log_probs(
                self.reference_model,
                better_input_ids,
                better_attention_mask,
            )
            better_ref_log_probs = (
                better_ref_log_probs * better_response_mask[:, 1:] * better_attention_mask[:, 1:]
            )
            better_ref_log_probs = better_ref_log_probs.sum(dim=1)

            worse_ref_log_probs = self.compute_log_probs(
                self.reference_model,
                worse_input_ids,
                worse_attention_mask,
            )
            worse_ref_log_probs = (
                worse_ref_log_probs * worse_response_mask[:, 1:] * worse_attention_mask[:, 1:]
            )
            worse_ref_log_probs = worse_ref_log_probs.sum(dim=1)

            better_log_ratio = better_sequence_log_probs - better_ref_log_probs
            worse_log_ratio = worse_sequence_log_probs - worse_ref_log_probs

            better_importance_weights = torch.exp(better_log_ratio)
            worse_importance_weights = torch.exp(worse_log_ratio)

        if debug:
            breakpoint()

        # Clip importance weights
        better_importance_weights = torch.clamp(better_importance_weights, min=1e-9, max=10)
        worse_importance_weights = torch.clamp(worse_importance_weights, min=1e-9, max=10)

        # DKL loss for both better and worse samples
        better_dkl_loss = (
            self.scale_coeff
            * better_importance_weights
            * better_log_ratio
            * better_sequence_log_probs
        )
        worse_dkl_loss = (
            self.scale_coeff * worse_importance_weights * worse_log_ratio * worse_sequence_log_probs
        )

        # Safety loss for both better and worse samples
        better_safety_loss = (
            better_importance_weights
            * better_sequence_log_probs
            * (better_cost * multipliers[None, :]).sum(dim=-1)
        ).sum()
        worse_safety_loss = (
            worse_importance_weights
            * worse_sequence_log_probs
            * (worse_cost * multipliers[None, :]).sum(dim=-1)
        ).sum()

        if better_reward is not None:
            better_reward_loss = (
                better_importance_weights * better_sequence_log_probs * better_reward
            ).sum()
            worse_reward_loss = (
                worse_importance_weights * worse_sequence_log_probs * worse_reward
            ).sum()
        else:
            better_reward_loss = 0
            worse_reward_loss = 0

        # Total loss
        dkl_loss = better_dkl_loss + worse_dkl_loss
        safety_loss = better_safety_loss + worse_safety_loss
        reward_loss = better_reward_loss + worse_reward_loss

        losses = dkl_loss + safety_loss - reward_loss
        loss = losses.mean()

        # Detach loss components for logging
        dkl_loss_d = dkl_loss.detach()
        safety_loss_d = safety_loss.detach()
        reward_loss_d = reward_loss.detach()
        # Calculate average costs for monitoring
        better_costs_avg = (better_cost * better_importance_weights[:, None]).mean()
        worse_costs_avg = (worse_cost * worse_importance_weights[:, None]).mean()
        # same with reward
        better_reward_avg = (better_reward * better_importance_weights[:, None]).mean()
        worse_reward_avg = (worse_reward * worse_importance_weights[:, None]).mean()

        return {
            'loss': loss,
            'dkl_loss': dkl_loss_d,
            'safety_loss': safety_loss_d,
            'reward_loss': reward_loss_d,
            'better_importance_weights': better_importance_weights.mean(),
            'worse_importance_weights': worse_importance_weights.mean(),
            'better_costs': better_costs_avg.item(),
            'worse_costs': worse_costs_avg.item(),
            'better_reward': better_reward_avg.item(),
            'worse_reward': worse_reward_avg.item(),
            'importance_weights': (
                better_importance_weights.mean() + worse_importance_weights.mean()
            )
            / 2,
        }

    def dual_step(
        self,
        slacks: torch.Tensor,
        multipliers: torch.Tensor,
        costs: torch.Tensor,
    ):
        multipliers = multipliers + self.args.dual_step_size * (
            slacks - 1 / (2 * self.args.resilient_coeff) * multipliers
        )
        # multiply by mask where costs are positive
        multipliers = multipliers * (costs > 0).float()
        multipliers = torch.clamp(multipliers, min=0)
        return multipliers

    def train_step(
        self,
        better_input_ids: torch.LongTensor,  # size = (B, L)
        better_attention_mask: torch.BoolTensor,  # size = (B, L)
        better_cost: torch.FloatTensor,  # size = (B, C)
        worse_input_ids: torch.LongTensor,  # size = (B, L)
        worse_attention_mask: torch.BoolTensor,  # size = (B, L)
        worse_cost: torch.FloatTensor,  # size = (B, C)
    ) -> dict[str, Any]:
        """Perform a single training step with preference data.

        Args:
            better_input_ids: The input ids of the better answer.
            better_attention_mask: The attention mask of the better answer.
            better_cost: The cost of the better answer.
            worse_input_ids: The input ids of the worse answer.
            worse_attention_mask: The attention mask of the worse answer.
            worse_cost: The cost of the worse answer.

        Returns:
            dict[str, Any]: training loss and other metrics
        """
        loss_dict = self.loss(
            better_input_ids=better_input_ids,
            better_attention_mask=better_attention_mask,
            worse_input_ids=worse_input_ids,
            worse_attention_mask=worse_attention_mask,
            better_cost=better_cost[:, :-1],
            worse_cost=worse_cost[:, :-1],
            multipliers=self.multipliers,
            better_reward=better_cost[:, -1],
            worse_reward=worse_cost[:, -1],
            debug=False,
        )

        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        if self.args.debug:
            # check if the model has gradients
            for name, param in self.model.module.named_parameters():
                if param.grad is not None:
                    print(f"{name} has a gradient")
                else:
                    print(f"{name} has no gradient")

            loss_dict = self.loss(
                better_input_ids=better_input_ids,
                better_attention_mask=better_attention_mask,
                worse_input_ids=worse_input_ids,
                worse_attention_mask=worse_attention_mask,
                better_cost=better_cost[:, :-1],
                worse_cost=worse_cost[:, :-1],
                multipliers=self.multipliers,
                better_reward=better_cost[:, -1],
                worse_reward=worse_cost[:, -1],
                debug=False,
            )

        with torch.no_grad():
            dkl_loss = loss_dict['dkl_loss'].mean()
            safety_loss = loss_dict['safety_loss'].mean()
            importance_weights = loss_dict['importance_weights']

            loss = get_all_reduce_mean(loss)
            dkl_loss = get_all_reduce_mean(dkl_loss)
            safety_loss = get_all_reduce_mean(safety_loss)

            importance_weights = get_all_reduce_mean(importance_weights)

        return {
            'train/loss': loss.item(),
            'train/dkl_loss': dkl_loss.item(),
            'train/safety_loss': safety_loss.item(),
            'train/importance_weights': importance_weights.item(),
            'train/better_costs': loss_dict['better_costs'],
            'train/worse_costs': loss_dict['worse_costs'],
        }

    def init_reference_model(self) -> None:
        """Initialize the reference model for KL calculations."""
        reference_model, _ = load_pretrained_models(
            self.args.model_name_or_path,
            model_max_length=self.args.max_length,
            padding_side='left',
            auto_model_type=AutoModelForCausalLM,
            trust_remote_code=self.args.trust_remote_code,
        )
        reference_model.requires_grad_(False)
        reference_model.eval()
        self.reference_model = reference_model.to(self.args.device)
