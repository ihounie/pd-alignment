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
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from safe_rlhf.datasets import MultiClassSafetyDataset
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.trainers import MultiDualTrainer
from safe_rlhf.utils import gather_log_probabilities, get_all_reduce_mean


class MultiPdAlignementTrainer(MultiDualTrainer):
    TRAINING_TYPE = 'multi_pd_alignment'
    DATASET_TYPE = MultiClassSafetyDataset

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
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        multipliers: torch.FloatTensor,
        costs: torch.FloatTensor,
        rewards: torch.FloatTensor,
        ref_sequence_log_probs: torch.FloatTensor,
        response_mask: torch.BoolTensor,  # size = (B, L)
        debug: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Loss function for the pdalignment algorithm.

        Args:
            better_input_ids (torch.LongTensor): The input ids of the better answer.
            better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
            worse_input_ids (torch.LongTensor): The input ids of the worse answer.
            worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.

        Returns:
            dict[str, torch.Tensor]: loss, reward, better sample reward, worse sample reward
        """
        # breakpoint()
        sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
            self.model.module,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_log_probs = sequence_log_probs * response_mask[:, 1:] * attention_mask[:, 1:]
        sequence_log_probs = sequence_log_probs.sum(dim=1)
        log_ratio = sequence_log_probs - ref_sequence_log_probs
        with torch.no_grad():
            importance_weights = torch.exp(log_ratio)
            # clip importance weights
            importance_weights = torch.clamp(importance_weights, min=None, max=10)
        if debug:
            breakpoint()
        # DKL loss
        dkl_loss = self.scale_coeff * importance_weights * 0.5 * (1 + log_ratio) ** 2
        # Safety loss
        safety_loss = (
            importance_weights
            * sequence_log_probs
            * (-rewards + (costs - self.args.safety_threshold) * multipliers[None, :]).sum(dim=-1)
        ).sum()

        # Total loss
        # 
        losses = safety_loss + dkl_loss
        # losses = -losses
        loss = losses.mean()

        # GET PARTS OF THE LOSS
        dkl_loss_d = dkl_loss.detach()
        safety_loss_d = safety_loss.detach()
        # breakpoint()
        infeasible = ((costs - self.args.safety_threshold) > 0).any(dim=-1).float()
        feasible = 1 - infeasible

        return {
            'loss': loss,
            'dkl_loss': dkl_loss_d,
            'safety_loss': safety_loss_d,
            'importance_weights': importance_weights,
            'costs': (costs * importance_weights[:, None]).mean().detach(),
            'rewards': (rewards * importance_weights[:, None]).mean().detach(),
            'importance_infeasible': (importance_weights* infeasible).mean().detach(),
            'importance_feasible': (importance_weights*feasible).mean().detach(),
            'feasible': feasible.mean().detach(),
        }

    def dual_step(
        self,
        slacks: torch.Tensor,
        multipliers: torch.Tensor,
        costs: torch.Tensor,
    ):
        # #breakpoint()
        multipliers = multipliers + self.args.dual_step_size * (
            slacks - 1 / (2 * self.args.resilient_coeff) * multipliers
        )
        # multiply by mask where costs are positive
        multipliers = multipliers * (costs > 0).float()
        multipliers = torch.clamp(multipliers, min=0)
        return multipliers

    def train_step(
        self,
        input_ids: torch.LongTensor,  # size = (B, L)
        attention_mask: torch.BoolTensor,  # size = (B, L)
        index: torch.LongTensor,  # size = (B,)
        response_mask: torch.BoolTensor,  # size = (B, L)
        labels: torch.LongTensor,  # size = (B,)
    ) -> dict[str, Any]:
        """Perform a single training step.

        Args:
            input_ids (torch.LongTensor): The input ids of the answer.
            attention_mask (torch.BoolTensor): The attention mask of the answer.
            index (torch.LongTensor): The index of the batch.
        Returns:
            dict[str, Any]: training loss, reward, learning rate
        """
        batch_costs = self.costs[index]
        if self.rewards is not None:
            batch_rewards = self.rewards[index]
        else:
            batch_rewards = 0.0
        batch_ref_sequence_log_probs = self.baseline_logprobs[index]
        loss_dict = self.loss(
            input_ids=input_ids,
            attention_mask=attention_mask,
            multipliers=self.multipliers,
            costs=batch_costs,
            rewards=batch_rewards,
            ref_sequence_log_probs=batch_ref_sequence_log_probs,
            response_mask=response_mask,
            debug=False,
        )
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()
        if self.args.debug:
            # check if the model has not none gradients
            for name, param in self.model.module.named_parameters():
                if param.grad is not None:
                    print(f"{name} has a gradient")
                else:
                    print(f"{name} has no gradient")
            ##breakpoint()
            loss_dict = self.loss(
                input_ids=input_ids,
                attention_mask=attention_mask,
                response_mask=response_mask,
                multipliers=self.multipliers,
                costs=batch_costs,
                rewards=batch_rewards,
                ref_sequence_log_probs=batch_ref_sequence_log_probs,
                debug=True,
            )

        with torch.no_grad():

            dkl_loss = loss_dict['dkl_loss'].mean()
            safety_loss = loss_dict['safety_loss'].mean()

            importance_weights = loss_dict['importance_weights'].mean()
            costs = loss_dict['costs'].mean()
            rewards = loss_dict['rewards'].mean()
            importance_infeasible = loss_dict['importance_infeasible'].mean()
            importance_feasible = loss_dict['importance_feasible'].mean()
            feasible = loss_dict['feasible'].mean()

            costs = get_all_reduce_mean(costs)
            rewards = get_all_reduce_mean(rewards)
            loss = get_all_reduce_mean(loss)
            dkl_loss = get_all_reduce_mean(dkl_loss)
            safety_loss = get_all_reduce_mean(safety_loss)
            importance_weights = get_all_reduce_mean(importance_weights)
            importance_infeasible = get_all_reduce_mean(importance_infeasible)
            importance_feasible = get_all_reduce_mean(importance_feasible)
            feasible = get_all_reduce_mean(feasible)
        return {
            'train/loss': loss.item(),
            'train/dkl_loss': dkl_loss.item(),
            'train/safety_loss': safety_loss.item(),
            'train/importance_weights': importance_weights.item(),
            'train/costs': costs.item(),
            'train/rewards': rewards.item(),
            'train/importance_infeasible': importance_infeasible.item(),
            'train/importance_feasible': importance_feasible.item(),
            'multipliers/0': self.multipliers[0].item(),
            'multipliers/1': self.multipliers[1].item(),
            'multipliers/2': self.multipliers[2].item(),
            'train/feasible': feasible.item(),
        }
