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

from safe_rlhf.datasets import PointwiseSafeDataset
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.trainers import DualTrainer
from safe_rlhf.utils import gather_log_probabilities, get_all_reduce_mean


class PdAlignementTrainer(DualTrainer):
    TRAINING_TYPE = 'pd_alignment'
    DATASET_TYPE = PointwiseSafeDataset

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
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "down_proj",
                    "up_proj",
                    "lm_head",
                ],
            ),
        )

    def init_engines(self) -> None:
        super().init_engines()
        '''
        self.model, *_ = deepspeed.initialize(
            model=self.model,
            config=self.ds_eval_config,
        )
        '''

    @staticmethod
    def compute_log_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(input_ids, attention_mask=attention_mask).logits
        # breakpoint()
        return gather_log_probabilities(logits[:, :-1], input_ids[:, 1:])

    def loss(  # pylint: disable=too-many-locals
        self,
        better_input_ids: torch.LongTensor,  # size = (B, L)
        better_attention_mask: torch.BoolTensor,  # size = (B, L)
        worse_input_ids: torch.LongTensor,  # size = (B, L)
        worse_attention_mask: torch.BoolTensor,  # size = (B, L)
        better_safe=torch.BoolTensor,
        worse_safe=torch.BoolTensor,
        multipliers=torch.FloatTensor,
        costs=torch.FloatTensor,
        rewards=torch.FloatTensor,
        ref_sequence_log_probs=torch.FloatTensor,
        response_masks=torch.BoolTensor,  # size = (B, L)
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
        assert better_input_ids.size(0) == worse_input_ids.size(0), 'batch size mismatch!'
        batch_size = better_input_ids.size(0)
        sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
            self.model.module,
            input_ids=torch.cat([better_input_ids, worse_input_ids], dim=0),
            attention_mask=torch.cat([better_attention_mask, worse_attention_mask], dim=0),
        )
        (
            better_sequence_log_probs,  # size = (B, L - 1)
            worse_sequence_log_probs,  # size = (B, L - 1)
        ) = sequence_log_probs.chunk(chunks=2, dim=0)

        better_log_prob = (
            better_sequence_log_probs * better_attention_mask[:, 1:] * response_masks[:, 1:]
        ).sum(dim=1)
        worse_log_prob = (
            worse_sequence_log_probs * worse_attention_mask[:, 1:] * response_masks[:, 1:]
        ).sum(dim=1)

        ref_better_log_prob = ref_sequence_log_probs[:, 0]
        ref_worse_log_prob = ref_sequence_log_probs[:, 1]
        better_log_ratio = better_log_prob - ref_better_log_prob
        worse_log_ratio = worse_log_prob - ref_worse_log_prob

        # DKL loss
        dkl_loss = -(
            self.scale_coeff * (better_log_ratio - 1) * torch.exp(better_log_ratio)
            + self.scale_coeff * (worse_log_ratio - 1) * torch.exp(worse_log_ratio)
        )

        # Safety loss
        safety_loss = (
            -self.args.resilient_coeff
            / 2
            * (
                torch.clamp(
                    torch.exp(better_log_ratio) * better_safe - self.args.safety_ratio_tol, 0, None
                )
                ** 2
                + torch.clamp(
                    torch.exp(worse_log_ratio) * worse_safe - self.args.safety_ratio_tol, 0, None
                )
                ** 2
            )
        )

        # Helpfullness loss
        helpfullness_loss = better_log_prob - worse_log_prob

        # Total loss
        losses = dkl_loss + safety_loss + helpfullness_loss
        losses = -losses
        loss = losses.mean()

        # GET PARTS OF THE LOSS
        dkl_loss_d = dkl_loss.detach()
        safety_loss_d = safety_loss.detach()
        helpfullness_loss_d = helpfullness_loss.detach()

        # GET RATIOS FOR BETTER, WORSE, SAFE, UNSAFE
        unsafe_sample_ratio = torch.cat(
            [better_log_ratio[~better_safe], worse_log_ratio[~worse_safe]], dim=0
        ).detach()
        safe_sample_ratio = torch.cat(
            [better_log_ratio[better_safe], worse_log_ratio[worse_safe]], dim=0
        ).detach()
        better_sample_ratio = better_log_ratio.detach()
        worse_sample_ratio = worse_log_ratio.detach()

        # reward = better_sample_reward + worse_sample_reward  # size = (B,)
        # reward_accuracy = (better_sample_ratio > worse_sample_ratio).float().mean()  # size = ()
        # reward_margin = better_sample_reward - worse_sample_reward  # size = (B,)

        return {
            'loss': loss,
            'dkl_loss': dkl_loss_d,
            'safety_loss': safety_loss_d,
            'helpfullness_loss': helpfullness_loss_d,
            'better_sample_ratio': better_sample_ratio,
            'worse_sample_ratio': worse_sample_ratio,
            'unsafe_sample_ratio': unsafe_sample_ratio,
            'safe_sample_ratio': safe_sample_ratio,
        }

    def dual_step(
        self,
        slacks: torch.Tensor,
        multipliers: torch.Tensor,
        costs: torch.Tensor,
    ):
        # breakpoint()
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
        worse_input_ids: torch.LongTensor,  # size = (B, L)
        worse_attention_mask: torch.BoolTensor,  # size = (B, L)
        better_safe: torch.BoolTensor,
        worse_safe: torch.BoolTensor,
        index: torch.LongTensor,
        response_masks: torch.BoolTensor,
    ) -> dict[str, Any]:
        """Perform a single training step.

        Args:
            better_input_ids (torch.LongTensor): The input ids of the better answer.
            better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
            worse_input_ids (torch.LongTensor): The input ids of the worse answer.
            worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.
            better_safe (torch.BoolTensor): The safety of the better answer.
            worse_safe (torch.BoolTensor): The safety of the worse answer.
            index (torch.LongTensor): The index of the batch.
        Returns:
            dict[str, Any]: training loss, reward, learning rate
        """
        batch_multipliers = self.multipliers[index]
        batch_costs = self.costs[index]
        batch_rewards = self.rewards[index]
        batch_ref_sequence_log_probs = self.baseline_logprobs[index]

        loss_dict = self.loss(
            better_input_ids=better_input_ids,
            better_attention_mask=better_attention_mask,
            worse_input_ids=worse_input_ids,
            worse_attention_mask=worse_attention_mask,
            response_masks=response_masks,
            better_safe=better_safe,
            worse_safe=worse_safe,
            multipliers=batch_multipliers,
            costs=batch_costs,
            rewards=batch_rewards,
            ref_sequence_log_probs=batch_ref_sequence_log_probs,
        )
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        with torch.no_grad():

            dkl_loss = loss_dict['dkl_loss'].mean()
            safety_loss = loss_dict['safety_loss'].mean()
            helpfullness_loss = loss_dict['helpfullness_loss'].mean()
            better_sample_ratio = loss_dict['better_sample_ratio'].mean()
            worse_sample_ratio = loss_dict['worse_sample_ratio'].mean()
            unsafe_sample_ratio = loss_dict['unsafe_sample_ratio'].mean()
            safe_sample_ratio = loss_dict['safe_sample_ratio'].mean()

            loss = get_all_reduce_mean(loss)
            dkl_loss = get_all_reduce_mean(dkl_loss)
            safety_loss = get_all_reduce_mean(safety_loss)
            helpfullness_loss = get_all_reduce_mean(helpfullness_loss)
            better_sample_ratio = get_all_reduce_mean(better_sample_ratio)
            worse_sample_ratio = get_all_reduce_mean(worse_sample_ratio)
            unsafe_sample_ratio = get_all_reduce_mean(unsafe_sample_ratio)
            safe_sample_ratio = get_all_reduce_mean(safe_sample_ratio)

        return {
            'train/loss': loss.item(),
            'train/dkl_loss': dkl_loss.item(),
            'train/safety_loss': safety_loss.item(),
            'train/helpfullness_loss': helpfullness_loss.item(),
            'train/better_sample_ratio': better_sample_ratio.item(),
            'train/worse_sample_ratio': worse_sample_ratio.item(),
            'train/unsafe_sample_ratio': unsafe_sample_ratio.item(),
            'train/safe_sample_ratio': safe_sample_ratio.item(),
        }
