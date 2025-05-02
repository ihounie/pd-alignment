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
from safe_rlhf.utils import gather_log_probabilities, get_all_reduce_mean, to_device


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
                lora_dropout=0.05,
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

    @staticmethod
    def compute_log_probs(
        model: AutoModelForCausalLM,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(input_ids, attention_mask=attention_mask).logits
        # #breakpoint()
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
        # clamp ratios to avoid nans
        clamp = 300
        better_log_ratio = torch.clamp(better_log_ratio, min=-clamp, max=clamp)
        worse_log_ratio = torch.clamp(worse_log_ratio, min=-clamp, max=clamp)
        if any(torch.isnan(better_log_ratio)) or any(torch.isnan(worse_log_ratio)):
            # breakpoint()
            print("NAN ratio detected")
            print(f"Better log prob: {better_log_prob}")
            print(f"Worse log prob: {worse_log_prob}")
            print(f"Ref better log prob: {ref_better_log_prob}")
            print(f"Ref worse log prob: {ref_worse_log_prob}")
            better_log_ratio = torch.nan_to_num(better_log_ratio, nan=-1e4)
            worse_log_ratio = torch.nan_to_num(worse_log_ratio, nan=-1e4)

        if False:  # self.args.use_supervised:
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
                        torch.exp(better_log_ratio) * better_safe - self.args.safety_ratio_tol,
                        0,
                        None,
                    )
                    ** 2
                    + torch.clamp(
                        torch.exp(worse_log_ratio) * worse_safe - self.args.safety_ratio_tol,
                        0,
                        None,
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
        else:  # DPO loss
            if False:
                # assign with probability proportional to costs
                label_better = torch.sigmoid(costs[:, 1] - costs[:, 0])
                # sample bernouli
                label_better = -1 + 2 * torch.bernoulli(label_better)
            else:
                label_better = (costs[:, 0] < costs[:, 1]).long() * 2 - 1

            prob = torch.sigmoid(costs[:, 0] - costs[:, 1])

            # loss = torch.log(torch.sigmoid(self.scale_coeff * (prob*label_better*better_log_ratio-(1-prob)*label_better*worse_log_ratio))).mean()
            loss = -(
                (costs[:, 0] - self.scale_coeff) * better_log_ratio
                + (costs[:, 1] - self.scale_coeff) * worse_log_ratio
            ).mean()
            # make sure the loss is not nan
            if torch.isnan(loss).any():
                # breakpoint()
                print(f"Loss is nan at batch")
            loss = torch.nan_to_num(loss, nan=0.0)
        with torch.no_grad():
            # GET RATIOS FOR BETTER, WORSE, SAFE, UNSAFE
            unsafe_sample_ratio = (better_log_ratio if ~label_better else worse_log_ratio).detach()
            safe_sample_ratio = (better_log_ratio if label_better else worse_log_ratio).detach()
            better_sample_ratio = better_log_ratio.detach()
            worse_sample_ratio = worse_log_ratio.detach()
            reward_accuracy = (
                (better_sample_ratio > worse_sample_ratio).float().detach()
            )  # size = ()
            # reward_margin = better_sample_reward - worse_sample_reward  # size = (B,)
            safety_accuracy = (
                ((label_better * better_log_ratio - label_better * worse_log_ratio) > 0)
                .float()
                .detach()
            )
            safety_margin = (
                label_better * better_log_ratio - label_better * worse_log_ratio
            ).detach()
            reward_margin = (better_sample_ratio - worse_sample_ratio).detach()

        return {
            'loss': loss,
            'reward_accuracy': reward_accuracy,
            'safety_accuracy': safety_accuracy,
            'better_sample_ratio': better_sample_ratio,
            'worse_sample_ratio': worse_sample_ratio,
            'unsafe_sample_ratio': unsafe_sample_ratio,
            'safe_sample_ratio': safe_sample_ratio,
            'reward_margin': reward_margin,
            'safety_margin': safety_margin,
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

    def eval(self, num_batches: int = 30) -> dict[str, Any]:
        """Evaluate the model."""
        # do a forward pass of 10 batches of the train_dataloader
        # and check if the logprobs match the reference model
        self.model.eval()
        counter = 0
        for batch in self.train_dataloader:
            batch = to_device(batch, self.args.device)
            ref_log_prob = self.baseline_logprobs[batch['index']]
            better_log_prob = (
                self.compute_log_probs(
                    self.model.module,
                    batch['better_input_ids'],
                    batch['better_attention_mask'],
                )
                * batch['better_attention_mask'][:, 1:]
                * batch['response_masks'][:, 1:]
            ).sum(dim=1)
            worse_log_prob = (
                self.compute_log_probs(
                    self.model.module,
                    batch['worse_input_ids'],
                    batch['worse_attention_mask'],
                )
                * batch['worse_attention_mask'][:, 1:]
                * batch['response_masks'][:, 1:]
            ).sum(dim=1)
            counter += 1
            try:
                assert torch.allclose(better_log_prob, ref_log_prob[:, 0])
                assert torch.allclose(worse_log_prob, ref_log_prob[:, 1])
            except:
                print(f"Better log prob: {better_log_prob}")
                print(f"Worse log prob: {worse_log_prob}")
                print(f"Ref better log prob: {ref_log_prob[:,0]}")
                print(f"Ref worse log prob: {ref_log_prob[:,1]}")
            if counter > 10:
                break
        # log multiplier stats
        multipliers = self.multipliers.detach().cpu().numpy()
        # zeros, max, min, mean, median, std
        multiplier_stats = {
            'zeros': (multipliers == 0).mean(),
            'max': multipliers.max(),
            'mean': multipliers.mean(),
            'std': multipliers.std(),
        }
        return multiplier_stats

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
        ##breakpoint()

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
            reward_accuracy = loss_dict['reward_accuracy'].mean()
            safety_accuracy = loss_dict['safety_accuracy'].mean()
            better_sample_ratio = loss_dict['better_sample_ratio'].mean()
            worse_sample_ratio = loss_dict['worse_sample_ratio'].mean()
            unsafe_sample_ratio = loss_dict['unsafe_sample_ratio'].mean()
            safe_sample_ratio = loss_dict['safe_sample_ratio'].mean()
            reward_margin = loss_dict['reward_margin'].mean()
            safety_margin = loss_dict['safety_margin'].mean()

            loss = get_all_reduce_mean(loss)
            reward_accuracy = get_all_reduce_mean(reward_accuracy)
            safety_accuracy = get_all_reduce_mean(safety_accuracy)
            better_sample_ratio = get_all_reduce_mean(better_sample_ratio)
            worse_sample_ratio = get_all_reduce_mean(worse_sample_ratio)
            unsafe_sample_ratio = get_all_reduce_mean(unsafe_sample_ratio)
            safe_sample_ratio = get_all_reduce_mean(safe_sample_ratio)
            reward_margin = get_all_reduce_mean(reward_margin)
            safety_margin = get_all_reduce_mean(safety_margin)

        return {
            'train/loss': loss.item(),
            'train/reward_accuracy': reward_accuracy.item(),
            'train/safety_accuracy': safety_accuracy.item(),
            'train/better_sample_ratio': better_sample_ratio.item(),
            'train/worse_sample_ratio': worse_sample_ratio.item(),
            'train/unsafe_sample_ratio': unsafe_sample_ratio.item(),
            'train/safe_sample_ratio': safe_sample_ratio.item(),
            'train/reward_margin': reward_margin.item(),
            'train/safety_margin': safety_margin.item(),
        }
