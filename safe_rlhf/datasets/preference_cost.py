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
"""Dataset class for preference training."""

from __future__ import annotations

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch

from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import format_prompt, right_padding


__all__ = [
    'PreferenceCostDataset',
    'PreferenceCostCollator',
    'PreferenceCostSample',
    'PreferenceCostBatch',
]


class PreferenceCostSample(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (L,)
    worse_input_ids: torch.LongTensor  # size = (L,)
    better_cost: torch.FloatTensor  # size = (C,)
    worse_cost: torch.FloatTensor  # size = (C,)


class PreferenceCostBatch(TypedDict, total=True):
    better_input_ids: torch.LongTensor  # size = (B, L)
    better_attention_mask: torch.BoolTensor  # size = (B, L)
    better_cost: torch.FloatTensor  # size = (B, C)

    worse_input_ids: torch.LongTensor  # size = (B, L)
    worse_attention_mask: torch.BoolTensor  # size = (B, L)
    worse_cost: torch.FloatTensor  # size = (B, C)


class PreferenceCostDataset(TokenizedDataset):
    def preprocess(self, raw_sample: RawSample) -> PreferenceCostSample:
        prompt = format_prompt(
            input=raw_sample.get('prompt', raw_sample.get('input', '')),
            eos_token=self.tokenizer.eos_token,
        )

        # New dataset format with safe and unsafe responses
        better_answer = raw_sample['safe_response']
        worse_answer = raw_sample['unsafe_response']
        better_cost = torch.tensor(raw_sample.get('safe_cost', [0.0]), dtype=torch.float)
        worse_cost = torch.tensor(raw_sample.get('unsafe_cost', [0.0]), dtype=torch.float)

        better_input_ids = self.tokenize(prompt + better_answer)
        if (
            better_input_ids[-1] != self.tokenizer.eos_token_id
            and len(better_input_ids) < self.tokenizer.model_max_length
        ):
            better_input_ids = torch.cat(
                [better_input_ids, torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)]
            )

        worse_input_ids = self.tokenize(prompt + worse_answer)
        if (
            worse_input_ids[-1] != self.tokenizer.eos_token_id
            and len(worse_input_ids) < self.tokenizer.model_max_length
        ):
            worse_input_ids = torch.cat(
                [worse_input_ids, torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)]
            )

        if (
            better_input_ids.size() == worse_input_ids.size()
            and torch.all(torch.eq(better_input_ids, worse_input_ids)).item()
        ):
            raise ValueError(
                'Two responses get the same `input_ids` after tokenization.\n\n'
                f'Prompt: {prompt}\n\n'
                f'Better answer: {better_answer}\n\n'
                f'Worse answer: {worse_answer}',
            )
        return {
            'better_input_ids': better_input_ids,  # size = (L,)
            'worse_input_ids': worse_input_ids,  # size = (L,)
            'better_cost': better_cost,  # size = (C,)
            'worse_cost': worse_cost,  # size = (C,)
        }

    def get_collator(self) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
        return PreferenceCostCollator(self.tokenizer.pad_token_id)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        data = self.data[index]
        if data is self._SENTINEL:
            raw_sample = self.rawdata[index]
            data = self.preprocess(raw_sample)
            self.data[index] = data
        # Add the index to the data dictionary
        data['index'] = index
        return data


class PreferenceCostCollator(CollatorBase):
    def __call__(self, samples: list[PreferenceCostSample]) -> PreferenceCostBatch:
        index_list = [s['index'] for s in samples]
        input_ids = [sample['better_input_ids'] for sample in samples] + [
            sample['worse_input_ids'] for sample in samples
        ]  # size = (2 * B, L)
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]  # size = (2 * B, L)

        costs = [sample['better_cost'] for sample in samples] + [
            sample['worse_cost'] for sample in samples
        ]  # size = (2 * B, C)

        input_ids = right_padding(input_ids, padding_value=self.pad_token_id)  # size = (2 * B, L)
        attention_mask = right_padding(attention_mask, padding_value=0)  # size = (2 * B, L)

        # Stack cost tensors
        costs = torch.stack(costs, dim=0)  # size = (2 * B, C)

        (
            better_input_ids,  # size = (B, L)
            worse_input_ids,  # size = (B, L)
        ) = input_ids.chunk(chunks=2, dim=0)
        (
            better_attention_mask,  # size = (B, L)
            worse_attention_mask,  # size = (B, L)
        ) = attention_mask.chunk(chunks=2, dim=0)
        (
            better_cost,  # size = (B, C)
            worse_cost,  # size = (B, C)
        ) = costs.chunk(chunks=2, dim=0)

        index = torch.tensor(index_list, dtype=torch.long)

        return {
            'better_input_ids': better_input_ids,  # size = (B, L)
            'better_attention_mask': better_attention_mask,  # size = (B, L)
            'better_cost': better_cost,  # size = (B, C)
            'worse_input_ids': worse_input_ids,  # size = (B, L)
            'worse_attention_mask': worse_attention_mask,  # size = (B, L)
            'worse_cost': worse_cost,  # size = (B, C)
            'index': index,  # size = (B,)
        }
