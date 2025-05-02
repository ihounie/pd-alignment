# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
# Based on existing dataset implementations
from __future__ import annotations

import json
from typing import Callable, ClassVar
from typing_extensions import TypedDict

import torch
import transformers

import datasets
from safe_rlhf.datasets.base import CollatorBase, RawDataset, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import format_prompt, right_padding


class MultiClassRawDataset(RawDataset):
    NAME: ClassVar[str] = 'multiclass_safety'
    ALIASES: ClassVar[set[str]] = {'multiclass_safe'}

    def __init__(self, path: str) -> None:
        """Initialize the dataset.

        Args:
            path: Hugging Face dataset path, format: "repo_id/dataset_name" or "repo_id/dataset_name:split"
                 Example: "organization/dataset" or "organization/dataset:train"
        """
        super().__init__()

        # Parse dataset path and split
        if ':' in path:
            dataset_path, split = path.split(':')
        else:
            dataset_path = path
            split = 'train'  # default to train split

        # Load the dataset from Hugging Face
        self.dataset = datasets.load_dataset(
            dataset_path,
            split=split,
        )

    def __getitem__(self, index: int) -> RawSample:
        item = self.dataset[index]
        return {'prompt': item['prompt'], 'response': item['response'], 'label': item['label']}

    def __len__(self) -> int:
        return len(self.dataset)


class MultiClassSafetySample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.FloatTensor  # size = (C,) where C is number of classes


class MultiClassSafetyBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    labels: torch.FloatTensor  # size = (B, C)


class MultiClassSafetyDataset(TokenizedDataset):
    NAME: ClassVar[str] = 'multiclass_safety'  # Add dataset name

    def __init__(
        self,
        dataset_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        lazy_tokenization: bool = False,
        seed: int = 42,
        max_length: int = 2048,
    ) -> None:
        self.max_length = max_length

        dataset_config = {
            'multiclass_safety': {
                'path': dataset_path,
                'proportion': 1.0,
            }
        }

        super().__init__(
            dataset_names_and_attributes=dataset_config,
            tokenizer=tokenizer,
            lazy_tokenization=lazy_tokenization,
            seed=seed,
        )
        self.num_respones = 1

    def preprocess(self, raw_sample: RawSample) -> MultiClassSafetySample:
        # Combine prompt and response
        prompt = format_prompt(input=raw_sample['prompt'], eos_token=self.tokenizer.eos_token)
        text = prompt + raw_sample['response']

        # Convert label to tensor
        labels = torch.FloatTensor(raw_sample['label'])

        # Tokenize with explicit max_length and truncation
        input_ids = self.tokenize(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        # Conditionally add EOS token if needed
        if input_ids[-1] != self.tokenizer.eos_token_id and len(input_ids) < self.max_length:
            input_ids = torch.cat(
                [input_ids, torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)]
            )

        prompt_ids = self.tokenize(
            prompt, max_length=self.max_length, truncation=True, padding=False
        )

        response_index = len(prompt_ids)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'response_index': response_index,
        }

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = True,
        max_length: int | None = None,
    ) -> torch.LongTensor:  # size = (L,)
        """Override tokenize method to handle long sequences properly."""
        if max_length is None:
            max_length = self.max_length

        return self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors='pt',
        )['input_ids'][0]

    def get_collator(self) -> Callable[[list[MultiClassSafetySample]], MultiClassSafetyBatch]:
        return MultiClassSafetyCollator(self.tokenizer.pad_token_id)

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


class MultiClassSafetyCollator(CollatorBase):
    def __call__(self, samples: list[MultiClassSafetySample]) -> MultiClassSafetyBatch:
        input_ids = right_padding(
            [sample['input_ids'] for sample in samples],
            padding_value=self.pad_token_id,
        )
        response_index = torch.tensor([s['response_index'] for s in samples], dtype=torch.long)

        attention_mask = input_ids.ne(self.pad_token_id)
        labels = torch.stack([sample['labels'] for sample in samples])
        index = torch.tensor([sample['index'] for sample in samples], dtype=torch.long)
        _, seq_length = input_ids.shape
        response_masks = (
            torch.arange(seq_length, device=input_ids.device)[None, :] >= response_index[:, None]
        )

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'index': index,
            'response_mask': response_masks,
        }
