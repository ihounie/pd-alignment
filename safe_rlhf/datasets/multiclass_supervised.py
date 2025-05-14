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
    NAME: ClassVar[str] = 'multiclass_supervised'
    ALIASES: ClassVar[set[str]] = {'multiclass'}

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


class MultiClassSupervisedSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.FloatTensor  # size = (C,) where C is number of classes


class MultiClassSupervisedBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    labels: torch.FloatTensor  # size = (B, C)


class MultiClassSupervisedDataset(TokenizedDataset):
    NAME: ClassVar[str] = 'multiclass_supervised'  # Add dataset name

    def __init__(
        self,
        dataset_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        lazy_tokenization: bool = False,
        seed: int = 42,
        max_length: int = 2048,  # Add max_length parameter with a reasonable default
    ) -> None:
        self.max_length = max_length  # Store max_length for use in tokenization

        dataset_config = {
            'multiclass_supervised': {
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
        self.num_responses = 1

    def preprocess(self, raw_sample: RawSample) -> MultiClassSupervisedSample:
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

        return {'input_ids': input_ids, 'labels': labels}  # size = (L,)  # size = (C,)

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

    def get_collator(
        self,
    ) -> Callable[[list[MultiClassSupervisedSample]], MultiClassSupervisedBatch]:
        return MultiClassSupervisedCollator(self.tokenizer.pad_token_id)


class MultiClassSupervisedCollator(CollatorBase):
    def __call__(self, samples: list[MultiClassSupervisedSample]) -> MultiClassSupervisedBatch:
        input_ids = right_padding(
            [sample['input_ids'] for sample in samples],
            padding_value=self.pad_token_id,
        )
        attention_mask = input_ids.ne(self.pad_token_id)
        labels = torch.stack([sample['labels'] for sample in samples])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
