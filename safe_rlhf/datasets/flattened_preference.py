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
"""Dataset class for flattened preference training, treating each response independently."""

from __future__ import annotations

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch
import transformers
from datasets import load_dataset
# Import the base Dataset class
from torch.utils.data import Dataset

from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import format_prompt, right_padding


__all__ = [
    'FlattenedPreferenceDataset',
    'FlattenedPreferenceCollator',
    'FlattenedPreferenceSample',
    'FlattenedPreferenceBatch',
]


class FlattenedPreferenceSample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    label: torch.FloatTensor  # size = (C,) cost vector
    response_index: int # Index where the response starts
    original_index: int
    is_safe_response: bool


class FlattenedPreferenceBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    labels: torch.FloatTensor  # size = (B, C) cost vector
    response_mask: torch.BoolTensor # size = (B, L)
    index: torch.LongTensor # Current batch indices
    original_index: torch.LongTensor
    is_safe_response: torch.BoolTensor


class FlattenedPreferenceDataset(TokenizedDataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        lazy_tokenization: bool = False,
        seed: int = 42,
        max_length: int = 2048,
    ) -> None:
        # We don't call super().__init__ here yet.
        # First, set up attributes needed by this class and load data.
        self.max_length = max_length
        self.dataset_path = dataset_path
        self.tokenizer = tokenizer
        self.lazy_tokenization = lazy_tokenization
        self.seed = seed
        self.num_responses = 1 # Each sample now has one response

        # Load and flatten data directly into self.rawdata
        self.rawdata = self._load_and_flatten_data()

        # --- Manually perform steps from TokenizedDataset.__init__ ---
        # Shuffle the raw data if a seed is provided
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed)
            indices = torch.randperm(len(self.rawdata), generator=generator).tolist()
            self.rawdata = [self.rawdata[i] for i in indices]
            print(f"Shuffled rawdata with seed {self.seed}")

        # Set up self.data for caching tokenized samples
        if self.lazy_tokenization:
            # Initialize cache with sentinels
            self.data: list[dict[str, torch.Tensor] | object] = [
                self._SENTINEL
            ] * len(self.rawdata)
            print("Initialized for lazy tokenization.")
        else:
            # Preprocess all data immediately
            print("Preprocessing all data upfront...")
            self.data: list[dict[str, torch.Tensor] | object] = [
                self.preprocess(raw_sample) for raw_sample in self.rawdata
            ]
            print("Preprocessing complete.")
        # -------------------------------------------------------------

        # No need to call super().__init__(...) as it tries to load raw datasets.
        # The necessary attributes (tokenizer, data, rawdata) are set manually.
        # If FlattenedPreferenceDataset inherited directly from torch.utils.data.Dataset,
        # we would call super().__init__() here. Since it inherits from TokenizedDataset,
        # which itself inherits from Dataset, we avoid calling the intermediate constructor.


    def _load_and_flatten_data(self) -> list[RawSample]:
        """Loads the original preference dataset and flattens it."""
        raw_data = []
        # Load the original preference dataset
        if ':' in self.dataset_path:
            path, split = self.dataset_path.split(':')
        else:
            path = self.dataset_path
            split = 'train' # Default to train split

        try:
            original_dataset = load_dataset(path, split=split)
        except Exception as e:
            # Add more informative error message
            print(f"Error loading dataset {path} (split: {split}): {e}")
            raise

        # Ensure tokenizer has pad token before formatting prompts, if needed by format_prompt
        # Although format_prompt might not use padding, it's good practice for consistency.
        if self.tokenizer.pad_token is None:
             if self.tokenizer.eos_token is not None:
                 self.tokenizer.pad_token = self.tokenizer.eos_token
                 print("Set tokenizer pad_token to eos_token for loading.")
             else:
                 # Handle cases where even eos_token is None, though rare for LLMs
                 # You might need to add a specific pad token depending on the model
                 print("Warning: tokenizer has no pad_token or eos_token.")


        for i, original_sample in enumerate(original_dataset):
            # Use the instance tokenizer now available
            prompt_formatted = format_prompt(
                 input=original_sample.get('prompt', original_sample.get('input', '')),
                 eos_token=self.tokenizer.eos_token, # Use instance tokenizer
            )

            # Validate expected fields exist
            if 'safe_response' not in original_sample or 'unsafe_response' not in original_sample:
                 print(f"Warning: Skipping sample {i} in {self.dataset_path} due to missing 'safe_response' or 'unsafe_response'.")
                 continue

            # Sample 1: Better response
            better_response = original_sample['safe_response']
            better_cost = original_sample.get('safe_cost', [0.0]) # Default cost if missing
            raw_data.append({
                'prompt': prompt_formatted, # Store formatted prompt
                'response': better_response,
                'label': better_cost,
                 # Store original index for potential debugging
                'original_index': i,
                'is_safe_response': True
            })

            # Sample 2: Worse response
            worse_response = original_sample['unsafe_response']
            worse_cost = original_sample.get('unsafe_cost', [0.0]) # Default cost if missing
            raw_data.append({
                'prompt': prompt_formatted, # Store formatted prompt
                'response': worse_response,
                'label': worse_cost,
                # Store original index
                'original_index': i,
                'is_safe_response': False
            })
        return raw_data


    def preprocess(self, raw_sample: RawSample) -> FlattenedPreferenceSample:
        # Prompt is already formatted during _load_and_flatten_data
        prompt = raw_sample['prompt'] # Retrieve pre-formatted prompt
        response_text = raw_sample['response']
        text = prompt + response_text

        # Tokenize the prompt to find its length for response_index
        # Important: Use the same tokenization settings as for the full text,
        # excluding padding for length calculation.
        prompt_ids = self.tokenize(
            prompt,
            max_length=self.max_length, # Apply max_length to prompt as well
            truncation=True,
            padding=False, # No padding for length calculation
        )
        response_index = len(prompt_ids)

        # Tokenize the combined text
        input_ids = self.tokenize(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False, # No padding during tokenization of individual sample
        )

        # Add EOS token if needed and space permits
        if (
            input_ids[-1] != self.tokenizer.eos_token_id
            and len(input_ids) < self.max_length
        ):
            input_ids = torch.cat(
                [input_ids, torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)]
            )

        # Convert label (cost) to tensor
        label = torch.tensor(raw_sample['label'], dtype=torch.float)
        original_idx = raw_sample['original_index']
        is_safe = raw_sample['is_safe_response']

        return {
            'input_ids': input_ids,  # size = (L,)
            'label': label,  # size = (C,)
            'response_index': response_index,
            'original_index': original_idx,
            'is_safe_response': is_safe,
        }

    def get_collator(self) -> Callable[[list[FlattenedPreferenceSample]], FlattenedPreferenceBatch]:
        return FlattenedPreferenceCollator(self.tokenizer.pad_token_id)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a tokenized data sample by index."""
        # If lazy, data needs preprocessing on the fly
        if self.lazy_tokenization:
            data = self.data[index]
            if data is self._SENTINEL:
                # Access rawdata using the index
                raw_sample = self.rawdata[index]
                data = self.preprocess(raw_sample)
                self.data[index] = data # Cache the result
            # else: data is already fetched/preprocessed from previous call
        else:
             # If not lazy, data is already preprocessed in self.data
             data = self.data[index]

        # Add the index *within the current dataset object* (0 to len(self)-1)
        # This is important for the collator
        final_data = data.copy() # Avoid modifying the cached dict
        final_data['current_index'] = index
        return final_data

    # __len__ remains the same
    def __len__(self) -> int:
        return len(self.rawdata)


class FlattenedPreferenceCollator(CollatorBase):
    def __call__(self, samples: list[FlattenedPreferenceSample]) -> FlattenedPreferenceBatch:
        # Extract data from samples
        input_ids_list = [sample['input_ids'] for sample in samples]
        labels_list = [sample['label'] for sample in samples]
        index_list = [sample['current_index'] for sample in samples]
        response_indices_list = [sample['response_index'] for sample in samples]
        original_indices_list = [sample['original_index'] for sample in samples]
        is_safe_response_list = [sample['is_safe_response'] for sample in samples]

        # Pad input_ids
        input_ids = right_padding(input_ids_list, padding_value=self.pad_token_id)

        # Create attention mask
        attention_mask = input_ids.ne(self.pad_token_id)

        # Stack labels
        labels = torch.stack(labels_list, dim=0)

        # Convert index list to tensor
        index = torch.tensor(index_list, dtype=torch.long)
        original_indices = torch.tensor(original_indices_list, dtype=torch.long)
        is_safe_response = torch.tensor(is_safe_response_list, dtype=torch.bool)

        # Create response_mask from response_indices
        seq_length = input_ids.shape[1]
        response_indices_tensor = torch.tensor(response_indices_list, dtype=torch.long)
        # Ensure response_indices are clamped if prompt was truncated to be > seq_length
        # This shouldn't happen if max_length is applied consistently, but as a safeguard:
        response_indices_tensor = torch.clamp(response_indices_tensor, max=seq_length -1)

        response_mask = torch.arange(seq_length, device=input_ids.device)[None, :] >= response_indices_tensor[:, None]
        response_mask = response_mask & attention_mask # Response mask should only be true where attention_mask is true


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'response_mask': response_mask,
            'index': index,
            'original_index': original_indices,
            'is_safe_response': is_safe_response,
        }

# Helper tokenize method if needed (can reuse from base or define here)
# Note: Using the inherited tokenize method from TokenizedDataset should suffice.
# If specific behavior is needed, uncomment and customize:
#    def tokenize(
#        self,
#        text: str,
#        add_special_tokens: bool = True,
#        padding: bool = False,
#        truncation: bool = True,
#        max_length: int | None = None,
#    ) -> torch.LongTensor:  # size = (L,)
#        """Tokenizes text."""
#        if max_length is None:
#            max_length = self.max_length
#
#        return self.tokenizer(
#            text,
#            add_special_tokens=add_special_tokens,
#            padding=padding,
#            max_length=max_length,
#            truncation=truncation,
#            return_tensors='pt',
#        )['input_ids'][0] 