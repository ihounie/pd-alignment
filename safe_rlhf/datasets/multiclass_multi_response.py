# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
# Based on existing dataset implementations
from __future__ import annotations

import json
from typing import Callable, ClassVar
from typing_extensions import TypedDict

import torch
import transformers
from tqdm import tqdm

import datasets
from safe_rlhf.datasets.base import CollatorBase, RawDataset, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import format_prompt, right_padding
from safe_rlhf.utils import gather_log_probabilities, get_all_reduce_mean


# Type hint for the structure loaded from JSONL
class MultiPromptRawSample(TypedDict, total=False):
    prompt: str
    generated_texts: list[str]
    label: list[float] | list[list[float]]  # Label can be per-prompt or per-response


class MultiClassRawDataset(RawDataset):
    NAME: ClassVar[str] = 'multiclass_multi_response_safety'
    ALIASES: ClassVar[set[str]] = {'multiclass_multi_response_safe'}

    num_responses: int
    dataset: list[MultiPromptRawSample]  # Store list of prompts with their responses

    def __init__(self, path: str) -> None:
        """Initialize the dataset from a JSONL file.

        Each line in the JSONL file should be a JSON object with at least
        'prompt' (str) and 'generated_texts' (list[str]) keys.
        An optional 'label' key can be included, either as a single list[float]
        (applied to all responses) or a list[list[float]] (one per response).

        Args:
            path: Path to the JSONL file.
        """
        super().__init__()
        self.path = path
        self._load_data()

    def _load_data(self) -> None:
        """Loads data from the JSONL file without flattening."""
        self.dataset = []
        num_responses = None
        print(f"Loading prompts from {self.path}...")
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading JSONL"):
                if not line.strip():
                    continue
                try:
                    item: MultiPromptRawSample = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {line.strip()} - Error: {e}")
                    continue

                prompt = item.get('prompt')
                responses = item.get('generated_texts')
                label = item.get('label')  # Optional

                if prompt is None or responses is None:
                    print(f"Skipping item due to missing 'prompt' or 'generated_texts': {item}")
                    continue
                if not isinstance(responses, list) or not responses:
                    print(
                        f"Skipping item due to invalid 'generated_texts' (must be non-empty list): {item}"
                    )
                    continue

                current_num_responses = len(responses)
                if num_responses is None:
                    num_responses = current_num_responses
                elif num_responses != current_num_responses:
                    # Option 1: Raise error (current behavior)
                    raise ValueError(
                        f"Inconsistent number of responses found for prompt: '{prompt[:50]}...'. "
                        f"Expected {num_responses}, got {current_num_responses}. "
                        f"Ensure all prompts have the same number of generated texts."
                    )
                    # Option 2: Skip item (alternative)
                    # print(f"Skipping item due to inconsistent number of responses ({current_num_responses} vs expected {num_responses}): {item}")
                    # continue

                # Validate label structure if present
                if label is not None:
                    if isinstance(label[0], list):
                        # Per-response labels: list[list[float]]
                        if len(label) != num_responses:
                            raise ValueError(
                                f"Number of labels ({len(label)}) does not match number of responses ({num_responses}) "
                                f"for prompt: '{prompt[:50]}...'"
                            )
                    # else: Assume per-prompt label: list[float]

                self.dataset.append(
                    {
                        'prompt': prompt,
                        'generated_texts': responses,
                        'label': label,  # Keep label as loaded (can be None)
                    }
                )

        if not self.dataset:
            raise ValueError(f"No valid data loaded from {self.path}")
        if num_responses is None:
            raise ValueError(
                f"Could not determine the number of responses from valid data in {self.path}"
            )

        self.num_responses = num_responses
        print(f"Loaded {len(self.dataset)} prompts, each with {self.num_responses} responses.")

    def __getitem__(self, index: int) -> MultiPromptRawSample:
        return self.dataset[index]

    def __len__(self) -> int:
        # Length is the number of prompts
        return len(self.dataset)


class MultiClassSafetySample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    labels: torch.FloatTensor  # size = (C,) where C is number of classes
    response_index: int  # Added


# This type is returned by Dataset.__getitem__ after adding the index
class MultiClassSafetySampleWithIndex(MultiClassSafetySample, total=True):
    index: int


# Represents the tokenized data for a single prompt and ALL its responses
class MultiResponseTokenizedSample(TypedDict, total=True):
    prompt_ids: torch.LongTensor  # size = (prompt_L,)
    responses_ids: list[torch.LongTensor]  # List of num_responses tensors, each size (response_L,)
    labels: torch.FloatTensor  # size = (num_responses, C)


# Type returned by Dataset.__getitem__ after adding the prompt index
class MultiResponseTokenizedSampleWithIndex(MultiResponseTokenizedSample, total=True):
    index: int  # Index of the prompt


# Batch structure: Responses are grouped by prompt
class MultiClassSafetyBatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, num_responses, L)
    attention_mask: torch.BoolTensor  # size = (B, num_responses, L)
    labels: torch.FloatTensor  # size = (B, num_responses, C)
    index: torch.LongTensor  # size = (B,) - Indices of the prompts in the batch
    response_mask: torch.BoolTensor  # size = (B, num_responses, L)


class MultiClassSafetyDataset(TokenizedDataset):
    NAME: ClassVar[str] = 'multiclass_safety'

    rawdata: MultiClassRawDataset
    num_responses: int
    tokenizer: transformers.PreTrainedTokenizer
    max_length: int
    _lazy: bool
    _data: list[MultiResponseTokenizedSample | None | object]
    num_classes: int  # Added num_classes attribute

    def __init__(
        self,
        dataset_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        num_classes: int,  # Added num_classes argument
        lazy_tokenization: bool = False,
        max_length: int = 2048,
    ) -> None:
        self.max_length = max_length
        self.tokenizer = tokenizer
        self._lazy = lazy_tokenization
        self.num_classes = num_classes  # Store num_classes

        self.rawdata = MultiClassRawDataset(path=dataset_path)
        self.num_responses = self.rawdata.num_responses

        # Initialize data storage based on number of prompts
        self._data: list[MultiResponseTokenizedSample | None | object]
        if not self._lazy:
            print(f"Tokenizing {len(self.rawdata)} prompts eagerly...")
            # Pass self.num_classes to preprocess implicitly via self
            self._data = [
                self.preprocess(raw_sample)
                for raw_sample in tqdm(self.rawdata, desc="Tokenizing Prompts")
            ]
        else:
            print(f"Using lazy tokenization for {len(self.rawdata)} prompts.")
            self._data = [self._SENTINEL] * len(self.rawdata)

    def preprocess(self, raw_sample: MultiPromptRawSample) -> MultiResponseTokenizedSample:
        prompt_text = format_prompt(input=raw_sample['prompt'], eos_token=self.tokenizer.eos_token)
        # Tokenize prompt (without special tokens initially)
        prompt_ids = self.tokenize(prompt_text, add_special_tokens=False)

        responses_text = raw_sample['generated_texts']
        # Tokenize each response (checking and adding EOS if needed)
        responses_ids = []
        for resp in responses_text:
            resp_ids = self.tokenize(resp, add_special_tokens=False)
            # Conditionally add EOS token if needed
            if (len(resp_ids) == 0 or resp_ids[-1] != self.tokenizer.eos_token_id) and len(
                resp_ids
            ) < self.max_length:
                resp_ids_with_eos = torch.cat(
                    [resp_ids, torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)]
                )
                responses_ids.append(resp_ids_with_eos)
            else:
                responses_ids.append(resp_ids)

        # Process labels
        raw_label = raw_sample.get('label')
        if raw_label is None:
            # If label is missing, create a default tensor of zeros
            print(
                f"Warning: Missing label for prompt: '{raw_sample['prompt'][:50]}...'. Creating default zero label."
            )
            # Shape: (num_responses, num_classes)
            labels_tensor = torch.zeros((self.num_responses, self.num_classes), dtype=torch.float32)
        else:
            # Process label if present
            try:
                if isinstance(raw_label[0], list):
                    # Per-response labels: list[list[float]], shape (num_responses, C)
                    if len(raw_label) != self.num_responses:
                        raise ValueError(
                            f"Number of labels ({len(raw_label)}) does not match num_responses ({self.num_responses})"
                        )
                    labels_tensor = torch.FloatTensor(raw_label)
                    if labels_tensor.shape[1] != self.num_classes:
                        raise ValueError(
                            f"Label dimension ({labels_tensor.shape[1]}) does not match num_classes ({self.num_classes})"
                        )
                else:
                    # Per-prompt label: list[float], shape (C,)
                    label_per_prompt = torch.FloatTensor(raw_label)
                    if len(label_per_prompt) != self.num_classes:
                        raise ValueError(
                            f"Label dimension ({len(label_per_prompt)}) does not match num_classes ({self.num_classes})"
                        )
                    # Duplicate for each response -> shape (num_responses, C)
                    labels_tensor = label_per_prompt.unsqueeze(0).repeat(self.num_responses, 1)
            except (TypeError, ValueError, IndexError) as e:
                # Catch potential errors during list access/conversion
                raise ValueError(
                    f"Invalid label format for prompt '{raw_sample['prompt'][:50]}...': {raw_label}. Error: {e}"
                ) from e

        # Optional: Check total length - Truncation happens in tokenize

        return {
            'prompt_ids': prompt_ids,
            'responses_ids': responses_ids,
            'labels': labels_tensor,  # Return the processed or default tensor
        }

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = True,
        max_length: int | None = None,
    ) -> torch.LongTensor:  # size = (L,)
        """Tokenizes text, ensuring truncation and handling max_length."""
        if max_length is None:
            max_length = self.max_length

        # Tokenizer returns dict, get 'input_ids', and squeeze to LongTensor
        tokenized = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_tensors='pt',
        )
        return tokenized['input_ids'][0]

    def get_collator(
        self,
    ) -> Callable[[list[MultiResponseTokenizedSampleWithIndex]], MultiClassSafetyBatch]:
        return MultiClassSafetyCollator(self.tokenizer.pad_token_id, self.num_responses)

    def __len__(self) -> int:
        """Return the total number of prompts."""
        return len(self.rawdata)

    def __getitem__(self, index: int) -> MultiResponseTokenizedSampleWithIndex:
        """Get a tokenized data sample for a prompt (with all its responses) by index."""
        data: MultiResponseTokenizedSample | None | object
        if not self._lazy:
            data = self._data[index]
            if data is None or data is self._SENTINEL:  # Check sentinel defensively
                raise IndexError(f"Tokenized data for index {index} is unexpectedly missing.")
        else:
            data = self._data[index]
            if data is self._SENTINEL:
                raw_sample = self.rawdata[index]
                data = self.preprocess(raw_sample)
                self._data[index] = data

        # Cast to expected type before adding index
        processed_data = data
        if not isinstance(processed_data, dict):
            # This case handles the _SENTINEL if lazy loading failed somehow, or unexpected type
            raise TypeError(
                f"Unexpected data type at index {index} after processing: {type(processed_data)}"
            )

        # Return a copy including the original prompt index
        final_data: MultiResponseTokenizedSampleWithIndex = {
            'prompt_ids': processed_data['prompt_ids'],
            'responses_ids': processed_data['responses_ids'],
            'labels': processed_data['labels'],
            'index': index,
        }
        return final_data


class MultiClassSafetyCollator(CollatorBase):
    num_responses: int  # Add num_responses attribute

    def __init__(self, pad_token_id: int, num_responses: int):
        super().__init__(pad_token_id)
        self.num_responses = num_responses

    # Input samples are list[MultiResponseTokenizedSampleWithIndex]
    def __call__(
        self, samples: list[MultiResponseTokenizedSampleWithIndex]
    ) -> MultiClassSafetyBatch:
        batch_prompt_indices = []
        batch_labels = []
        sequences_to_pad = []  # Will collect lists of sequences (one list per prompt)

        eos_token_id = (
            torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None
            else None
        )
        if eos_token_id is None:
            # Fallback or error if EOS is critical and missing
            print("Warning: EOS token ID not found in tokenizer. Concatenating without EOS.")

        for sample in samples:
            prompt_ids = sample['prompt_ids']
            responses_ids = sample['responses_ids']  # List of tensors
            labels = sample['labels']  # Tensor (num_responses, C)
            prompt_index = sample['index']

            batch_prompt_indices.append(prompt_index)
            batch_labels.append(labels)  # Append the (num_responses, C) tensor

            current_prompt_sequences = []
            for i in range(self.num_responses):
                resp_ids = responses_ids[i]
                # Concatenate prompt + response + EOS (if available)
                if eos_token_id is not None:
                    # Move tensors to the same device if necessary before cat
                    # Assuming tensors are on CPU initially
                    sequence = torch.cat((prompt_ids, resp_ids, eos_token_id))
                else:
                    sequence = torch.cat((prompt_ids, resp_ids))

                # Truncate sequence if it exceeds max_length (using dataset's max_length)
                # Note: This truncation might happen *after* concatenation
                # It might be better to truncate prompt/response *before* concatenating
                # Check max_length usage in tokenize/preprocess
                # Assuming tokenize already truncated prompt/response based on max_length

                current_prompt_sequences.append(sequence)

            sequences_to_pad.append(current_prompt_sequences)

        # Pad sequences: Need to pad each prompt's list of sequences together,
        # then stack the results.
        batch_input_ids_list = []
        batch_attn_mask_list = []
        batch_resp_mask_list = []

        # Determine max length across all sequences in the batch for uniform padding
        max_len = 0
        for prompt_seqs in sequences_to_pad:
            for seq in prompt_seqs:
                max_len = max(max_len, len(seq))

        for i, prompt_seqs in enumerate(sequences_to_pad):
            prompt_ids_len = len(samples[i]['prompt_ids'])

            # Pad all responses for the current prompt to max_len
            padded_seqs_tensor = right_padding(
                prompt_seqs, padding_value=self.pad_token_id, max_len=max_len
            )  # Shape: (num_responses, max_len)
            batch_input_ids_list.append(padded_seqs_tensor)

            attn_mask = padded_seqs_tensor.ne(self.pad_token_id)
            batch_attn_mask_list.append(attn_mask)

            # Create response mask for this prompt (num_responses, max_len)
            # Mask is True for response tokens (including EOS, excluding padding)
            response_mask = torch.zeros_like(attn_mask, dtype=torch.bool)
            # Create range tensor [0, 1, ..., max_len - 1]
            col_indices = torch.arange(max_len)
            # Response starts after prompt_ids_len
            response_mask[:, prompt_ids_len:] = True
            # Mask out padding tokens within the response part
            response_mask = response_mask & attn_mask
            batch_resp_mask_list.append(response_mask)

        # Stack results into batch tensors
        input_ids = torch.stack(batch_input_ids_list)  # (B, num_responses, L)
        attention_mask = torch.stack(batch_attn_mask_list)  # (B, num_responses, L)
        response_mask = torch.stack(batch_resp_mask_list)  # (B, num_responses, L)
        labels = torch.stack(batch_labels)  # (B, num_responses, C)
        index = torch.tensor(batch_prompt_indices, dtype=torch.long)  # (B,)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'index': index,
            'response_mask': response_mask,
        }


# Renamed trainer class
class MultiClassMultiResponsePdAlignementTrainer(MultiDualTrainer):
    TRAINING_TYPE = 'multiclass_multiresponse_pd_alignment'  # Updated type string
    # Updated DATASET_TYPE to the correct class
    DATASET_TYPE = MultiClassSafetyDataset

    model: deepspeed.DeepSpeedEngine
    # reference_model: deepspeed.DeepSpeedEngine # Reference model is loaded on-demand in eval/baseline

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

        # Call super().__init__ from the correct base class
        # This will handle dataset initialization via init_datasets
        super().__init__(args, ds_train_config)

        # Convert safety threshold to tensor here for consistent device placement
        self.safety_threshold = torch.tensor(self.args.safety_threshold, device=self.args.device)

    # init_models is inherited from MultiDualTrainer but needs PEFT application here
