"""Dataset class for pointwise safe training."""

from __future__ import annotations

from typing import Callable
from typing_extensions import TypedDict  # Python 3.10+

import torch

from safe_rlhf.datasets.base import CollatorBase, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import format_prompt, right_padding


__all__ = [
    'PointwiseSafeDataset',
    'PointwiseSafeCollator',
    'PointwiseSafeSample',
    'PointwiseSafeBatch',
]


class PointwiseSafeSample(TypedDict, total=True):
    """
    Represents a single tokenized sample for our pointwise safe dataset.
    All 'safe'/'dual' fields and lambdas are stored as floats (e.g. 0.0, 1.0).
    """

    prompt_input_ids: torch.LongTensor  # size = (L_prompt,)
    better_input_ids: torch.LongTensor  # size = (L_better,)
    worse_input_ids: torch.LongTensor  # size = (L_worse,)
    response_index: int  # New field to store the precomputed response index

    # Additional scalar fields (floats):
    better_safe: float
    worse_safe: float
    better_dual: float
    worse_dual: float

    # Lambdas (initialized to zero here, can be learned later if needed):
    better_lambda: float
    worse_lambda: float


class PointwiseSafeBatch(TypedDict, total=True):
    """
    Represents a collated batch of samples.
    Each input_ids and attention_mask is shape (B, L).
    Each float field is shape (B,).
    """

    prompt_input_ids: torch.LongTensor  # size = (B, L_prompt)
    prompt_attention_mask: torch.BoolTensor  # size = (B, L_prompt)

    better_input_ids: torch.LongTensor  # size = (B, L_better)
    better_attention_mask: torch.BoolTensor  # size = (B, L_better)

    worse_input_ids: torch.LongTensor  # size = (B, L_worse)
    worse_attention_mask: torch.BoolTensor  # size = (B, L_worse)

    better_safe: torch.FloatTensor  # size = (B,)
    worse_safe: torch.FloatTensor  # size = (B,)
    better_dual: torch.FloatTensor  # size = (B,)
    worse_dual: torch.FloatTensor  # size = (B,)
    better_lambda: torch.FloatTensor  # size = (B,)
    worse_lambda: torch.FloatTensor  # size = (B,)


class PointwiseSafeDataset(TokenizedDataset):
    """
    A dataset class similar to PreferenceDataset but with extra fields:
      - prompt, better_answer, worse_answer
      - better_safe, worse_safe
      - better_dual, worse_dual
      - lambdas (initialized to zero)
    """

    def preprocess(self, raw_sample: RawSample) -> PointwiseSafeSample:
        """
        Convert the raw sample into a tokenized sample, extracting fields
        and storing floats for safety/dual/lambda.

        RawSample could have:
          - raw_sample['input']         -> prompt text
          - raw_sample['answer']        -> better_answer text
          - raw_sample['other_answer']  -> worse_answer text
          - raw_sample['is_safe']       -> better_safe (bool or int)
          - raw_sample['is_other_safe'] -> worse_safe (bool or int)
          - raw_sample['safer']         -> better_dual (bool or int)
          - etc. Adjust as needed.

        All non-text fields are turned into floats (e.g. 0.0 or 1.0).
        Lambdas are explicitly set to 0.0.
        """
        prompt_text = (
            format_prompt(
                input=raw_sample['input'],
                eos_token=self.tokenizer.eos_token,
            )
            if 'input' in raw_sample
            else ""
        )

        better_answer_text = raw_sample.get('answer', "")
        worse_answer_text = raw_sample.get('other_answer', "")

        # Convert these to floats (0.0 or 1.0) or your chosen logic
        better_safe = float(raw_sample.get('is_safe', 0))
        worse_safe = float(raw_sample.get('is_other_safe', 0))

        # Tokenize each field
        better_input_ids = self.tokenize(
            prompt_text + better_answer_text + self.tokenizer.eos_token
        )
        worse_input_ids = self.tokenize(prompt_text + worse_answer_text + self.tokenizer.eos_token)

        # Compute response index by finding first difference
        min_len = min(len(better_input_ids), len(worse_input_ids))
        differences = [i for i in range(min_len) if better_input_ids[i] != worse_input_ids[i]]
        if not differences:
            raise ValueError("Found identical sequences in better and worse inputs")
        response_index = differences[0]

        # Return typed dict with everything we need
        return {
            'better_input_ids': better_input_ids,
            'worse_input_ids': worse_input_ids,
            'better_safe': better_safe,
            'worse_safe': worse_safe,
            'response_index': response_index,
        }

    def get_collator(self) -> Callable[[list[PointwiseSafeSample]], PointwiseSafeBatch]:
        """
        Return the collator that handles padding all three types of token IDs,
        plus merges float fields into float tensors.
        """
        return PointwiseSafeCollator(self.tokenizer.pad_token_id)

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


class PointwiseSafeCollator(CollatorBase):
    """
    Collator that:
      - pads the prompt_input_ids, better_input_ids, and worse_input_ids
      - splits them back into separate batch tendata['index'] = torch.tensor(index)sors
      - converts the safe/dual/lambda floats to tensors
    """

    def __call__(self, samples: list[PointwiseSafeSample]) -> PointwiseSafeBatch:
        """
        1. Collect prompt_input_ids, better_input_ids, worse_input_ids
        2. Pad them up to (B, L) each
        3. Build attention masks
        4. Convert float fields to Tensors
        5. Return a dictionary conforming to PointwiseSafeBatch
        """
        # 1) Gather IDs for all samples, then pad them in one shot
        better_ids_list = [s['better_input_ids'] for s in samples]
        worse_ids_list = [s['worse_input_ids'] for s in samples]
        index_list = [s['index'] for s in samples]
        response_index = torch.tensor([s['response_index'] for s in samples], dtype=torch.long)

        input_ids = better_ids_list + worse_ids_list  # size = (2 * B, L)
        attention_mask = [
            input_id.new_ones(input_id.size(), dtype=torch.bool) for input_id in input_ids
        ]  # size = (2 * B, L)

        input_ids = right_padding(input_ids, padding_value=self.pad_token_id)  # size = (2 * B, L)
        attention_mask = right_padding(attention_mask, padding_value=0)  # size = (2 * B, L)

        (
            better_input_ids,  # size = (B, L)
            worse_input_ids,  # size = (B, L)
        ) = input_ids.chunk(chunks=2, dim=0)
        (
            better_attention_mask,  # size = (B, L)
            worse_attention_mask,  # size = (B, L)
        ) = attention_mask.chunk(chunks=2, dim=0)

        # Create response masks
        _, seq_length = better_input_ids.shape
        response_masks = (
            torch.arange(seq_length, device=better_input_ids.device)[None, :]
            >= response_index[:, None]
        )

        # 3) Convert the other fields to tensors
        better_safe = torch.tensor([s['better_safe'] for s in samples], dtype=torch.float)
        worse_safe = torch.tensor([s['worse_safe'] for s in samples], dtype=torch.float)
        indexes = torch.tensor(index_list, dtype=torch.long)
        # 4) Return everything as a single batch dictionary
        return {
            'better_input_ids': better_input_ids,
            'better_attention_mask': better_attention_mask,
            'worse_input_ids': worse_input_ids,
            'worse_attention_mask': worse_attention_mask,
            'better_safe': better_safe,
            'worse_safe': worse_safe,
            'index': indexes,
            'response_masks': response_masks,
        }
