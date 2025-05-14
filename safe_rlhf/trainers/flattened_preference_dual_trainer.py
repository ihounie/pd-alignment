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
"""Trainer for dual optimization using flattened preference datasets."""

from __future__ import annotations

from typing import ClassVar
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


# Ensure FlattenedPreferenceDataset is updated to include response_mask for eval compatibility
from safe_rlhf.datasets import FlattenedPreferenceDataset, MultiClassSafetyDataset # Import MultiClassSafetyDataset
from safe_rlhf.trainers.multi_dual_trainer import MultiDualTrainer

__all__ = ['FlattenedPreferenceDualTrainer']


class FlattenedPreferenceDualTrainer(MultiDualTrainer):
    """
    Trainer that adapts MultiDualTrainer to use FlattenedPreferenceDataset for training
    and a specified EVAL_DATASET_TYPE (defaulting to MultiClassSafetyDataset) for evaluation.

    Training dataset provides individual prompt-response samples with a 'label' field (cost vector).
    Evaluation dataset is handled by EVAL_DATASET_TYPE.
    """

    DATASET_TYPE: ClassVar[type[FlattenedPreferenceDataset]] = FlattenedPreferenceDataset
    EVAL_DATASET_TYPE: ClassVar[type[MultiClassSafetyDataset]] = MultiClassSafetyDataset # Specify eval dataset type

    # The abstract methods `loss` and `train_step` from MultiDualTrainer
    # must be implemented by a concrete class that inherits from this
    # FlattenedPreferenceDualTrainer. This class primarily configures the dataset type.

    def init_datasets(self, tokenizer=None) -> None:
        """Initialize training and evaluation datasets with specific types."""
        if tokenizer is None:
            tokenizer = self.tokenizer

        # Training dataset uses self.DATASET_TYPE (FlattenedPreferenceDataset)
        train_dataset = self.DATASET_TYPE(
            self.args.train_datasets,
            tokenizer=tokenizer,
            # lazy_tokenization is handled by FlattenedPreferenceDataset's __init__
            # seed is handled by FlattenedPreferenceDataset's __init__
            # max_length is handled by FlattenedPreferenceDataset's __init__
        )

        if self.args.need_eval:
            if self.args.eval_datasets is None and self.args.eval_split_ratio is not None:
                # This split logic might be tricky if train_dataset is already flattened.
                # For now, assume eval_datasets is provided.
                # If splitting is needed, it should ideally happen on the raw, unflattened data.
                raise NotImplementedError(
                    "Splitting from a FlattenedPreferenceDataset for evaluation is not straightforward. "
                    "Please provide a separate eval_datasets path."
                )
                # train_dataset, eval_dataset = train_dataset.split_train_test(
                #     split_ratio=self.args.eval_split_ratio,
                # )
                # # Ensure eval_dataset is of the correct type if split this way - would require re-wrapping
                # eval_dataset = self.EVAL_DATASET_TYPE( # This line would be problematic
                #     eval_dataset.rawdata, # This is not how EVAL_DATASET_TYPE is initialized
                #     tokenizer=tokenizer, lazy_tokenization=False, seed=self.args.seed
                # )
            elif self.args.eval_datasets is not None and self.args.eval_split_ratio is None:
                # Evaluation dataset uses self.EVAL_DATASET_TYPE
                eval_dataset = self.EVAL_DATASET_TYPE(
                    self.args.eval_datasets, 
                    tokenizer=tokenizer, 
                    # lazy_tokenization specific to MultiClassSafetyDataset / its base
                    # seed specific to MultiClassSafetyDataset / its base
                    # max_length specific to MultiClassSafetyDataset / its base
                )
            else:
                raise ValueError(
                    'Either `eval_datasets` must be provided or (`eval_datasets` is None and `eval_split_ratio` is None if need_eval=False)'
                )

            self.eval_dataloader = DataLoader(
                eval_dataset,
                collate_fn=eval_dataset.get_collator(),
                sampler=DistributedSampler(eval_dataset, shuffle=False), # Usually False for eval
                batch_size=self.args.per_device_eval_batch_size,
            )
        else:
            self.eval_dataloader = None

        self.train_dataloader = DataLoader(
            train_dataset,
            collate_fn=train_dataset.get_collator(),
            sampler=DistributedSampler(train_dataset, shuffle=True),
            batch_size=self.args.per_device_train_batch_size,
        )

    # No explicit overrides for init_costs, init_rewards, etc., are made here,
    # assuming the base class's "indicator" cost logic (using batch['labels'])
    # and other initializations are compatible or will be correctly configured
    # via arguments.

    # Potential considerations for advanced use:
    # - If using `args.run_closed_form_dual=True` with `args.sample_responses_for_dual=False`,
    #   the format of `self.costs` (derived from the flattened dataset's labels) might
    #   need adjustment to be compatible with the `DualOptimizer`, which expects a
    #   dimension for 'responses per prompt'. 