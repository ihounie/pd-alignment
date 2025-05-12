from __future__ import annotations

from typing import Callable, ClassVar, List
from typing_extensions import TypedDict

import datasets
import torch
import transformers
from safe_rlhf.datasets.base import CollatorBase, RawDataset, RawSample, TokenizedDataset
from safe_rlhf.datasets.utils import right_padding

__all__ = [
    'PubMedQARawDataset',
    'PubMedQADataset',
    'PubMedQACollator',
    'PubMedQASample',
    'PubMedQABatch',
]


LABEL2IDX: dict[str, int] = {
    'yes': 0,
    'no': 1,
    'maybe': 2,
}
NUM_LABELS: int = len(LABEL2IDX)


class PubMedQARawDataset(RawDataset):
    """Raw PubMedQA dataset loader.

    The expected *path* string follows the pattern:
        ``repo_id[:subset][:split]``
    where
        repo_id  – huggingface dataset repository (default: ``qiaojin/PubMedQA``)
        subset   – one of ``pqa_labeled``, ``pqa_artificial``, ``pqa_unlabeled`` (default: ``pqa_labeled``)
        split    – dataset split recognised by 🤗 *datasets* (default: ``train``)

    Example usages
    -------------
        path = "qiaojin/PubMedQA:pqa_labeled:train"
        path = "qiaojin/PubMedQA:pqa_artificial"
        path = "qiaojin/PubMedQA"  # equivalent to the default above
    """

    NAME: ClassVar[str] = 'pubmedqa'
    ALIASES: ClassVar[set[str]] = {'PubMedQA', 'pubmed_qa'}

    def __init__(self, path: str | None = None) -> None:  # noqa: D401
        super().__init__()

        if path is None:
            path = 'qiaojin/PubMedQA:pqa_labeled:train'

        # Parse path
        repo_id, subset, split = self._parse_path(path)

        # ------------------------------------------------------------------
        # Handle special case: the `pqa_labeled` subset published on HF only
        # contains a single "train" split that mixes both official training
        # and test items.  The official test ids (and their ground-truth
        # answers) are released separately in `pubmedqa/data/test_ground_truth.json`.
        #
        # When the user requests `...:pqa_labeled:test`, we therefore need to
        #  1) load the *train* split from the HF dataset, and
        #  2) filter it down to the samples whose PubMed IDs appear in the
        #     ground-truth json file.
        #
        # Likewise, when the user requests `...:pqa_labeled:train`, we must
        # exclude those test-set samples so that *only* the official training
        # questions remain.
        # ------------------------------------------------------------------

        # Always load the available HF split ("train") first.
        hf_split = 'train' if subset == 'pqa_labeled' else split
        self.dataset = datasets.load_dataset(repo_id, subset, split=hf_split)

        # Apply train/test filtering for pqa_labeled using ground-truth IDs.
        if subset == 'pqa_labeled' and split in {'train', 'test'}:
            try:
                from pathlib import Path
                import json

                gt_path = (
                    Path(__file__)  # safe_rlhf/datasets/pubmedqa.py
                    .resolve()
                    .parent  # datasets
                    .parent  # safe_rlhf
                    .parent  # project root
                    / 'pubmedqa' / 'data' / 'test_ground_truth.json'
                )
                with gt_path.open('r', encoding='utf-8') as f:
                    gt_ids: set[int] = {int(pid) for pid in json.load(f).keys()}
            except FileNotFoundError:  # pragma: no cover – makes debugging easier
                raise FileNotFoundError(
                    'PubMedQA test_ground_truth.json not found. Expected at '
                    f'{gt_path}. Please ensure the file is available.'
                ) from None

            if split == 'test':
                self.dataset = self.dataset.filter(lambda x: x['pubid'] in gt_ids)
            else:  # requested "train"
                self.dataset = self.dataset.filter(lambda x: x['pubid'] not in gt_ids)

    @staticmethod
    def _parse_path(path: str) -> tuple[str, str, str]:
        """Split the *path* string into (repo_id, subset, split)."""
        parts = path.split(':')
        # Ensure we have at least repo id
        repo_id = parts[0] if parts[0] else 'qiaojin/PubMedQA'
        subset = parts[1] if len(parts) >= 2 and parts[1] else 'pqa_labeled'
        split = parts[2] if len(parts) >= 3 and parts[2] else 'train'
        return repo_id, subset, split

    # ---------------------------------------------------------------------
    # Mandatory RawDataset API
    # ---------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self.dataset)

    def __getitem__(self, index: int) -> RawSample:  # noqa: D401
        item = self.dataset[index]
        question: str = item['question']
        # The *context* field is a list of strings – join them with whitespace.
        context_list: List[str] = item.get('context', [])
        # Some references store context as dict with 'contexts' etc; handle generically.
        if isinstance(context_list, list):
            context_text = ' '.join(context_list)
        else:  # fallback – just str()
            context_text = str(context_list)

        prompt = f'Question: {question}\nContext: {context_text}\nAnswer:'

        final_decision: str = item['final_decision'].lower()
        if final_decision not in LABEL2IDX:
            raise ValueError(f'Unknown label `{final_decision}` in PubMedQA dataset.')
        label_idx = LABEL2IDX[final_decision]
        # One-hot encode label for BCEWithLogitsLoss style training used in the codebase
        label_vec = [0.0] * NUM_LABELS
        label_vec[label_idx] = 1.0

        return {
            'prompt': prompt,  # type: ignore[typeddict-item]
            'label': label_vec,  # type: ignore[typeddict-item]
        }


class PubMedQASample(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (L,)
    attention_mask: torch.BoolTensor  # size = (L,)
    labels: torch.FloatTensor  # size = (NUM_LABELS,)


class PubMedQABatch(TypedDict, total=True):
    input_ids: torch.LongTensor  # size = (B, L)
    attention_mask: torch.BoolTensor  # size = (B, L)
    labels: torch.FloatTensor  # size = (B, NUM_LABELS)


class PubMedQADataset(TokenizedDataset):
    """Tokenised dataset for PubMedQA classification."""

    NAME: ClassVar[str] = 'pubmedqa_supervised'

    def __init__(
        self,
        dataset_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        lazy_tokenization: bool = False,
        seed: int = 42,
        max_length: int = 512,
    ) -> None:
        self.max_length = max_length
        super().__init__(
            dataset_names_and_attributes={
                'pubmedqa': {
                    'path': dataset_path,
                    'proportion': 1.0,
                }
            },
            tokenizer=tokenizer,
            lazy_tokenization=lazy_tokenization,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # TokenizedDataset API – preprocess each raw sample into tensors
    # ------------------------------------------------------------------
    def preprocess(self, raw_sample: RawSample) -> PubMedQASample:  # type: ignore[override]
        prompt: str = raw_sample['prompt']  # type: ignore[index]
        label_vec: List[float] = raw_sample['label']  # type: ignore[index]

        # Tokenise; we do *not* add the assistant answer (yes/no/maybe) – model should infer.
        input_ids = self.tokenize(
            prompt,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_length,
            padding=False,
        )
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.FloatTensor(label_vec),
        }

    # We override tokenise to ensure max_length
    def tokenize(self, *args, **kwargs):  # type: ignore[override]
        if 'max_length' not in kwargs or kwargs['max_length'] is None:
            kwargs['max_length'] = self.max_length
        kwargs.setdefault('return_tensors', 'pt')
        res = self.tokenizer(*args, **kwargs)
        return res['input_ids'][0]

    # ------------------------------------------------------------------
    def get_collator(self) -> Callable[[list[PubMedQASample]], PubMedQABatch]:  # type: ignore[override]
        return PubMedQACollator(self.tokenizer.pad_token_id)


class PubMedQACollator(CollatorBase):
    """Pad sequences to right and stack labels."""

    def __call__(self, samples: list[PubMedQASample]) -> PubMedQABatch:  # type: ignore[override]
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