from __future__ import annotations

from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DistributedSampler

from safe_rlhf.datasets.pubmedqa import PubMedQADataset, NUM_LABELS
from safe_rlhf.trainers import SupervisedTrainer
from safe_rlhf.utils import get_all_reduce_mean, is_main_process, to_device


class PubMedQATrainer(SupervisedTrainer):
    """Trainer for supervised fine-tuning on PubMedQA (yes / no / maybe)."""

    TRAINING_TYPE = 'pubmedqa_supervised'
    DATASET_TYPE = PubMedQADataset
    MODEL_TYPE = AutoModelForSequenceClassification

    def __init__(self, args, ds_config):  # noqa: D401
        # The parent *MultiClassTrainer* sets ``num_labels`` from args, we mimic that here.
        self.num_labels = NUM_LABELS
        super().__init__(args, ds_config)

        # -----------------------------
        # KL-divergence regularization
        # -----------------------------
        self.kl_coef: float = getattr(args, 'kl_coef', 0.01)

        # Load reference LM *and its tokenizer* to guarantee vocabulary consistency.
        ref_model_path = getattr(args, 'reference_model_name_or_path', args.model_name_or_path)

        self.reference_tokenizer = AutoTokenizer.from_pretrained(
            ref_model_path,
            trust_remote_code=getattr(args, 'trust_remote_code', False),
        )

        self.reference_lm = AutoModelForCausalLM.from_pretrained(
            ref_model_path,
            trust_remote_code=getattr(args, 'trust_remote_code', False),
        ).to(args.device)
        self.reference_lm.eval()
        for param in self.reference_lm.parameters():
            param.requires_grad = False

        # Pre-compute token ids in the reference tokenizer vocabulary.
        def _get_single_token(text_variants: list[str]) -> int:
            for txt in text_variants:
                ids = self.reference_tokenizer.encode(txt, add_special_tokens=False)
                if len(ids) == 1:
                    return ids[0]
            raise ValueError(f'Unable to find single-token encoding for any of {text_variants}')

        self._yes_token_id = _get_single_token([' yes', 'yes'])
        self._no_token_id = _get_single_token([' no', 'no'])
        self._maybe_token_id = _get_single_token([' maybe', 'maybe'])

    # ------------------------------------------------------------------
    # SupervisedTrainer hooks
    # ------------------------------------------------------------------
    @property
    def extra_model_kwargs(self) -> dict[str, Any]:  # type: ignore[override]
        return {
            'num_labels': NUM_LABELS,
            'problem_type': 'single_label_classification',  # force CE loss
        }

    # ------------------------------------------------------------------
    def loss(self, model: AutoModelForSequenceClassification, inputs: Dict[str, torch.Tensor], return_outputs: bool = False):  # type: ignore[override]
        # Labels are provided as one-hot – convert to indices for CrossEntropy.
        labels_one_hot = inputs.pop('labels')  # shape (B, C)
        labels = torch.argmax(labels_one_hot, dim=-1)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def train_step(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        labels: torch.FloatTensor,
        **_: Any,
    ) -> Dict[str, Any]:
        """Single optimisation step."""
        labels_idx = torch.argmax(labels, dim=-1)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_idx,
        )
        ce_loss_scalar = outputs.loss  # Cross-entropy loss from the current model

        # ------------------------------------------------------------------
        # KL divergence w.r.t. *token* log-probs of the base language model
        # ------------------------------------------------------------------
        # Map any tokens not in reference vocab (e.g., pad token from classification tokenizer)
        ref_vocab_size = self.reference_lm.get_input_embeddings().num_embeddings
        input_ids_ref = input_ids.clone()
        unk_id = (
            self.reference_tokenizer.unk_token_id
            if self.reference_tokenizer.unk_token_id is not None
            else 0
        )
        input_ids_ref[input_ids_ref >= ref_vocab_size] = unk_id

        with torch.no_grad():
            lm_out = self.reference_lm(input_ids=input_ids_ref, attention_mask=attention_mask)

        # Gather logits of the *next-token* prediction for each sequence. We
        # take the last non-padding token position as the context.
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        next_token_logits = lm_out.logits[batch_indices, seq_lengths]  # (B, V)

        # Extract logits for the label tokens.
        ref_token_logits = torch.stack(
            [
                next_token_logits[:, self._yes_token_id],
                next_token_logits[:, self._no_token_id],
                next_token_logits[:, self._maybe_token_id],
            ],
            dim=-1,
        )  # (B, 3)

        log_probs = F.log_softmax(outputs.logits.float(), dim=-1)  # (B, 3)
        ref_probs = F.softmax(ref_token_logits.float(), dim=-1)  # (B, 3)
        kl_loss = F.kl_div(log_probs, ref_probs, reduction='none')
        kl_loss = (kl_loss-1)**2
        kl_loss = kl_loss.sum(dim=-1).mean()

        # Total loss combines task loss and KL penalty
        loss = ce_loss_scalar + self.kl_coef * kl_loss

        self.model.backward(loss)
        self.model.step()

        preds_idx = torch.argmax(outputs.logits, dim=-1)
        accuracy = (preds_idx == labels_idx).float().mean()

        # All-reduce scalars
        ce_loss_scalar = get_all_reduce_mean(ce_loss_scalar)
        kl_loss = get_all_reduce_mean(kl_loss)
        loss = get_all_reduce_mean(loss)
        accuracy = get_all_reduce_mean(accuracy)

        return {
            'train/loss': loss.item(),
            'train/ce_loss': ce_loss_scalar.item(),
            'train/kl_loss': kl_loss.item(),
            'train/accuracy': accuracy.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }

    ############################################################
    # Helper: evaluate metrics for a given dataloader           #
    ############################################################
    @torch.no_grad()
    def _evaluate_loader(self, dataloader: torch.utils.data.DataLoader, prefix: str) -> Dict[str, Any]:
        """Run evaluation on *dataloader* and return metrics with given prefix."""
        self.set_eval()

        all_preds: list[int] = []
        all_labels: list[int] = []
        last_batch = None

        # Accumulators for per-label loss
        loss_yes = torch.tensor(0.0, device=self.args.device)
        loss_no = torch.tensor(0.0, device=self.args.device)
        count_yes = torch.tensor(0, device=self.args.device)
        count_no = torch.tensor(0, device=self.args.device)

        yes_probs_list: list[float] = []
        binary_labels_list: list[int] = []  # 1 = yes, 0 = no

        for batch in tqdm(
            dataloader,
            desc=f'Evaluating ({prefix})',
            disable=not is_main_process(),
            position=1,
            leave=False,
        ):
            last_batch = batch
            batch = to_device(batch, self.args.device)
            labels_idx = torch.argmax(batch['labels'], dim=-1)
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
            )
            logits = outputs.logits
            preds_idx = torch.argmax(logits, dim=-1)
            all_preds.extend(preds_idx.cpu().tolist())
            all_labels.extend(labels_idx.cpu().tolist())

            # Yes probability for AUC (only if ground truth not maybe)
            probs = F.softmax(logits.float(), dim=-1)
            prob_yes = probs[:, 0]
            mask_binary = labels_idx != 2
            if mask_binary.any():
                yes_probs_list.extend(prob_yes[mask_binary].cpu().tolist())
                binary_labels_list.extend((labels_idx[mask_binary] == 0).cpu().int().tolist())

            # Per-sample CE loss (no reduction)
            ce_loss = F.cross_entropy(logits.float(), labels_idx, reduction='none')
            mask_yes = labels_idx == 0
            mask_no = labels_idx == 1
            if mask_yes.any():
                loss_yes += ce_loss[mask_yes].sum()
                count_yes += mask_yes.sum()
            if mask_no.any():
                loss_no += ce_loss[mask_no].sum()
                count_no += mask_no.sum()

        if last_batch is None:
            self.logger.print(f'WARNING: `{prefix}` dataloader is empty.')
            return {}

        preds_tensor = torch.tensor(all_preds, device=self.args.device)
        labels_tensor = torch.tensor(all_labels, device=self.args.device)

        # Filter out maybe labels and predictions
        mask_yesno = (labels_tensor != 2) & (preds_tensor != 2)
        labels_yesno = labels_tensor[mask_yesno]
        preds_yesno = preds_tensor[mask_yesno]

        if dist.is_initialized():
            dist.all_reduce(loss_yes, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_no, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_yes, op=dist.ReduceOp.SUM)
            dist.all_reduce(count_no, op=dist.ReduceOp.SUM)

        # Average losses
        avg_loss_yes = loss_yes / count_yes if count_yes > 0 else torch.tensor(0.0, device=self.args.device)
        avg_loss_no = loss_no / count_no if count_no > 0 else torch.tensor(0.0, device=self.args.device)

        # Accuracy (yes/no only)
        acc = (labels_yesno == preds_yesno).float().mean() if len(labels_yesno) > 0 else torch.tensor(0.0, device=self.args.device)
        acc = get_all_reduce_mean(acc)

        # Confusion matrix
        confusion_yesno = torch.zeros(2, 2, device=self.args.device)
        if len(labels_yesno) > 0:
            for t, p in zip(labels_yesno, preds_yesno):
                confusion_yesno[int(t.item()), int(p.item())] += 1
        if dist.is_initialized():
            dist.all_reduce(confusion_yesno, op=dist.ReduceOp.SUM)

        tp = confusion_yesno[0, 0]
        fp = confusion_yesno[1, 0]
        fn = confusion_yesno[0, 1]
        tn = confusion_yesno[1, 1]

        maybe_gt = (labels_tensor == 2).sum()
        maybe_pred = (preds_tensor == 2).sum()
        if dist.is_initialized():
            dist.all_reduce(maybe_gt, op=dist.ReduceOp.SUM)
            dist.all_reduce(maybe_pred, op=dist.ReduceOp.SUM)

        precision_yes = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=self.args.device)
        recall_yes = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=self.args.device)
        f1_yes = 2 * precision_yes * recall_yes / (precision_yes + recall_yes) if (precision_yes + recall_yes) > 0 else torch.tensor(0.0, device=self.args.device)

        # AUC
        try:
            from sklearn.metrics import roc_auc_score  # type: ignore
            auc_yes = roc_auc_score(binary_labels_list, yes_probs_list) if len(set(binary_labels_list)) == 2 else 0.0
        except Exception:
            auc_yes = 0.0

        # Type-I (false positive) and Type-II (false negative) error rates
        type1_error_rate = fp / (fp + tn) if (fp + tn) > 0 else torch.tensor(0.0, device=self.args.device)
        type2_error_rate = fn / (fn + tp) if (fn + tp) > 0 else torch.tensor(0.0, device=self.args.device)

        metrics: Dict[str, Any] = {
            f'{prefix}/accuracy': acc.item(),
            f'{prefix}/true_positives': tp.item(),
            f'{prefix}/false_positives': fp.item(),
            f'{prefix}/false_negatives': fn.item(),
            f'{prefix}/true_negatives': tn.item(),
            f'{prefix}/loss_yes': avg_loss_yes.item(),
            f'{prefix}/loss_no': avg_loss_no.item(),
            f'{prefix}/maybe_ground_truth_count': maybe_gt.item(),
            f'{prefix}/maybe_prediction_count': maybe_pred.item(),
            f'{prefix}/precision_yes': precision_yes.item(),
            f'{prefix}/recall_yes': recall_yes.item(),
            f'{prefix}/f1_yes': f1_yes.item(),
            f'{prefix}/auc_yes': auc_yes,
            f'{prefix}/type1_error_rate': type1_error_rate.item(),  # False positive rate
            f'{prefix}/type2_error_rate': type2_error_rate.item(),  # False negative rate
        }

        # Print examples (only for main process and prefix == 'test')
        if prefix in {'test', 'eval'} and is_main_process():
            max_num_rows = 3
            texts = self.tokenizer.batch_decode(
                last_batch['input_ids'][:max_num_rows], skip_special_tokens=True
            )
            pred_lbl = [all_preds[i] for i in range(max_num_rows)]
            true_lbl = [all_labels[i] for i in range(max_num_rows)]
            self.logger.print_table(
                title=f"{prefix.capitalize()} evaluation: accuracy = {acc.item():.4f}",
                columns=['text', 'pred', 'true'],
                rows=list(zip(texts, pred_lbl, true_lbl)),
                max_num_rows=max_num_rows,
            )

        self.set_train()
        return metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def eval(self) -> dict[str, Any]:  # type: ignore[override]
        """Compute metrics on both *test* (eval_dataloader) and *train* splits."""
        if self.eval_dataloader is None:
            return {}

        # --- Evaluate on test/validation split ----------------------------------
        metrics_test = self._evaluate_loader(self.eval_dataloader, prefix='test')

        # --- Evaluate on training split -----------------------------------------
        # Build a deterministic (shuffle=False) dataloader for the training dataset
        train_dataset = self.train_dataloader.dataset  # type: ignore[attr-defined]
        train_sampler = DistributedSampler(train_dataset, shuffle=False)
        train_eval_loader = torch.utils.data.DataLoader(
            train_dataset,
            collate_fn=train_dataset.get_collator(),  # type: ignore[attr-defined]
            sampler=train_sampler,
            batch_size=self.args.per_device_eval_batch_size,
            num_workers=self.args.dataloader_workers,
            pin_memory=True,
        )

        metrics_train = self._evaluate_loader(train_eval_loader, prefix='train')

        # Merge dictionaries (train + test)
        metrics: Dict[str, Any] = {**metrics_test, **metrics_train}
        return metrics 