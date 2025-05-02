from __future__ import annotations

from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from safe_rlhf.datasets import MultiClassSupervisedDataset
from safe_rlhf.trainers import SupervisedTrainer
from safe_rlhf.utils import get_all_reduce_mean, is_main_process, to_device


class MultiClassTrainer(SupervisedTrainer):
    """Trainer for multiclass classification."""

    TRAINING_TYPE = 'multiclass'
    DATASET_TYPE = MultiClassSupervisedDataset
    MODEL_TYPE = AutoModelForSequenceClassification

    def __init__(self, args, ds_config):
        self.num_labels = args.num_labels
        super().__init__(args, ds_config)

    @property
    def extra_model_kwargs(self) -> dict[str, Any]:
        """Extra keyword arguments for initializing the model."""
        return {
            'num_labels': self.num_labels,
        }

    def loss(
        self,
        model: AutoModelForSequenceClassification,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        # Get labels from inputs
        labels = inputs.pop("labels")

        # Forward pass - the model will automatically compute the loss
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        if self.eval_dataloader is None:
            return {}

        self.set_eval()
        all_predictions = []
        all_labels = []
        batch = None
        for batch in tqdm(
            self.eval_dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
            position=1,
            leave=False,
        ):
            batch = to_device(batch, self.args.device)
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )

            predictions = torch.argmax(outputs.logits, dim=-1)
            labels_indices = torch.argmax(batch['labels'], dim=-1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels_indices.cpu().tolist())

        if batch is None:
            self.logger.print('WARNING: `eval_dataloader` is empty.')
            return {}

        # Convert lists to tensors for computation and move to correct device
        all_predictions = torch.tensor(all_predictions, device=self.args.device)
        all_labels = torch.tensor(all_labels, device=self.args.device)

        # Calculate overall accuracy
        accuracy = (all_predictions == all_labels).float().mean()
        accuracy = get_all_reduce_mean(accuracy)

        # Calculate per-class accuracy
        per_class_accuracy = {}
        for class_idx in range(self.num_labels):
            class_mask = all_labels == class_idx
            if class_mask.sum() > 0:  # Avoid division by zero
                class_acc = (all_predictions[class_mask] == class_idx).float().mean()
                class_acc = get_all_reduce_mean(class_acc)
                per_class_accuracy[f'eval/accuracy_class_{class_idx}'] = class_acc.item()

        # Calculate confusion matrix
        confusion_matrix = torch.zeros(self.num_labels, self.num_labels, device=self.args.device)
        for t, p in zip(all_labels, all_predictions):
            confusion_matrix[t, p] += 1

        # All-reduce confusion matrix across processes
        if dist.is_initialized():
            dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)

        self.set_train()

        # Combine all metrics
        info = {
            'eval/accuracy': accuracy.item(),
            **per_class_accuracy,
            'eval/confusion_matrix': confusion_matrix.tolist(),
        }

        if is_main_process():
            # Print some examples from the last batch
            max_num_rows = 3
            texts = self.tokenizer.batch_decode(
                batch['input_ids'][:max_num_rows],
                skip_special_tokens=True,
            )
            pred_labels = predictions[:max_num_rows].tolist()
            true_labels = batch['labels'][:max_num_rows].tolist()

            title = f'Evaluation: accuracy = {accuracy.item():.4f}'
            self.logger.print_table(
                title=title,
                columns=['text', 'predicted', 'true'],
                rows=tuple(zip(texts, pred_labels, true_labels)),
                max_num_rows=max_num_rows,
            )

        return info

    def train_step(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        labels: torch.LongTensor,
        **kwargs,
    ) -> dict[str, Any]:
        """Perform a single training step.

        Args:
            input_ids: The input ids of shape (batch_size, sequence_length)
            attention_mask: The attention mask of shape (batch_size, sequence_length)
            labels: The labels of shape (batch_size,)

        Returns:
            dict[str, Any]: Training metrics
        """
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        loss, outputs = self.loss(self.model, inputs, return_outputs=True)

        self.model.backward(loss)
        self.model.step()

        predictions = torch.argmax(outputs.logits, dim=-1)  # Shape: (B,)
        # Convert one-hot labels to class indices
        labels_indices = torch.argmax(labels, dim=-1)  # Shape: (B,)
        accuracy = (predictions == labels_indices).float().mean()

        loss = get_all_reduce_mean(loss)
        accuracy = get_all_reduce_mean(accuracy)

        return {
            'train/loss': loss.item(),
            'train/accuracy': accuracy.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }
