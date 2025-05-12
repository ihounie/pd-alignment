from __future__ import annotations

from typing import Any, Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

from safe_rlhf.algorithms.pubmedqa.trainer import PubMedQATrainer
from safe_rlhf.utils import get_all_reduce_mean, is_main_process, to_device


class PubMedQAConstrainedTrainer(PubMedQATrainer):
    """PubMedQA trainer with per-label CE constraints enforced via dual ascent."""

    TRAINING_TYPE = 'pubmedqa_constrained'

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(self, args, ds_config):  # noqa: D401
        # Read constraint hyper-parameters from *args*
        self.ce_threshold_yes: float = args.ce_threshold_yes
        self.ce_threshold_no: float = args.ce_threshold_no
        self.dual_step_size: float = args.dual_step_size
        # Linearly warm up the dual-step size from 0 to the target value over this many epochs.
        # If set to 0, warm-up is disabled and *dual_step_size* is used from the first epoch.
        self.dual_warmup_epochs: int = getattr(args, 'dual_warmup_epochs', 0)
        # Weight decay applied to Lagrange multipliers at each update (0 = none).
        self.dual_weight_decay: float = getattr(args, 'dual_weight_decay', 0.0)
        # Update multipliers every N training batches instead of once per epoch.
        # If set to 0, defaults to one full epoch (legacy behaviour)
        self.dual_update_interval: int = getattr(args, 'dual_update_interval', 0)
        # Lagrange multipliers (lambda >= 0)
        self.lambda_yes: float = getattr(args, 'lambda_yes_init', 0.0)
        self.lambda_no: float = getattr(args, 'lambda_no_init', 0.0)

        # Accumulators for an epoch (initialised in *train*)
        self._epoch_yes_loss_sum: torch.Tensor | None = None
        self._epoch_no_loss_sum: torch.Tensor | None = None
        self._epoch_yes_count: torch.Tensor | None = None
        self._epoch_no_count: torch.Tensor | None = None

        super().__init__(args, ds_config)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------
    def _reset_epoch_stats(self) -> None:
        device = self.args.device
        self._epoch_yes_loss_sum = torch.tensor(0.0, device=device)
        self._epoch_no_loss_sum = torch.tensor(0.0, device=device)
        self._epoch_yes_count = torch.tensor(0, device=device)
        self._epoch_no_count = torch.tensor(0, device=device)

    # Alias for backwards compatibility
    _reset_stats = _reset_epoch_stats

    # ------------------------------------------------------------------
    def train_step(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.BoolTensor,
        labels: torch.FloatTensor,
        **_: Any,
    ) -> Dict[str, Any]:
        """Single constrained optimisation step."""
        labels_idx = torch.argmax(labels, dim=-1)  # (B,)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # Per-sample cross entropy, no reduction
        ce_losses = F.cross_entropy(logits.float(), labels_idx, reduction='none')  # (B,)

        # Masks for individual classes
        mask_yes = labels_idx == 0
        mask_no = labels_idx == 1

        # Per-label averaged CE losses for this batch (0 if not present)
        ce_yes = ce_losses[mask_yes].mean() if mask_yes.any() else torch.tensor(0.0, device=logits.device)
        ce_no = ce_losses[mask_no].mean() if mask_no.any() else torch.tensor(0.0, device=logits.device)

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

        # Augment the objective with Lagrange terms
        penalty = 0.0
        if self.lambda_yes > 0 or ce_yes.item() > 0:
            penalty = penalty + self.lambda_yes * (ce_yes - self.ce_threshold_yes)
        if self.lambda_no > 0 or ce_no.item() > 0:
            penalty = penalty + self.lambda_no * (ce_no - self.ce_threshold_no)

        # Primary loss is the mean CE across all samples + penalty + KL regularization
        loss = ce_losses.mean() + penalty + self.kl_coef * kl_loss

        # Back-prop & optimisation step
        self.model.backward(loss)
        self.model.step()

        # Accuracy for monitoring
        preds_idx = torch.argmax(logits, dim=-1)
        accuracy = (preds_idx == labels_idx).float().mean()

        # Update epoch accumulators (used for multiplier updates at epoch end)
        self._epoch_yes_loss_sum += ce_losses[mask_yes].sum()
        self._epoch_no_loss_sum += ce_losses[mask_no].sum()
        self._epoch_yes_count += mask_yes.sum()
        self._epoch_no_count += mask_no.sum()

        # Synchronise statistics across processes
        accuracy = get_all_reduce_mean(accuracy)
        ce_yes_sync = get_all_reduce_mean(ce_yes)
        ce_no_sync = get_all_reduce_mean(ce_no)
        loss_sync = get_all_reduce_mean(loss)
        kl_loss_sync = get_all_reduce_mean(kl_loss)

        return {
            'train/loss': loss_sync.item(),
            'train/accuracy': accuracy.item(),
            'train/ce_yes': ce_yes_sync.item(),
            'train/ce_no': ce_no_sync.item(),
            'train/kl_loss': kl_loss_sync.item(),
            'train/lambda_yes': self.lambda_yes,
            'train/lambda_no': self.lambda_no,
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }

    # ------------------------------------------------------------------
    def _update_multipliers(self) -> Dict[str, float]:
        """Perform a projected dual ascent step on the Lagrange multipliers."""
        # Aggregate across distributed workers
        yes_loss_sum = self._epoch_yes_loss_sum.clone()
        no_loss_sum = self._epoch_no_loss_sum.clone()
        yes_count = self._epoch_yes_count.clone()
        no_count = self._epoch_no_count.clone()

        if dist.is_initialized():
            dist.all_reduce(yes_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(no_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(yes_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(no_count, op=dist.ReduceOp.SUM)

        # Compute average CE for the epoch (avoid div/0)
        avg_yes = (yes_loss_sum / yes_count) if yes_count > 0 else torch.tensor(0.0, device=self.args.device)
        avg_no = (no_loss_sum / no_count) if no_count > 0 else torch.tensor(0.0, device=self.args.device)

        # Linear warm-up of the dual step size
        if self.dual_warmup_epochs > 0 and hasattr(self, 'current_epoch'):
            warmup_factor = min((self.current_epoch + 1) / self.dual_warmup_epochs, 1.0)
        else:
            warmup_factor = 1.0

        effective_step = self.dual_step_size * warmup_factor

        # Apply weight decay to current multipliers before the ascent step
        self.lambda_yes = self.lambda_yes * (1.0 - self.dual_weight_decay)
        self.lambda_no = self.lambda_no * (1.0 - self.dual_weight_decay)

        # Dual ascent update with decayed base
        self.lambda_yes = max(0.0, self.lambda_yes + effective_step * (avg_yes.item() - self.ce_threshold_yes))
        self.lambda_no = max(0.0, self.lambda_no + effective_step * (avg_no.item() - self.ce_threshold_no))

        return {
            'epoch/avg_ce_yes': avg_yes.item(),
            'epoch/avg_ce_no': avg_no.item(),
            'epoch/lambda_yes': self.lambda_yes,
            'epoch/lambda_no': self.lambda_no,
            'epoch/dual_step_effective': effective_step,
        }

    # ------------------------------------------------------------------
    def train(self) -> None:  # type: ignore[override]
        """Train with constrained optimisation."""
        self.logger.print('***** Running constrained training *****')

        progress_bar = tqdm(
            total=self.args.epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.args.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )

        # Determine interval: default to full epoch if not specified or <=0
        update_interval = self.dual_update_interval if self.dual_update_interval > 0 else len(self.train_dataloader)

        for epoch in range(self.args.epochs):
            # Track current epoch index for warm-up schedule
            self.current_epoch = epoch
            # Reset stats & set train mode
            self._reset_stats()
            self.set_train()

            for batch_idx, batch in enumerate(self.train_dataloader, start=1):
                info = self.train_step(**to_device(batch, self.args.device))
                torch.cuda.empty_cache()

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.args.epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )
                progress_bar.update(1)

                info['train/epoch'] = self.global_step / len(self.train_dataloader)
                self.logger.log(info, step=self.global_step)

                # Perform multiplier update at specified batch interval
                if batch_idx % update_interval == 0:
                    metrics = self._update_multipliers()
                    self.logger.log(metrics, step=self.global_step)
                    self._reset_stats()

                # Checkpointing, evaluation strategy etc. handled by subclass or caller.

            # --- End of epoch: update multipliers for leftover stats -------------
            if (self._epoch_yes_count.item() > 0) or (self._epoch_no_count.item() > 0):
                epoch_metrics = self._update_multipliers()
                self.logger.log(epoch_metrics, step=self.global_step)
                self._reset_stats()

            # Optionally run evaluation using inherited *eval* method
            if self.args.need_eval and self.args.eval_strategy == 'epoch' and epoch % self.args.eval_interval == 0:
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.args.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)

            # Update progress bar title for next epoch
            if epoch + 1 < self.args.epochs:
                progress_bar.set_description(f'Training {epoch + 2}/{self.args.epochs} epoch')

            self.model.tput_timer.update_epoch_count() 