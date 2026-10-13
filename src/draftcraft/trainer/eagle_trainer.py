"""PyTorch training loop engine for EAGLE speculative draft heads."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from draftcraft.core.config import EagleDraftHeadConfig, EagleTrainConfig
from draftcraft.core.exceptions import TrainingDivergenceError
from draftcraft.models.draft_head import EagleDraftHead
from draftcraft.models.loss import EagleDistillationLoss
from draftcraft.trainer.checkpoint import CheckpointManager
from draftcraft.trainer.dataset import EagleBatch, EagleDataset


class EagleTrainer:
    """Orchestrates model training, gradient accumulation, and evaluation."""

    def __init__(
        self,
        model: EagleDraftHead,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        train_config: Optional[EagleTrainConfig] = None,
        device: Optional[torch.device] = None,
        step_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.config = train_config or EagleTrainConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.step_callback = step_callback

        self.loss_fn = EagleDistillationLoss(self.config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=EagleDataset.collate_fn,
        )

        total_steps = len(self.train_loader) * self.config.epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps, 1),
            eta_min=self.config.learning_rate * 0.05,
        )

    def train(self) -> Dict[str, Any]:
        """Execute the full training loop over configured epochs."""
        global_step = 0
        best_loss = float("inf")
        history: List[Dict[str, float]] = []

        self.model.train()
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            accumulated_loss = 0.0
            self.optimizer.zero_grad()

            for step_idx, batch in enumerate(self.train_loader, start=1):
                global_step += 1

                # Move tensors to device
                in_h = batch.hidden_states.to(self.device)
                in_tok = batch.input_ids.to(self.device)
                tgt_h = batch.target_hidden_states.to(self.device)
                tgt_tok = batch.target_token_ids.to(self.device)

                # Forward pass
                pred_h, logits = self.model(in_h, in_tok)

                # Compute loss
                loss, metrics = self.loss_fn(pred_h, tgt_h, logits, tgt_tok)

                if torch.isnan(loss) or torch.isinf(loss):
                    raise TrainingDivergenceError(f"Loss diverged to {loss.item()} at step {global_step}")

                # Scale loss for gradient accumulation
                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()
                accumulated_loss += loss.item()

                # Step optimizer when accumulation threshold reached
                if step_idx % self.config.gradient_accumulation_steps == 0 or step_idx == len(self.train_loader):
                    if self.config.max_grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                step_info = {
                    "epoch": epoch,
                    "step": global_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    **metrics,
                }
                history.append(step_info)

                if self.step_callback:
                    self.step_callback(step_info)

                # Periodic evaluation
                if global_step % self.config.eval_interval == 0 and self.eval_dataset is not None:
                    eval_metrics = self.evaluate()
                    if eval_metrics["eval_loss"] < best_loss:
                        best_loss = eval_metrics["eval_loss"]
                        CheckpointManager.save_checkpoint(
                            model=self.model,
                            output_dir=checkpoint_dir,
                            checkpoint_name="best_draft_head",
                            config=self.model.config,
                            optimizer=self.optimizer,
                            epoch=epoch,
                            step=global_step,
                            metrics=eval_metrics,
                        )

        # Save final checkpoint
        final_ckpt = CheckpointManager.save_checkpoint(
            model=self.model,
            output_dir=checkpoint_dir,
            checkpoint_name="final_draft_head",
            config=self.model.config,
            optimizer=self.optimizer,
            epoch=self.config.epochs,
            step=global_step,
            metrics={"final_train_loss": history[-1]["total_loss"] if history else 0.0},
        )

        return {
            "total_steps": global_step,
            "final_loss": history[-1]["total_loss"] if history else 0.0,
            "final_checkpoint": str(final_ckpt),
            "history": history,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation pass over eval_dataset."""
        if self.eval_dataset is None:
            return {"eval_loss": 0.0, "eval_top1_accuracy": 0.0}

        self.model.eval()
        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=EagleDataset.collate_fn,
        )

        total_loss = 0.0
        total_acc = 0.0
        batches = 0

        for batch in eval_loader:
            in_h = batch.hidden_states.to(self.device)
            in_tok = batch.input_ids.to(self.device)
            tgt_h = batch.target_hidden_states.to(self.device)
            tgt_tok = batch.target_token_ids.to(self.device)

            pred_h, logits = self.model(in_h, in_tok)
            loss, metrics = self.loss_fn(pred_h, tgt_h, logits, tgt_tok)

            total_loss += metrics["total_loss"]
            total_acc += metrics["top1_accuracy"]
            batches += 1

        self.model.train()
        return {
            "eval_loss": total_loss / max(batches, 1),
            "eval_top1_accuracy": total_acc / max(batches, 1),
        }
