"""Loss functions for EAGLE speculative draft head distillation."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from draftcraft.core.config import EagleTrainConfig


class EagleDistillationLoss(nn.Module):
    """Computes combined hidden state regression and token classification loss."""

    def __init__(self, config: Optional[EagleTrainConfig] = None):
        super().__init__()
        self.config = config or EagleTrainConfig()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        predicted_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        predicted_logits: torch.Tensor,
        target_token_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the weighted distillation loss.

        Args:
            predicted_hidden_states: [Batch, Seq_Len, Hidden_Dim] from draft head.
            target_hidden_states: [Batch, Seq_Len, Hidden_Dim] ground truth from target LLM.
            predicted_logits: [Batch, Seq_Len, Vocab_Size] token logits from draft head.
            target_token_ids: [Batch, Seq_Len] ground truth next-token IDs.

        Returns:
            Tuple of (total_loss_tensor, metrics_dict).
        """
        # Hidden state distillation loss (MSE)
        distill_loss = self.mse_loss(predicted_hidden_states, target_hidden_states)

        # Next-token prediction loss (CrossEntropy)
        # Flatten batch and sequence dimensions for CrossEntropyLoss
        vocab_size = predicted_logits.size(-1)
        flat_logits = predicted_logits.view(-1, vocab_size)
        flat_targets = target_token_ids.view(-1)
        ce_loss = self.ce_loss(flat_logits, flat_targets)

        # Weighted composite loss
        w_distill = self.config.distillation_loss_weight
        w_ce = self.config.token_ce_loss_weight
        total_loss = w_distill * distill_loss + w_ce * ce_loss

        # Calculate top-1 accuracy on valid tokens
        valid_mask = flat_targets != -100
        if valid_mask.sum() > 0:
            pred_tokens = torch.argmax(flat_logits, dim=-1)
            correct = (pred_tokens[valid_mask] == flat_targets[valid_mask]).sum().item()
            total_valid = valid_mask.sum().item()
            top1_accuracy = correct / max(total_valid, 1)
        else:
            top1_accuracy = 0.0

        metrics = {
            "total_loss": float(total_loss.item()),
            "distillation_loss": float(distill_loss.item()),
            "ce_loss": float(ce_loss.item()),
            "top1_accuracy": float(top1_accuracy),
        }

        return total_loss, metrics
