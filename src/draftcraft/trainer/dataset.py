"""PyTorch dataset abstractions and synthetic data pipelines for EAGLE training."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset


@dataclass
class EagleBatch:
    """Batch container for EAGLE training step."""
    hidden_states: torch.Tensor       # [B, S, Target_Hidden_Dim]
    input_ids: torch.Tensor           # [B, S]
    target_hidden_states: torch.Tensor  # [B, S, Target_Hidden_Dim]
    target_token_ids: torch.Tensor    # [B, S]


class EagleDataset(Dataset):
    """PyTorch Dataset pairing input features with next-token teacher targets."""

    def __init__(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
    ):
        """Initialize dataset.

        Args:
            hidden_states: Tensor of shape [Num_Samples, Seq_Len + 1, Hidden_Dim].
            token_ids: Tensor of shape [Num_Samples, Seq_Len + 1].
        """
        assert hidden_states.size(0) == token_ids.size(0), "Sample count mismatch"
        assert hidden_states.size(1) == token_ids.size(1), "Sequence length mismatch"
        assert hidden_states.size(1) >= 2, "Sequence length must be at least 2 for autoregression"

        self.hidden_states = hidden_states
        self.token_ids = token_ids

    def __len__(self) -> int:
        return self.hidden_states.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Input is sequence [0 ... L-1]
        input_h = self.hidden_states[idx, :-1, :]
        input_tok = self.token_ids[idx, :-1]

        # Target is shifted sequence [1 ... L]
        target_h = self.hidden_states[idx, 1:, :]
        target_tok = self.token_ids[idx, 1:]

        return {
            "hidden_states": input_h,
            "input_ids": input_tok,
            "target_hidden_states": target_h,
            "target_token_ids": target_tok,
        }

    @staticmethod
    def collate_fn(batch_list: List[Dict[str, torch.Tensor]]) -> EagleBatch:
        """Collate list of sample dictionaries into an EagleBatch."""
        b_h = torch.stack([item["hidden_states"] for item in batch_list], dim=0)
        b_tok = torch.stack([item["input_ids"] for item in batch_list], dim=0)
        b_target_h = torch.stack([item["target_hidden_states"] for item in batch_list], dim=0)
        b_target_tok = torch.stack([item["target_token_ids"] for item in batch_list], dim=0)

        return EagleBatch(
            hidden_states=b_h,
            input_ids=b_tok,
            target_hidden_states=b_target_h,
            target_token_ids=b_target_tok,
        )


class SyntheticFeatureGenerator:
    """Generates synthetic hidden state trajectories for testing and headless runs."""

    @staticmethod
    def generate_synthetic_dataset(
        num_samples: int = 32,
        seq_len: int = 32,
        hidden_dim: int = 512,
        vocab_size: int = 1000,
        seed: int = 42,
    ) -> EagleDataset:
        """Create a synthetic dataset simulating LLM hidden state dynamics."""
        torch.manual_seed(seed)
        total_len = seq_len + 1

        # Simulate auto-regressive hidden state continuity via random walk
        # Start states
        start_h = torch.randn(num_samples, 1, hidden_dim) * 0.1
        # Increments
        steps_h = torch.randn(num_samples, total_len - 1, hidden_dim) * 0.05
        # Cumulative trajectory
        full_h = torch.cat([start_h, steps_h], dim=1).cumsum(dim=1)

        # Synthesize correlated token IDs
        # Project hidden state down to vocab logits
        proj = torch.randn(hidden_dim, vocab_size) * 0.05
        logits = torch.matmul(full_h, proj)
        tokens = torch.argmax(logits, dim=-1)

        return EagleDataset(hidden_states=full_h, token_ids=tokens)
