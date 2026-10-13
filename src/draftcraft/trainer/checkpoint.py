"""Checkpoint management and serialization for EAGLE draft heads."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from draftcraft.core.config import EagleDraftHeadConfig
from draftcraft.core.exceptions import CheckpointError


class CheckpointManager:
    """Handles saving, loading, and metadata inspection of draft head checkpoints."""

    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        output_dir: Union[str, Path],
        checkpoint_name: str,
        config: Optional[EagleDraftHeadConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        step: int = 0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Path:
        """Save model weights and metadata to directory.

        Creates:
        - <checkpoint_name>.pt (PyTorch state_dict)
        - <checkpoint_name>_metadata.json (training info and architecture config)
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        weights_file = out_path / f"{checkpoint_name}.pt"
        meta_file = out_path / f"{checkpoint_name}_metadata.json"

        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
        }
        if optimizer is not None:
            checkpoint_payload["optimizer_state_dict"] = optimizer.state_dict()

        torch.save(checkpoint_payload, weights_file)

        # Save metadata JSON
        metadata: Dict[str, Any] = {
            "checkpoint_name": checkpoint_name,
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
            "weights_filename": weights_file.name,
        }
        if config is not None:
            metadata["config"] = config.model_dump()

        meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return weights_file

    @staticmethod
    def load_checkpoint(
        model: nn.Module,
        checkpoint_path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Any]:
        """Load state dict into model and optional optimizer."""
        p = Path(checkpoint_path)
        if not p.exists():
            raise CheckpointError(f"Checkpoint file not found: {p}")

        try:
            payload = torch.load(p, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state_dict"])

            if optimizer is not None and "optimizer_state_dict" in payload:
                optimizer.load_state_dict(payload["optimizer_state_dict"])

            return {
                "epoch": payload.get("epoch", 0),
                "step": payload.get("step", 0),
                "metrics": payload.get("metrics", {}),
            }
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint from {p}: {e}") from e
