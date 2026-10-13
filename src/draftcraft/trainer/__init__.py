"""EAGLE draft head training and dataset orchestration module."""

from draftcraft.trainer.checkpoint import CheckpointManager
from draftcraft.trainer.dataset import EagleBatch, EagleDataset, SyntheticFeatureGenerator
from draftcraft.trainer.eagle_trainer import EagleTrainer

__all__ = [
    "CheckpointManager",
    "EagleBatch",
    "EagleDataset",
    "EagleTrainer",
    "SyntheticFeatureGenerator",
]
