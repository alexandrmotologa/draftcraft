"""EAGLE neural models, loss functions, and tree attention modules."""

from draftcraft.models.draft_head import EagleDecoderLayer, EagleDraftHead, RMSNorm
from draftcraft.models.loss import EagleDistillationLoss
from draftcraft.models.tree_attention import TreeAttentionBuilder

__all__ = [
    "EagleDecoderLayer",
    "EagleDraftHead",
    "EagleDistillationLoss",
    "RMSNorm",
    "TreeAttentionBuilder",
]
