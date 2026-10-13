"""Unit tests for EAGLE draft head architecture, attention, and loss."""

import pytest
import torch
from draftcraft.core.config import EagleDraftHeadConfig, EagleTrainConfig
from draftcraft.models.draft_head import EagleDecoderLayer, EagleDraftHead, RMSNorm
from draftcraft.models.loss import EagleDistillationLoss
from draftcraft.models.tree_attention import TreeAttentionBuilder


def test_rmsnorm():
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 8, 64)
    out = norm(x)
    assert out.shape == x.shape
    # Check that variance along hidden dim is close to 1
    var = out.pow(2).mean(-1)
    assert torch.allclose(var, torch.ones_like(var), atol=1e-2)


def test_eagle_decoder_layer_forward():
    layer = EagleDecoderLayer(
        hidden_dim=128,
        num_heads=4,
        intermediate_dim=256,
    )
    x = torch.randn(2, 6, 128)
    out = layer(x)
    assert out.shape == (2, 6, 128)


def test_eagle_draft_head_shapes():
    config = EagleDraftHeadConfig(
        hidden_dim=128,
        target_hidden_dim=256,
        num_layers=2,
        num_heads=4,
        intermediate_dim=256,
        vocab_size=1000,
    )
    head = EagleDraftHead(config)

    batch_size = 2
    seq_len = 8
    target_h = torch.randn(batch_size, seq_len, 256)
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))

    pred_h, logits = head(target_h, token_ids)
    assert pred_h.shape == (batch_size, seq_len, 256)
    assert logits.shape == (batch_size, seq_len, 1000)


def test_generate_draft_candidates():
    config = EagleDraftHeadConfig(
        hidden_dim=64,
        target_hidden_dim=128,
        num_layers=1,
        num_heads=2,
        intermediate_dim=128,
        vocab_size=500,
    )
    head = EagleDraftHead(config)

    init_h = torch.randn(1, 1, 128)
    init_tok = 42

    candidates, hidden_states = head.generate_draft_candidates(
        initial_hidden_state=init_h,
        initial_token_id=init_tok,
        lookahead_k=4,
        temperature=0.0,
    )

    assert len(candidates) == 4
    assert hidden_states.shape == (1, 4, 128)
    for tok in candidates:
        assert isinstance(tok, int)
        assert 0 <= tok < 500


def test_tree_attention_builder():
    paths = [[0, 1, 2], [0, 1, 3], [0, 4]]
    mask = TreeAttentionBuilder.build_tree_mask(tree_paths=paths, prefix_len=5)
    # Total nodes in candidates: 0, 1, 2, 3, 4 -> 5 nodes
    # Total sequence length: 5 (prefix) + 5 (candidates) = 10
    assert mask.shape == (10, 10)
    assert mask.dtype == torch.bool

    # Prefix tokens attend causally to prefix tokens
    assert mask[0, 0].item() is True
    assert mask[0, 1].item() is False

    # Candidate node 0 (idx 5) attends to all prefix tokens (0..4)
    assert mask[5, 0].item() is True
    assert mask[5, 4].item() is True

    # Candidate node 2 (idx 7) should attend to ancestor 1 (idx 6) and 0 (idx 5)
    assert mask[7, 5].item() is True
    assert mask[7, 6].item() is True
    # But should NOT attend to node 3 (idx 8) or node 4 (idx 9)
    assert mask[7, 8].item() is False
    assert mask[7, 9].item() is False


def test_distillation_loss():
    train_cfg = EagleTrainConfig(distillation_loss_weight=0.6, token_ce_loss_weight=0.4)
    loss_fn = EagleDistillationLoss(train_cfg)

    pred_h = torch.randn(2, 4, 64, requires_grad=True)
    target_h = torch.randn(2, 4, 64)
    logits = torch.randn(2, 4, 200, requires_grad=True)
    targets = torch.randint(0, 200, (2, 4))

    loss, metrics = loss_fn(pred_h, target_h, logits, targets)
    assert loss.item() > 0.0
    assert "distillation_loss" in metrics
    assert "ce_loss" in metrics
    assert "top1_accuracy" in metrics

    # Test backward gradient flow
    loss.backward()
    assert pred_h.grad is not None
    assert logits.grad is not None
