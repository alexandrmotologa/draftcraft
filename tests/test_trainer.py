"""Unit tests for dataset pipeline, checkpoint manager, and trainer."""

import pytest
import torch
from draftcraft.core.config import EagleDraftHeadConfig, EagleTrainConfig
from draftcraft.models.draft_head import EagleDraftHead
from draftcraft.trainer.checkpoint import CheckpointManager
from draftcraft.trainer.dataset import EagleDataset, SyntheticFeatureGenerator
from draftcraft.trainer.eagle_trainer import EagleTrainer


def test_synthetic_dataset_generation():
    dataset = SyntheticFeatureGenerator.generate_synthetic_dataset(
        num_samples=8,
        seq_len=16,
        hidden_dim=64,
        vocab_size=100,
    )
    assert len(dataset) == 8
    sample = dataset[0]
    assert sample["hidden_states"].shape == (16, 64)
    assert sample["input_ids"].shape == (16,)
    assert sample["target_hidden_states"].shape == (16, 64)
    assert sample["target_token_ids"].shape == (16,)


def test_checkpoint_save_and_load(tmp_path):
    config = EagleDraftHeadConfig(
        hidden_dim=32,
        target_hidden_dim=64,
        num_layers=1,
        num_heads=2,
        intermediate_dim=64,
        vocab_size=100,
    )
    model1 = EagleDraftHead(config)
    ckpt_path = CheckpointManager.save_checkpoint(
        model=model1,
        output_dir=tmp_path,
        checkpoint_name="test_head",
        config=config,
        epoch=1,
        step=10,
        metrics={"eval_loss": 0.42},
    )
    assert ckpt_path.exists()

    model2 = EagleDraftHead(config)
    info = CheckpointManager.load_checkpoint(model2, ckpt_path)
    assert info["epoch"] == 1
    assert info["step"] == 10
    assert info["metrics"]["eval_loss"] == 0.42

    # Verify weights match exactly
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_eagle_trainer_execution(tmp_path):
    head_config = EagleDraftHeadConfig(
        hidden_dim=32,
        target_hidden_dim=64,
        num_layers=1,
        num_heads=2,
        intermediate_dim=64,
        vocab_size=100,
    )
    model = EagleDraftHead(head_config)

    train_data = SyntheticFeatureGenerator.generate_synthetic_dataset(
        num_samples=16,
        seq_len=8,
        hidden_dim=64,
        vocab_size=100,
        seed=1,
    )
    eval_data = SyntheticFeatureGenerator.generate_synthetic_dataset(
        num_samples=8,
        seq_len=8,
        hidden_dim=64,
        vocab_size=100,
        seed=2,
    )

    train_config = EagleTrainConfig(
        learning_rate=1e-3,
        batch_size=4,
        epochs=2,
        eval_interval=2,
        checkpoint_dir=str(tmp_path),
    )

    trainer = EagleTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        train_config=train_config,
    )

    result = trainer.train()
    assert result["total_steps"] > 0
    assert result["final_loss"] > 0.0
    assert len(result["history"]) > 0
    assert (tmp_path / "final_draft_head.pt").exists()
