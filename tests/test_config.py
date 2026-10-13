"""Unit tests for configuration schemas and validation."""

import pytest
from draftcraft.core.config import (
    DraftCraftConfig,
    EagleDraftHeadConfig,
    EagleTrainConfig,
    ModelPairConfig,
    ProfilerConfig,
)
from draftcraft.core.exceptions import ConfigurationError
from draftcraft.core.types import DeviceType, DomainCategory


def test_default_config_initialization():
    config = DraftCraftConfig()
    assert config.version == "0.1.0"
    assert config.model_pair.target_hidden_dim == 4096
    assert config.profiler.min_lookahead == 1
    assert config.profiler.max_lookahead == 8
    assert config.draft_head.hidden_dim == 1024
    assert config.trainer.epochs == 3


def test_invalid_lookahead_range():
    with pytest.raises(ConfigurationError):
        ProfilerConfig(min_lookahead=8, max_lookahead=2)


def test_json_serialization_roundtrip(tmp_path):
    config = DraftCraftConfig(
        model_pair=ModelPairConfig(
            target_model_name_or_path="Qwen/Qwen2.5-7B",
            target_hidden_dim=3584,
            vocab_size=152064,
            device=DeviceType.CPU,
        ),
        profiler=ProfilerConfig(
            min_lookahead=2,
            max_lookahead=6,
            domain=DomainCategory.CODE,
        ),
    )
    json_path = tmp_path / "draftcraft_config.json"
    config.save_json(json_path)

    loaded = DraftCraftConfig.from_json(json_path)
    assert loaded.model_pair.target_model_name_or_path == "Qwen/Qwen2.5-7B"
    assert loaded.model_pair.target_hidden_dim == 3584
    assert loaded.model_pair.vocab_size == 152064
    assert loaded.profiler.domain == DomainCategory.CODE
    assert loaded.profiler.min_lookahead == 2
    assert loaded.profiler.max_lookahead == 6
