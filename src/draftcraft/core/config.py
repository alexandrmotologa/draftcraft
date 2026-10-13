"""Configuration schemas and serialization facilities for DraftCraft."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, model_validator

from draftcraft.core.exceptions import ConfigurationError
from draftcraft.core.types import DeviceType, DomainCategory, DraftStrategy


class ModelPairConfig(BaseModel):
    """Specification for target and draft model pairing."""
    target_model_name_or_path: str = Field(
        default="meta-llama/Llama-3-8B-Instruct",
        description="Target model identifier or local directory path."
    )
    draft_model_name_or_path: Optional[str] = Field(
        default=None,
        description="Draft model identifier, local path, or None for standalone EAGLE head."
    )
    target_hidden_dim: int = Field(
        default=4096,
        gt=0,
        description="Hidden state vector dimension of the target LLM."
    )
    draft_hidden_dim: int = Field(
        default=1024,
        gt=0,
        description="Hidden state vector dimension of the draft model or draft head."
    )
    vocab_size: int = Field(
        default=128256,
        gt=0,
        description="Vocabulary size used by the target tokenizer."
    )
    target_layers: int = Field(
        default=32,
        gt=0,
        description="Total layer count of the target model."
    )
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Execution device: cpu, cuda, directml, or auto."
    )
    dtype: str = Field(
        default="float32",
        description="Tensor numerical precision (float32, float16, bfloat16)."
    )


class ProfilerConfig(BaseModel):
    """Profiling parameters for empirical speculative decoding analysis."""
    min_lookahead: int = Field(default=1, ge=1, le=16)
    max_lookahead: int = Field(default=8, ge=1, le=16)
    num_eval_tokens: int = Field(default=1000, gt=0)
    batch_size: int = Field(default=1, ge=1)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    draft_latency_ms: float = Field(
        default=3.5,
        gt=0.0,
        description="Average execution time for one draft proposal step in milliseconds."
    )
    target_latency_ms: float = Field(
        default=28.0,
        gt=0.0,
        description="Average execution time for one target verification step in milliseconds."
    )
    acceptance_threshold: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Minimum target acceptance rate to qualify as viable speculation."
    )
    tree_widths: List[int] = Field(
        default_factory=lambda: [1, 2, 4, 8],
        description="Candidate tree expansion branching factors to evaluate."
    )
    domain: DomainCategory = Field(
        default=DomainCategory.GENERAL,
        description="Evaluation corpus domain."
    )

    @model_validator(mode="after")
    def validate_lookahead_range(self) -> "ProfilerConfig":
        if self.min_lookahead > self.max_lookahead:
            raise ConfigurationError(
                f"min_lookahead ({self.min_lookahead}) cannot exceed max_lookahead ({self.max_lookahead})"
            )
        if self.draft_latency_ms >= self.target_latency_ms:
            # Latency ratio must be favorable for speculative speedup
            pass
        return self


class EagleDraftHeadConfig(BaseModel):
    """Architectural specification for an EAGLE speculative draft head."""
    hidden_dim: int = Field(default=1024, gt=0)
    target_hidden_dim: int = Field(default=4096, gt=0)
    num_layers: int = Field(default=2, ge=1, le=8)
    num_heads: int = Field(default=8, ge=1)
    intermediate_dim: int = Field(default=2048, gt=0)
    vocab_size: int = Field(default=32000, gt=0)
    max_position_embeddings: int = Field(default=2048, gt=0)
    rms_norm_eps: float = Field(default=1e-6, gt=0.0)
    use_residual_connection: bool = Field(default=True)
    dropout: float = Field(default=0.0, ge=0.0, le=0.5)


class EagleTrainConfig(BaseModel):
    """Hyperparameter and runtime configuration for training EAGLE draft heads."""
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-2, ge=0.0)
    batch_size: int = Field(default=8, ge=1)
    seq_len: int = Field(default=128, ge=16)
    epochs: int = Field(default=3, ge=1)
    warmup_steps: int = Field(default=50, ge=0)
    gradient_accumulation_steps: int = Field(default=2, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    distillation_loss_weight: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Weight assigned to MSE loss between target and draft hidden states."
    )
    token_ce_loss_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight assigned to Cross-Entropy loss on next-token prediction."
    )
    checkpoint_dir: str = Field(default="checkpoints")
    eval_interval: int = Field(default=25, ge=1)
    save_interval: int = Field(default=100, ge=1)
    device: DeviceType = Field(default=DeviceType.AUTO)


class DraftCraftConfig(BaseModel):
    """Consolidated master configuration for DraftCraft workflows."""
    version: str = Field(default="0.1.0")
    model_pair: ModelPairConfig = Field(default_factory=ModelPairConfig)
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    draft_head: EagleDraftHeadConfig = Field(default_factory=EagleDraftHeadConfig)
    trainer: EagleTrainConfig = Field(default_factory=EagleTrainConfig)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a JSON string."""
        return self.model_dump_json(indent=indent)

    def save_json(self, path: Union[str, Path]) -> None:
        """Write configuration to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, data: Union[str, Path]) -> "DraftCraftConfig":
        """Load configuration from JSON string or file path."""
        p = Path(data)
        if p.exists() and p.is_file():
            content = p.read_text(encoding="utf-8")
            return cls.model_validate_json(content)
        return cls.model_validate_json(data)
