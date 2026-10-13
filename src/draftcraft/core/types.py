"""Core types and enumeration primitives for DraftCraft."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DeviceType(str, Enum):
    """Supported execution hardware platforms."""
    CPU = "cpu"
    CUDA = "cuda"
    DIRECTML = "directml"
    ROCM = "rocm"
    AUTO = "auto"


class DraftStrategy(str, Enum):
    """Speculative draft candidate generation strategies."""
    GREEDY_CHAIN = "greedy_chain"
    TREE_SPECULATION = "tree_speculation"
    DYNAMIC_EAGLE = "dynamic_eagle"
    MEDUSA_HEADS = "medusa_heads"


class AlignmentStatus(str, Enum):
    """Status flag for tokenizer compatibility analysis."""
    FULLY_ALIGNED = "fully_aligned"
    SUBSET_COMPATIBLE = "subset_compatible"
    VOCAB_MISMATCH = "vocab_mismatch"
    TOKENIZER_DRIFT = "tokenizer_drift"


class DomainCategory(str, Enum):
    """Benchmark corpus evaluation domains."""
    GENERAL = "general"
    CODE = "code"
    REASONING = "reasoning"
    FACTUAL_QA = "factual_qa"
    CREATIVE = "creative"


class TokenVerificationSummary(BaseModel):
    """Summary metrics of tokenizer comparison."""
    target_vocab_size: int
    draft_vocab_size: int
    shared_tokens_count: int
    overlap_ratio: float
    special_tokens_matched: bool
    status: AlignmentStatus
    divergent_sample_count: int = 0
    subword_prefix_consistency: float = 1.0
    details: Dict[str, str] = Field(default_factory=dict)


class AcceptanceMetrics(BaseModel):
    """Empirical acceptance rate and throughput metrics."""
    alpha_acceptance_rate: float
    lookahead_depth: int
    expected_accepted_tokens: float
    target_latency_ms: float
    draft_latency_ms: float
    latency_ratio: float
    theoretical_speedup: float
    wallclock_speedup: float
    tokens_per_second_baseline: float
    tokens_per_second_speculative: float
    acceptance_by_position: List[float] = Field(default_factory=list)
    rejection_reasons: Dict[str, int] = Field(default_factory=dict)


class MatrixPoint(BaseModel):
    """Single point in the speculative decoding configuration matrix."""
    lookahead_depth: int
    tree_width: int
    latency_ratio: float
    acceptance_rate: float
    expected_tokens: float
    speedup: float
    is_pareto_optimal: bool = False


class MatrixEvaluationResult(BaseModel):
    """Comprehensive multi-parameter matrix evaluation result."""
    points: List[MatrixPoint]
    optimal_point: MatrixPoint
    recommended_depth: int
    recommended_tree_width: int
    peak_speedup: float
    evaluation_domain: DomainCategory
