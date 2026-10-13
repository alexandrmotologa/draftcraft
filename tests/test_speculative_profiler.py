"""Unit tests for empirical speculative profiler and matrix evaluation."""

import pytest
from draftcraft.core.config import ProfilerConfig
from draftcraft.core.types import DomainCategory
from draftcraft.profiler.matrix_evaluator import MatrixEvaluator
from draftcraft.profiler.speculative_profiler import SpeculativeProfiler


def test_speculative_profiler_simulated_stream():
    profiler = SpeculativeProfiler(
        ProfilerConfig(
            draft_latency_ms=3.0,
            target_latency_ms=25.0,
            max_lookahead=4,
        )
    )
    metrics = profiler.profile_simulated_stream(
        base_acceptance_rate=0.75,
        lookahead_depth=4,
        num_steps=30,
        vocab_size=1000,
    )
    assert 0.0 < metrics.alpha_acceptance_rate <= 1.0
    assert metrics.expected_accepted_tokens >= 1.0
    assert metrics.theoretical_speedup > 1.0
    assert len(metrics.acceptance_by_position) == 4
    assert metrics.tokens_per_second_speculative > 0.0


def test_profile_domain_benchmarks():
    profiler = SpeculativeProfiler()
    metrics_code = profiler.profile_domain_benchmarks(
        domain=DomainCategory.CODE,
        lookahead_depth=3,
    )
    assert metrics_code.alpha_acceptance_rate > 0.5
    assert metrics_code.theoretical_speedup > 1.0


def test_matrix_evaluator():
    evaluator = MatrixEvaluator(
        ProfilerConfig(
            min_lookahead=1,
            max_lookahead=5,
            draft_latency_ms=3.0,
            target_latency_ms=25.0,
            tree_widths=[1, 2],
        )
    )
    result = evaluator.evaluate_matrix(
        domain=DomainCategory.GENERAL,
        alpha=0.75,
    )
    assert len(result.points) == 5 * 2  # 5 depths * 2 tree widths
    assert result.peak_speedup > 1.2
    assert 1 <= result.recommended_depth <= 5
    assert result.optimal_point is not None

    # Verify at least one point has Pareto optimality flag
    pareto_count = sum(1 for p in result.points if p.is_pareto_optimal)
    assert pareto_count >= 1
