"""Unit tests for speculative decoding acceptance rate mathematics."""

import math
import pytest
import torch
from draftcraft.profiler.acceptance_math import AcceptanceMath


def test_total_variation_distance_identical():
    p = torch.tensor([0.2, 0.5, 0.3])
    q = torch.tensor([0.2, 0.5, 0.3])
    tv = AcceptanceMath.total_variation_distance(p, q)
    assert abs(tv - 0.0) < 1e-6

    alpha = AcceptanceMath.exact_acceptance_probability(p, q)
    assert abs(alpha - 1.0) < 1e-6


def test_total_variation_distance_disjoint():
    p = torch.tensor([1.0, 0.0])
    q = torch.tensor([0.0, 1.0])
    tv = AcceptanceMath.total_variation_distance(p, q)
    assert abs(tv - 1.0) < 1e-6

    alpha = AcceptanceMath.exact_acceptance_probability(p, q)
    assert abs(alpha - 0.0) < 1e-6


def test_rejection_sample_step_accept():
    # If p has high probability on proposed token, should frequently accept
    p = torch.tensor([0.9, 0.1])
    q = torch.tensor([0.5, 0.5])
    # Seed generator for deterministic test
    torch.manual_seed(42)
    accepted, token = AcceptanceMath.rejection_sample_step(p, q, proposed_token=0)
    assert accepted is True
    assert token == 0


def test_expected_tokens_chain():
    # alpha = 0 -> only 1 token (target fallback)
    assert abs(AcceptanceMath.expected_tokens_chain(0.0, 4) - 1.0) < 1e-5

    # alpha = 1 -> K + 1 tokens
    assert abs(AcceptanceMath.expected_tokens_chain(1.0, 4) - 5.0) < 1e-5

    # alpha = 0.5, K = 2: (1 - 0.5^3) / (1 - 0.5) = (1 - 0.125) / 0.5 = 0.875 / 0.5 = 1.75
    expected = AcceptanceMath.expected_tokens_chain(0.5, 2)
    assert abs(expected - 1.75) < 1e-5


def test_expected_tokens_positional():
    positional = [0.8, 0.6, 0.4]
    # E[tau] = 1 + 0.8 + (0.8 * 0.6) + (0.8 * 0.6 * 0.4) = 1 + 0.8 + 0.48 + 0.192 = 2.472
    res = AcceptanceMath.expected_tokens_chain(0.5, 3, positional_alphas=positional)
    assert abs(res - 2.472) < 1e-5


def test_compute_speedup():
    # expected_tokens = 2.5, K = 3, draft = 3ms, target = 25ms, overhead = 0.5ms
    # spec step time = 3*3 + 25 + 0.5 = 34.5ms
    # baseline time = 2.5 * 25 = 62.5ms
    # speedup = 62.5 / 34.5 ~= 1.811
    speedup, step_ms, tps = AcceptanceMath.compute_speedup(
        expected_tokens=2.5,
        lookahead_depth=3,
        draft_latency_ms=3.0,
        target_latency_ms=25.0,
        verification_overhead_ms=0.5,
    )
    assert abs(speedup - (62.5 / 34.5)) < 1e-3
    assert abs(step_ms - 34.5) < 1e-3
    assert tps > 0.0


def test_find_optimal_lookahead():
    # With fast draft model (2ms) vs slow target (30ms) and alpha=0.8, optimal K should be between 3 and 7
    best_k, peak_speedup = AcceptanceMath.find_optimal_lookahead(
        alpha=0.8,
        draft_latency_ms=2.0,
        target_latency_ms=30.0,
        max_lookahead=8,
    )
    assert 2 <= best_k <= 8
    assert peak_speedup > 1.5
