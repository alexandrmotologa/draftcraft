"""Unit tests for temperature sensitivity profiler."""

import pytest
from draftcraft.profiler.temperature_profiler import TemperatureProfiler


def test_temperature_profiler_greedy_peak():
    res = TemperatureProfiler.profile_temperature_curve(
        base_greedy_alpha=0.88,
        lookahead_depth=4,
        draft_latency_ms=3.5,
        target_latency_ms=28.0,
    )
    # Peak speedup should occur at T=0.0 (greedy)
    assert res.peak_temperature == 0.0
    assert res.peak_speedup > 2.0
    assert len(res.points) == 7

    # As temperature rises, acceptance rate should decrease
    p0 = res.points[0]
    p_last = res.points[-1]
    assert p0.acceptance_rate > p_last.acceptance_rate
    assert p0.speedup > p_last.speedup


def test_temperature_profiler_break_even():
    # If base greedy alpha is very low and draft latency is unfavorable, it will break even early
    res = TemperatureProfiler.profile_temperature_curve(
        base_greedy_alpha=0.45,
        lookahead_depth=4,
        draft_latency_ms=6.0,
        target_latency_ms=20.0,
    )
    # Some point will have is_viable False
    unviable_count = sum(1 for p in res.points if not p.is_viable)
    assert unviable_count > 0
