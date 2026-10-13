"""Temperature sensitivity profiler for speculative decoding across sampling temperatures."""

from typing import Dict, List, Optional
from pydantic import BaseModel
import torch
import torch.nn.functional as F

from draftcraft.profiler.acceptance_math import AcceptanceMath


class TemperaturePoint(BaseModel):
    """Speculative performance metric at a specific temperature setting."""
    temperature: float
    acceptance_rate: float
    expected_tokens: float
    speedup: float
    is_viable: bool


class TemperatureProfileResult(BaseModel):
    """Full temperature sweep results and critical thresholds."""
    points: List[TemperaturePoint]
    break_even_temperature: Optional[float]
    peak_speedup: float
    peak_temperature: float
    recommendation: str


class TemperatureProfiler:
    """Profiles acceptance rate degradation and speedup across temperature values."""

    @classmethod
    def profile_temperature_curve(
        cls,
        base_greedy_alpha: float = 0.88,
        lookahead_depth: int = 4,
        draft_latency_ms: float = 3.5,
        target_latency_ms: float = 28.0,
        temperatures: Optional[List[float]] = None,
        entropy_divergence_slope: float = 0.22,
    ) -> TemperatureProfileResult:
        """Evaluate speculative decoding across temperature values from greedy to high entropy.

        Args:
            base_greedy_alpha: Acceptance rate at T=0.0 (top-1 argmax agreement).
            lookahead_depth: Speculative lookahead depth K.
            draft_latency_ms: Latency per draft token in ms.
            target_latency_ms: Latency per target token in ms.
            temperatures: List of temperatures to evaluate. Default: [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0].
            entropy_divergence_slope: Sensitivity of acceptance decay to temperature increase.

        Returns:
            TemperatureProfileResult containing the sweep curve and break-even analysis.
        """
        temps = temperatures or [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0]
        points: List[TemperaturePoint] = []

        peak_speedup = 0.0
        peak_temp = 0.0
        break_even_t: Optional[float] = None

        for t in temps:
            if t <= 0.01:
                # Greedy mode: highest acceptance rate
                alpha = base_greedy_alpha
            else:
                # Softmax smoothing increases sampling variance and lowers effective alpha
                decay = entropy_divergence_slope * (t ** 0.85)
                alpha = max(0.15, min(base_greedy_alpha, base_greedy_alpha - decay))

            expected_tokens = AcceptanceMath.expected_tokens_chain(alpha, lookahead_depth)
            speedup, _, _ = AcceptanceMath.compute_speedup(
                expected_tokens=expected_tokens,
                lookahead_depth=lookahead_depth,
                draft_latency_ms=draft_latency_ms,
                target_latency_ms=target_latency_ms,
            )

            is_viable = speedup > 1.0
            if not is_viable and break_even_t is None and t > 0.0:
                break_even_t = t

            if speedup > peak_speedup:
                peak_speedup = speedup
                peak_temp = t

            points.append(TemperaturePoint(
                temperature=t,
                acceptance_rate=round(alpha, 4),
                expected_tokens=round(expected_tokens, 2),
                speedup=round(speedup, 2),
                is_viable=is_viable,
            ))

        # Recommendation
        if break_even_t is not None:
            rec = f"Speculative acceleration viable below T={break_even_t:.1f}. Beyond this temperature, sampling entropy reduces speedup below 1.0x."
        else:
            rec = f"Speculative acceleration viable across all tested temperatures up to T={max(temps):.1f} (Peak {peak_speedup:.2f}x at T={peak_temp:.1f})."

        return TemperatureProfileResult(
            points=points,
            break_even_temperature=break_even_t,
            peak_speedup=round(peak_speedup, 2),
            peak_temperature=peak_temp,
            recommendation=rec,
        )
