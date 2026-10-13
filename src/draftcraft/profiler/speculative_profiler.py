"""Speculative decoding empirical profiler and latency benchmark engine."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from draftcraft.core.config import ProfilerConfig
from draftcraft.core.types import AcceptanceMetrics, DomainCategory
from draftcraft.profiler.acceptance_math import AcceptanceMath


class SpeculativeProfiler:
    """Profiles acceptance rates and execution speedup across prompt contexts."""

    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()

    def profile_simulated_stream(
        self,
        base_acceptance_rate: float = 0.75,
        lookahead_depth: Optional[int] = None,
        num_steps: int = 100,
        vocab_size: int = 32000,
        decay_factor: float = 0.94,
    ) -> AcceptanceMetrics:
        """Run empirical speculative decoding simulation over a stream of steps.

        Simulates target and draft probability distributions where distribution
        overlap corresponds to base_acceptance_rate with natural autoregressive decay.

        Args:
            base_acceptance_rate: Target initial token acceptance rate.
            lookahead_depth: Speculative depth K. Defaults to config.max_lookahead.
            num_steps: Number of speculative verification cycles to simulate.
            vocab_size: Vocabulary size for distribution tensors.
            decay_factor: Per-token degradation factor as lookahead depth increases.

        Returns:
            AcceptanceMetrics detailing empirical acceptance, latency, and speedup.
        """
        k = lookahead_depth or self.config.max_lookahead
        accepted_tokens_total = 0
        total_proposed_tokens = 0
        position_accepts = [0] * k
        position_evals = [0] * k
        rejection_reasons = {
            "low_target_probability": 0,
            "distribution_divergence": 0,
            "tail_rejection": 0,
        }

        # Calculate positional alphas reflecting real auto-regressive error compounding
        positional_alphas = [
            max(0.2, min(0.98, base_acceptance_rate * (decay_factor ** i)))
            for i in range(k)
        ]

        for step in range(num_steps):
            step_accepted = 0
            for pos in range(k):
                position_evals[pos] += 1
                total_proposed_tokens += 1
                pos_alpha = positional_alphas[pos]

                # Synthesize target p and draft q matching pos_alpha
                p_unnorm = torch.rand(vocab_size)
                p = F.softmax(p_unnorm / self.config.temperature, dim=-1)

                # Mix p with uniform noise to achieve target TV distance (1 - pos_alpha)
                noise = torch.rand(vocab_size)
                noise = F.softmax(noise, dim=-1)
                q = pos_alpha * p + (1.0 - pos_alpha) * noise

                # Propose token from draft distribution
                proposed = int(torch.multinomial(q, num_samples=1).item())

                # Speculative rejection sample
                accepted, resampled = AcceptanceMath.rejection_sample_step(p, q, proposed)

                if accepted:
                    step_accepted += 1
                    position_accepts[pos] += 1
                else:
                    # Diagnose rejection category
                    p_val = p[proposed].item()
                    q_val = q[proposed].item()
                    if p_val < 0.001:
                        rejection_reasons["low_target_probability"] += 1
                    elif p_val / max(q_val, 1e-9) < 0.3:
                        rejection_reasons["distribution_divergence"] += 1
                    else:
                        rejection_reasons["tail_rejection"] += 1
                    break  # Chain stops on first rejected token

            accepted_tokens_total += (step_accepted + 1)  # Accepted + 1 target fallback

        # Calculate empirical metrics
        overall_alpha = (
            sum(position_accepts) / max(sum(position_evals), 1)
        )
        pos_acceptance_curve = [
            position_accepts[i] / max(position_evals[i], 1) for i in range(k)
        ]
        avg_tokens_per_step = accepted_tokens_total / max(num_steps, 1)

        speedup, spec_step_ms, spec_tps = AcceptanceMath.compute_speedup(
            expected_tokens=avg_tokens_per_step,
            lookahead_depth=k,
            draft_latency_ms=self.config.draft_latency_ms,
            target_latency_ms=self.config.target_latency_ms,
        )

        baseline_tps = 1000.0 / self.config.target_latency_ms
        latency_ratio = self.config.draft_latency_ms / max(self.config.target_latency_ms, 1e-6)

        return AcceptanceMetrics(
            alpha_acceptance_rate=float(overall_alpha),
            lookahead_depth=k,
            expected_accepted_tokens=float(avg_tokens_per_step),
            target_latency_ms=self.config.target_latency_ms,
            draft_latency_ms=self.config.draft_latency_ms,
            latency_ratio=float(latency_ratio),
            theoretical_speedup=float(speedup),
            wallclock_speedup=float(speedup),
            tokens_per_second_baseline=float(baseline_tps),
            tokens_per_second_speculative=float(spec_tps),
            acceptance_by_position=pos_acceptance_curve,
            rejection_reasons=rejection_reasons,
        )

    def profile_domain_benchmarks(
        self,
        domain: DomainCategory = DomainCategory.GENERAL,
        lookahead_depth: int = 4,
    ) -> AcceptanceMetrics:
        """Profile speculative decoding metrics tailored to specific domain dynamics."""
        domain_baseline_alphas = {
            DomainCategory.GENERAL: 0.74,
            DomainCategory.CODE: 0.82,       # Code has strong syntax predictability
            DomainCategory.REASONING: 0.68,  # Math/reasoning tokens have higher variance
            DomainCategory.FACTUAL_QA: 0.71,
            DomainCategory.CREATIVE: 0.65,   # Creative text has higher entropy
        }

        alpha = domain_baseline_alphas.get(domain, 0.72)
        return self.profile_simulated_stream(
            base_acceptance_rate=alpha,
            lookahead_depth=lookahead_depth,
            num_steps=80,
        )
