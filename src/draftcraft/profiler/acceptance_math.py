"""Authentic speculative decoding mathematics and speedup formulations."""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


class AcceptanceMath:
    """Rigorous mathematical calculations for speculative decoding."""

    @staticmethod
    def total_variation_distance(p: torch.Tensor, q: torch.Tensor) -> float:
        """Compute Total Variation (TV) distance between target p and draft q.

        TV(p, q) = 0.5 * sum(|p(x) - q(x)|)
        """
        assert p.shape == q.shape, "Distributions must have matching shapes"
        tv = 0.5 * torch.sum(torch.abs(p - q)).item()
        return float(tv)

    @staticmethod
    def exact_acceptance_probability(p: torch.Tensor, q: torch.Tensor) -> float:
        """Calculate theoretical token acceptance probability alpha.

        alpha = sum(min(p(x), q(x))) = 1 - TV(p, q)
        """
        assert p.shape == q.shape, "Distributions must have matching shapes"
        min_probs = torch.minimum(p, q)
        alpha = float(torch.sum(min_probs).item())
        return max(0.0, min(1.0, alpha))

    @staticmethod
    def rejection_sample_step(
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
        proposed_token: int,
    ) -> Tuple[bool, int]:
        """Perform authentic speculative rejection sampling for a single token.

        Leviathan et al. (2023) / Chen et al. (2023):
        Accept with probability min(1, p(x) / q(x)).
        If rejected, resample from max(0, p(x) - q(x)) / norm.

        Args:
            target_probs: Target distribution tensor [V].
            draft_probs: Draft distribution tensor [V].
            proposed_token: Token index proposed by draft model.

        Returns:
            Tuple of (accepted: bool, final_token: int).
        """
        p_val = target_probs[proposed_token].item()
        q_val = draft_probs[proposed_token].item()

        acceptance_ratio = min(1.0, p_val / max(q_val, 1e-12))
        rand_val = torch.rand(1).item()

        if rand_val <= acceptance_ratio:
            return True, proposed_token

        # Rejected: resample from residual distribution
        residual = torch.clamp(target_probs - draft_probs, min=0.0)
        residual_sum = torch.sum(residual).item()

        if residual_sum > 1e-9:
            residual_probs = residual / residual_sum
            resampled_token = int(torch.multinomial(residual_probs, num_samples=1).item())
        else:
            resampled_token = int(torch.multinomial(target_probs, num_samples=1).item())

        return False, resampled_token

    @staticmethod
    def expected_tokens_chain(
        alpha: float,
        lookahead_depth: int,
        positional_alphas: Optional[List[float]] = None,
    ) -> float:
        """Calculate expected accepted tokens per step E[tau].

        For homogeneous alpha:
            E[tau] = (1 - alpha^(K+1)) / (1 - alpha)
        For positional alphas [alpha_1, ..., alpha_K]:
            E[tau] = 1 + sum_{i=1}^K prod_{j=1}^i alpha_j
        """
        if positional_alphas is not None and len(positional_alphas) > 0:
            k = min(lookahead_depth, len(positional_alphas))
            expected = 1.0
            cum_prod = 1.0
            for i in range(k):
                cum_prod *= max(0.0, min(1.0, positional_alphas[i]))
                expected += cum_prod
            return float(expected)

        if alpha >= 0.9999:
            return float(lookahead_depth + 1)
        if alpha <= 0.0001:
            return 1.0

        num = 1.0 - math.pow(alpha, lookahead_depth + 1)
        den = 1.0 - alpha
        return float(num / den)

    @staticmethod
    def expected_tokens_tree(
        alpha: float,
        depth: int,
        tree_width: int,
    ) -> float:
        """Calculate expected tokens per step for tree speculative decoding.

        In tree decoding, multiple candidate branches are verified in parallel.
        The probability that at least one branch of depth d is accepted equals:
        1 - (1 - alpha^d)^w
        """
        if depth <= 0 or tree_width <= 0:
            return 1.0

        expected = 1.0
        for d in range(1, depth + 1):
            prob_chain = math.pow(alpha, d)
            prob_any_accepted = 1.0 - math.pow(max(0.0, 1.0 - prob_chain), tree_width)
            expected += prob_any_accepted
        return float(expected)

    @staticmethod
    def compute_speedup(
        expected_tokens: float,
        lookahead_depth: int,
        draft_latency_ms: float,
        target_latency_ms: float,
        verification_overhead_ms: float = 0.5,
    ) -> Tuple[float, float, float]:
        """Compute theoretical speedup, speculative latency, and throughput.

        Returns:
            Tuple of (speedup_ratio, spec_step_latency_ms, tokens_per_second).
        """
        spec_step_time_ms = (
            lookahead_depth * draft_latency_ms
            + target_latency_ms
            + verification_overhead_ms
        )
        baseline_time_for_same_tokens_ms = expected_tokens * target_latency_ms

        speedup = baseline_time_for_same_tokens_ms / max(spec_step_time_ms, 1e-6)
        tokens_per_sec = (expected_tokens / (spec_step_time_ms / 1000.0))

        return float(speedup), float(spec_step_time_ms), float(tokens_per_sec)

    @classmethod
    def find_optimal_lookahead(
        cls,
        alpha: float,
        draft_latency_ms: float,
        target_latency_ms: float,
        max_lookahead: int = 16,
    ) -> Tuple[int, float]:
        """Find lookahead depth K* that maximizes speculative speedup.

        Returns:
            Tuple of (optimal_depth, peak_speedup).
        """
        best_k = 1
        best_speedup = 0.0

        for k in range(1, max_lookahead + 1):
            expected = cls.expected_tokens_chain(alpha, k)
            speedup, _, _ = cls.compute_speedup(
                expected, k, draft_latency_ms, target_latency_ms
            )
            if speedup > best_speedup:
                best_speedup = speedup
                best_k = k

        return best_k, best_speedup
