"""Speculative decoding matrix evaluator and Pareto frontier optimization."""

from typing import List, Optional, Tuple
from draftcraft.core.config import ProfilerConfig
from draftcraft.core.types import DomainCategory, MatrixEvaluationResult, MatrixPoint
from draftcraft.profiler.acceptance_math import AcceptanceMath


class MatrixEvaluator:
    """Evaluates multi-dimensional speculative decoding configuration matrix."""

    def __init__(self, config: Optional[ProfilerConfig] = None):
        self.config = config or ProfilerConfig()

    def evaluate_matrix(
        self,
        domain: DomainCategory = DomainCategory.GENERAL,
        alpha: Optional[float] = None,
        draft_latency_ms: Optional[float] = None,
        target_latency_ms: Optional[float] = None,
    ) -> MatrixEvaluationResult:
        """Compute the full configuration matrix and find optimal Pareto trade-offs.

        Args:
            domain: Evaluation domain category.
            alpha: Base acceptance rate (if None, derived from domain default).
            draft_latency_ms: Draft step latency in ms.
            target_latency_ms: Target step latency in ms.

        Returns:
            MatrixEvaluationResult with all matrix points, Pareto points, and recommendations.
        """
        d_latency = draft_latency_ms or self.config.draft_latency_ms
        t_latency = target_latency_ms or self.config.target_latency_ms
        latency_ratio = d_latency / max(t_latency, 1e-6)

        domain_alphas = {
            DomainCategory.GENERAL: 0.74,
            DomainCategory.CODE: 0.82,
            DomainCategory.REASONING: 0.68,
            DomainCategory.FACTUAL_QA: 0.71,
            DomainCategory.CREATIVE: 0.65,
        }
        effective_alpha = alpha if alpha is not None else domain_alphas.get(domain, 0.72)

        points: List[MatrixPoint] = []
        depth_range = range(self.config.min_lookahead, self.config.max_lookahead + 1)
        tree_widths = self.config.tree_widths or [1, 2, 4]

        best_point: Optional[MatrixPoint] = None
        max_speedup = 0.0

        for depth in depth_range:
            for width in tree_widths:
                if width == 1:
                    # Greedy linear chain
                    expected_tokens = AcceptanceMath.expected_tokens_chain(
                        alpha=effective_alpha,
                        lookahead_depth=depth,
                    )
                else:
                    # Tree speculative decoding
                    expected_tokens = AcceptanceMath.expected_tokens_tree(
                        alpha=effective_alpha,
                        depth=depth,
                        tree_width=width,
                    )

                # Compute speedup
                speedup, _, _ = AcceptanceMath.compute_speedup(
                    expected_tokens=expected_tokens,
                    lookahead_depth=depth,
                    draft_latency_ms=d_latency,
                    target_latency_ms=t_latency,
                )

                point = MatrixPoint(
                    lookahead_depth=depth,
                    tree_width=width,
                    latency_ratio=round(latency_ratio, 4),
                    acceptance_rate=round(effective_alpha, 4),
                    expected_tokens=round(expected_tokens, 3),
                    speedup=round(speedup, 3),
                    is_pareto_optimal=False,
                )
                points.append(point)

                if speedup > max_speedup:
                    max_speedup = speedup
                    best_point = point

        # Mark Pareto optimal points (no other point has higher speedup with smaller or equal depth)
        self._calculate_pareto_optimality(points)

        assert best_point is not None, "Evaluation produced no valid matrix points"

        return MatrixEvaluationResult(
            points=points,
            optimal_point=best_point,
            recommended_depth=best_point.lookahead_depth,
            recommended_tree_width=best_point.tree_width,
            peak_speedup=round(max_speedup, 2),
            evaluation_domain=domain,
        )

    @staticmethod
    def _calculate_pareto_optimality(points: List[MatrixPoint]) -> None:
        """Flag Pareto frontier points balancing depth vs speedup."""
        # For each unique depth, the point with highest speedup is candidate
        depth_best: dict[int, MatrixPoint] = {}
        for p in points:
            if p.lookahead_depth not in depth_best or p.speedup > depth_best[p.lookahead_depth].speedup:
                depth_best[p.lookahead_depth] = p

        current_max_speedup = 0.0
        for depth in sorted(depth_best.keys()):
            pt = depth_best[depth]
            if pt.speedup > current_max_speedup:
                pt.is_pareto_optimal = True
                current_max_speedup = pt.speedup
