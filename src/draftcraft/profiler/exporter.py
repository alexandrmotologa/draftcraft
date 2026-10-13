"""Production runtime configuration and benchmark report exporter."""

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from draftcraft.core.types import MatrixEvaluationResult
from draftcraft.profiler.temperature_profiler import TemperatureProfileResult
from draftcraft.profiler.vram_calculator import VRAMBreakdown


class BenchmarkExporter:
    """Exports benchmark matrices and runtime configurations to Markdown, JSON, and CSV."""

    @staticmethod
    def to_markdown(
        matrix_result: MatrixEvaluationResult,
        vram_breakdown: Optional[VRAMBreakdown] = None,
        temp_result: Optional[TemperatureProfileResult] = None,
    ) -> str:
        """Generate a GitHub Markdown summary report."""
        lines = [
            "# DraftCraft Speculative Decoding Evaluation Report",
            "",
            f"**Domain Category:** {matrix_result.evaluation_domain.value.upper()}",
            f"**Peak Acceleration:** {matrix_result.peak_speedup:.2f}x speedup",
            f"**Recommended Configuration:** Lookahead Depth K* = {matrix_result.recommended_depth}, Tree Width W* = {matrix_result.recommended_tree_width}",
            "",
            "## Speculative Matrix Summary",
            "",
            "| Depth (K) | Tree Width (W) | Latency Ratio | Exp. Tokens | Speedup | Pareto Status |",
            "|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]

        for pt in matrix_result.points:
            pareto_str = "**OPTIMAL**" if pt.is_pareto_optimal else "-"
            lines.append(
                f"| {pt.lookahead_depth} | {pt.tree_width} | {pt.latency_ratio:.3f} | {pt.expected_tokens:.2f} | {pt.speedup:.2f}x | {pareto_str} |"
            )

        if vram_breakdown is not None:
            lines.extend([
                "",
                "## VRAM Memory Budget",
                "",
                f"- **Target Weights:** {vram_breakdown.target_weights_gb:.2f} GB",
                f"- **Draft Head Weights:** {vram_breakdown.draft_head_weights_gb:.2f} GB",
                f"- **Total KV Cache:** {vram_breakdown.total_kv_cache_gb:.2f} GB",
                f"- **Total Allocation:** {vram_breakdown.total_vram_gb:.2f} GB / {vram_breakdown.gpu_vram_capacity_gb:.1f} GB",
                f"- **Headroom:** {vram_breakdown.headroom_gb:.2f} GB ({'Fits within budget' if vram_breakdown.fits_in_gpu else 'Exceeds budget'})",
                f"- **Recommendation:** {vram_breakdown.recommendation}",
            ])

        if temp_result is not None:
            lines.extend([
                "",
                "## Temperature Sensitivity",
                "",
                f"- **Peak Speedup:** {temp_result.peak_speedup:.2f}x at T={temp_result.peak_temperature:.1f}",
                f"- **Break-even Threshold:** {f'T={temp_result.break_even_temperature:.2f}' if temp_result.break_even_temperature else 'None (viable across tested range)'}",
                f"- **Guidance:** {temp_result.recommendation}",
            ])

        return "\n".join(lines) + "\n"

    @staticmethod
    def to_csv(matrix_result: MatrixEvaluationResult) -> str:
        """Serialize matrix evaluation points to CSV string."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["depth_k", "tree_width_w", "latency_ratio", "acceptance_rate", "expected_tokens", "speedup", "is_pareto_optimal"])
        for pt in matrix_result.points:
            writer.writerow([
                pt.lookahead_depth,
                pt.tree_width,
                pt.latency_ratio,
                pt.acceptance_rate,
                pt.expected_tokens,
                pt.speedup,
                pt.is_pareto_optimal,
            ])
        return output.getvalue()

    @staticmethod
    def to_vllm_config(
        matrix_result: MatrixEvaluationResult,
        target_model_name: str = "meta-llama/Llama-3-8B-Instruct",
        draft_model_or_head: str = "checkpoints/final_draft_head.pt",
    ) -> Dict[str, Any]:
        """Generate production vLLM and SGLang speculative decoding configuration."""
        opt = matrix_result.optimal_point
        return {
            "model": target_model_name,
            "speculative_model": draft_model_or_head,
            "num_speculative_tokens": opt.lookahead_depth,
            "speculative_draft_tensor_parallel_size": 1,
            "speculative_max_model_len": 4096,
            "speculative_strategy": "tree" if opt.tree_width > 1 else "chain",
            "speculative_tree_branching_factor": opt.tree_width,
            "expected_speedup": opt.speedup,
            "engine": "vllm",
        }

    @classmethod
    def save_report(
        cls,
        output_path: Union[str, Path],
        matrix_result: MatrixEvaluationResult,
        vram_breakdown: Optional[VRAMBreakdown] = None,
        temp_result: Optional[TemperatureProfileResult] = None,
    ) -> None:
        """Save report to file based on extension (.md, .json, .csv)."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix.lower()

        if ext == ".md":
            content = cls.to_markdown(matrix_result, vram_breakdown, temp_result)
            p.write_text(content, encoding="utf-8")
        elif ext == ".csv":
            content = cls.to_csv(matrix_result)
            p.write_text(content, encoding="utf-8")
        elif ext == ".json":
            payload = {
                "matrix": matrix_result.model_dump(),
                "vram": vram_breakdown.model_dump() if vram_breakdown else None,
                "temperature": temp_result.model_dump() if temp_result else None,
                "vllm_config": cls.to_vllm_config(matrix_result),
            }
            p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            raise ValueError(f"Unsupported export format: {ext} (supported: .md, .csv, .json)")
