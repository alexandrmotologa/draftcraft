"""Unit tests for benchmark report and runtime configuration exporter."""

import json
from pathlib import Path
import pytest
from draftcraft.core.config import ProfilerConfig
from draftcraft.core.types import DomainCategory
from draftcraft.profiler.exporter import BenchmarkExporter
from draftcraft.profiler.matrix_evaluator import MatrixEvaluator
from draftcraft.profiler.vram_calculator import VRAMCalculator


@pytest.fixture
def matrix_result():
    evaluator = MatrixEvaluator(ProfilerConfig(min_lookahead=1, max_lookahead=4))
    return evaluator.evaluate_matrix(domain=DomainCategory.CODE, alpha=0.8)


def test_export_markdown(matrix_result):
    vram = VRAMCalculator.calculate_memory(gpu_capacity_gb=12.0)
    md = BenchmarkExporter.to_markdown(matrix_result, vram_breakdown=vram)
    assert "# DraftCraft Speculative Decoding Evaluation Report" in md
    assert "Peak Acceleration:" in md
    assert "VRAM Memory Budget" in md


def test_export_csv(matrix_result):
    csv_str = BenchmarkExporter.to_csv(matrix_result)
    assert "depth_k,tree_width_w,latency_ratio" in csv_str
    lines = csv_str.strip().split("\n")
    assert len(lines) > 4


def test_export_vllm_config(matrix_result):
    cfg = BenchmarkExporter.to_vllm_config(matrix_result)
    assert cfg["engine"] == "vllm"
    assert "num_speculative_tokens" in cfg
    assert cfg["num_speculative_tokens"] == matrix_result.recommended_depth


def test_save_report_file(tmp_path, matrix_result):
    json_path = tmp_path / "report.json"
    BenchmarkExporter.save_report(json_path, matrix_result)
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "matrix" in data
    assert "vllm_config" in data
