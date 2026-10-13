"""Integration tests for Typer CLI commands."""

import json
from pathlib import Path
import pytest
from typer.testing import CliRunner
from draftcraft.cli.main import app

runner = CliRunner()


def test_cli_info():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "DraftCraft System Environment" in result.stdout
    assert "PyTorch Version" in result.stdout


def test_cli_check_tokenizer():
    result = runner.invoke(app, ["check-tokenizer", "--target-size", "32000", "--draft-size", "32000"])
    assert result.exit_code == 0
    assert "FULLY_ALIGNED" in result.stdout
    assert "100.00%" in result.stdout


def test_cli_profile():
    result = runner.invoke(app, ["profile", "--alpha", "0.8", "--depth", "3", "--steps", "20", "--preset", "rtx-4090"])
    assert result.exit_code == 0
    assert "Speculative Decoding Performance Profile" in result.stdout
    assert "Empirical Wallclock Speedup" in result.stdout


def test_cli_matrix():
    result = runner.invoke(app, ["matrix", "--min-depth", "1", "--max-depth", "3", "--preset", "rtx-4060-8gb"])
    assert result.exit_code == 0
    assert "Speculative Decoding Matrix" in result.stdout
    assert "Pareto Optimal Configuration Found" in result.stdout


def test_cli_vram():
    result = runner.invoke(app, ["vram", "--preset", "rtx-4090", "--params", "8.0"])
    assert result.exit_code == 0
    assert "VRAM & KV Cache Budget Analyzer" in result.stdout
    assert "BUDGET VERIFIED" in result.stdout


def test_cli_temp_profile():
    result = runner.invoke(app, ["temp-profile", "--alpha", "0.85", "--depth", "3"])
    assert result.exit_code == 0
    assert "Temperature Sensitivity Sweep" in result.stdout
    assert "Peak Acceleration:" in result.stdout


def test_cli_export(tmp_path):
    report_file = tmp_path / "cli_report.md"
    result = runner.invoke(app, ["export", "--output", str(report_file), "--alpha", "0.8"])
    assert result.exit_code == 0
    assert report_file.exists()
    assert "Successfully exported benchmark report" in result.stdout


def test_cli_train(tmp_path):
    out_dir = str(tmp_path / "cli_ckpts")
    result = runner.invoke(app, [
        "train",
        "--epochs", "1",
        "--samples", "16",
        "--batch-size", "4",
        "--hidden-dim", "64",
        "--target-dim", "128",
        "--vocab-size", "200",
        "--out-dir", out_dir,
    ])
    assert result.exit_code == 0
    assert "Training Completed Successfully" in result.stdout
