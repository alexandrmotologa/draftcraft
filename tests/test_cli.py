"""Integration tests for Typer CLI commands."""

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
    result = runner.invoke(app, ["profile", "--alpha", "0.8", "--depth", "3", "--steps", "20"])
    assert result.exit_code == 0
    assert "Speculative Decoding Performance Profile" in result.stdout
    assert "Empirical Wallclock Speedup" in result.stdout


def test_cli_matrix():
    result = runner.invoke(app, ["matrix", "--min-depth", "1", "--max-depth", "3"])
    assert result.exit_code == 0
    assert "Speculative Decoding Matrix" in result.stdout
    assert "Pareto Optimal Configuration Found" in result.stdout


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
