"""Unit tests for VRAM calculator and hardware presets."""

import pytest
from draftcraft.profiler.vram_calculator import HARDWARE_PRESETS, PrecisionType, VRAMCalculator


def test_vram_calculator_rtx_4090():
    # 8B model with Q4_K_M on 24GB RTX 4090 should fit easily with abundant headroom
    breakdown = VRAMCalculator.calculate_memory(
        target_param_count_billions=8.0,
        target_precision=PrecisionType.Q4_K_M,
        context_len=4096,
        lookahead_k=4,
        gpu_capacity_gb=24.0,
    )
    assert breakdown.fits_in_gpu is True
    assert breakdown.headroom_gb > 10.0
    assert breakdown.target_weights_gb < 5.0
    assert breakdown.draft_head_weights_gb > 0.5
    assert breakdown.max_context_length_tokens > 10000


def test_vram_calculator_fp16_exceed_8gb():
    # 8B model in FP16 weighs ~15GB, will exceed 8GB RTX 4060
    breakdown = VRAMCalculator.calculate_memory(
        target_param_count_billions=8.0,
        target_precision=PrecisionType.FP16,
        context_len=4096,
        gpu_capacity_gb=8.0,
    )
    assert breakdown.fits_in_gpu is False
    assert breakdown.headroom_gb < 0.0


def test_hardware_presets_presence():
    assert "rtx-4090" in HARDWARE_PRESETS
    assert "rtx-4060-8gb" in HARDWARE_PRESETS
    assert "apple-m3-max" in HARDWARE_PRESETS
    preset = HARDWARE_PRESETS["rtx-4090"]
    assert preset.vram_gb == 24.0
    assert preset.bandwidth_gb_s > 900.0
