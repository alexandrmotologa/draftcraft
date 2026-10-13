"""VRAM memory footprint and KV cache allocation calculator for speculative decoding."""

from enum import Enum
from typing import Dict, Optional
from pydantic import BaseModel, Field


class PrecisionType(str, Enum):
    """Numerical precision formats and their average bytes per parameter."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"
    Q4_K_M = "q4_k_m"
    INT4 = "int4"
    Q2_K = "q2_k"


# Average bytes per parameter including quantization metadata (scales, zeros)
PRECISION_BYTES_PER_PARAM: Dict[PrecisionType, float] = {
    PrecisionType.FP32: 4.0,
    PrecisionType.FP16: 2.0,
    PrecisionType.BF16: 2.0,
    PrecisionType.FP8: 1.0,
    PrecisionType.INT8: 1.0,
    PrecisionType.Q4_K_M: 0.5625,  # 4.5 bits/param
    PrecisionType.INT4: 0.5,
    PrecisionType.Q2_K: 0.32,      # 2.56 bits/param
}


class HardwarePreset(BaseModel):
    """Consumer and workstation GPU hardware specifications."""
    name: str
    vram_gb: float
    bandwidth_gb_s: float
    typical_target_latency_ms: float
    typical_draft_latency_ms: float


HARDWARE_PRESETS: Dict[str, HardwarePreset] = {
    "rtx-4090": HardwarePreset(
        name="NVIDIA GeForce RTX 4090",
        vram_gb=24.0,
        bandwidth_gb_s=1008.0,
        typical_target_latency_ms=18.5,
        typical_draft_latency_ms=2.4,
    ),
    "rtx-4080": HardwarePreset(
        name="NVIDIA GeForce RTX 4080 (16GB)",
        vram_gb=16.0,
        bandwidth_gb_s=717.0,
        typical_target_latency_ms=23.0,
        typical_draft_latency_ms=2.9,
    ),
    "rtx-4070-ti": HardwarePreset(
        name="NVIDIA GeForce RTX 4070 Ti (12GB)",
        vram_gb=12.0,
        bandwidth_gb_s=504.0,
        typical_target_latency_ms=27.5,
        typical_draft_latency_ms=3.4,
    ),
    "rtx-4060-ti-16gb": HardwarePreset(
        name="NVIDIA GeForce RTX 4060 Ti (16GB)",
        vram_gb=16.0,
        bandwidth_gb_s=288.0,
        typical_target_latency_ms=36.0,
        typical_draft_latency_ms=4.8,
    ),
    "rtx-4060-8gb": HardwarePreset(
        name="NVIDIA GeForce RTX 4060 (8GB)",
        vram_gb=8.0,
        bandwidth_gb_s=272.0,
        typical_target_latency_ms=38.0,
        typical_draft_latency_ms=5.0,
    ),
    "rtx-3060-12gb": HardwarePreset(
        name="NVIDIA GeForce RTX 3060 (12GB)",
        vram_gb=12.0,
        bandwidth_gb_s=360.0,
        typical_target_latency_ms=42.0,
        typical_draft_latency_ms=5.5,
    ),
    "apple-m3-max": HardwarePreset(
        name="Apple Silicon M3 Max (36GB Unified)",
        vram_gb=36.0,
        bandwidth_gb_s=300.0,
        typical_target_latency_ms=28.0,
        typical_draft_latency_ms=3.6,
    ),
    "apple-m3-pro": HardwarePreset(
        name="Apple Silicon M3 Pro (18GB Unified)",
        vram_gb=18.0,
        bandwidth_gb_s=150.0,
        typical_target_latency_ms=44.0,
        typical_draft_latency_ms=5.8,
    ),
}


class VRAMBreakdown(BaseModel):
    """Detailed memory allocation breakdown."""
    target_weights_gb: float
    draft_head_weights_gb: float
    kv_cache_baseline_gb: float
    kv_cache_speculative_overhead_gb: float
    total_kv_cache_gb: float
    activations_and_cuda_overhead_gb: float
    total_vram_gb: float
    gpu_vram_capacity_gb: float
    headroom_gb: float
    fits_in_gpu: bool
    max_context_length_tokens: int
    recommendation: str


class VRAMCalculator:
    """Computes exact VRAM footprints and evaluates consumer GPU compatibility."""

    @classmethod
    def calculate_memory(
        cls,
        target_param_count_billions: float = 8.0,
        target_precision: PrecisionType = PrecisionType.Q4_K_M,
        target_layers: int = 32,
        target_hidden_dim: int = 4096,
        target_heads: int = 32,
        head_hidden_dim: int = 1024,
        head_layers: int = 2,
        vocab_size: int = 128256,
        context_len: int = 4096,
        batch_size: int = 1,
        lookahead_k: int = 4,
        tree_width: int = 1,
        gpu_capacity_gb: float = 12.0,
    ) -> VRAMBreakdown:
        """Calculate exact VRAM usage for target LLM and speculative draft head.

        Args:
            target_param_count_billions: Total parameters of the base model (e.g. 8.0 for 8B).
            target_precision: Quantization/precision of base model weights.
            target_layers: Number of transformer layers in base model.
            target_hidden_dim: Hidden dimension of base model.
            target_heads: Number of attention heads (or KV heads if GQA).
            head_hidden_dim: Draft head hidden dimension.
            head_layers: Number of transformer decoder layers in draft head.
            vocab_size: Target vocabulary size.
            context_len: Current active context sequence length in tokens.
            batch_size: Active generation batch size.
            lookahead_k: Speculative lookahead depth K.
            tree_width: Tree branching width W (1 for greedy chain).
            gpu_capacity_gb: Total physical GPU VRAM in GB.

        Returns:
            VRAMBreakdown object with component memory metrics and budget feasibility.
        """
        # 1. Target model weights memory
        bytes_per_param = PRECISION_BYTES_PER_PARAM.get(target_precision, 2.0)
        target_weights_bytes = (target_param_count_billions * 1e9) * bytes_per_param
        target_weights_gb = target_weights_bytes / (1024 ** 3)

        # 2. EAGLE Draft Head weights (always stored in FP16/BF16 = 2 bytes)
        # Components:
        # fc_in: target_hidden * head_hidden
        # embed: vocab_size * head_hidden
        # fusion: (2 * head_hidden) * head_hidden
        # layers * (qkv_out: 4 * head_hidden^2 + swiglu: 3 * head_hidden * (2 * head_hidden))
        # fc_out: head_hidden * target_hidden
        # lm_head: target_hidden * vocab_size
        head_params = (
            target_hidden_dim * head_hidden_dim +
            vocab_size * head_hidden_dim +
            (2 * head_hidden_dim) * head_hidden_dim +
            head_layers * (4 * (head_hidden_dim ** 2) + 6 * (head_hidden_dim ** 2)) +
            head_hidden_dim * target_hidden_dim +
            target_hidden_dim * vocab_size
        )
        head_weights_bytes = head_params * 2.0  # FP16
        draft_weights_gb = head_weights_bytes / (1024 ** 3)

        # 3. KV Cache Memory (Key + Value, 2 elements per token, 2 bytes FP16 per elem)
        # KV cache per token: 2 * layers * heads * (hidden_dim // heads) * 2 bytes = 4 * layers * hidden_dim bytes
        head_dim_size = target_hidden_dim // target_heads
        bytes_per_token_kv = 2 * target_layers * target_heads * head_dim_size * 2  # FP16 KV cache

        baseline_kv_bytes = bytes_per_token_kv * context_len * batch_size
        baseline_kv_gb = baseline_kv_bytes / (1024 ** 3)

        # Speculative decoding KV cache expansion:
        # In tree or chain speculative decoding, the verification pass buffers lookahead slots:
        # Chain needs K additional slots; Tree needs K * tree_width candidate branches
        spec_extra_tokens = lookahead_k * tree_width
        spec_kv_overhead_bytes = bytes_per_token_kv * spec_extra_tokens * batch_size
        spec_kv_overhead_gb = spec_kv_overhead_bytes / (1024 ** 3)
        total_kv_gb = baseline_kv_gb + spec_kv_overhead_gb

        # 4. Activation buffer & CUDA runtime driver overhead (approx 0.6 GB baseline + 0.2 GB for verification pass)
        activations_gb = 0.6 + (0.05 * spec_extra_tokens)

        # 5. Total VRAM
        total_vram_gb = target_weights_gb + draft_weights_gb + total_kv_gb + activations_gb
        headroom_gb = gpu_capacity_gb - total_vram_gb
        fits = headroom_gb >= 0.0

        # Calculate max theoretical context length within remaining memory
        available_for_kv_gb = gpu_capacity_gb - (target_weights_gb + draft_weights_gb + activations_gb)
        if available_for_kv_gb > 0:
            bytes_available_for_kv = available_for_kv_gb * (1024 ** 3)
            max_context = int(bytes_available_for_kv // max(bytes_per_token_kv * batch_size, 1))
        else:
            max_context = 0

        # Recommendation note
        if fits and headroom_gb >= 2.0:
            rec = f"Comfortable fit: {headroom_gb:.2f} GB headroom available. Up to {max_context:,} context tokens supported."
        elif fits:
            rec = f"Tight fit: {headroom_gb:.2f} GB headroom remaining. Recommend context limit <= {context_len} tokens."
        else:
            rec = f"Exceeds VRAM budget by {abs(headroom_gb):.2f} GB. Consider quantized weights (Q4_K_M) or reducing context length."

        return VRAMBreakdown(
            target_weights_gb=round(target_weights_gb, 3),
            draft_head_weights_gb=round(draft_weights_gb, 3),
            kv_cache_baseline_gb=round(baseline_kv_gb, 3),
            kv_cache_speculative_overhead_gb=round(spec_kv_overhead_gb, 3),
            total_kv_cache_gb=round(total_kv_gb, 3),
            activations_and_cuda_overhead_gb=round(activations_gb, 3),
            total_vram_gb=round(total_vram_gb, 3),
            gpu_vram_capacity_gb=float(gpu_capacity_gb),
            headroom_gb=round(headroom_gb, 3),
            fits_in_gpu=fits,
            max_context_length_tokens=max_context,
            recommendation=rec,
        )
