"""Domain evaluation corpora sampler for tokenizer and speculative profiling."""

from typing import Dict, List
from draftcraft.core.types import DomainCategory

_DOMAIN_CORPUS: Dict[DomainCategory, List[str]] = {
    DomainCategory.GENERAL: [
        "Speculative decoding accelerates large language model inference without altering the output distribution.",
        "A smaller draft model generates candidate tokens in sequence, which the target model verifies in parallel.",
        "Consumer grade GPUs often have enough memory bandwidth to run lightweight draft heads alongside quantized models.",
        "By amortizing memory access across multiple speculative tokens, inference throughput increases substantially.",
        "Modern neural network architectures benefit from batching memory transfers during auto-regressive generation.",
    ],
    DomainCategory.CODE: [
        "def compute_acceptance_rate(accepted: int, proposed: int) -> float:\n    return accepted / max(proposed, 1)",
        "class DraftDecoder(torch.nn.Module):\n    def __init__(self, hidden_dim: int):\n        super().__init__()",
        "SELECT user_id, COUNT(token_id) as total_tokens FROM inference_logs WHERE latency_ms < 15.0 GROUP BY user_id;",
        "async def stream_tokens(generator: AsyncIterator[str]) -> None:\n    async for token in generator:\n        yield token",
        '{"model": "llama-3-8b", "temperature": 0.7, "max_tokens": 128, "speculation": {"depth": 4, "strategy": "eagle"}}',
    ],
    DomainCategory.REASONING: [
        "Premise 1: If acceptance rate alpha is 0.75 and lookahead depth is 4, expected accepted tokens equal (1 - 0.75^5) / (1 - 0.75).",
        "Step 1: Calculate cumulative probability product across consecutive draft tokens.\nStep 2: Reject on first failure.",
        "Let draft latency be d = 3.5ms and target verification latency be t = 28ms. The speedup ratio is tau * t / (d * K + t).",
        "To maximize tokens per second, find the root derivative of speedup S(K) with respect to lookahead depth K.",
        "Because verification evaluates candidate tokens in a single forward pass, KV cache reuse is mathematically exact.",
    ],
    DomainCategory.FACTUAL_QA: [
        "Question: What is EAGLE speculative decoding?\nAnswer: EAGLE extrapolates features to predict future hidden states.",
        "Question: Does speculative decoding alter the final sampling distribution?\nAnswer: No, rejection sampling preserves it exactly.",
        "Question: What determines speculative speedup?\nAnswer: Token acceptance rate and the ratio of draft latency to target latency.",
        "Question: What happens when a draft token is rejected?\nAnswer: The token is resampled from the adjusted target probability distribution.",
        "Question: How many transformer layers are in a standard EAGLE draft head?\nAnswer: Typically 1 to 2 transformer decoder layers.",
    ],
    DomainCategory.CREATIVE: [
        "The server rack hummed quietly in the basement, its cooling fans pulsing in rhythm with incoming requests.",
        "Through the quiet streets of the harbor city, fog rolled over cobblestones like vapor from liquid nitrogen.",
        "Lines of code flickered across the curved monitor, tracing paths through multidimensional embedding spaces.",
        "In the quiet hours before dawn, the telemetry dashboard showed acceptance rates climbing toward steady equilibrium.",
        "The observatory dome opened with a low mechanical rumble, framing a crisp field of distant binary stars.",
    ],
}


class CorpusSampler:
    """Provides categorized prompt corpora for tokenizer validation and profiling."""

    @classmethod
    def get_samples(cls, domain: DomainCategory, limit: int = 5) -> List[str]:
        """Retrieve sample prompts for the given domain."""
        samples = _DOMAIN_CORPUS.get(domain, _DOMAIN_CORPUS[DomainCategory.GENERAL])
        return samples[:limit]

    @classmethod
    def get_all_domains(cls) -> Dict[DomainCategory, List[str]]:
        """Retrieve all domain corpora."""
        return _DOMAIN_CORPUS.copy()
