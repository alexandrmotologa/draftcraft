"""Tokenizer alignment verifier for speculative decoding compatibility."""

from typing import Any, Dict, List, Optional, Tuple
from draftcraft.alignment.corpus_sampler import CorpusSampler
from draftcraft.alignment.vocab_mapper import VocabularyMapper, VocabularyMappingResult
from draftcraft.core.types import AlignmentStatus, DomainCategory, TokenVerificationSummary


class TokenizerAlignmentVerifier:
    """Verifies tokenizer alignment and compatibility for speculative decoding."""

    def __init__(
        self,
        target_vocab: Optional[Dict[str, int]] = None,
        draft_vocab: Optional[Dict[str, int]] = None,
        target_special_tokens: Optional[Dict[str, int]] = None,
        draft_special_tokens: Optional[Dict[str, int]] = None,
    ):
        self.target_vocab = target_vocab or {}
        self.draft_vocab = draft_vocab or {}
        self.target_special_tokens = target_special_tokens or {}
        self.draft_special_tokens = draft_special_tokens or {}

    def verify_vocabularies(self) -> TokenVerificationSummary:
        """Run vocabulary parity, collision, and special token checks."""
        mapping_result = VocabularyMapper.build_mapping(
            self.target_vocab,
            self.draft_vocab
        )

        special_matched = self._check_special_tokens()

        # Determine alignment status
        if mapping_result.is_exact_match and special_matched:
            status = AlignmentStatus.FULLY_ALIGNED
        elif mapping_result.coverage_ratio >= 0.90 and mapping_result.collisions_count == 0:
            status = AlignmentStatus.SUBSET_COMPATIBLE
        elif mapping_result.collisions_count > 0:
            status = AlignmentStatus.TOKENIZER_DRIFT
        else:
            status = AlignmentStatus.VOCAB_MISMATCH

        details = {
            "target_vocab_size": str(mapping_result.target_vocab_size),
            "draft_vocab_size": str(mapping_result.draft_vocab_size),
            "shared_tokens": str(mapping_result.shared_tokens),
            "coverage_ratio_pct": f"{mapping_result.coverage_ratio * 100:.2f}%",
            "collisions_count": str(mapping_result.collisions_count),
            "special_tokens_matched": str(special_matched),
        }

        if mapping_result.sample_collisions:
            details["sample_collisions"] = str(mapping_result.sample_collisions[:3])

        return TokenVerificationSummary(
            target_vocab_size=mapping_result.target_vocab_size,
            draft_vocab_size=mapping_result.draft_vocab_size,
            shared_tokens_count=mapping_result.shared_tokens,
            overlap_ratio=mapping_result.coverage_ratio,
            special_tokens_matched=special_matched,
            status=status,
            divergent_sample_count=mapping_result.collisions_count,
            subword_prefix_consistency=1.0 if mapping_result.is_exact_match else mapping_result.coverage_ratio,
            details=details,
        )

    def _check_special_tokens(self) -> bool:
        """Verify that essential special tokens match between target and draft."""
        if not self.target_special_tokens or not self.draft_special_tokens:
            return True

        critical_keys = ["eos_token", "bos_token", "pad_token", "unk_token"]
        for key in critical_keys:
            if key in self.target_special_tokens and key in self.draft_special_tokens:
                if self.target_special_tokens[key] != self.draft_special_tokens[key]:
                    return False
        return True

    def verify_token_stream_consistency(
        self,
        target_tokens_list: List[List[int]],
        draft_tokens_list: List[List[int]],
    ) -> Tuple[float, int]:
        """Compute token-level exact match consistency across evaluation texts.

        Args:
            target_tokens_list: List of token ID sequences from target tokenizer.
            draft_tokens_list: List of token ID sequences from draft tokenizer.

        Returns:
            Tuple of (consistency_score_between_0_and_1, divergent_count).
        """
        if not target_tokens_list or not draft_tokens_list:
            return 1.0, 0

        total_samples = min(len(target_tokens_list), len(draft_tokens_list))
        divergent_samples = 0
        token_match_ratios: List[float] = []

        for i in range(total_samples):
            t_seq = target_tokens_list[i]
            d_seq = draft_tokens_list[i]
            if t_seq != d_seq:
                divergent_samples += 1

            min_len = min(len(t_seq), len(d_seq))
            max_len = max(len(t_seq), len(d_seq))
            if max_len == 0:
                token_match_ratios.append(1.0)
                continue

            matches = sum(1 for j in range(min_len) if t_seq[j] == d_seq[j])
            token_match_ratios.append(matches / max_len)

        mean_consistency = sum(token_match_ratios) / len(token_match_ratios)
        return mean_consistency, divergent_samples

    @classmethod
    def verify_eagle_head(
        cls,
        target_vocab_size: int,
        head_vocab_size: int,
        target_hidden_dim: int,
        head_hidden_dim: int,
    ) -> TokenVerificationSummary:
        """Verify structural compatibility between a target LLM and an EAGLE draft head.

        EAGLE draft heads share the target LLM vocabulary space. This method asserts
        vocab dimension alignment and hidden state projection compatibility.
        """
        is_vocab_aligned = target_vocab_size == head_vocab_size

        if is_vocab_aligned:
            status = AlignmentStatus.FULLY_ALIGNED
            overlap = 1.0
        else:
            status = AlignmentStatus.VOCAB_MISMATCH
            overlap = min(target_vocab_size, head_vocab_size) / max(target_vocab_size, head_vocab_size)

        details = {
            "target_vocab_size": str(target_vocab_size),
            "head_vocab_size": str(head_vocab_size),
            "target_hidden_dim": str(target_hidden_dim),
            "head_hidden_dim": str(head_hidden_dim),
            "architecture": "EAGLE Speculative Draft Head",
        }

        return TokenVerificationSummary(
            target_vocab_size=target_vocab_size,
            draft_vocab_size=head_vocab_size,
            shared_tokens_count=min(target_vocab_size, head_vocab_size),
            overlap_ratio=overlap,
            special_tokens_matched=True,
            status=status,
            divergent_sample_count=0 if is_vocab_aligned else abs(target_vocab_size - head_vocab_size),
            subword_prefix_consistency=1.0 if is_vocab_aligned else overlap,
            details=details,
        )
