"""Vocabulary mapping and token collision analyzer for target and draft models."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class VocabularyMappingResult:
    """Summary of mapping between target and draft token vocabularies."""
    target_vocab_size: int
    draft_vocab_size: int
    shared_tokens: int
    coverage_ratio: float
    is_exact_match: bool
    draft_to_target_map: Dict[int, int] = field(default_factory=dict)
    target_to_draft_map: Dict[int, int] = field(default_factory=dict)
    unmapped_draft_tokens: List[int] = field(default_factory=list)
    unmapped_target_tokens: List[int] = field(default_factory=list)
    collisions_count: int = 0
    sample_collisions: List[Tuple[str, int, int]] = field(default_factory=list)


class VocabularyMapper:
    """Maps token IDs between non-identical vocabularies and detects ID drift."""

    @staticmethod
    def build_mapping(
        target_vocab: Dict[str, int],
        draft_vocab: Dict[str, int],
        max_sample_collisions: int = 10,
    ) -> VocabularyMappingResult:
        """Analyze vocabulary overlap and construct bidirectional ID mapping.

        Args:
            target_vocab: Dictionary mapping string tokens to target integer IDs.
            draft_vocab: Dictionary mapping string tokens to draft integer IDs.
            max_sample_collisions: Max collision examples to record.

        Returns:
            VocabularyMappingResult containing mapping tables and diagnostic metrics.
        """
        target_vocab_size = len(target_vocab)
        draft_vocab_size = len(draft_vocab)

        draft_to_target: Dict[int, int] = {}
        target_to_draft: Dict[int, int] = {}
        unmapped_draft: List[int] = []
        unmapped_target: List[int] = []
        collisions: List[Tuple[str, int, int]] = []

        shared_tokens = 0
        target_token_set: Set[str] = set(target_vocab.keys())

        for token_str, draft_id in draft_vocab.items():
            if token_str in target_vocab:
                target_id = target_vocab[token_str]
                draft_to_target[draft_id] = target_id
                target_to_draft[target_id] = draft_id
                shared_tokens += 1
                if target_id != draft_id:
                    if len(collisions) < max_sample_collisions:
                        collisions.append((token_str, target_id, draft_id))
            else:
                unmapped_draft.append(draft_id)

        for token_str, target_id in target_vocab.items():
            if token_str not in draft_vocab:
                unmapped_target.append(target_id)

        # Total collision count
        collisions_count = sum(
            1 for token_str, draft_id in draft_vocab.items()
            if token_str in target_vocab and target_vocab[token_str] != draft_id
        )

        is_exact = (
            target_vocab_size == draft_vocab_size
            and shared_tokens == target_vocab_size
            and collisions_count == 0
        )

        coverage = shared_tokens / max(target_vocab_size, 1)

        return VocabularyMappingResult(
            target_vocab_size=target_vocab_size,
            draft_vocab_size=draft_vocab_size,
            shared_tokens=shared_tokens,
            coverage_ratio=coverage,
            is_exact_match=is_exact,
            draft_to_target_map=draft_to_target,
            target_to_draft_map=target_to_draft,
            unmapped_draft_tokens=unmapped_draft,
            unmapped_target_tokens=unmapped_target,
            collisions_count=collisions_count,
            sample_collisions=collisions,
        )

    @staticmethod
    def translate_draft_tokens(
        draft_token_ids: List[int],
        mapping: VocabularyMappingResult,
    ) -> Tuple[List[int], bool]:
        """Translate a sequence of draft token IDs into target token IDs.

        Args:
            draft_token_ids: Sequence of draft model token IDs.
            mapping: VocabularyMappingResult containing draft_to_target_map.

        Returns:
            Tuple of (translated_ids, all_mapped_successfully).
        """
        translated: List[int] = []
        all_success = True
        for token_id in draft_token_ids:
            if token_id in mapping.draft_to_target_map:
                translated.append(mapping.draft_to_target_map[token_id])
            else:
                all_success = False
                translated.append(-1)
        return translated, all_success
