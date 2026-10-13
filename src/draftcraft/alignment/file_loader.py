"""Loader and cross-verifier for Hugging Face and local tokenizer.json files."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from draftcraft.alignment.tokenizer_verifier import TokenizerAlignmentVerifier
from draftcraft.alignment.vocab_mapper import VocabularyMapper, VocabularyMappingResult
from draftcraft.core.exceptions import TokenizerAlignmentError
from draftcraft.core.types import AlignmentStatus, TokenVerificationSummary


class TokenizerFileLoader:
    """Parses real Hugging Face tokenizer.json files and extracts vocabularies."""

    @classmethod
    def load_vocab_from_file(cls, file_path: Union[str, Path]) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Load vocabulary dictionary and special tokens from tokenizer.json.

        Args:
            file_path: Path to tokenizer.json file.

        Returns:
            Tuple of (vocab_dict: Dict[str, int], special_tokens_dict: Dict[str, int]).
        """
        p = Path(file_path)
        if not p.exists():
            raise TokenizerAlignmentError(f"Tokenizer file not found: {p}")

        try:
            content = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise TokenizerAlignmentError(f"Failed to parse JSON in {p}: {e}") from e

        vocab: Dict[str, int] = {}
        special_tokens: Dict[str, int] = {}

        # 1. Inspect model.vocab structure
        model_sec = content.get("model", {})
        model_vocab = model_sec.get("vocab", {})

        if isinstance(model_vocab, dict):
            # Standard BPE / WordPiece dictionary mapping token_str -> token_id
            vocab.update(model_vocab)
        elif isinstance(model_vocab, list):
            # Unigram or SentencePiece format: list of [token_str, score]
            for idx, item in enumerate(model_vocab):
                if isinstance(item, (list, tuple)) and len(item) > 0:
                    vocab[str(item[0])] = idx
                elif isinstance(item, str):
                    vocab[item] = idx

        # 2. Inspect added_tokens for special tokens
        added_tokens = content.get("added_tokens", [])
        for token_entry in added_tokens:
            if isinstance(token_entry, dict):
                token_str = token_entry.get("content")
                token_id = token_entry.get("id")
                is_special = token_entry.get("special", False)
                if token_str is not None and token_id is not None:
                    vocab[str(token_str)] = int(token_id)
                    if is_special:
                        special_tokens[str(token_str)] = int(token_id)

        if not vocab:
            raise TokenizerAlignmentError(f"No tokens found in tokenizer file: {p}")

        return vocab, special_tokens

    @classmethod
    def cross_verify_files(
        cls,
        target_file: Union[str, Path],
        draft_file: Union[str, Path],
        export_remap_path: Optional[Union[str, Path]] = None,
    ) -> TokenVerificationSummary:
        """Verify alignment between two real tokenizer files and optionally export remap table."""
        target_vocab, target_special = cls.load_vocab_from_file(target_file)
        draft_vocab, draft_special = cls.load_vocab_from_file(draft_file)

        verifier = TokenizerAlignmentVerifier(
            target_vocab=target_vocab,
            draft_vocab=draft_vocab,
            target_special_tokens=target_special,
            draft_special_tokens=draft_special,
        )
        summary = verifier.verify_vocabularies()

        if export_remap_path:
            mapping_result = VocabularyMapper.build_mapping(target_vocab, draft_vocab)
            remap_payload = {
                "version": "1.0",
                "target_vocab_size": summary.target_vocab_size,
                "draft_vocab_size": summary.draft_vocab_size,
                "overlap_ratio": summary.overlap_ratio,
                "is_lossless": summary.status == AlignmentStatus.FULLY_ALIGNED,
                "draft_to_target_index_map": mapping_result.draft_to_target_map,
                "collisions_count": mapping_result.collisions_count,
            }
            out_p = Path(export_remap_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(json.dumps(remap_payload, indent=2), encoding="utf-8")

        return summary
