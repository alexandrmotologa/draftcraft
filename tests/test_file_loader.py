"""Unit tests for tokenizer JSON file loading and cross-verification."""

import json
import pytest
from draftcraft.alignment.file_loader import TokenizerFileLoader
from draftcraft.core.exceptions import TokenizerAlignmentError
from draftcraft.core.types import AlignmentStatus


def test_load_vocab_from_file_bpe(tmp_path):
    tokenizer_data = {
        "version": "1.0",
        "model": {
            "type": "BPE",
            "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3, "world": 4},
        },
        "added_tokens": [
            {"id": 100, "content": "<|special|>", "special": True}
        ],
    }
    tok_path = tmp_path / "tokenizer.json"
    tok_path.write_text(json.dumps(tokenizer_data), encoding="utf-8")

    vocab, special = TokenizerFileLoader.load_vocab_from_file(tok_path)
    assert len(vocab) == 6
    assert vocab["hello"] == 3
    assert vocab["<|special|>"] == 100
    assert "<|special|>" in special


def test_load_vocab_from_file_unigram(tmp_path):
    tokenizer_data = {
        "version": "1.0",
        "model": {
            "type": "Unigram",
            "vocab": [["<unk>", 0.0], ["the", -1.0], ["quick", -2.0]],
        },
    }
    tok_path = tmp_path / "tokenizer_unigram.json"
    tok_path.write_text(json.dumps(tokenizer_data), encoding="utf-8")

    vocab, special = TokenizerFileLoader.load_vocab_from_file(tok_path)
    assert len(vocab) == 3
    assert vocab["the"] == 1


def test_cross_verify_files_and_export_remap(tmp_path):
    target_data = {
        "model": {"vocab": {"a": 0, "b": 1, "c": 2}}
    }
    draft_data = {
        "model": {"vocab": {"a": 10, "b": 11, "c": 12}}
    }
    t_path = tmp_path / "target.json"
    d_path = tmp_path / "draft.json"
    remap_path = tmp_path / "remap.json"

    t_path.write_text(json.dumps(target_data), encoding="utf-8")
    d_path.write_text(json.dumps(draft_data), encoding="utf-8")

    summary = TokenizerFileLoader.cross_verify_files(
        target_file=t_path,
        draft_file=d_path,
        export_remap_path=remap_path,
    )
    assert summary.overlap_ratio == 1.0
    assert summary.status == AlignmentStatus.TOKENIZER_DRIFT  # IDs differ
    assert remap_path.exists()

    remap_content = json.loads(remap_path.read_text(encoding="utf-8"))
    assert remap_content["target_vocab_size"] == 3
    assert remap_content["draft_to_target_index_map"]["10"] == 0


def test_load_missing_file_raises():
    with pytest.raises(TokenizerAlignmentError):
        TokenizerFileLoader.load_vocab_from_file("non_existent_file.json")
