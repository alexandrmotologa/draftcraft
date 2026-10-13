"""Unit tests for tokenizer alignment and vocabulary mapping."""

import pytest
from draftcraft.alignment.corpus_sampler import CorpusSampler
from draftcraft.alignment.tokenizer_verifier import TokenizerAlignmentVerifier
from draftcraft.alignment.vocab_mapper import VocabularyMapper
from draftcraft.core.types import AlignmentStatus, DomainCategory


def test_vocabulary_mapper_identical():
    vocab = {"<pad>": 0, "hello": 1, "world": 2, "<eos>": 3}
    result = VocabularyMapper.build_mapping(vocab, vocab)
    assert result.is_exact_match is True
    assert result.coverage_ratio == 1.0
    assert result.collisions_count == 0
    assert result.shared_tokens == 4


def test_vocabulary_mapper_with_collisions():
    target_vocab = {"hello": 1, "world": 2, "!": 3}
    draft_vocab = {"hello": 10, "world": 20, "universe": 30}
    result = VocabularyMapper.build_mapping(target_vocab, draft_vocab)
    assert result.is_exact_match is False
    assert result.shared_tokens == 2
    assert result.collisions_count == 2
    assert len(result.unmapped_draft_tokens) == 1
    assert len(result.unmapped_target_tokens) == 1


def test_translate_draft_tokens():
    target_vocab = {"hello": 1, "world": 2}
    draft_vocab = {"hello": 10, "world": 20, "other": 30}
    result = VocabularyMapper.build_mapping(target_vocab, draft_vocab)

    translated, success = VocabularyMapper.translate_draft_tokens([10, 20], result)
    assert success is True
    assert translated == [1, 2]

    translated_err, success_err = VocabularyMapper.translate_draft_tokens([10, 30], result)
    assert success_err is False
    assert translated_err == [1, -1]


def test_tokenizer_verifier_status():
    vocab_a = {"token_a": 0, "token_b": 1, "token_c": 2}
    verifier = TokenizerAlignmentVerifier(vocab_a, vocab_a)
    summary = verifier.verify_vocabularies()
    assert summary.status == AlignmentStatus.FULLY_ALIGNED
    assert summary.overlap_ratio == 1.0

    vocab_b = {"token_a": 10, "token_b": 11}  # Different IDs
    verifier_drift = TokenizerAlignmentVerifier(vocab_a, vocab_b)
    summary_drift = verifier_drift.verify_vocabularies()
    assert summary_drift.status == AlignmentStatus.TOKENIZER_DRIFT
    assert summary_drift.divergent_sample_count == 2


def test_verify_eagle_head():
    summary_aligned = TokenizerAlignmentVerifier.verify_eagle_head(
        target_vocab_size=128256,
        head_vocab_size=128256,
        target_hidden_dim=4096,
        head_hidden_dim=1024,
    )
    assert summary_aligned.status == AlignmentStatus.FULLY_ALIGNED
    assert summary_aligned.overlap_ratio == 1.0

    summary_mismatched = TokenizerAlignmentVerifier.verify_eagle_head(
        target_vocab_size=128256,
        head_vocab_size=32000,
        target_hidden_dim=4096,
        head_hidden_dim=1024,
    )
    assert summary_mismatched.status == AlignmentStatus.VOCAB_MISMATCH


def test_corpus_sampler_all_domains():
    for domain in DomainCategory:
        samples = CorpusSampler.get_samples(domain)
        assert len(samples) > 0
        assert isinstance(samples[0], str)
