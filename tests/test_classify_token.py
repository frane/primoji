"""Tests for the shared classify_token function.

This is the single source of truth for tier classification, used by
training data preparation, inference, and trace display.
"""

from __future__ import annotations

import pytest

from primoji import Tokenizer
from primoji.primitives import PRIMITIVES, get_primitive_by_name
from primoji.utils import (
    _IDS,
    SpecialTokens,
    classify_token,
    classify_token_name,
    TIER_EMOJI,
    TIER_WORD,
    TIER_PRIMITIVE,
    TIER_STRUCTURAL,
    TIER_BYTE,
)
from primoji.vocabulary import COMMON_WORD_TOKENS, TIER1_DIRECT_EMOJI


class TestClassifyToken:
    """classify_token must return the correct tier for every token type."""

    def test_emoji_tokens(self) -> None:
        for tid in list(TIER1_DIRECT_EMOJI.values())[:10]:
            assert classify_token(tid) == TIER_EMOJI

    def test_primitive_tokens(self) -> None:
        for p in PRIMITIVES:
            assert classify_token(p.id) == TIER_PRIMITIVE, (
                f"Primitive {p.name} (ID {p.id}) classified as {classify_token_name(p.id)}"
            )

    def test_all_140_primitives(self) -> None:
        """Every one of the 140 primitives must classify as TIER_PRIMITIVE."""
        prim_count = sum(1 for p in PRIMITIVES if classify_token(p.id) == TIER_PRIMITIVE)
        assert prim_count == len(PRIMITIVES) == 140

    def test_word_tokens(self) -> None:
        for word, tid in list(COMMON_WORD_TOKENS.items())[:10]:
            assert classify_token(tid) == TIER_WORD, (
                f"Word '{word}' (ID {tid}) classified as {classify_token_name(tid)}"
            )

    def test_special_tokens_are_structural(self) -> None:
        assert classify_token(SpecialTokens.BOS) == TIER_STRUCTURAL
        assert classify_token(SpecialTokens.EOS) == TIER_STRUCTURAL
        assert classify_token(SpecialTokens.PAD) == TIER_STRUCTURAL

    def test_byte_tokens(self) -> None:
        assert classify_token(SpecialTokens.BYTES_START) == TIER_BYTE
        assert classify_token(SpecialTokens.BYTES_END) == TIER_BYTE
        byte_offset = _IDS["BYTE_OFFSET"]
        assert classify_token(byte_offset) == TIER_BYTE      # byte 0x00
        assert classify_token(byte_offset + 255) == TIER_BYTE  # byte 0xFF

    def test_classify_token_name_matches(self) -> None:
        """String and numeric classification must agree."""
        test_ids = [0, PRIMITIVES[0].id, SpecialTokens.EOS, _IDS["BYTE_OFFSET"]]
        for tid in test_ids:
            numeric = classify_token(tid)
            name = classify_token_name(tid)
            expected_name = {TIER_EMOJI: "emoji", TIER_WORD: "word",
                             TIER_PRIMITIVE: "prim", TIER_STRUCTURAL: "struct",
                             TIER_BYTE: "byte"}[numeric]
            assert name == expected_name


class TestClassifyTokenConsistentWithEncode:
    """classify_token must agree with the tokenizer's encode output."""

    @pytest.fixture
    def tok(self) -> Tokenizer:
        return Tokenizer(fuzzy=False)

    def test_content_words_encode_to_primitive_tier(self, tok: Tokenizer) -> None:
        for word in ["water", "fire", "think", "good", "love", "health"]:
            ids = tok.encode(word)
            assert len(ids) == 1
            assert classify_token(ids[0]) == TIER_PRIMITIVE, (
                f"'{word}' encodes to ID {ids[0]} but classifies as {classify_token_name(ids[0])}"
            )

    def test_grammar_words_encode_to_word_tier(self, tok: Tokenizer) -> None:
        for word in ["is", "not", "the", "with", "was"]:
            ids = tok.encode(word)
            if not ids:  # articles get dropped
                continue
            assert len(ids) == 1
            assert classify_token(ids[0]) == TIER_WORD, (
                f"'{word}' encodes to ID {ids[0]} but classifies as {classify_token_name(ids[0])}"
            )
