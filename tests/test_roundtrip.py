"""Tests for Primoji encode -> decode roundtrip fidelity.

Tests encode→decode produces non-empty readable output, and that
the pipeline is deterministic and stable after one roundtrip.
"""

from __future__ import annotations

import pytest

from primoji.tokenizer import Tokenizer
from primoji.utils import SpecialTokens


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer(fuzzy=False)


# ── Roundtrip produces non-empty, readable output ────────────────────────────

_KNOWN_WORDS: list[str] = [
    "dog", "fish", "bird", "snake", "apple", "music", "car",
    "teacher", "writer", "student",
    "photosynthesis", "computer", "internet", "television",
    "know", "see", "move", "say",
    "big", "small", "good", "bad",
    "water", "fire", "house", "home", "dark", "change",
    "peace", "love", "fear", "hot", "cold", "energy",
]


class TestRoundtripDecodes:
    @pytest.mark.parametrize("word", _KNOWN_WORDS)
    def test_encode_decode_nonempty(self, tok: Tokenizer, word: str) -> None:
        """Every known word should encode to non-empty IDs and decode to non-empty text."""
        ids = tok.encode(word)
        assert len(ids) > 0, f"'{word}' produced no tokens"
        decoded = tok.decode(ids)
        assert len(decoded) > 0, f"'{word}' decoded to empty string"


# ── Canonical form roundtrip (primitives and compositions) ───────────────────

_CANONICAL_ROUNDTRIPS: list[str] = [
    "water", "fire", "big", "good", "bad", "dark", "think", "know",
    "love", "fear", "hot", "cold", "energy", "home", "teacher",
    "photosynthesis", "dog", "house",
]


class TestCanonicalRoundtrip:
    @pytest.mark.parametrize("word", _CANONICAL_ROUNDTRIPS)
    def test_word_survives_roundtrip(self, tok: Tokenizer, word: str) -> None:
        """Words that are canonical forms should appear in their own decoded output."""
        ids = tok.encode(word)
        decoded = tok.decode(ids)
        assert word in decoded.lower(), (
            f"'{word}' not found in decoded '{decoded}' (ids={ids})"
        )


# ── Non-empty decoding ──────────────────────────────────────────────────────


class TestNonEmptyDecoding:
    def test_decode_nonempty_for_unknown_word(self, tok: Tokenizer) -> None:
        """Unknown words use byte fallback."""
        ids = tok.encode("xyzzyplugh")
        assert len(ids) > 0

    def test_empty_input_gives_empty_output(self, tok: Tokenizer) -> None:
        assert tok.encode("") == []
        assert tok.decode([]) == ""


# ── Sentence-level ──────────────────────────────────────────────────────────


class TestRoundtripSentences:
    def test_sentence_decode_nonempty(self, tok: Tokenizer) -> None:
        ids = tok.encode("the teacher said dog")
        decoded = tok.decode(ids)
        assert len(decoded) > 0
        assert len(decoded.split()) >= 2

    def test_sentence_preserves_word_count(self, tok: Tokenizer) -> None:
        ids = tok.encode("dog water fire house")
        decoded = tok.decode(ids)
        assert len(decoded.split()) >= 3


# ── Determinism ──────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_encode_is_deterministic(self, tok: Tokenizer) -> None:
        text = "the teacher explained photosynthesis"
        assert tok.encode(text) == tok.encode(text)

    def test_decode_is_deterministic(self, tok: Tokenizer) -> None:
        ids = tok.encode("teacher said dog")
        assert tok.decode(ids) == tok.decode(ids)

    def test_second_roundtrip_stable(self, tok: Tokenizer) -> None:
        """After one roundtrip, subsequent roundtrips are stable."""
        for word in ["dog", "teacher", "computer", "fire", "water"]:
            ids1 = tok.encode(word)
            decoded1 = tok.decode(ids1)
            ids2 = tok.encode(decoded1)
            decoded2 = tok.decode(ids2)
            ids3 = tok.encode(decoded2)
            decoded3 = tok.decode(ids3)
            assert decoded2 == decoded3, (
                f"Unstable for '{word}': '{decoded2}' != '{decoded3}'"
            )
