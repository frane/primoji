"""Tests for Primoji encode -> decode roundtrip fidelity.

Verifies that encoding text to token IDs and decoding back produces
semantically recognizable English. Uses words from the hardcoded dictionary.

Note: Primoji provides SEMANTIC equivalence, not verbatim. "small" may decode
to "little" (both map to the SMALL primitive). This is by design — Level 1
fidelity guarantees concept survival, not exact wording.
"""

from __future__ import annotations

import pytest

from primoji.tokenizer import Tokenizer


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer()


# ── Roundtrip: words that should survive as themselves ────────────────────────
# These are words that ARE the canonical reverse-lookup form.

_EXACT_ROUNDTRIP_WORDS: list[str] = [
    # Tier 1 direct emoji (exact roundtrip via dictionary reverse lookup)
    "dog", "cat", "fish", "tree", "fire", "water", "bird", "snake",
    "book", "star", "apple", "flower", "music", "house", "car",
    # Composed words (exact roundtrip — these are the canonical forms)
    "teacher", "writer", "student",
    "photosynthesis", "computer", "internet", "telephone", "television",
    # Verbs that are the canonical form for their primitive
    "think", "know", "want", "feel", "see", "move", "write",
    "connect", "live", "grow",
    # Adjectives that are the canonical form
    "good", "bad", "dark", "near", "far", "long",
    # Other canonical forms
    "war", "peace", "change", "home", "path",
]


class TestExactRoundtrip:
    @pytest.mark.parametrize("word", _EXACT_ROUNDTRIP_WORDS)
    def test_exact_roundtrip(self, tok: Tokenizer, word: str) -> None:
        """Words that are the canonical form should roundtrip exactly."""
        ids = tok.encode(word)
        decoded = tok.decode(ids)
        assert word in decoded.lower(), (
            f"Roundtrip failed for '{word}': encoded={ids}, decoded='{decoded}'"
        )


# ── Roundtrip: semantic equivalence (word may change form) ───────────────────

_SEMANTIC_ROUNDTRIP: list[tuple[str, list[str]]] = [
    # (input_word, [any_of_these_acceptable_outputs])
    ("say", ["say", "said", "explained"]),
    ("said", ["say", "said", "explained"]),
    ("big", ["big", "large"]),
    ("large", ["big", "large"]),
    ("small", ["small", "little"]),
    ("little", ["small", "little"]),
    ("knowledge", ["know", "knowledge"]),
    ("education", ["teach", "education"]),
    ("life", ["live", "life"]),
    ("death", ["die", "death"]),
    ("creation", ["create", "creation"]),
    ("destruction", ["destroy", "destruction"]),
    ("beginning", ["begin", "beginning"]),
    ("end", ["end", "stop"]),
    ("bright", ["bright", "light"]),
    # Verbs where the noun form is the canonical reverse lookup
    ("teach", ["teach", "education"]),
    ("destroy", ["destroy", "destruction"]),
    ("create", ["create", "creation"]),
    ("die", ["die", "death"]),
]


class TestSemanticRoundtrip:
    @pytest.mark.parametrize("word,acceptable", _SEMANTIC_ROUNDTRIP)
    def test_semantic_equivalence(
        self, tok: Tokenizer, word: str, acceptable: list[str]
    ) -> None:
        """Encoding then decoding should produce a semantically equivalent word."""
        ids = tok.encode(word)
        decoded = tok.decode(ids).lower()
        assert any(a in decoded for a in acceptable), (
            f"Semantic roundtrip failed for '{word}': decoded='{decoded}', "
            f"expected one of {acceptable}"
        )


# ── Non-empty input always produces non-empty output ─────────────────────────


class TestNonEmptyDecoding:
    def test_decode_nonempty_for_known_word(self, tok: Tokenizer) -> None:
        for word in ["dog", "teacher", "photosynthesis", "fire", "know"]:
            ids = tok.encode(word)
            decoded = tok.decode(ids)
            assert len(decoded) > 0, f"Decoded output for '{word}' is empty"

    def test_decode_nonempty_for_unknown_word(self, tok: Tokenizer) -> None:
        """Unknown words use byte fallback and should produce non-empty token IDs."""
        ids = tok.encode("xyzzyplugh")
        assert len(ids) > 0

    def test_empty_input_gives_empty_output(self, tok: Tokenizer) -> None:
        assert tok.encode("") == []
        assert tok.decode([]) == ""


# ── Sentence-level roundtrip ────────────────────────────────────────────────


class TestRoundtripSentences:
    def test_simple_sentence(self, tok: Tokenizer) -> None:
        """A simple sentence of known words should produce recognizable decoded text."""
        ids = tok.encode("the teacher said dog")
        decoded = tok.decode(ids)
        lower = decoded.lower()
        assert "teacher" in lower or "teach" in lower
        assert "dog" in lower

    def test_sentence_preserves_word_count_approximately(self, tok: Tokenizer) -> None:
        """Decoded output should have a reasonable number of words."""
        ids = tok.encode("dog cat fish bird snake")
        decoded = tok.decode(ids)
        words = decoded.split()
        assert len(words) >= 3


# ── Determinism ──────────────────────────────────────────────────────────────


class TestDeterminism:
    def test_encode_is_deterministic(self, tok: Tokenizer) -> None:
        """Same input should always produce the same token IDs."""
        text = "the teacher explained photosynthesis"
        ids1 = tok.encode(text)
        ids2 = tok.encode(text)
        assert ids1 == ids2

    def test_decode_is_deterministic(self, tok: Tokenizer) -> None:
        """Same token IDs should always produce the same text."""
        ids = tok.encode("teacher said dog")
        text1 = tok.decode(ids)
        text2 = tok.decode(ids)
        assert text1 == text2
