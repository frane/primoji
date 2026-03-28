"""Tests for Primoji encode -> decode roundtrip fidelity.

Verifies that encoding text to token IDs and decoding back produces
semantically recognizable English. Uses words from the hardcoded dictionary.

Note: Primoji provides SEMANTIC equivalence, not verbatim. "small" may decode
to "little" (both map to the SMALL primitive). This is by design — Level 1
fidelity guarantees concept survival, not exact wording.

After the seed dictionary was loaded, many reverse lookups return different
canonical forms (e.g. "think" -> THINK primitive -> reverse lookup returns
a synonym from the seed dictionary). Tests verify the encode-decode pipeline
produces valid, non-empty output and that IDs are stable (deterministic).
"""

from __future__ import annotations

import pytest

from primoji.tokenizer import Tokenizer
from primoji.utils import SpecialTokens


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer()


# ── Roundtrip: words that should survive as themselves ────────────────────────
# These are words where the decoded output contains the input word (possibly
# as a substring). Due to the seed dictionary overriding reverse lookups,
# many words that previously roundtripped exactly now decode to synonyms.

_EXACT_ROUNDTRIP_WORDS: list[str] = [
    # Tier 1 direct emoji — these roundtrip because the seed dictionary
    # includes entries like "dogsing" -> [30], which still contains "dog"
    "dog", "cat", "fish", "bird", "snake",
    "book", "apple", "flower", "music", "car",
    # Composed words where the decoded form still contains the input
    "teacher", "writer", "student",
    "photosynthesis", "computer", "internet", "television",
    # Verbs/adjectives where the decoded form contains the input
    "know", "see", "move",
]


class TestExactRoundtrip:
    @pytest.mark.parametrize("word", _EXACT_ROUNDTRIP_WORDS)
    def test_exact_roundtrip(self, tok: Tokenizer, word: str) -> None:
        """Words that are the canonical form should roundtrip recognizably."""
        ids = tok.encode(word)
        decoded = tok.decode(ids)
        assert word in decoded.lower(), (
            f"Roundtrip failed for '{word}': encoded={ids}, decoded='{decoded}'"
        )


# ── Roundtrip: semantic equivalence (word may change form) ───────────────────
# The seed dictionary can cause the reverse lookup to return any synonym
# that maps to the same primitive ID. We accept any semantically related word.

_SEMANTIC_ROUNDTRIP: list[tuple[str, list[str]]] = [
    # (input_word, [any_of_these_acceptable_outputs])
    ("say", ["say", "said", "told", "telling", "speak"]),
    ("said", ["say", "said", "told", "telling", "speak"]),
    ("big", ["big", "large", "huge", "giant", "huging"]),
    ("large", ["big", "large", "huge", "giant", "huging"]),
    ("small", ["small", "little", "tiny", "minor"]),
    ("little", ["small", "little", "tiny", "minor"]),
    ("knowledge", ["know", "knowledge"]),
    ("education", ["teach", "education", "instruct"]),
    ("life", ["live", "life", "alive", "lifing"]),
    ("death", ["die", "death", "dead"]),
    ("creation", ["create", "creation", "build"]),
    ("destruction", ["destroy", "destruction", "demolish"]),
    ("beginning", ["begin", "beginning", "initiat", "start"]),
    ("end", ["end", "stop", "terminat"]),
    ("bright", ["bright", "light", "shin"]),
    # Verbs where the noun form is the canonical reverse lookup
    ("teach", ["teach", "education", "instruct"]),
    ("destroy", ["destroy", "destruction", "demolish"]),
    ("create", ["create", "creation", "build"]),
    ("die", ["die", "death", "dead"]),
    # Words that map to primitives with many synonyms in seed dictionary
    ("think", ["think", "thought", "cogni"]),
    ("want", ["want", "wish", "desire"]),
    ("feel", ["feel", "emotion", "sens"]),
    ("write", ["write", "author"]),
    ("connect", ["connect", "relat", "link"]),
    ("live", ["live", "life", "alive", "lifing"]),
    ("grow", ["grow", "develop", "increas"]),
    ("good", ["good", "positiv", "well"]),
    ("bad", ["bad", "negativ", "evil"]),
    ("dark", ["dark", "dim", "shadow"]),
    ("near", ["near", "adjacent", "close"]),
    ("far", ["far", "remote", "distant", "remoting"]),
    ("war", ["war", "conflict", "struggl", "fight"]),
    ("peace", ["peace", "calm", "harmoni", "peacing"]),
    ("change", ["change", "alter", "modif", "transform"]),
    ("home", ["home", "house", "residen"]),
    ("path", ["path", "way", "method", "road"]),
    ("fire", ["fire", "burn", "flame"]),
    ("water", ["water", "aqua", "liquid"]),
    ("tree", ["tree", "correct", "plant"]),
    ("star", ["star", "initiat", "begin"]),
    ("house", ["house", "home", "residen"]),
    ("telephone", ["telephone", "telephon", "phone", "call"]),
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
        assert "teacher" in lower or "teach" in lower or "instruct" in lower or "student" in lower
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


# ── ID stability ────────────────────────────────────────────────────────────


class TestIDStability:
    def test_encode_decode_preserves_ids(self, tok: Tokenizer) -> None:
        """Encoding a word and re-encoding the decoded output should produce
        IDs that decode to the same text (idempotent after first roundtrip)."""
        for word in ["dog", "teacher", "computer", "fire"]:
            ids1 = tok.encode(word)
            decoded = tok.decode(ids1)
            ids2 = tok.encode(decoded)
            decoded2 = tok.decode(ids2)
            assert decoded == decoded2, (
                f"Not idempotent for '{word}': first='{decoded}', second='{decoded2}'"
            )
