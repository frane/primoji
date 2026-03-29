"""Stress tests that try to break the tokenizer.

These don't test specific known bugs. They throw unexpected input
at the system and verify it doesn't crash, corrupt data, or produce
nonsense.
"""

from __future__ import annotations

import random
import string

import pytest

from primoji import Tokenizer
from primoji.byte_fallback import BYTES_START_ID, BYTES_END_ID, BYTE_TOKEN_OFFSET
from primoji.utils import SpecialTokens


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer(fuzzy=False)


# ── Adversarial inputs ───────────────────────────────────────────────────────


class TestAdversarialInputs:
    """Inputs designed to break things."""

    def test_null_bytes(self, tok: Tokenizer) -> None:
        ids = tok.encode("hello\x00world")
        assert len(ids) > 0
        assert SpecialTokens.UNK not in ids

    def test_only_punctuation(self, tok: Tokenizer) -> None:
        ids = tok.encode("...!!??,,")
        assert len(ids) > 0

    def test_only_newlines(self, tok: Tokenizer) -> None:
        ids = tok.encode("\n\n\n")
        # Whitespace-only normalizes to empty
        assert isinstance(ids, list)

    def test_very_long_word(self, tok: Tokenizer) -> None:
        word = "a" * 10000
        ids = tok.encode(word)
        decoded = tok.decode(ids)
        assert decoded == word

    def test_very_long_sentence(self, tok: Tokenizer) -> None:
        sentence = " ".join(["dog"] * 1000)
        ids = tok.encode(sentence)
        assert len(ids) == 1000  # 1000 dog IDs, no separators

    def test_mixed_scripts(self, tok: Tokenizer) -> None:
        text = "Hello Привет 你好 مرحبا こんにちは"
        ids = tok.encode(text)
        decoded = tok.decode(ids)
        # Every word should survive roundtrip through byte fallback
        for word in ["Hello", "Привет", "你好", "مرحبا", "こんにちは"]:
            assert word.lower() in decoded.lower() or word in decoded

    def test_emoji_in_text(self, tok: Tokenizer) -> None:
        ids = tok.encode("I love 🐕 dogs")
        assert len(ids) > 0
        assert SpecialTokens.UNK not in ids

    def test_numbers_with_commas(self, tok: Tokenizer) -> None:
        ids = tok.encode("The population is 1,234,567")
        assert len(ids) > 0

    def test_urls(self, tok: Tokenizer) -> None:
        ids = tok.encode("Visit example.com for more info")
        decoded = tok.decode(ids)
        assert len(decoded) > 0

    def test_repeated_apostrophes(self, tok: Tokenizer) -> None:
        ids = tok.encode("don'''t can''''t")
        assert len(ids) > 0
        assert SpecialTokens.UNK not in ids

    def test_tab_separated(self, tok: Tokenizer) -> None:
        ids = tok.encode("col1\tcol2\tcol3")
        assert len(ids) > 0

    def test_single_character_words(self, tok: Tokenizer) -> None:
        for c in "abcdefghijklmnopqrstuvwxyz":
            ids = tok.encode(c)
            assert SpecialTokens.UNK not in ids

    def test_all_digits(self, tok: Tokenizer) -> None:
        ids = tok.encode("0123456789")
        assert len(ids) == 10  # one token per digit

    def test_mixed_case_contraction(self, tok: Tokenizer) -> None:
        ids_lower = tok.encode("don't")
        ids_upper = tok.encode("DON'T")
        assert ids_lower == ids_upper


# ── Decode robustness ────────────────────────────────────────────────────────


class TestDecodeRobustness:
    """Decoder must handle garbage IDs without crashing."""

    def test_decode_empty(self, tok: Tokenizer) -> None:
        assert tok.decode([]) == ""

    def test_decode_single_id(self, tok: Tokenizer) -> None:
        # Every valid ID should decode to something
        for tid in [0, 100, 500, 1200, 1300]:
            result = tok.decode([tid])
            assert isinstance(result, str)

    def test_decode_out_of_range_id(self, tok: Tokenizer) -> None:
        # Should not crash on IDs beyond vocab
        result = tok.decode([99999])
        assert isinstance(result, str)

    def test_decode_byte_region_without_end(self, tok: Tokenizer) -> None:
        # Malformed byte region: START but no END
        result = tok.decode([BYTES_START_ID, BYTE_TOKEN_OFFSET + 65])
        assert isinstance(result, str)  # should not crash

    def test_decode_special_tokens_skipped(self, tok: Tokenizer) -> None:
        result = tok.decode([SpecialTokens.BOS, 0, SpecialTokens.EOS])
        assert isinstance(result, str)
        assert len(result) > 0  # the middle token (ID 0) should decode


# ── Consistency ──────────────────────────────────────────────────────────────


class TestConsistency:
    """Properties that must always hold."""

    def test_encode_decode_does_not_grow(self, tok: Tokenizer) -> None:
        """Repeated encode-decode should not produce ever-longer output."""
        text = "The dog ate food"
        for _ in range(5):
            ids = tok.encode(text)
            text = tok.decode(ids)
        # After 5 roundtrips, length should be stable (not exploding)
        assert len(text) < 200

    def test_encode_preserves_word_count_roughly(self, tok: Tokenizer) -> None:
        """A sentence with N content words should produce roughly N or more tokens."""
        text = "dog cat fish bird snake"  # 5 content words
        ids = tok.encode(text)
        assert len(ids) >= 5

    def test_byte_fallback_length_predictable(self, tok: Tokenizer) -> None:
        """Byte fallback for an ASCII word of length N produces N+2 tokens."""
        word = "xyzzy"  # 5 ASCII chars, definitely unknown
        ids = tok.encode(word)
        assert ids[0] == BYTES_START_ID
        assert ids[-1] == BYTES_END_ID
        assert len(ids) == len(word) + 2

    def test_fuzz_100_random_sentences(self, tok: Tokenizer) -> None:
        """100 random sentences encode and decode without crashing."""
        rng = random.Random(123)
        words = ["dog", "cat", "the", "is", "big", "water", "fire",
                 "xyzzy", "123", "don't", "hello", "world"]
        for _ in range(100):
            n = rng.randint(1, 20)
            sentence = " ".join(rng.choices(words, k=n))
            ids = tok.encode(sentence)
            decoded = tok.decode(ids)
            assert isinstance(decoded, str)
            assert SpecialTokens.UNK not in ids
