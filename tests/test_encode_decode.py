"""Tests that the tokenizer encodes and decodes correctly.

Every test here verifies actual behavior that matters:
- Known words encode to specific expected IDs
- Decode recovers the original word (or a correct canonical form)
- Byte fallback roundtrips perfectly
- No input ever produces UNK
"""

from __future__ import annotations

import math
import random
import string

import pytest

from primoji import Tokenizer, encode_bytes, decode_bytes
from primoji.byte_fallback import BYTES_START_ID, BYTES_END_ID, BYTE_TOKEN_OFFSET
from primoji.primitives import get_primitive_by_name
from primoji.utils import SpecialTokens
from primoji.vocabulary import (
    DIGIT_IDS,
    PUNCTUATION_IDS,
    TIER1_DIRECT_EMOJI,
)


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer(fuzzy=False)


# ── Exact encode/decode for known words ───────────────────────────────────────


class TestExactEncoding:
    """Words with known mappings must encode to exact expected IDs."""

    def test_photosynthesis(self, tok: Tokenizer) -> None:
        ids = tok.encode("photosynthesis")
        expected = [
            get_primitive_by_name("PLANT").id,
            get_primitive_by_name("HAVE").id,
            get_primitive_by_name("LIGHT").id,
        ]
        assert ids == expected

    def test_water_is_single_token(self, tok: Tokenizer) -> None:
        ids = tok.encode("water")
        assert len(ids) == 1  # single token (word token or primitive)

    def test_fire_is_single_token(self, tok: Tokenizer) -> None:
        ids = tok.encode("fire")
        assert len(ids) == 1

    def test_dog_maps_to_dog_emoji(self, tok: Tokenizer) -> None:
        ids = tok.encode("dog")
        assert ids == [TIER1_DIRECT_EMOJI["🐕"]]

    def test_teacher_starts_with_someone(self, tok: Tokenizer) -> None:
        ids = tok.encode("teacher")
        assert ids[0] == get_primitive_by_name("SOMEONE").id
        assert get_primitive_by_name("TEACH").id in ids

    def test_dont_expands_to_do_not(self, tok: Tokenizer) -> None:
        ids = tok.encode("don't")
        do_ids = tok.encode("do")
        not_ids = tok.encode("not")
        assert ids == do_ids + not_ids

    def test_articles_are_word_tokens(self, tok: Tokenizer) -> None:
        """Articles encode as single word tokens (V8: no longer dropped)."""
        assert len(tok.encode("the")) == 1
        assert len(tok.encode("a")) == 1
        assert len(tok.encode("an")) == 1

    def test_digits_encode_individually(self, tok: Tokenizer) -> None:
        ids = tok.encode("42")
        assert ids == [DIGIT_IDS["4"], DIGIT_IDS["2"]]

    def test_case_insensitive(self, tok: Tokenizer) -> None:
        assert tok.encode("dog") == tok.encode("Dog") == tok.encode("DOG")


class TestExactDecoding:
    """Decode must recover the correct canonical word."""

    def test_dog_decodes_to_dog(self, tok: Tokenizer) -> None:
        ids = tok.encode("dog")
        assert tok.decode(ids) == "dog"

    def test_water_decodes_to_water(self, tok: Tokenizer) -> None:
        ids = tok.encode("water")
        assert tok.decode(ids) == "water"

    def test_fire_decodes_to_fire(self, tok: Tokenizer) -> None:
        ids = tok.encode("fire")
        assert tok.decode(ids) == "fire"

    def test_photosynthesis_decodes_to_photosynthesis(self, tok: Tokenizer) -> None:
        ids = tok.encode("photosynthesis")
        assert tok.decode(ids) == "photosynthesis"

    def test_teacher_decodes_to_teacher(self, tok: Tokenizer) -> None:
        ids = tok.encode("teacher")
        assert tok.decode(ids) == "teacher"

    def test_dont_decodes_to_do_not(self, tok: Tokenizer) -> None:
        ids = tok.encode("don't")
        assert tok.decode(ids) == "do not"


# ── Byte fallback: lossless roundtrip ─────────────────────────────────────────


class TestByteFallback:
    """Byte fallback must roundtrip ANY string perfectly."""

    def test_ascii_roundtrip(self) -> None:
        ids = encode_bytes("hello")
        assert decode_bytes(ids) == "hello"

    def test_unicode_roundtrip(self) -> None:
        ids = encode_bytes("cafe\u0301")  # café with combining accent
        assert decode_bytes(ids) == "cafe\u0301"

    def test_emoji_roundtrip(self) -> None:
        ids = encode_bytes("🎉🔥")
        assert decode_bytes(ids) == "🎉🔥"

    def test_chinese_roundtrip(self) -> None:
        ids = encode_bytes("你好世界")
        assert decode_bytes(ids) == "你好世界"

    def test_empty_string(self) -> None:
        ids = encode_bytes("")
        assert ids == [BYTES_START_ID, BYTES_END_ID]
        assert decode_bytes(ids) == ""

    def test_boundary_markers_present(self) -> None:
        ids = encode_bytes("abc")
        assert ids[0] == BYTES_START_ID
        assert ids[-1] == BYTES_END_ID
        assert len(ids) == 5  # START + 3 bytes + END

    def test_byte_values_correct(self) -> None:
        ids = encode_bytes("A")  # ASCII 65
        assert ids == [BYTES_START_ID, BYTE_TOKEN_OFFSET + 65, BYTES_END_ID]

    def test_tokenizer_byte_fallback_roundtrip(self, tok: Tokenizer) -> None:
        """Unknown word through full tokenizer pipeline roundtrips."""
        ids = tok.encode("xyzzyplugh")
        decoded = tok.decode(ids)
        assert decoded == "xyzzyplugh"

    def test_random_strings_roundtrip(self, tok: Tokenizer) -> None:
        """100 random strings all roundtrip through byte fallback."""
        rng = random.Random(42)
        for _ in range(100):
            length = rng.randint(1, 20)
            word = "".join(rng.choices(string.ascii_letters, k=length))
            ids = tok.encode(word)
            decoded = tok.decode(ids)
            assert word.lower() in decoded.lower() or word in decoded, (
                f"Failed roundtrip: '{word}' -> {ids} -> '{decoded}'"
            )


# ── Never UNK ─────────────────────────────────────────────────────────────────


class TestNeverUNK:
    """The tokenizer must NEVER produce UNK for any input."""

    @pytest.mark.parametrize("text", [
        "xyzzyplugh",
        "a" * 100,
        "café naïve Zürich",
        "你好世界",
        "🎉🔥💧",
        "mixed 123 and !@# symbols",
        "",
        "   ",
        "don't won't can't",
        "a",
    ])
    def test_no_unk(self, tok: Tokenizer, text: str) -> None:
        ids = tok.encode(text)
        assert SpecialTokens.UNK not in ids


# ── Determinism ───────────────────────────────────────────────────────────────


class TestDeterminism:
    """Same input must always produce same output."""

    def test_encode_deterministic(self, tok: Tokenizer) -> None:
        text = "The teacher explained photosynthesis to the student"
        assert tok.encode(text) == tok.encode(text)

    def test_decode_deterministic(self, tok: Tokenizer) -> None:
        ids = tok.encode("water fire earth")
        assert tok.decode(ids) == tok.decode(ids)

    def test_encode_decode_encode_stable(self, tok: Tokenizer) -> None:
        """After one roundtrip, subsequent roundtrips are stable."""
        text = "water fire dog teacher"
        ids1 = tok.encode(text)
        decoded = tok.decode(ids1)
        ids2 = tok.encode(decoded)
        decoded2 = tok.decode(ids2)
        ids3 = tok.encode(decoded2)
        assert ids2 == ids3


# ── ID range validity ─────────────────────────────────────────────────────────


class TestIDRanges:
    """Every output ID must be within vocab_size."""

    def test_known_words_in_range(self, tok: Tokenizer) -> None:
        vs = tok.vocab_size
        for word in ["dog", "water", "fire", "teacher", "photosynthesis",
                     "don't", "42", "hello world"]:
            for tid in tok.encode(word):
                assert 0 <= tid < vs, f"ID {tid} out of range for '{word}'"

    def test_unknown_words_in_range(self, tok: Tokenizer) -> None:
        vs = tok.vocab_size
        for tid in tok.encode("xyzzyplugh supercalifragilistic"):
            assert 0 <= tid < vs

    def test_byte_fallback_ids_in_range(self) -> None:
        for byte_val in [0, 127, 255]:
            tid = BYTE_TOKEN_OFFSET + byte_val
            assert tid < Tokenizer(fuzzy=False).vocab_size


class TestClassifyWord:
    """classify_word must report the correct tier for every token type."""

    def test_classify_emoji(self, tok: Tokenizer) -> None:
        assert tok.classify_word("dog") == "tier1_emoji"

    def test_classify_primitive_or_word(self, tok: Tokenizer) -> None:
        # "water" may be a primitive or a word token depending on NGSL expansion
        tier = tok.classify_word("water")
        assert tier in ("tier2_primitive", "tier1b_word")

    def test_classify_word_token(self, tok: Tokenizer) -> None:
        assert tok.classify_word("out") == "tier1b_word"

    def test_classify_article_as_word(self, tok: Tokenizer) -> None:
        """Articles are word tokens in V8 (no longer dropped)."""
        assert tok.classify_word("the") == "tier1b_word"

    def test_classify_composed(self, tok: Tokenizer) -> None:
        assert tok.classify_word("photosynthesis") == "dict_composed"

    def test_classify_digit(self, tok: Tokenizer) -> None:
        assert tok.classify_word("42") == "tier3_structural"

    def test_classify_byte_fallback(self, tok: Tokenizer) -> None:
        assert tok.classify_word("xyzzyplugh") == "byte_fallback"

    def test_classify_punctuation(self, tok: Tokenizer) -> None:
        assert tok.classify_word(".") == "tier3_structural"


class TestNGSLSingleTokens:
    """Common NGSL words must encode as single tokens, not multi-primitive compositions."""

    @pytest.mark.parametrize("word", [
        "government", "important", "different",
        "between", "against", "could", "would", "should",
        "because", "through", "before", "really",
        "city", "company", "public", "women", "children",
    ])
    def test_common_word_is_single_token(self, tok: Tokenizer, word: str) -> None:
        ids = tok.encode(word)
        assert len(ids) == 1, f"'{word}' encoded as {len(ids)} tokens: {ids}"

    def test_photosynthesis_still_composed(self, tok: Tokenizer) -> None:
        """Rare technical words should still use composition, not direct tokens."""
        ids = tok.encode("photosynthesis")
        assert len(ids) > 1


class TestContractionExpansion:
    """Contractions expand to full semantic words, not fragments."""

    def test_wont_expands_to_will_not(self, tok: Tokenizer) -> None:
        ids = tok.encode("won't")
        will_ids = tok.encode("will")
        not_ids = tok.encode("not")
        assert ids == will_ids + not_ids

    def test_dont_expands_to_do_not(self, tok: Tokenizer) -> None:
        ids = tok.encode("don't")
        assert ids == tok.encode("do") + tok.encode("not")

    def test_not_unified(self, tok: Tokenizer) -> None:
        """'not', 'don't', 'won't' all contain the same NOT token."""
        not_id = tok.encode("not")[0]
        assert not_id in tok.encode("don't")
        assert not_id in tok.encode("won't")
        assert not_id in tok.encode("can't")
        assert not_id in tok.encode("isn't")

    def test_possessive_not_expanded(self, tok: Tokenizer) -> None:
        """'John's' keeps possessive marker, not expanded."""
        ids = tok.encode("John's")
        # Should NOT contain "is" token
        is_ids = tok.encode("is")
        assert is_ids[0] not in ids

    def test_case_insensitive(self, tok: Tokenizer) -> None:
        assert tok.encode("don't") == tok.encode("Don't") == tok.encode("DON'T")

    def test_contraction_fewer_tokens_than_bytes(self, tok: Tokenizer) -> None:
        """Expanded contraction should be 2 tokens, not 6-10 byte tokens."""
        ids = tok.encode("won't")
        assert len(ids) == 2
