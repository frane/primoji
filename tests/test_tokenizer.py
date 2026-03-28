"""Tests for the Primoji Tokenizer class.

Covers encode/decode basics, vocab_size, describe(), empty/punctuation/number
handling, and the 4-tier fallback pipeline (exact lookup, contraction tokens,
fuzzy match, byte fallback).
"""

from __future__ import annotations

import pytest

from primoji.byte_fallback import BYTES_END_ID, BYTES_START_ID, is_byte_region_token
from primoji.primitives import get_primitive_by_name
from primoji.tokenizer import Tokenizer
from primoji.utils import SpecialTokens
from primoji.vocabulary import (
    CONTRACTION_TOKENS,
    DIGIT_IDS,
    MATH_OP_IDS,
    PUNCTUATION_IDS,
)


@pytest.fixture
def tok() -> Tokenizer:
    """Create a fresh Tokenizer instance with fuzzy matching enabled."""
    return Tokenizer(fuzzy=True)


@pytest.fixture
def tok_no_fuzzy() -> Tokenizer:
    """Create a Tokenizer with fuzzy matching disabled."""
    return Tokenizer(fuzzy=False)


# ── vocab_size ───────────────────────────────────────────────────────────────


class TestVocabSize:
    def test_vocab_size_is_positive_int(self, tok: Tokenizer) -> None:
        assert isinstance(tok.vocab_size, int)
        assert tok.vocab_size > 0

    def test_vocab_size_is_2062(self, tok: Tokenizer) -> None:
        """Vocabulary should be exactly 2062 (1806 base + 256 bytes)."""
        assert tok.vocab_size == 2062

    def test_vocab_size_is_in_expected_range(self, tok: Tokenizer) -> None:
        """Vocabulary should be in a reasonable range."""
        assert 1500 < tok.vocab_size < 3000


# ── encode basics ────────────────────────────────────────────────────────────


class TestEncode:
    def test_encode_returns_list_of_ints(self, tok: Tokenizer) -> None:
        ids: list[int] = tok.encode("dog")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_encode_known_word(self, tok: Tokenizer) -> None:
        """'dog' should encode to its Tier 1 emoji ID [0]."""
        ids = tok.encode("dog")
        assert ids == [0]

    def test_encode_composed_word(self, tok: Tokenizer) -> None:
        """'teacher' should compose as SOMEONE + TEACH."""
        someone_id = get_primitive_by_name("SOMEONE").id
        teach_id = get_primitive_by_name("TEACH").id
        ids = tok.encode("teacher")
        assert ids == [someone_id, teach_id]

    def test_encode_sentence(self, tok: Tokenizer) -> None:
        """A full sentence should return a non-empty list of IDs."""
        ids = tok.encode("The teacher explained photosynthesis")
        assert len(ids) > 0

    def test_encode_drops_articles(self, tok: Tokenizer) -> None:
        """Articles 'the', 'a', 'an' map to empty lists and should be dropped."""
        ids_the = tok.encode("the")
        assert ids_the == []

    def test_encode_case_insensitive(self, tok: Tokenizer) -> None:
        """Encoding should be case-insensitive."""
        ids_lower = tok.encode("dog")
        ids_upper = tok.encode("Dog")
        ids_all_caps = tok.encode("DOG")
        assert ids_lower == ids_upper == ids_all_caps

    def test_encode_preserves_semantic_content(self, tok: Tokenizer) -> None:
        """Encoding 'photosynthesis' should produce the PLANT+HAVE+LIGHT sequence."""
        plant_id = get_primitive_by_name("PLANT").id
        have_id = get_primitive_by_name("HAVE").id
        light_id = get_primitive_by_name("LIGHT").id
        ids = tok.encode("photosynthesis")
        assert ids == [plant_id, have_id, light_id]

    def test_encode_unknown_word_uses_byte_fallback(self, tok: Tokenizer) -> None:
        """A truly unknown word should fall back to byte encoding, not UNK."""
        ids = tok.encode("xyzzyplugh")
        assert SpecialTokens.UNK not in ids
        assert BYTES_START_ID in ids
        assert BYTES_END_ID in ids


# ── empty string ─────────────────────────────────────────────────────────────


class TestEmptyString:
    def test_encode_empty_string(self, tok: Tokenizer) -> None:
        assert tok.encode("") == []

    def test_encode_whitespace_only(self, tok: Tokenizer) -> None:
        assert tok.encode("   ") == []

    def test_decode_empty_list(self, tok: Tokenizer) -> None:
        assert tok.decode([]) == ""


# ── decode basics ────────────────────────────────────────────────────────────


class TestDecode:
    def test_decode_known_word(self, tok: Tokenizer) -> None:
        text = tok.decode([0])
        assert "dog" in text.lower()

    def test_decode_composed(self, tok: Tokenizer) -> None:
        """Decoding SOMEONE+TEACH should produce 'teacher'."""
        someone_id = get_primitive_by_name("SOMEONE").id
        teach_id = get_primitive_by_name("TEACH").id
        text = tok.decode([someone_id, teach_id])
        assert "teacher" in text.lower()

    def test_decode_returns_string(self, tok: Tokenizer) -> None:
        result = tok.decode([0, 1, 2])
        assert isinstance(result, str)

    def test_decode_with_verbatim_flag_returns_string(self, tok: Tokenizer) -> None:
        """verbatim=True currently falls back to canonical but should still work."""
        result = tok.decode([0], verbatim=True)
        assert isinstance(result, str)
        assert len(result) > 0


# ── punctuation ──────────────────────────────────────────────────────────────


class TestPunctuation:
    def test_encode_period(self, tok: Tokenizer) -> None:
        ids = tok.encode("dog.")
        assert 0 in ids
        assert PUNCTUATION_IDS["."] in ids

    def test_encode_comma(self, tok: Tokenizer) -> None:
        ids = tok.encode("dog, cat")
        assert PUNCTUATION_IDS[","] in ids

    def test_encode_question_mark(self, tok: Tokenizer) -> None:
        ids = tok.encode("dog?")
        assert PUNCTUATION_IDS["?"] in ids

    def test_encode_exclamation(self, tok: Tokenizer) -> None:
        ids = tok.encode("fire!")
        assert PUNCTUATION_IDS["!"] in ids

    def test_punctuation_ids_are_in_structural_range(self) -> None:
        """All punctuation IDs should be in the structural range (1606+)."""
        for punc, tid in PUNCTUATION_IDS.items():
            assert tid >= 1606, f"Punctuation '{punc}' has ID {tid} below structural base"


# ── numbers in text ──────────────────────────────────────────────────────────


class TestNumbers:
    def test_encode_standalone_number(self, tok: Tokenizer) -> None:
        ids = tok.encode("42")
        assert DIGIT_IDS["4"] in ids
        assert DIGIT_IDS["2"] in ids

    def test_encode_number_in_sentence(self, tok: Tokenizer) -> None:
        ids = tok.encode("the cat ate 3 fish")
        assert DIGIT_IDS["3"] in ids

    def test_encode_decimal_number(self, tok: Tokenizer) -> None:
        ids = tok.encode("3.14")
        assert DIGIT_IDS["3"] in ids
        assert DIGIT_IDS["1"] in ids
        assert DIGIT_IDS["4"] in ids
        assert PUNCTUATION_IDS["."] in ids

    def test_digit_ids_are_in_structural_range(self) -> None:
        """All digit IDs should be in the structural range (1606+)."""
        for digit, tid in DIGIT_IDS.items():
            assert tid >= 1606, f"Digit '{digit}' has ID {tid} below structural base"


# ── describe ─────────────────────────────────────────────────────────────────


class TestDescribe:
    def test_describe_returns_string(self, tok: Tokenizer) -> None:
        desc = tok.describe(0)
        assert isinstance(desc, str)

    def test_describe_includes_id(self, tok: Tokenizer) -> None:
        desc = tok.describe(0)
        assert "0" in desc

    def test_describe_primitive(self, tok: Tokenizer) -> None:
        """Describing a primitive token should mention its name."""
        someone_id = get_primitive_by_name("SOMEONE").id
        desc = tok.describe(someone_id)
        assert "SOMEONE" in desc

    def test_describe_digit(self, tok: Tokenizer) -> None:
        desc = tok.describe(DIGIT_IDS["5"])
        assert "digit" in desc.lower() or "5" in desc

    def test_describe_unknown_id(self, tok: Tokenizer) -> None:
        """An ID outside the vocabulary should still return something (not crash)."""
        desc = tok.describe(99999)
        assert isinstance(desc, str)


# ── 4-tier pipeline ─────────────────────────────────────────────────────────


class TestTieredPipeline:
    def test_tier1_exact_lookup(self, tok: Tokenizer) -> None:
        """Known dictionary words go through Tier 1 with no byte tokens."""
        ids = tok.encode("water")
        assert len(ids) > 0
        assert not any(is_byte_region_token(tid) for tid in ids)

    def test_tier2_dedicated_contraction(self, tok: Tokenizer) -> None:
        """Dedicated contractions get a single contraction token."""
        ids = tok.encode("don't")
        expected_id = CONTRACTION_TOKENS["don't"]
        assert expected_id in ids

    def test_tier2_contraction_suffix_split(self, tok: Tokenizer) -> None:
        """Uncommon contractions get split, suffix becomes contraction token."""
        ids = tok.encode("he'd")
        suffix_id = CONTRACTION_TOKENS["'d"]
        assert suffix_id in ids

    def test_tier3_fuzzy_match(self, tok: Tokenizer) -> None:
        """Correctable typo (4+ chars, unique candidate) uses Tier 3."""
        ids_typo = tok.encode("watir")
        ids_correct = tok.encode("water")
        assert ids_typo == ids_correct

    def test_tier3_disabled_falls_through_to_bytes(self, tok_no_fuzzy: Tokenizer) -> None:
        """With fuzzy=False, typos go to byte fallback instead."""
        ids = tok_no_fuzzy.encode("watir")
        assert BYTES_START_ID in ids
        assert BYTES_END_ID in ids

    def test_tier4_byte_fallback_for_unknown(self, tok: Tokenizer) -> None:
        """Truly unknown words get byte-encoded."""
        ids = tok.encode("xyzzyplugh")
        assert BYTES_START_ID in ids
        assert BYTES_END_ID in ids

    def test_never_unk(self, tok: Tokenizer) -> None:
        """The tokenizer should NEVER produce UNK tokens."""
        garbage_strings = [
            "xyzzyplugh",
            "qwrtybnm",
            "asdfghjkl",
            "zxcvbnm123",
            "a" * 50,
        ]
        for text in garbage_strings:
            ids = tok.encode(text)
            assert SpecialTokens.UNK not in ids, (
                f"UNK found in encoding of '{text}': {ids}"
            )

    def test_byte_fallback_roundtrip(self, tok: Tokenizer) -> None:
        """Unknown word byte-encoded then decoded should recover the word."""
        ids = tok.encode("xyzzy")
        decoded = tok.decode(ids)
        assert "xyzzy" in decoded.lower()

    def test_mixed_known_and_unknown(self, tok: Tokenizer) -> None:
        """Sentence with known and unknown words encodes without UNK."""
        ids = tok.encode("the dog ate xyzzyplugh")
        assert SpecialTokens.UNK not in ids
        assert len(ids) > 0

    def test_contraction_tokens_are_in_range(self) -> None:
        """Contraction token IDs should be in the 1579-1605 range."""
        for contraction, tid in CONTRACTION_TOKENS.items():
            assert 1579 <= tid <= 1605, (
                f"Contraction '{contraction}' has ID {tid} outside 1579-1605 range"
            )

    def test_special_tokens_ids(self) -> None:
        """Special token IDs should match the expected values."""
        assert SpecialTokens.BOS == 1800
        assert SpecialTokens.EOS == 1801
        assert SpecialTokens.PAD == 1802
        assert SpecialTokens.UNK == 1803
        assert SpecialTokens.BYTES_START == 1804
        assert SpecialTokens.BYTES_END == 1805
