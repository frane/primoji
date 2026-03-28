"""Tests for the lightweight Preprocessor.

The preprocessor handles Unicode normalization, apostrophe standardization,
contraction splitting, and word tokenization. It does NOT lowercase or
spell-correct -- those are handled downstream by the tokenizer tiers.
"""

from __future__ import annotations

import pytest

from primoji.preprocessor import APOSTROPHE_VARIANTS, Preprocessor
from primoji.vocabulary import CONTRACTION_SUFFIXES, DEDICATED_CONTRACTIONS


@pytest.fixture
def preprocessor() -> Preprocessor:
    return Preprocessor()


# ── Unicode normalization ───────────────────────────────────────────────────


class TestUnicodeNormalization:
    def test_right_single_quotation_mark_normalized(self, preprocessor: Preprocessor) -> None:
        """RIGHT SINGLE QUOTATION MARK (U+2019) should become ASCII apostrophe."""
        text = "don\u2019t"
        result = preprocessor.normalize_unicode(text)
        assert result == "don't"

    def test_left_single_quotation_mark_normalized(self, preprocessor: Preprocessor) -> None:
        """LEFT SINGLE QUOTATION MARK (U+2018) should become ASCII apostrophe."""
        text = "\u2018hello\u2019"
        result = preprocessor.normalize_unicode(text)
        assert result == "'hello'"

    def test_modifier_letter_apostrophe_normalized(self, preprocessor: Preprocessor) -> None:
        """MODIFIER LETTER APOSTROPHE (U+02BC) should become ASCII apostrophe."""
        text = "don\u02BCt"
        result = preprocessor.normalize_unicode(text)
        assert result == "don't"

    def test_fullwidth_apostrophe_normalized(self, preprocessor: Preprocessor) -> None:
        """FULLWIDTH APOSTROPHE (U+FF07) should become ASCII apostrophe."""
        text = "don\uFF07t"
        result = preprocessor.normalize_unicode(text)
        assert result == "don't"

    def test_grave_accent_normalized(self, preprocessor: Preprocessor) -> None:
        """GRAVE ACCENT (U+0060) should become ASCII apostrophe."""
        text = "don\u0060t"
        result = preprocessor.normalize_unicode(text)
        assert result == "don't"

    def test_nfc_normalization(self, preprocessor: Preprocessor) -> None:
        """Text should be NFC normalized (e.g. combining accent composed)."""
        # "cafe\u0301" (e + combining acute) should normalize to "caf\u00e9"
        text = "cafe\u0301"
        result = preprocessor.normalize_unicode(text)
        assert result == "caf\u00e9"

    def test_all_apostrophe_variants_covered(self) -> None:
        """All documented apostrophe variants should be in the map."""
        expected_variants = {"\u2018", "\u2019", "\u02BC", "\uFF07", "\u0060"}
        assert expected_variants == set(APOSTROPHE_VARIANTS.keys())


# ── Contraction splitting ───────────────────────────────────────────────────


class TestContractionSplitting:
    def test_dedicated_contraction_preserved(self, preprocessor: Preprocessor) -> None:
        """Dedicated contractions like \"don't\" stay as a single token."""
        result = preprocessor.split_contraction("don't")
        assert result == ["don't"]

    def test_dedicated_contraction_case_insensitive(self, preprocessor: Preprocessor) -> None:
        """Dedicated contraction check is case-insensitive."""
        result = preprocessor.split_contraction("Don't")
        assert result == ["Don't"]

    def test_uncommon_contraction_split(self, preprocessor: Preprocessor) -> None:
        """Uncommon contractions like \"shan't\" are split at the apostrophe."""
        result = preprocessor.split_contraction("shan't")
        assert result == ["shan", "'t"]

    def test_possessive_split(self, preprocessor: Preprocessor) -> None:
        """Possessive \"John's\" is split into [\"John\", \"'s\"]."""
        result = preprocessor.split_contraction("John's")
        assert result == ["John", "'s"]

    def test_would_contraction_split(self, preprocessor: Preprocessor) -> None:
        """\"he'd\" splits to [\"he\", \"'d\"]."""
        result = preprocessor.split_contraction("he'd")
        assert result == ["he", "'d"]

    def test_will_contraction_split_uncommon(self, preprocessor: Preprocessor) -> None:
        """\"she'll\" is not in dedicated contractions, so splits."""
        # "she'll" is not in DEDICATED_CONTRACTIONS (only i'll is)
        if "she'll" not in DEDICATED_CONTRACTIONS:
            result = preprocessor.split_contraction("she'll")
            assert result == ["she", "'ll"]

    def test_no_apostrophe_passes_through(self, preprocessor: Preprocessor) -> None:
        """Word without apostrophe returns as-is."""
        result = preprocessor.split_contraction("hello")
        assert result == ["hello"]

    def test_all_dedicated_contractions_are_preserved(self, preprocessor: Preprocessor) -> None:
        """Every dedicated contraction should pass through unsplit."""
        for contraction in DEDICATED_CONTRACTIONS:
            result = preprocessor.split_contraction(contraction)
            assert result == [contraction], (
                f"Dedicated contraction '{contraction}' was split"
            )


# ── Word tokenization ──────────────────────────────────────────────────────


class TestWordTokenization:
    def test_basic_sentence(self, preprocessor: Preprocessor) -> None:
        """Simple sentence splits on whitespace with punctuation separated."""
        result = preprocessor.tokenize_words("The dog ran.")
        assert "The" in result
        assert "dog" in result
        assert "ran" in result
        assert "." in result

    def test_punctuation_separated(self, preprocessor: Preprocessor) -> None:
        """Trailing punctuation should be a separate token."""
        result = preprocessor.tokenize_words("hello, world!")
        assert "," in result
        assert "!" in result

    def test_apostrophe_preserved_in_word(self, preprocessor: Preprocessor) -> None:
        """Apostrophes inside words should NOT be split off."""
        result = preprocessor.tokenize_words("don't")
        assert "don't" in result


# ── Full preprocess pipeline ────────────────────────────────────────────────


class TestPreprocessPipeline:
    def test_case_is_preserved(self, preprocessor: Preprocessor) -> None:
        """The preprocessor does NOT lowercase -- case is preserved."""
        result = preprocessor.preprocess("The Dog")
        assert "The" in result
        assert "Dog" in result

    def test_empty_input_returns_empty(self, preprocessor: Preprocessor) -> None:
        assert preprocessor.preprocess("") == []

    def test_whitespace_only_returns_empty(self, preprocessor: Preprocessor) -> None:
        assert preprocessor.preprocess("   ") == []

    def test_tab_and_newline_returns_empty(self, preprocessor: Preprocessor) -> None:
        assert preprocessor.preprocess("\t\n  ") == []

    def test_basic_sentence(self, preprocessor: Preprocessor) -> None:
        result = preprocessor.preprocess("The dog is big")
        assert "The" in result
        assert "dog" in result
        assert "is" in result
        assert "big" in result

    def test_contraction_in_sentence(self, preprocessor: Preprocessor) -> None:
        """Dedicated contraction in a sentence stays as one token."""
        result = preprocessor.preprocess("I don't know")
        assert "don't" in result

    def test_uncommon_contraction_splits_in_sentence(self, preprocessor: Preprocessor) -> None:
        """Uncommon contractions get split in the pipeline."""
        result = preprocessor.preprocess("He'd go")
        assert "He" in result
        assert "'d" in result

    def test_unicode_apostrophe_normalized_then_handled(self, preprocessor: Preprocessor) -> None:
        """Unicode apostrophe in contraction normalizes then handles correctly."""
        # "don\u2019t" → normalize → "don't" → recognized as dedicated contraction
        result = preprocessor.preprocess("I don\u2019t know")
        assert "don't" in result

    def test_punctuation_separated_in_pipeline(self, preprocessor: Preprocessor) -> None:
        result = preprocessor.preprocess("Hello, world!")
        assert "," in result
        assert "!" in result
        assert "Hello" in result
        assert "world" in result

    def test_multiple_spaces_collapsed(self, preprocessor: Preprocessor) -> None:
        """Multiple spaces should be collapsed before tokenization."""
        result = preprocessor.preprocess("dog   cat")
        assert result == ["dog", "cat"]
