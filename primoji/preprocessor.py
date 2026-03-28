"""Lightweight text preprocessor for primoji tokenization.

Handles Unicode normalization, contraction splitting, and word tokenization.
Does NOT perform spell correction — that's handled by the FuzzyMatcher in
the tokenizer's Tier 3, and unknown words fall through to byte encoding.

Design principle: be maximally permissive. Let the tiered tokenizer
handle noise rather than trying to fix it here.
"""

from __future__ import annotations

import re
import unicodedata

from primoji.vocabulary import CONTRACTION_SUFFIXES, DEDICATED_CONTRACTIONS

# Apostrophe variants to normalize → standard ASCII apostrophe
APOSTROPHE_VARIANTS: dict[str, str] = {
    "\u2018": "'",  # LEFT SINGLE QUOTATION MARK
    "\u2019": "'",  # RIGHT SINGLE QUOTATION MARK
    "\u02BC": "'",  # MODIFIER LETTER APOSTROPHE
    "\uFF07": "'",  # FULLWIDTH APOSTROPHE
    "\u0060": "'",  # GRAVE ACCENT (sometimes used as apostrophe)
}

_APOSTROPHE_RE = re.compile(
    "[" + "".join(re.escape(c) for c in APOSTROPHE_VARIANTS) + "]"
)


class Preprocessor:
    """Lightweight text preprocessor.

    Normalizes Unicode, splits contractions, and tokenizes words.
    Does NOT lowercase or spell-correct — those are handled downstream.
    """

    def preprocess(self, text: str) -> list[str]:
        """Normalize and tokenize text into words.

        Pipeline:
        1. Unicode NFC normalization
        2. Apostrophe variant normalization
        3. Word tokenization (split on whitespace + punctuation)
        4. Contraction handling (dedicated tokens or apostrophe split)

        Args:
            text: Raw input text.

        Returns:
            List of normalized word strings ready for tiered lookup.
        """
        text = self.normalize_unicode(text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        words = self.tokenize_words(text)

        # Handle contractions in each word
        result: list[str] = []
        for word in words:
            result.extend(self.split_contraction(word))
        return result

    def normalize_unicode(self, text: str) -> str:
        """NFC normalize and standardize apostrophe variants.

        Args:
            text: Raw input text.

        Returns:
            NFC-normalized text with standardized apostrophes.
        """
        text = unicodedata.normalize("NFC", text)
        text = _APOSTROPHE_RE.sub("'", text)
        return text

    def split_contraction(self, word: str) -> list[str]:
        """Split a contraction at the apostrophe boundary.

        If the word is a dedicated contraction (top ~20), return as-is.
        Otherwise split: "wouldn't" → ["wouldn", "'t"]

        Args:
            word: A single word potentially containing an apostrophe.

        Returns:
            List of 1 or 2 strings.
        """
        if "'" not in word:
            return [word]

        # Check if it's a dedicated whole-contraction token
        if word.lower() in DEDICATED_CONTRACTIONS:
            return [word]

        # Find the apostrophe and try to match a known suffix
        apos_idx = word.index("'")
        suffix = word[apos_idx:]
        if suffix.lower() in CONTRACTION_SUFFIXES:
            prefix = word[:apos_idx]
            if prefix:
                return [prefix, suffix]

        # No known suffix pattern — return as-is
        return [word]

    def tokenize_words(self, text: str) -> list[str]:
        """Split text into words, preserving punctuation as separate tokens.

        Preserves apostrophes within words for contraction handling.

        Args:
            text: Normalized input text.

        Returns:
            List of word/punctuation tokens.
        """
        from primoji.utils import simple_word_tokenize

        return simple_word_tokenize(text)
