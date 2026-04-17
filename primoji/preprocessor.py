"""Lightweight text preprocessor for primoji tokenization.

Handles Unicode normalization, contraction expansion, possessive splitting,
slash/hyphen compound splitting, and word tokenization.
"""

from __future__ import annotations

import re
import unicodedata


# Apostrophe variants to normalize
APOSTROPHE_VARIANTS: dict[str, str] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u02BC": "'",
    "\uFF07": "'",
    "\u0060": "'",
}

_APOSTROPHE_RE = re.compile(
    "[" + "".join(re.escape(c) for c in APOSTROPHE_VARIANTS) + "]"
)

# Semantic contraction expansion table
CONTRACTION_EXPANSIONS: dict[str, list[str]] = {
    "won't": ["will", "not"],
    "don't": ["do", "not"],
    "can't": ["can", "not"],
    "isn't": ["is", "not"],
    "aren't": ["are", "not"],
    "wasn't": ["was", "not"],
    "weren't": ["were", "not"],
    "couldn't": ["could", "not"],
    "wouldn't": ["would", "not"],
    "shouldn't": ["should", "not"],
    "doesn't": ["does", "not"],
    "didn't": ["did", "not"],
    "hasn't": ["has", "not"],
    "haven't": ["have", "not"],
    "hadn't": ["had", "not"],
    "mustn't": ["must", "not"],
    "needn't": ["need", "not"],
    "shan't": ["shall", "not"],
    "mightn't": ["might", "not"],
    "i'm": ["i", "am"],
    "it's": ["it", "is"],
    "he's": ["he", "is"],
    "she's": ["she", "is"],
    "that's": ["that", "is"],
    "there's": ["there", "is"],
    "here's": ["here", "is"],
    "what's": ["what", "is"],
    "who's": ["who", "is"],
    "how's": ["how", "is"],
    "where's": ["where", "is"],
    "i've": ["i", "have"],
    "you've": ["you", "have"],
    "we've": ["we", "have"],
    "they've": ["they", "have"],
    "could've": ["could", "have"],
    "would've": ["would", "have"],
    "should've": ["should", "have"],
    "might've": ["might", "have"],
    "must've": ["must", "have"],
    "i'll": ["i", "will"],
    "you'll": ["you", "will"],
    "he'll": ["he", "will"],
    "she'll": ["she", "will"],
    "it'll": ["it", "will"],
    "we'll": ["we", "will"],
    "they'll": ["they", "will"],
    "that'll": ["that", "will"],
    "you're": ["you", "are"],
    "we're": ["we", "are"],
    "they're": ["they", "are"],
    "i'd": ["i", "would"],
    "you'd": ["you", "would"],
    "he'd": ["he", "would"],
    "she'd": ["she", "would"],
    "we'd": ["we", "would"],
    "they'd": ["they", "would"],
    "it'd": ["it", "would"],
    "let's": ["let", "us"],
    "ain't": ["am", "not"],
}


class Preprocessor:
    """Lightweight text preprocessor.

    Normalizes Unicode, expands contractions to full words, tokenizes.
    """

    def preprocess(self, text: str) -> list[str]:
        """Normalize and tokenize text into words.

        Pipeline:
        1. Unicode NFC normalization
        2. Apostrophe variant normalization
        3. Word tokenization
        4. Slash splitting ("and/or" -> ["and", "or"], URLs preserved)
        5. Hyphen splitting ("insulin-deprived" -> ["insulin", "deprived"])
        6. Contraction expansion ("won't" -> ["will", "not"])
        7. Possessive splitting ("John's" -> ["John", "<POSSESSIVE>"])
        """
        text = self.normalize_unicode(text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        words = self.tokenize_words(text)

        result: list[str] = []
        for word in words:
            # Split slashes first (but not in URLs), then hyphens, then contractions
            slash_parts = self.split_slash(word)
            for spart in slash_parts:
                hyp_parts = self.split_hyphen(spart)
                for part in hyp_parts:
                    result.extend(self.expand_contraction(part))
        return result

    @staticmethod
    def split_slash(word: str) -> list[str]:
        """Split slash-separated compounds into components.

        "and/or" -> ["and", "or"]
        "input/output" -> ["input", "output"]
        "http://example.com" -> ["http://example.com"] (URL preserved)
        "TCP/IP" -> ["TCP", "IP"]
        """
        if "/" not in word:
            return [word]
        # Don't split URLs (contains ://)
        if "://" in word:
            return [word]
        parts = [p for p in word.split("/") if p]
        return parts if parts else [word]

    @staticmethod
    def split_hyphen(word: str) -> list[str]:
        """Split hyphenated words into components.

        "insulin-deprived" -> ["insulin", "deprived"]
        "self-driving" -> ["self", "driving"]
        "x-ray" -> ["x", "ray"]
        "non-" -> ["non"] (trailing hyphen stripped)
        """
        if "-" not in word:
            return [word]
        parts = [p for p in word.split("-") if p]
        return parts if parts else [word]

    def normalize_unicode(self, text: str) -> str:
        """NFC normalize and standardize apostrophe variants."""
        text = unicodedata.normalize("NFC", text)
        text = _APOSTROPHE_RE.sub("'", text)
        return text

    def expand_contraction(self, word: str) -> list[str]:
        """Expand a contraction to its full semantic components.

        "won't" -> ["will", "not"]
        "John's" -> ["John", "'s"] (possessive, not expanded)
        "dog" -> ["dog"] (not a contraction)
        """
        if "'" not in word:
            return [word]

        lower = word.lower()

        # Check expansion table
        if lower in CONTRACTION_EXPANSIONS:
            return CONTRACTION_EXPANSIONS[lower]

        # Possessive 's: split into base + possessive marker
        if lower.endswith("'s"):
            base = word[:-2]
            if base:
                return [base, "<POSSESSIVE>"]

        # Plural possessive: workers' -> workers + possessive
        if word.endswith("'") and len(word) > 1 and word[-2].isalpha():
            base = word[:-1]
            if base:
                return [base, "<POSSESSIVE>"]

        # Unknown apostrophe pattern: return as-is
        return [word]

    def tokenize_words(self, text: str) -> list[str]:
        """Split text into words, preserving apostrophes within words."""
        from primoji.utils import simple_word_tokenize
        return simple_word_tokenize(text)
