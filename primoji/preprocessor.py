"""Lightweight text preprocessor for primoji tokenization.

Handles Unicode normalization, contraction expansion, and word tokenization.
Contractions are EXPANDED to full words ("won't" -> ["will", "not"]),
not split into fragments ("wo" + "n't").
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
        4. Contraction expansion ("won't" -> ["will", "not"])
        """
        text = self.normalize_unicode(text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        words = self.tokenize_words(text)

        result: list[str] = []
        for word in words:
            result.extend(self.expand_contraction(word))
        return result

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

        # Possessive 's: keep as structural token, don't expand
        if lower.endswith("'s"):
            base = word[:-2]
            if base:
                return [base, "'s"]

        # Unknown apostrophe pattern: return as-is
        return [word]

    def tokenize_words(self, text: str) -> list[str]:
        """Split text into words, preserving apostrophes within words."""
        from primoji.utils import simple_word_tokenize
        return simple_word_tokenize(text)
