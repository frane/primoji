"""Conservative fuzzy matching for typo correction (Tier 3).

Uses edit distance 1 with strict constraints:
- Only correct words NOT in the dictionary (non-word errors only)
- Only correct if exactly ONE candidate exists (unique match)
- Only correct words with 4+ characters (short words too ambiguous)
- Never correct words that ARE real words (no real-word error correction)

Optional dependency: pip install primoji[spelling]
Falls back to brute-force Levenshtein if symspellpy not installed.
"""

from __future__ import annotations


class FuzzyMatcher:
    """Conservative spell corrector for the primoji tokenizer."""

    def __init__(self, vocabulary_words: set[str]) -> None:
        """Initialize with the set of known vocabulary words.

        Args:
            vocabulary_words: Set of all words in the primoji dictionary.
        """
        self._vocab_words = vocabulary_words
        self._symspell = None
        self._try_load_symspell()

    def _try_load_symspell(self) -> None:
        """Try to load SymSpell for fast fuzzy matching."""
        try:
            from symspellpy import SymSpell, Verbosity

            self._symspell = SymSpell(max_dictionary_edit_distance=1)
            for word in self._vocab_words:
                self._symspell.create_dictionary_entry(word, 1)
            self._verbosity = Verbosity
        except ImportError:
            self._symspell = None

    def correct(self, word: str) -> str | None:
        """Attempt to correct a misspelled word.

        Returns the corrected word if ALL conditions are met:
        1. word is NOT in the vocabulary (it's unknown)
        2. len(word) >= 4 (short words too ambiguous)
        3. Exactly ONE vocabulary word exists within edit distance 1
        4. The candidate is unambiguous

        Returns None if no safe correction exists (word should fall
        through to byte fallback).

        Args:
            word: A single unknown word to attempt correction on.

        Returns:
            Corrected word string, or None if no safe correction.
        """
        # Don't correct known words or short words
        if word in self._vocab_words or len(word) < 4:
            return None

        if self._symspell is not None:
            return self._correct_symspell(word)
        return self._correct_brute_force(word)

    def _correct_symspell(self, word: str) -> str | None:
        """Correct using SymSpell (fast, O(1) lookup)."""
        suggestions = self._symspell.lookup(
            word, self._verbosity.ALL, max_edit_distance=1
        )
        # Filter to only candidates that are actually in our vocab
        candidates = [s.term for s in suggestions if s.term in self._vocab_words]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def _correct_brute_force(self, word: str) -> str | None:
        """Correct using brute-force Levenshtein (fallback when no SymSpell)."""
        candidates: list[str] = []
        for vw in self._vocab_words:
            if _edit_distance_one(word, vw):
                candidates.append(vw)
                if len(candidates) > 1:
                    return None  # Ambiguous — bail early
        if len(candidates) == 1:
            return candidates[0]
        return None

    @property
    def is_available(self) -> bool:
        """Whether SymSpell is installed and functional."""
        return self._symspell is not None


def _edit_distance_one(a: str, b: str) -> bool:
    """Check if two strings are within edit distance 1 (Levenshtein).

    Handles insertions, deletions, substitutions, and transpositions.

    Args:
        a: First string.
        b: Second string.

    Returns:
        True if edit distance is exactly 1.
    """
    la, lb = len(a), len(b)

    if abs(la - lb) > 1:
        return False

    if la == lb:
        diffs = 0
        for i in range(la):
            if a[i] != b[i]:
                diffs += 1
                if diffs > 1:
                    if diffs == 2 and i > 0 and a[i - 1] == b[i] and a[i] == b[i - 1]:
                        if a[i + 1 :] == b[i + 1 :]:
                            return True
                    return False
        return diffs == 1

    shorter, longer = (a, b) if la < lb else (b, a)
    diffs = 0
    i, j = 0, 0
    while i < len(shorter) and j < len(longer):
        if shorter[i] != longer[j]:
            diffs += 1
            if diffs > 1:
                return False
            j += 1
        else:
            i += 1
            j += 1
    return True
