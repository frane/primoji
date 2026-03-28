"""Tests for the FuzzyMatcher (Tier 3 conservative typo correction).

The FuzzyMatcher only corrects when ALL conditions are met:
  1. Word is NOT already in the vocabulary
  2. Word has 4+ characters (short words too ambiguous)
  3. Exactly ONE candidate within edit distance 1
  4. The candidate is unambiguous
"""

from __future__ import annotations

import pytest

from primoji.fuzzy import FuzzyMatcher, _edit_distance_one


# ── Test vocabulary for FuzzyMatcher ────────────────────────────────────────

_TEST_VOCAB: set[str] = {
    "water", "fire", "tree", "house", "dog", "cat", "fish",
    "teacher", "student", "computer", "internet", "book",
    "the", "and", "for", "bat", "car", "bar",
}


@pytest.fixture
def matcher() -> FuzzyMatcher:
    """Create a FuzzyMatcher with the test vocabulary."""
    return FuzzyMatcher(_TEST_VOCAB)


# ── Basic correction ───────────────────────────────────────────────────────


class TestBasicCorrection:
    def test_simple_typo_corrected(self, matcher: FuzzyMatcher) -> None:
        """'watir' is 1 edit from 'water', unique, 5 chars => corrected."""
        result = matcher.correct("watir")
        assert result == "water"

    def test_substitution_typo(self, matcher: FuzzyMatcher) -> None:
        """'hoase' is 1 edit from 'house', unique, 5 chars => corrected."""
        result = matcher.correct("hoase")
        assert result == "house"

    def test_insertion_typo(self, matcher: FuzzyMatcher) -> None:
        """'treee' is 1 edit from 'tree', unique, 5 chars => corrected."""
        result = matcher.correct("treee")
        assert result == "tree"

    def test_deletion_typo(self, matcher: FuzzyMatcher) -> None:
        """'techer' is 1 edit from 'teacher', unique, 6 chars => corrected."""
        result = matcher.correct("techer")
        assert result == "teacher"


# ── Returns None when correction is unsafe ─────────────────────────────────


class TestReturnsNone:
    def test_known_word_returns_none(self, matcher: FuzzyMatcher) -> None:
        """Words already in vocab should NOT be corrected."""
        assert matcher.correct("water") is None
        assert matcher.correct("dog") is None

    def test_short_word_returns_none(self, matcher: FuzzyMatcher) -> None:
        """Words with fewer than 4 characters are never corrected."""
        assert matcher.correct("teh") is None
        assert matcher.correct("dg") is None
        assert matcher.correct("fo") is None

    def test_ambiguous_candidates_returns_none(self) -> None:
        """When multiple candidates exist at distance 1, return None."""
        # Build a vocab where "tost" is 1 edit from both "test" and "tost"
        # won't work — need two distinct words both at distance 1 from input
        vocab = {"test", "text", "tent", "toast"}
        m = FuzzyMatcher(vocab)
        # Force brute-force so we don't depend on SymSpell internals
        m._symspell = None
        # "teat" is 1 edit from "test" (a→s), "text" (a→x), "tent" (a→n) => 3 candidates
        result = m.correct("teat")
        assert result is None

    def test_no_candidate_returns_none(self, matcher: FuzzyMatcher) -> None:
        """Completely unknown word with no close match returns None."""
        result = matcher.correct("xyzzyplugh")
        assert result is None

    def test_three_char_word_even_with_unique_match(self, matcher: FuzzyMatcher) -> None:
        """Even if a 3-char word has a unique candidate, still return None."""
        # "dag" is 1 edit from "dog" but only 3 chars
        assert matcher.correct("dag") is None


# ── Brute-force fallback ───────────────────────────────────────────────────


class TestBruteForceFallback:
    def test_works_without_symspellpy(self) -> None:
        """FuzzyMatcher should work even if symspellpy is not installed."""
        matcher = FuzzyMatcher({"water", "fire", "house", "teacher"})
        # Force brute-force path by disabling symspell
        matcher._symspell = None
        result = matcher.correct("watir")
        assert result == "water"

    def test_brute_force_ambiguous_returns_none(self) -> None:
        """Brute-force path should also return None for ambiguous matches."""
        matcher = FuzzyMatcher({"test", "text", "tent"})
        matcher._symspell = None
        # "teat" is 1 edit from both "test" and "text" => ambiguous
        result = matcher.correct("teat")
        assert result is None

    def test_brute_force_no_match_returns_none(self) -> None:
        matcher = FuzzyMatcher({"water"})
        matcher._symspell = None
        result = matcher.correct("zzzzz")
        assert result is None


# ── is_available property ──────────────────────────────────────────────────


class TestIsAvailable:
    def test_is_available_is_bool(self, matcher: FuzzyMatcher) -> None:
        assert isinstance(matcher.is_available, bool)


# ── _edit_distance_one helper ──────────────────────────────────────────────


class TestEditDistanceOne:
    def test_identical_strings(self) -> None:
        assert not _edit_distance_one("hello", "hello")

    def test_single_substitution(self) -> None:
        assert _edit_distance_one("cat", "bat")

    def test_single_insertion(self) -> None:
        assert _edit_distance_one("cat", "cats")

    def test_single_deletion(self) -> None:
        assert _edit_distance_one("cats", "cat")

    def test_transposition(self) -> None:
        assert _edit_distance_one("teh", "the")

    def test_two_substitutions_returns_false(self) -> None:
        assert not _edit_distance_one("cat", "dog")

    def test_length_diff_too_large(self) -> None:
        assert not _edit_distance_one("a", "abc")

    def test_empty_vs_single_char(self) -> None:
        assert _edit_distance_one("", "a")

    def test_both_empty(self) -> None:
        assert not _edit_distance_one("", "")
