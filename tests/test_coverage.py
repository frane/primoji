"""Tests that verify tokenizer coverage on real text.

These catch dictionary gaps that cause excessive byte fallback.
"""

from __future__ import annotations

import pytest

from primoji import Tokenizer
from primoji.byte_fallback import BYTES_START_ID


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer(fuzzy=False)


class TestCommonWordsEncoded:
    """The 100 most common English words must NOT go to byte fallback."""

    # Source: Oxford English Corpus top 100
    COMMON_WORDS = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their",
        "what", "so", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know",
        "take", "people", "into", "year", "your", "good", "some", "could",
        "them", "see", "other", "than", "then", "now", "look", "only", "come",
        "its", "over", "think", "also", "back", "after", "use", "two", "how",
        "our", "work", "first", "well", "way", "even", "new", "want",
        "because", "any", "these", "give", "day", "most", "us",
    ]

    @pytest.mark.parametrize("word", COMMON_WORDS)
    def test_common_word_not_byte_fallback(self, tok: Tokenizer, word: str) -> None:
        ids = tok.encode(word)
        assert BYTES_START_ID not in ids, (
            f"Common word '{word}' hit byte fallback: {ids}"
        )


class TestEducationalVocab:
    """Key FineWeb-Edu vocabulary must encode without byte fallback."""

    EDUCATIONAL_WORDS = [
        "photosynthesis", "evolution", "democracy", "temperature",
        "science", "mathematics", "history", "biology", "chemistry",
        "physics", "geography", "government", "population", "environment",
        "technology", "university", "student", "teacher", "experiment",
    ]

    @pytest.mark.parametrize("word", EDUCATIONAL_WORDS)
    def test_educational_word_not_byte_fallback(self, tok: Tokenizer, word: str) -> None:
        ids = tok.encode(word)
        assert BYTES_START_ID not in ids, (
            f"Educational word '{word}' hit byte fallback: {ids}"
        )


class TestSentenceCoverage:
    """Real sentences should not have too many byte-fallback tokens."""

    SAMPLE_SENTENCES = [
        "The teacher explained how photosynthesis converts light energy into chemical energy.",
        "Water evaporates when heated and rises into the atmosphere as water vapor.",
        "The United States declared independence from Great Britain in 1776.",
        "Scientists use the scientific method to test their hypotheses.",
        "The population of the world has grown significantly in the last century.",
    ]

    @pytest.mark.parametrize("sentence", SAMPLE_SENTENCES)
    def test_sentence_mostly_covered(self, tok: Tokenizer, sentence: str) -> None:
        ids = tok.encode(sentence)
        byte_count = sum(1 for tid in ids if tid == BYTES_START_ID)
        total = len(ids)
        byte_fraction = byte_count / total if total > 0 else 0
        # Less than 40% of tokens should be byte-fallback START markers
        assert byte_fraction < 0.4, (
            f"Too many byte fallback tokens ({byte_fraction:.0%}): {sentence[:60]}..."
        )
