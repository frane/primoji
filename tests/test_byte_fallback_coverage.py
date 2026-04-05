"""Tests that common English words never hit byte fallback.

These catch the exact bug found in v6: 46 top-3K English words
(place, force, material, social, great, end, business, etc.)
were falling through to byte fallback because they were missing
from both the dictionary and common_words list.
"""

from __future__ import annotations

import pytest

from primoji import Tokenizer
from primoji.byte_fallback import BYTES_START_ID


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer(fuzzy=False)


class TestTop200NeverByteFallback:
    """The 200 most frequent English words must NEVER be byte fallback."""

    TOP_200 = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their",
        "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
        "me", "when", "make", "can", "like", "time", "no", "just", "him",
        "know", "take", "people", "into", "year", "your", "good", "some",
        "could", "them", "see", "other", "than", "then", "now", "look", "only",
        "come", "its", "over", "think", "also", "back", "after", "use", "two",
        "how", "our", "work", "first", "well", "way", "even", "new", "want",
        "because", "any", "these", "give", "day", "most", "us",
        "great", "between", "being", "under", "never", "place", "same",
        "while", "last", "long", "very", "still", "own", "point", "form",
        "end", "where", "much", "again", "case", "right", "old", "run",
        "high", "world", "home", "small", "kind", "hand", "large", "turn",
        "number", "every", "move", "state", "group", "begin", "side", "part",
        "change", "keep", "found", "play", "close", "far", "night", "real",
        "city", "line", "open", "name", "country", "school", "power",
        "water", "fact", "head", "order", "system", "different", "land",
        "question", "area", "social", "room", "face", "public", "read",
        "family", "body", "level", "problem", "field", "force", "reason",
        "set", "study", "light", "human", "business", "few", "able",
        "children", "important", "table", "example", "during", "history",
        "material", "local", "young", "mother", "result",
    ]

    @pytest.mark.parametrize("word", TOP_200)
    def test_word_not_byte_fallback(self, tok: Tokenizer, word: str) -> None:
        ids = tok.encode(word)
        assert BYTES_START_ID not in ids, (
            f"Top-200 word '{word}' hit byte fallback: {ids}. "
            f"Add it to dictionary or common_words.json."
        )


class TestFineWebEduVocab:
    """Words that appear frequently in FineWeb-Edu must not be byte fallback."""

    FINEWEB_WORDS = [
        # Scientific
        "photosynthesis", "evolution", "temperature", "species",
        "population", "environment", "ecosystem", "climate",
        "gravity", "electricity", "chemical", "molecule",
        # Educational
        "education", "university", "student", "teacher", "research",
        "experiment", "evidence", "analysis", "theory",
        # Social/political
        "government", "democracy", "economy", "society",
        "culture", "religion", "philosophy", "law",
        # The v6 bug words (previously byte fallback)
        "place", "force", "material", "social", "great", "end",
        "business", "fact", "future", "reason", "study", "action",
        "side", "together", "close", "past",
        # The v7 fixes (mapped to primitives/compositions)
        "distant", "identical", "beneath", "remote", "extent",
        "studied", "studying", "ruled", "ruling", "lands",
        "locations", "element", "fighter", "disability",
    ]

    @pytest.mark.parametrize("word", FINEWEB_WORDS)
    def test_word_not_byte_fallback(self, tok: Tokenizer, word: str) -> None:
        ids = tok.encode(word)
        assert BYTES_START_ID not in ids, (
            f"FineWeb-Edu word '{word}' hit byte fallback: {ids}. "
            f"This word needs a dictionary entry, composition, or word token."
        )


class TestPunctuationSeparation:
    """Words with attached punctuation must not cause byte fallback.

    Bug found: benchmark used s.split() which kept punctuation attached,
    making 'world.' classify as byte fallback even though 'world' was fine.
    The encode() pipeline correctly separates punctuation.
    """

    @pytest.mark.parametrize("text,expected_no_bytes", [
        ("world.", True),
        ("however,", True),
        ("system,", True),
        ("example,", True),
        ("history.", True),
    ])
    def test_encode_separates_punctuation(
        self, tok: Tokenizer, text: str, expected_no_bytes: bool
    ) -> None:
        ids = tok.encode(text)
        has_bytes = BYTES_START_ID in ids
        assert not has_bytes, (
            f"encode('{text}') hit byte fallback: {ids}. "
            f"Preprocessor should split punctuation from word."
        )
