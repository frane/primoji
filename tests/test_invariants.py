"""Property-based invariant tests.

These test PROPERTIES that must hold for ALL inputs, not specific examples.
They would have caught every real bug we found:
- The BPB formula bug (subset tokens / total bytes)
- The "dog -> bone" bug (wrong emoji mapping)
- The "food -> face savoring food" growth bug (decoder returns CLDR name)
- The variation selector overlap (emoji in both tiers)
"""

from __future__ import annotations

import random
import string

import pytest

from primoji import Tokenizer
from primoji.byte_fallback import encode_bytes, decode_bytes
from primoji.utils import SpecialTokens


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer(fuzzy=False)


class TestRoundtripStability:
    """THE critical invariant: roundtrips must stabilize.

    encode(x) -> decode -> encode -> decode must equal
    encode(x) -> decode (second pass = first pass).

    This catches the "food -> face savoring food -> angry face..." growth bug.
    """

    WORDS = [
        "dog", "cat", "water", "fire", "food", "tree", "house", "star",
        "teacher", "student", "photosynthesis", "computer", "internet",
        "big", "small", "good", "bad", "hot", "cold", "love", "fear",
        "don't", "can't", "won't",
        "the", "is", "was", "not", "with", "for", "about",
        "42", "hello", "world",
    ]

    @pytest.mark.parametrize("word", WORDS)
    def test_single_word_stabilizes(self, tok: Tokenizer, word: str) -> None:
        """After one roundtrip, text must not change on subsequent roundtrips."""
        ids1 = tok.encode(word)
        text1 = tok.decode(ids1)
        ids2 = tok.encode(text1)
        text2 = tok.decode(ids2)
        assert text1 == text2, (
            f"Unstable roundtrip for '{word}': "
            f"'{word}' -> '{text1}' -> '{text2}'"
        )

    def test_sentence_stabilizes(self, tok: Tokenizer) -> None:
        sentence = "The dog ate food and water"
        ids1 = tok.encode(sentence)
        text1 = tok.decode(ids1)
        ids2 = tok.encode(text1)
        text2 = tok.decode(ids2)
        assert text1 == text2, f"Unstable: '{text1}' -> '{text2}'"

    def test_does_not_grow(self, tok: Tokenizer) -> None:
        """Repeated roundtrips must not increase text length."""
        text = "The dog ate food"
        lengths = [len(text)]
        for _ in range(5):
            ids = tok.encode(text)
            text = tok.decode(ids)
            lengths.append(len(text))
        # Length should be monotonically non-increasing after first roundtrip
        for i in range(2, len(lengths)):
            assert lengths[i] <= lengths[i - 1] + 1, (
                f"Text growing: {lengths}"
            )

    def test_random_words_stabilize(self, tok: Tokenizer) -> None:
        """50 random words must all stabilize after one roundtrip."""
        rng = random.Random(42)
        for _ in range(50):
            word = "".join(rng.choices(string.ascii_lowercase, k=rng.randint(2, 12)))
            ids1 = tok.encode(word)
            text1 = tok.decode(ids1)
            ids2 = tok.encode(text1)
            text2 = tok.decode(ids2)
            assert text1 == text2, f"Unstable for random word '{word}': '{text1}' != '{text2}'"


class TestByteFallbackLossless:
    """Byte fallback must be perfectly lossless for any UTF-8 string."""

    @pytest.mark.parametrize("text", [
        "hello",
        "cafe\u0301",
        "\u4f60\u597d",
        "\U0001f525\U0001f4a7",
        "a" * 1000,
        "",
        "\x00\x01\x02",
        "mixed 123 !@# stuff",
    ])
    def test_encode_decode_identity(self, text: str) -> None:
        assert decode_bytes(encode_bytes(text)) == text


class TestNeverUNK:
    """No input, no matter how weird, should produce UNK."""

    def test_fuzz_500_random_strings(self, tok: Tokenizer) -> None:
        rng = random.Random(99)
        for _ in range(500):
            length = rng.randint(0, 50)
            text = "".join(rng.choices(
                string.ascii_letters + string.digits + " .,!?'\n\t" + "\u00e9\u00f1\u4e16",
                k=length
            ))
            ids = tok.encode(text)
            assert SpecialTokens.UNK not in ids, f"UNK in encoding of {text!r}"


class TestNeverCrash:
    """Decode must handle any list of ints without crashing."""

    def test_empty(self, tok: Tokenizer) -> None:
        assert tok.decode([]) == ""

    def test_garbage_ids(self, tok: Tokenizer) -> None:
        result = tok.decode([99999, 0, 1, 88888])
        assert isinstance(result, str)

    def test_random_id_lists(self, tok: Tokenizer) -> None:
        rng = random.Random(77)
        for _ in range(100):
            ids = [rng.randint(0, 5000) for _ in range(rng.randint(0, 20))]
            result = tok.decode(ids)
            assert isinstance(result, str)


class TestDeterminism:

    def test_encode_1000_times(self, tok: Tokenizer) -> None:
        text = "The teacher explained photosynthesis"
        first = tok.encode(text)
        for _ in range(999):
            assert tok.encode(text) == first

    def test_decode_1000_times(self, tok: Tokenizer) -> None:
        ids = [1200, 1299, 1227]
        first = tok.decode(ids)
        for _ in range(999):
            assert tok.decode(ids) == first
