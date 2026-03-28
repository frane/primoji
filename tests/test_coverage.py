"""Stub tests for future FineWeb-Edu coverage testing.

Contains placeholder tests and basic dictionary/tokenizer coverage checks.
The full coverage test will run against 1,000 random FineWeb-Edu sentences
once the dataset is tokenized (see scripts/coverage_test.py).
"""

from __future__ import annotations

import pytest

from primoji.dictionary import Dictionary
from primoji.tokenizer import Tokenizer
from primoji.utils import SpecialTokens


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer()


@pytest.fixture
def dictionary() -> Dictionary:
    return Dictionary()


# ── Placeholder for future FineWeb-Edu coverage ─────────────────────────────


class TestFineWebCoverage:
    def test_placeholder(self) -> None:
        """Placeholder: full FineWeb-Edu coverage testing not yet implemented.

        TODO: Download 1,000 random FineWeb-Edu sentences, tokenize each,
        measure coverage (% of words that get non-UNK encodings), and assert
        coverage exceeds a target threshold (e.g. 70%+). See scripts/coverage_test.py.
        """
        assert True


# ── Dictionary loading ───────────────────────────────────────────────────────


class TestDictionaryLoading:
    def test_bootstrap_dictionary_has_entries(self, dictionary: Dictionary) -> None:
        """The bootstrap dictionary should have a meaningful number of entries."""
        assert dictionary.size() > 0

    def test_bootstrap_dictionary_minimum_size(self, dictionary: Dictionary) -> None:
        """Bootstrap dictionary should contain at least 50 hardcoded mappings."""
        assert dictionary.size() >= 50

    def test_dictionary_contains_common_nouns(self, dictionary: Dictionary) -> None:
        common_nouns = ["dog", "cat", "fish", "fire", "water", "house"]
        for word in common_nouns:
            assert dictionary.contains(word), f"Dictionary missing common noun '{word}'"

    def test_dictionary_contains_verbs(self, dictionary: Dictionary) -> None:
        verbs = ["think", "know", "feel", "see", "say", "move", "grow", "teach"]
        for word in verbs:
            assert dictionary.contains(word), f"Dictionary missing verb '{word}'"

    def test_dictionary_contains_adjectives(self, dictionary: Dictionary) -> None:
        adjectives = ["big", "small", "good", "bad", "dark"]
        for word in adjectives:
            assert dictionary.contains(word), f"Dictionary missing adjective '{word}'"

    def test_dictionary_contains_composed_concepts(self, dictionary: Dictionary) -> None:
        concepts = ["photosynthesis", "computer", "internet", "teacher"]
        for word in concepts:
            assert dictionary.contains(word), f"Dictionary missing concept '{word}'"

    def test_dictionary_lookup_returns_list(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("dog")
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_dictionary_lookup_missing_word_returns_none(self, dictionary: Dictionary) -> None:
        assert dictionary.lookup("xyzzyplugh") is None

    def test_dictionary_reverse_lookup(self, dictionary: Dictionary) -> None:
        """Forward then reverse lookup should return a word mapping to the same IDs."""
        ids = dictionary.lookup("dog")
        assert ids is not None
        word = dictionary.reverse_lookup(ids)
        assert word is not None, "Reverse lookup returned None for dog's IDs"
        # The reverse lookup may return a different word that maps to the same IDs
        # (e.g. a synonym from the seed dictionary). Verify it maps back.
        reverse_ids = dictionary.lookup(word)
        assert reverse_ids == ids


# ── Word type coverage ───────────────────────────────────────────────────────


class TestWordTypeCoverage:
    def test_concrete_nouns(self, tok: Tokenizer) -> None:
        """Concrete nouns (Tier 1 emoji) should encode to short sequences."""
        nouns = ["dog", "cat", "tree", "fire", "water", "apple", "star"]
        for noun in nouns:
            ids = tok.encode(noun)
            assert len(ids) >= 1, f"'{noun}' produced no tokens"
            assert SpecialTokens.UNK not in ids, f"'{noun}' produced UNK"

    def test_action_verbs(self, tok: Tokenizer) -> None:
        """Common verbs should map to primitive IDs, not UNK."""
        verbs = ["think", "know", "feel", "see", "say", "move", "grow"]
        for verb in verbs:
            ids = tok.encode(verb)
            assert len(ids) >= 1
            assert SpecialTokens.UNK not in ids, f"Verb '{verb}' produced UNK"

    def test_adjectives(self, tok: Tokenizer) -> None:
        """Common adjectives should map to descriptor primitives, not UNK."""
        adjectives = ["big", "small", "good", "bad", "dark", "bright"]
        for adj in adjectives:
            ids = tok.encode(adj)
            assert len(ids) >= 1
            assert SpecialTokens.UNK not in ids, f"Adjective '{adj}' produced UNK"

    def test_function_words(self, tok: Tokenizer) -> None:
        """Function words should either be dropped (empty) or mapped, never UNK."""
        function_words = ["the", "a", "an", "is", "not", "can", "if", "because"]
        for fw in function_words:
            ids = tok.encode(fw)
            # May be empty (articles) or mapped — but never UNK
            assert SpecialTokens.UNK not in ids, f"Function word '{fw}' produced UNK"

    def test_abstract_concepts(self, tok: Tokenizer) -> None:
        """Abstract concepts should encode via composition, not UNK."""
        concepts = ["knowledge", "education", "society", "war", "peace", "life", "death"]
        for concept in concepts:
            ids = tok.encode(concept)
            assert len(ids) >= 1
            assert SpecialTokens.UNK not in ids, f"Concept '{concept}' produced UNK"

    def test_composed_technical_terms(self, tok: Tokenizer) -> None:
        """Technical terms should compose to multi-token sequences."""
        terms = ["photosynthesis", "computer", "internet", "telephone", "television"]
        for term in terms:
            ids = tok.encode(term)
            assert len(ids) >= 2, f"'{term}' should compose to 2+ tokens, got {len(ids)}"
            assert SpecialTokens.UNK not in ids, f"'{term}' produced UNK"
