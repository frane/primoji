"""Tests for compositional embedding alias map.

Verifies that grammar words map to correct primitive decompositions
and that the alias map builds correctly from the tokenizer.
"""

from __future__ import annotations

import pytest

from primoji import Tokenizer
from primoji.alias_map import GRAMMAR_ALIASES, build_alias_map
from primoji.primitives import get_primitive_by_name


@pytest.fixture
def tok() -> Tokenizer:
    return Tokenizer(fuzzy=False)


@pytest.fixture
def alias_map(tok: Tokenizer) -> dict[int, list[int]]:
    return build_alias_map(tok.encode)


class TestGrammarAliases:
    """Grammar aliases must map to valid primitives."""

    def test_all_primitive_names_exist(self) -> None:
        """Every primitive name in GRAMMAR_ALIASES must be a real primitive."""
        for word, prim_names in GRAMMAR_ALIASES.items():
            for name in prim_names:
                p = get_primitive_by_name(name)
                assert p is not None, f"'{word}' references unknown primitive '{name}'"

    def test_is_maps_to_be_and_now(self) -> None:
        assert GRAMMAR_ALIASES["is"] == ["BE", "NOW"]

    def test_was_maps_to_be_and_before(self) -> None:
        assert GRAMMAR_ALIASES["was"] == ["BE", "BEFORE"]

    def test_not_maps_to_not(self) -> None:
        assert GRAMMAR_ALIASES["not"] == ["NOT"]

    def test_pronouns_removed_in_v8(self) -> None:
        """V8: pronouns removed from aliases (NSM primitives too coarse)."""
        for pronoun in ["i", "me", "you", "he", "she", "it", "we", "they"]:
            assert pronoun not in GRAMMAR_ALIASES

    def test_prepositions_added_in_v8(self) -> None:
        """V8: prepositions added as aliases."""
        for prep in ["in", "on", "at", "to", "from", "of"]:
            assert prep in GRAMMAR_ALIASES

    def test_disability_not_in_aliases(self) -> None:
        """Content word compositions should NOT be grammar aliases."""
        assert "disability" not in GRAMMAR_ALIASES
        assert "fighter" not in GRAMMAR_ALIASES
        assert "water" not in GRAMMAR_ALIASES


class TestBuildAliasMap:
    """build_alias_map must produce valid token ID -> primitive ID mappings."""

    def test_alias_map_has_entries(self, alias_map: dict) -> None:
        assert len(alias_map) >= 70, f"Expected 70+ aliases, got {len(alias_map)}"

    def test_all_values_are_primitive_ids(self, alias_map: dict) -> None:
        """Every ID in the alias map values must be a valid primitive token ID."""
        from primoji.primitives import PRIMITIVES
        prim_ids = {p.id for p in PRIMITIVES}
        for tok_id, prim_ids_list in alias_map.items():
            for pid in prim_ids_list:
                assert pid in prim_ids, (
                    f"Alias token {tok_id} references non-primitive ID {pid}"
                )

    def test_grammar_words_are_word_tokens(self, tok: Tokenizer) -> None:
        """Grammar words must encode to word tokens, not primitives."""
        for word in ["is", "not", "this", "with", "was", "if", "can", "they"]:
            tier = tok.classify_word(word)
            assert tier == "tier1b_word", f"'{word}' should be tier1b_word, got {tier}"

    def test_content_words_are_primitives(self, tok: Tokenizer) -> None:
        """Content words must encode to primitives, not word tokens."""
        for word in ["water", "think", "good", "fire", "love", "health"]:
            tier = tok.classify_word(word)
            assert tier == "tier2_primitive", f"'{word}' should be tier2_primitive, got {tier}"

    def test_is_and_was_share_be_primitive(self, alias_map: dict, tok: Tokenizer) -> None:
        """'is' and 'was' must share the BE primitive in their alias."""
        be_id = get_primitive_by_name("BE").id
        is_id = tok.encode("is")[0]
        was_id = tok.encode("was")[0]
        assert be_id in alias_map[is_id], "'is' alias missing BE"
        assert be_id in alias_map[was_id], "'was' alias missing BE"
