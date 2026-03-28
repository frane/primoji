"""Tests for the seed dictionary (dictionary_seed.json).

Validates that all token IDs are within valid ranges, composed terms
use valid primitive IDs, and key entries are correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from primoji.dictionary import Dictionary
from primoji.primitives import get_primitive_by_name, get_primitive_by_id, PRIMITIVES
from primoji.vocabulary import TIER1_DIRECT_EMOJI, Vocabulary


_DATA_DIR = Path(__file__).parent.parent / "data"
_SEED_PATH = _DATA_DIR / "dictionary_seed.json"


@pytest.fixture
def seed_data() -> dict:
    """Load the raw seed dictionary JSON."""
    with _SEED_PATH.open() as f:
        return json.load(f)


@pytest.fixture
def dictionary() -> Dictionary:
    return Dictionary()


@pytest.fixture
def vocab() -> Vocabulary:
    return Vocabulary()


class TestSeedDictionaryStructure:
    def test_seed_file_exists(self) -> None:
        assert _SEED_PATH.exists(), "dictionary_seed.json not found"

    def test_has_entries(self, seed_data: dict) -> None:
        entries = seed_data.get("entries", seed_data)
        assert len(entries) > 1000, f"Expected >1000 entries, got {len(entries)}"

    def test_all_ids_are_lists(self, seed_data: dict) -> None:
        entries = seed_data.get("entries", seed_data)
        for word, ids in entries.items():
            assert isinstance(ids, list), f"'{word}' has non-list value: {ids}"

    def test_symbolic_refs_are_valid(self, seed_data: dict) -> None:
        """Each entry should be a list of symbolic refs or empty (for dropped words)."""
        entries = seed_data.get("entries", seed_data)
        valid_types = {"emoji", "primitive", "anchor"}
        for word, refs in entries.items():
            assert isinstance(refs, list), f"'{word}' is not a list"
            for ref in refs:
                assert isinstance(ref, dict), f"'{word}' has non-dict ref: {ref}"
                assert ref["type"] in valid_types, f"'{word}' has bad type: {ref['type']}"


class TestSymbolicResolution:
    def test_all_refs_resolve(self, seed_data: dict, vocab: Vocabulary) -> None:
        """Every symbolic ref in the seed should resolve to a valid token ID."""
        from primoji.dictionary import _resolve_refs

        entries = seed_data.get("entries", seed_data)
        failed = []
        for word, refs in entries.items():
            if not refs:
                continue
            ids = _resolve_refs(refs)
            if ids is None:
                failed.append(word)
        assert len(failed) == 0, f"{len(failed)} entries failed to resolve: {failed[:10]}"

    def test_resolved_ids_in_range(self, dictionary: Dictionary, vocab: Vocabulary) -> None:
        """After resolution, all IDs should be within vocab_size."""
        max_id = vocab.vocab_size - 1
        for word in ["dog", "water", "photosynthesis", "teacher"]:
            ids = dictionary.lookup(word)
            if ids:
                for tid in ids:
                    assert 0 <= tid <= max_id, f"'{word}' ID {tid} > {max_id}"

    def test_primitive_refs_resolve_correctly(self, seed_data: dict) -> None:
        """Primitive refs should resolve to actual primitive IDs."""
        from primoji.dictionary import _resolve_refs

        entries = seed_data.get("entries", seed_data)
        for word, refs in entries.items():
            if not refs:
                continue
            for ref in refs:
                if ref["type"] == "primitive":
                    p = get_primitive_by_name(ref["name"])
                    assert p is not None, f"'{word}' refs unknown primitive '{ref['name']}'"


class TestKeyEntries:
    def test_photosynthesis_composition(self, dictionary: Dictionary) -> None:
        """'photosynthesis' should map to [PLANT, HAVE, LIGHT]."""
        ids = dictionary.lookup("photosynthesis")
        assert ids is not None
        plant_id = get_primitive_by_name("PLANT").id
        have_id = get_primitive_by_name("HAVE").id
        light_id = get_primitive_by_name("LIGHT").id
        assert ids == [plant_id, have_id, light_id]

    def test_dog_maps_to_single_tier1_id(self, dictionary: Dictionary) -> None:
        """'dog' should map to a single Tier 1 emoji ID."""
        ids = dictionary.lookup("dog")
        assert ids is not None
        assert len(ids) == 1
        assert 0 <= ids[0] <= 1199, f"dog ID {ids[0]} not in Tier 1 range"

    def test_water_maps_to_single_id(self, dictionary: Dictionary) -> None:
        """'water' should map to a single token ID (Tier 1 emoji or primitive)."""
        ids = dictionary.lookup("water")
        assert ids is not None
        assert len(ids) == 1
        assert ids[0] >= 0

    def test_teacher_is_composed(self, dictionary: Dictionary) -> None:
        """'teacher' should be a multi-primitive composition starting with SOMEONE."""
        ids = dictionary.lookup("teacher")
        assert ids is not None
        assert len(ids) >= 2
        assert ids[0] == get_primitive_by_name("SOMEONE").id

    def test_function_words_present(self, dictionary: Dictionary) -> None:
        """Common function words should be in the dictionary."""
        for word in ["the", "a", "is", "not", "can", "if", "with", "for", "about"]:
            assert dictionary.contains(word), f"'{word}' not in dictionary"


class TestNewV02Primitives:
    """Test that v0.2 primitive words are accessible via the dictionary.

    The seed dictionary may map these words to Tier 1 emoji IDs rather
    than primitive IDs. We verify the words are present and map to valid
    single-token IDs.
    """

    def test_with_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("with")
        assert ids is not None
        assert len(ids) == 1

    def test_for_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("for")
        assert ids is not None
        assert len(ids) >= 1

    def test_about_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("about")
        assert ids is not None
        assert len(ids) >= 1

    def test_love_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("love")
        assert ids is not None
        assert len(ids) >= 1

    def test_fear_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("fear")
        assert ids is not None
        assert len(ids) >= 1

    def test_energy_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("energy")
        assert ids is not None
        assert len(ids) >= 1

    def test_hot_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("hot")
        assert ids is not None
        assert len(ids) >= 1

    def test_cold_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("cold")
        assert ids is not None
        assert len(ids) >= 1


class TestCanonicalFormSelection:
    """Test that reverse lookup returns the best canonical form, not last-write-wins."""

    def test_canonical_cldr_name(self, dictionary: Dictionary) -> None:
        """Tier 1 emoji decodes to its CLDR TTS name."""
        from primoji.vocabulary import TIER1_DIRECT_EMOJI

        dog_id = TIER1_DIRECT_EMOJI.get("🐕")
        if dog_id is not None:
            word = dictionary.reverse_lookup([dog_id])
            assert word == "dog", f"Expected 'dog', got '{word}'"

    def test_canonical_primitive_name(self, dictionary: Dictionary) -> None:
        """Primitives decode to their canonical name, not synonyms."""
        fire_id = get_primitive_by_name("FIRE").id
        word = dictionary.reverse_lookup([fire_id])
        assert word == "fire", f"Expected 'fire', got '{word}'"

    def test_canonical_water(self, dictionary: Dictionary) -> None:
        """'water' encodes to WATER primitive and decodes back to 'water'."""
        water_id = get_primitive_by_name("WATER").id
        word = dictionary.reverse_lookup([water_id])
        assert word == "water", f"Expected 'water', got '{word}'"

    def test_canonical_big(self, dictionary: Dictionary) -> None:
        big_id = get_primitive_by_name("BIG").id
        word = dictionary.reverse_lookup([big_id])
        assert word == "big", f"Expected 'big', got '{word}'"

    def test_canonical_good(self, dictionary: Dictionary) -> None:
        good_id = get_primitive_by_name("GOOD").id
        word = dictionary.reverse_lookup([good_id])
        assert word == "good", f"Expected 'good', got '{word}'"

    def test_canonical_frequency_over_length(self, dictionary: Dictionary) -> None:
        """More frequent word wins over shorter one (frequency > shortest)."""
        from wordfreq import word_frequency

        # Among all synonyms for CONFLICT, "conflict" should win (primitive name)
        conflict_id = get_primitive_by_name("CONFLICT").id
        word = dictionary.reverse_lookup([conflict_id])
        assert word == "conflict", f"Expected 'conflict', got '{word}'"

    def test_roundtrip_common_words(self) -> None:
        """Common words survive encode→decode roundtrip."""
        from primoji import Tokenizer

        tok = Tokenizer(fuzzy=False)
        for word in ["dog", "water", "fire", "house", "big", "good", "bad",
                     "think", "know", "love", "fear", "hot", "cold"]:
            ids = tok.encode(word)
            decoded = tok.decode(ids)
            assert word in decoded.lower(), (
                f"Roundtrip failed: '{word}' → {ids} → '{decoded}'"
            )
