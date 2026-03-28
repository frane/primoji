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

    def test_all_ids_are_ints(self, seed_data: dict) -> None:
        entries = seed_data.get("entries", seed_data)
        for word, ids in entries.items():
            for tid in ids:
                assert isinstance(tid, int), f"'{word}' has non-int ID: {tid}"


class TestIDRangeValidity:
    def test_no_id_exceeds_vocab_size(self, seed_data: dict, vocab: Vocabulary) -> None:
        """Every token ID in the dictionary must be < vocab_size."""
        max_id = vocab.vocab_size - 1
        entries = seed_data.get("entries", seed_data)
        for word, ids in entries.items():
            for tid in ids:
                assert 0 <= tid <= max_id, (
                    f"'{word}' has ID {tid} outside valid range [0, {max_id}]"
                )

    def test_composed_terms_use_valid_primitive_ids(self, seed_data: dict) -> None:
        """Multi-token entries that use primitive-range IDs should reference real primitives."""
        entries = seed_data.get("entries", seed_data)
        for word, ids in entries.items():
            for tid in ids:
                if 1200 <= tid <= 1331:
                    prim = get_primitive_by_id(tid)
                    assert prim is not None, (
                        f"'{word}' references primitive ID {tid} which doesn't exist"
                    )


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

    def test_water_maps_to_primitive(self, dictionary: Dictionary) -> None:
        """'water' should map to the WATER primitive."""
        ids = dictionary.lookup("water")
        assert ids is not None
        water_id = get_primitive_by_name("WATER").id
        assert ids == [water_id]

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
    """Test that v0.2 primitives are accessible via the dictionary."""

    def test_with_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("with")
        assert ids is not None
        assert ids == [get_primitive_by_name("WITH").id]

    def test_for_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("for")
        assert ids is not None
        assert ids == [get_primitive_by_name("FOR").id]

    def test_about_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("about")
        assert ids is not None
        assert ids == [get_primitive_by_name("ABOUT").id]

    def test_love_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("love")
        assert ids is not None
        assert ids == [get_primitive_by_name("LOVE").id]

    def test_fear_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("fear")
        assert ids is not None
        assert ids == [get_primitive_by_name("FEAR").id]

    def test_energy_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("energy")
        assert ids is not None
        assert ids == [get_primitive_by_name("ENERGY").id]

    def test_hot_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("hot")
        assert ids is not None
        assert ids == [get_primitive_by_name("HOT").id]

    def test_cold_in_dictionary(self, dictionary: Dictionary) -> None:
        ids = dictionary.lookup("cold")
        assert ids is not None
        assert ids == [get_primitive_by_name("COLD").id]
