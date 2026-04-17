"""Tests for vocabulary and primitive integrity.

Catches real structural problems: ID collisions, missing primitives,
emoji/primitive overlaps, broken symbolic resolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from primoji.primitives import PRIMITIVES, get_primitive_by_name, get_primitive_by_id
from primoji.vocabulary import (
    TIER1_DIRECT_EMOJI,
    TIER2_PRIMITIVES,
    DIGIT_IDS,
    MATH_OP_IDS,
    PUNCTUATION_IDS,
    ANCHOR_TOKENS,
    Vocabulary,
)
from primoji.utils import SpecialTokens
from primoji.byte_fallback import BYTE_TOKEN_OFFSET

import primoji as _primoji_pkg

_DATA_DIR = Path(_primoji_pkg.__file__).parent / "data"


class TestPrimitives:
    """Verify the 132 primitives from primitives.json are loaded correctly."""

    def test_count(self) -> None:
        assert len(PRIMITIVES) == 140

    def test_ids_sequential(self) -> None:
        ids = [p.id for p in PRIMITIVES]
        assert ids == list(range(1200, 1340))

    def test_v02_primes_present(self) -> None:
        """Key v0.2 additions must be loadable."""
        for name in ["WITH", "FOR", "ABOUT", "ENERGY", "COLOR",
                     "HOT", "COLD", "LOVE", "FEAR", "HEAR",
                     "HERE", "DONT_WANT", "MOMENT", "BE_SOMEWHERE"]:
            p = get_primitive_by_name(name)
            assert p is not None, f"Primitive {name} missing"

    def test_v01_removals_gone(self) -> None:
        """Primitives removed in v0.2 must not exist."""
        for name in ["HIERARCHY", "LONG", "LESS"]:
            assert get_primitive_by_name(name) is None, f"{name} should be gone"

    def test_v02_renames(self) -> None:
        """Renamed primitives must work under new names."""
        assert get_primitive_by_name("THERE_IS") is not None  # was EXIST
        assert get_primitive_by_name("FOR_SOME_TIME") is not None  # was CONTINUE
        assert get_primitive_by_name("EXIST") is None
        assert get_primitive_by_name("CONTINUE") is None

    def test_all_emoji_unique(self) -> None:
        emoji = [p.emoji for p in PRIMITIVES]
        assert len(emoji) == len(set(emoji)), "Duplicate emoji in primitives"

    def test_roundtrip_by_id(self) -> None:
        for p in PRIMITIVES:
            assert get_primitive_by_id(p.id) is p


class TestNoIDCollisions:
    """No two different token types should share the same ID."""

    def test_tier1_and_tier2_no_overlap(self) -> None:
        t1_ids = set(TIER1_DIRECT_EMOJI.values())
        t2_ids = set(p.id for p in PRIMITIVES)
        overlap = t1_ids & t2_ids
        assert len(overlap) == 0, f"Tier1/Tier2 ID overlap: {overlap}"

    def test_no_emoji_in_both_tiers(self) -> None:
        """No emoji character should appear in both Tier 1 and Tier 2."""
        def normalize(s: str) -> str:
            return s.replace("\ufe0e", "").replace("\ufe0f", "")

        t1_emoji = {normalize(e) for e in TIER1_DIRECT_EMOJI}
        t2_emoji = {normalize(p.emoji) for p in PRIMITIVES}
        overlap = t1_emoji & t2_emoji
        assert len(overlap) == 0, f"Emoji in both tiers: {overlap}"

    def test_all_ids_below_vocab_size(self) -> None:
        v = Vocabulary()
        vs = v.vocab_size
        for tid in v._id_to_token:
            assert 0 <= tid < vs, f"ID {tid} >= vocab_size {vs}"


class TestSymbolicSeed:
    """The dictionary seed must use symbolic refs that resolve correctly."""

    def test_seed_exists(self) -> None:
        assert (_DATA_DIR / "dictionary_seed.json").exists()

    def test_seed_is_symbolic(self) -> None:
        with open(_DATA_DIR / "dictionary_seed.json") as f:
            data = json.load(f)
        assert data.get("format") == "symbolic"

    def test_all_primitive_refs_resolve(self) -> None:
        with open(_DATA_DIR / "dictionary_seed.json") as f:
            data = json.load(f)
        bad = []
        for word, refs in data["entries"].items():
            for ref in refs:
                if ref.get("type") == "primitive":
                    if get_primitive_by_name(ref["name"]) is None:
                        bad.append((word, ref["name"]))
        assert len(bad) == 0, f"Unresolvable primitive refs: {bad[:10]}"

    def test_all_emoji_refs_resolve(self) -> None:
        with open(_DATA_DIR / "dictionary_seed.json") as f:
            data = json.load(f)
        bad = []
        for word, refs in data["entries"].items():
            for ref in refs:
                if ref.get("type") == "emoji":
                    if ref["char"] not in TIER1_DIRECT_EMOJI:
                        bad.append((word, ref["char"]))
        assert len(bad) == 0, f"Unresolvable emoji refs: {bad[:10]}"
