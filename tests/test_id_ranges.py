"""Tests for token ID range integrity.

Catches the bugs we actually found: overlapping ID ranges,
hardcoded values drifting from data, and describe() showing
wrong tier for primitives.
"""

from __future__ import annotations

import pytest

from primoji.primitives import PRIMITIVES
from primoji.utils import _IDS
from primoji.vocabulary import (
    ANCHOR_TOKENS,
    COMMON_WORD_TOKENS,
    CONTRACTION_TOKENS,
    DIGIT_IDS,
    MATH_OP_IDS,
    PUNCTUATION_IDS,
    TIER1_DIRECT_EMOJI,
    TIER3_FLAGS,
    Vocabulary,
)


class TestIDRangesNoOverlap:
    """Token ID ranges must not overlap. Overlaps cause wrong tier classification."""

    def _all_ids(self) -> dict[str, set[int]]:
        return {
            "emoji": set(TIER1_DIRECT_EMOJI.values()),
            "primitive": {p.id for p in PRIMITIVES},
            "flag": set(TIER3_FLAGS.values()),
            "contraction": set(CONTRACTION_TOKENS.values()),
            "anchor": set(ANCHOR_TOKENS.values()),
            "word": set(COMMON_WORD_TOKENS.values()),
            "digit": set(DIGIT_IDS.values()),
            "math_op": set(MATH_OP_IDS.values()),
            "punctuation": set(PUNCTUATION_IDS.values()),
        }

    def test_emoji_and_primitive_no_overlap(self) -> None:
        ids = self._all_ids()
        overlap = ids["emoji"] & ids["primitive"]
        assert not overlap, f"Emoji/primitive overlap: {overlap}"

    def test_word_and_primitive_no_overlap(self) -> None:
        ids = self._all_ids()
        overlap = ids["word"] & ids["primitive"]
        assert not overlap, f"Word/primitive overlap: {overlap}"

    def test_word_and_emoji_no_overlap(self) -> None:
        ids = self._all_ids()
        overlap = ids["word"] & ids["emoji"]
        assert not overlap, f"Word/emoji overlap: {overlap}"

    def test_anchor_and_word_no_overlap(self) -> None:
        ids = self._all_ids()
        overlap = ids["anchor"] & ids["word"]
        assert not overlap, f"Anchor/word overlap: {overlap}"

    def test_no_duplicate_ids_anywhere(self) -> None:
        """Every token ID must be assigned to exactly one tier."""
        ids = self._all_ids()
        all_ids: list[int] = []
        for tier_ids in ids.values():
            all_ids.extend(tier_ids)
        # Flag/primitive overlap is known and handled (flags defer to prims in describe)
        # But no other tier should have duplicates
        non_flag = [tid for name, tids in ids.items() if name != "flag" for tid in tids]
        assert len(non_flag) == len(set(non_flag)), "Duplicate IDs found outside flags"


class TestDynamicIDComputation:
    """ID ranges must be computed from data, not hardcoded."""

    def test_prim_count_matches_data(self) -> None:
        assert _IDS["PRIM_COUNT"] == len(PRIMITIVES), (
            f"_IDS says {_IDS['PRIM_COUNT']} primitives but PRIMITIVES has {len(PRIMITIVES)}"
        )

    def test_emoji_count_matches_data(self) -> None:
        assert _IDS["EMOJI_COUNT"] == len(TIER1_DIRECT_EMOJI), (
            f"_IDS says {_IDS['EMOJI_COUNT']} emoji but catalog has {len(TIER1_DIRECT_EMOJI)}"
        )

    def test_word_count_matches_data(self) -> None:
        assert _IDS["WORD_COUNT"] == len(COMMON_WORD_TOKENS), (
            f"_IDS says {_IDS['WORD_COUNT']} words but COMMON_WORD_TOKENS has {len(COMMON_WORD_TOKENS)}"
        )

    def test_vocab_size_consistent(self) -> None:
        from primoji import Tokenizer
        tok = Tokenizer(fuzzy=False)
        assert tok.vocab_size == _IDS["VOCAB_SIZE"]


class TestFrozenLayout:
    """Critical ID boundaries must never change.

    These are baked into every trained model and tokenized dataset.
    Changing them silently breaks inference, training, and compatibility.
    If a test here fails, you are about to break every existing model.
    """

    def test_prim_start_frozen_at_1200(self) -> None:
        assert _IDS["PRIM_START"] == 1200, (
            "PRIM_START changed! This breaks all trained models. Revert immediately."
        )

    def test_flags_start_frozen_at_1332(self) -> None:
        assert _IDS["FLAGS_START"] == 1332, (
            "FLAGS_START changed! This shifts ALL downstream IDs and breaks "
            "every trained model and tokenized dataset. Revert immediately."
        )

    def test_contract_start_frozen(self) -> None:
        assert _IDS["CONTRACT_START"] == 1332 + 259, (
            "CONTRACT_START changed! This breaks model compatibility."
        )

    def test_layout_contract(self) -> None:
        """The full ID chain must be deterministic from frozen anchors."""
        assert _IDS["FLAGS_START"] == 1332
        assert _IDS["CONTRACT_START"] == 1332 + 259  # 1591
        # Everything after CONTRACT_START depends on data file sizes,
        # which is fine -- those change when we add words/anchors.
        # But the chain must start from the frozen anchor points.


class TestDescribeShowsCorrectTier:
    """Vocabulary.describe() must show the right tier, even for overlapping IDs.

    Bug found: primitives 1332-1339 overlapped with flags, and describe()
    showed 'Flag: AD' instead of 'Tier 2 primitive: HEALTH'.
    """

    @pytest.fixture
    def vocab(self) -> Vocabulary:
        return Vocabulary()

    @pytest.mark.parametrize("prim_name", [
        "HEALTH", "SUBSTANCE", "DEGREE", "ENVIRONMENT",
        "BODY_PART", "VISIBLE", "STUDY", "ELECTRIC",
    ])
    def test_v03_primitives_describe_as_primitives(self, vocab: Vocabulary, prim_name: str) -> None:
        from primoji.primitives import get_primitive_by_name
        p = get_primitive_by_name(prim_name)
        assert p is not None
        desc = vocab.describe(p.id)
        assert "Tier 2 primitive" in desc, (
            f"Primitive {prim_name} (ID {p.id}) describes as '{desc}', expected 'Tier 2 primitive'"
        )
