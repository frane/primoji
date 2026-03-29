"""Decoder: token IDs → English text.

Decodes by tier, matching the encode architecture. Does NOT depend on
dictionary reverse lookup as primary path. Each token is decoded by its
tier (emoji catalog name, primitive name, structural token string, byte
fallback reconstruction).

Dictionary reverse lookup is only used as an OPTIONAL enhancement for
multi-token composition pretty-printing (e.g., showing "photosynthesis"
instead of "plant have light").
"""

from __future__ import annotations

import json
from pathlib import Path

from primoji.byte_fallback import (
    BYTES_END_ID,
    BYTES_START_ID,
    BYTE_TOKEN_OFFSET,
    is_byte_boundary,
    is_byte_token,
)
from primoji.primitives import get_primitive_by_id
from primoji.utils import SpecialTokens, _IDS
from primoji.vocabulary import (
    ANCHOR_TOKENS,
    COMMON_WORD_TOKENS,
    CONTRACTION_TOKENS,
    DIGIT_IDS,
    MATH_OP_IDS,
    PUNCTUATION_IDS,
)

_DATA_DIR = Path(__file__).parent.parent / "data"


# ── Tier-based reverse lookups (NOT dictionary-dependent) ─────────────────────

def _build_tier1_names() -> dict[int, str]:
    """Load ID → CLDR name for Tier 1 emoji."""
    path = _DATA_DIR / "emoji_catalog.json"
    if path.exists():
        with path.open() as f:
            data = json.load(f)
        return {e["id"]: e["name"].lower() for e in data["emoji"]}
    return {}


def _build_anchor_names() -> dict[int, str]:
    """Load ID → proper noun name for anchors."""
    return {v: k for k, v in ANCHOR_TOKENS.items()}


def _build_word_names() -> dict[int, str]:
    """Load ID → word for common word tokens."""
    return {v: k for k, v in COMMON_WORD_TOKENS.items()}


def _build_contraction_names() -> dict[int, str]:
    return {v: k for k, v in CONTRACTION_TOKENS.items()}


def _build_structural_names() -> dict[int, str]:
    names: dict[int, str] = {}
    for ch, tid in DIGIT_IDS.items():
        names[tid] = ch
    for op, tid in MATH_OP_IDS.items():
        if tid not in names:
            names[tid] = op
    for p, tid in PUNCTUATION_IDS.items():
        names[tid] = p
    return names


_TIER1_NAMES: dict[int, str] = _build_tier1_names()
_ANCHOR_NAMES: dict[int, str] = _build_anchor_names()
_WORD_NAMES: dict[int, str] = _build_word_names()
_CONTRACTION_NAMES: dict[int, str] = _build_contraction_names()
_STRUCTURAL_NAMES: dict[int, str] = _build_structural_names()


class Decoder:
    """Decode Primoji token IDs back to English.

    Primary decode path uses tier-based lookups (catalog, primitives,
    structural). Dictionary reverse lookup is optional for multi-token
    composition pretty-printing.
    """

    def __init__(self, vocabulary: "Vocabulary", dictionary: "Dictionary") -> None:
        self._vocab = vocabulary
        self._dict = dictionary

    def decode_canonical(self, ids: list[int]) -> str:
        """Decode token IDs to canonical English.

        Decodes each token by its tier. For multi-token sequences, attempts
        dictionary reverse lookup for composition names (e.g., "photosynthesis")
        before falling back to per-token decoding.

        Args:
            ids: List of token IDs.

        Returns:
            Canonical English text.
        """
        words: list[str] = []
        i = 0
        while i < len(ids):
            tid = ids[i]

            # Skip non-boundary special tokens
            if SpecialTokens.is_special(tid) and not is_byte_boundary(tid):
                i += 1
                continue

            # Byte fallback region: BYTES_START ... BYTES_END
            if tid == BYTES_START_ID:
                j = i + 1
                while j < len(ids) and ids[j] != BYTES_END_ID:
                    j += 1
                byte_vals = []
                for k in range(i + 1, j):
                    byte_vals.append(ids[k] - BYTE_TOKEN_OFFSET)
                try:
                    words.append(bytes(byte_vals).decode("utf-8", errors="replace"))
                except Exception:
                    words.append("<bytes?>")
                i = j + 1
                continue

            # Try multi-token composition from dictionary (longest match first)
            composed = self._try_composition(ids, i)
            if composed is not None:
                word, length = composed
                words.append(word)
                i += length
                continue

            # Single token: dictionary canonical form first, then tier name
            dict_word = self._dict.reverse_lookup([tid])
            if dict_word is not None:
                words.append(dict_word)
                i += 1
                continue

            # Fallback: tier-based name (CLDR, primitive, structural)
            word = self._decode_single(tid)
            if word is not None:
                words.append(word)
            i += 1

        return " ".join(words)

    def _decode_single(self, tid: int) -> str | None:
        """Decode a single token ID by its tier."""
        # Tier 1a: emoji catalog name
        name = _TIER1_NAMES.get(tid)
        if name is not None:
            return name

        # Tier 2: primitive name
        prim = get_primitive_by_id(tid)
        if prim is not None:
            return prim.name.lower()

        # Tier 1b: common word token
        word = _WORD_NAMES.get(tid)
        if word is not None:
            return word

        # Contraction
        contraction = _CONTRACTION_NAMES.get(tid)
        if contraction is not None:
            return contraction

        # Anchor
        anchor = _ANCHOR_NAMES.get(tid)
        if anchor is not None:
            return anchor

        # Structural (digit, math op, punctuation)
        structural = _STRUCTURAL_NAMES.get(tid)
        if structural is not None:
            return structural

        return None

    def _try_composition(self, ids: list[int], start: int) -> tuple[str, int] | None:
        """Try to match a multi-token composition in the dictionary.

        Tries longest match first (up to 5 tokens). Only matches sequences
        that don't contain byte tokens.

        Returns:
            (word, length) tuple if found, None otherwise.
        """
        for length in range(min(5, len(ids) - start), 1, -1):
            subseq = ids[start : start + length]
            if any(is_byte_token(t) or is_byte_boundary(t) for t in subseq):
                continue
            word = self._dict.reverse_lookup(subseq)
            if word is not None:
                return (word, length)
        return None

    def decode_semantic(self, ids: list[int]) -> str:
        """Decode token IDs to best-effort semantic English."""
        return self.decode_canonical(ids)
