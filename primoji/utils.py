"""Utility helpers for Primoji tokenizer.

Provides Unicode emoji catalog access, text normalization, and shared constants.
ID ranges are computed dynamically from data files in _compute_ids().
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import emoji as emoji_lib

_DATA_DIR = Path(__file__).parent / "data"


# ── Dynamic ID computation ────────────────────────────────────────────────────

def _compute_ids() -> dict[str, int]:
    """Compute all token ID boundaries from actual data file sizes."""
    # Tier 2: primitives (compute from data)
    prim_path = _DATA_DIR / "primitives.json"
    if prim_path.exists():
        with prim_path.open() as f:
            prim_count = len(json.load(f)["primitives"])
    else:
        prim_count = 140

    # Tier 1: emoji catalog
    catalog_path = _DATA_DIR / "emoji_catalog.json"
    if catalog_path.exists():
        with catalog_path.open() as f:
            emoji_count = len(json.load(f)["emoji"])
    else:
        emoji_count = 1200

    flags_count = 259
    contractions_count = 27  # dead slots: kept for frozen layout compatibility
    structural_count = 10 + 14 + 60 + 2 + 38 + 27  # digits + math ops + punctuation + POSSESSIVE/ORDINAL + ordinals + abbreviations

    # Dynamic: anchors
    anchor_path = _DATA_DIR / "proper_noun_anchors.json"
    if anchor_path.exists():
        with anchor_path.open() as f:
            anchor_count = json.load(f)["total_count"]
    else:
        anchor_count = 0

    # Dynamic: common word tokens (Tier 1b)
    words_path = _DATA_DIR / "common_words.json"
    if words_path.exists():
        with words_path.open() as f:
            word_count = json.load(f)["total_count"]
    else:
        word_count = 0

    # !! FROZEN LAYOUT -- DO NOT CHANGE !!
    # These boundaries are baked into every trained model and tokenized dataset.
    # Changing them silently breaks inference and training compatibility.
    # Primitives 1332-1339 overlap with the first 8 flag slots; this is a
    # known compromise handled in vocabulary.py (primitives take priority).
    prim_start = 1200
    flags_start = 1332  # NOT prim_start + prim_count -- intentionally frozen
    contract_start = flags_start + flags_count
    anchor_start = contract_start + contractions_count
    struct_start = anchor_start + anchor_count
    word_start = struct_start + structural_count
    special_start = word_start + word_count

    return {
        "EMOJI_COUNT": emoji_count,
        "PRIM_START": prim_start,
        "PRIM_COUNT": prim_count,
        "FLAGS_START": flags_start,
        "CONTRACT_START": contract_start,
        "ANCHOR_START": anchor_start,
        "ANCHOR_COUNT": anchor_count,
        "STRUCT_START": struct_start,
        "WORD_START": word_start,
        "WORD_COUNT": word_count,
        "BOS": special_start,
        "EOS": special_start + 1,
        "PAD": special_start + 2,
        "UNK": special_start + 3,
        "BYTES_START": special_start + 4,
        "BYTES_END": special_start + 5,
        "BYTE_OFFSET": special_start + 6,
        "VOCAB_SIZE": special_start + 6 + 256,
    }


_IDS = _compute_ids()


# ── Special token IDs ─────────────────────────────────────────────────────────

class SpecialTokens:
    """Reserved special token IDs (dynamically computed)."""

    BOS: int = _IDS["BOS"]
    EOS: int = _IDS["EOS"]
    PAD: int = _IDS["PAD"]
    UNK: int = _IDS["UNK"]
    BYTES_START: int = _IDS["BYTES_START"]
    BYTES_END: int = _IDS["BYTES_END"]

    ALL: dict[str, int] = {
        "BOS": _IDS["BOS"],
        "EOS": _IDS["EOS"],
        "PAD": _IDS["PAD"],
        "UNK": _IDS["UNK"],
        "BYTES_START": _IDS["BYTES_START"],
        "BYTES_END": _IDS["BYTES_END"],
    }

    @classmethod
    def is_special(cls, token_id: int) -> bool:
        """Check if a token ID is a special token."""
        return token_id in cls.ALL.values()

    @classmethod
    def name_of(cls, token_id: int) -> str | None:
        """Get the name of a special token by ID."""
        for name, tid in cls.ALL.items():
            if tid == token_id:
                return name
        return None


# ── Token tier classification ────────────────────────────────────────────────

# Tier IDs for tier embeddings (numeric, used in training data)
TIER_EMOJI = 0
TIER_WORD = 1
TIER_PRIMITIVE = 2
TIER_STRUCTURAL = 3
TIER_BYTE = 4

# Precomputed boundaries (avoid repeated dict lookups)
_PRIM_START = _IDS["PRIM_START"]
_PRIM_END = _PRIM_START + _IDS["PRIM_COUNT"] - 1
_WORD_START = _IDS["WORD_START"]
_WORD_END = _WORD_START + _IDS["WORD_COUNT"]


def classify_token(tid: int) -> int:
    """Classify a token ID into its tier (numeric).

    Returns one of: TIER_EMOJI (0), TIER_WORD (1), TIER_PRIMITIVE (2),
    TIER_STRUCTURAL (3), TIER_BYTE (4).

    Note: IDs 1332-1339 overlap between primitives and flags.
    Primitives take priority (checked first).
    """
    from primoji.byte_fallback import is_byte_token, is_byte_boundary

    if 0 <= tid < _PRIM_START:
        return TIER_EMOJI
    if _PRIM_START <= tid <= _PRIM_END:
        return TIER_PRIMITIVE  # checked before flags -- primitives win on overlap
    if is_byte_token(tid) or is_byte_boundary(tid):
        return TIER_BYTE
    if _WORD_START <= tid < _WORD_END:
        return TIER_WORD
    return TIER_STRUCTURAL


def classify_token_name(tid: int) -> str:
    """Classify a token ID into its tier (string label)."""
    names = {TIER_EMOJI: "emoji", TIER_WORD: "word", TIER_PRIMITIVE: "prim",
             TIER_STRUCTURAL: "struct", TIER_BYTE: "byte"}
    return names[classify_token(tid)]


# ── Emoji utilities ───────────────────────────────────────────────────────────

def is_emoji(char: str) -> bool:
    """Check if a string is a single emoji or emoji sequence."""
    return emoji_lib.is_emoji(char)


def get_all_emoji() -> list[str]:
    """Return a list of all Unicode emoji from the emoji library."""
    return sorted(emoji_lib.EMOJI_DATA.keys())


def emoji_name(char: str) -> str | None:
    """Get the CLDR short name for an emoji."""
    demojized = emoji_lib.demojize(char, delimiters=("", ""))
    if demojized == char:
        return None
    return demojized.replace("_", " ").strip(":")


# ── Text normalization ────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for tokenization: collapse whitespace, strip."""
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def is_punctuation(char: str) -> bool:
    """Check if a character is punctuation."""
    if len(char) != 1:
        return False
    cat = unicodedata.category(char)
    return cat.startswith("P") or (cat.startswith("S") and char in "+-=<>|~^")


def simple_word_tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer.

    Preserves apostrophes within words (contractions).
    """
    tokens: list[str] = []
    for chunk in text.split():
        while chunk and is_punctuation(chunk[0]) and chunk[0] != "'":
            tokens.append(chunk[0])
            chunk = chunk[1:]
        trailing: list[str] = []
        while chunk and is_punctuation(chunk[-1]) and chunk[-1] != "'":
            trailing.append(chunk[-1])
            chunk = chunk[:-1]
        if chunk:
            tokens.append(chunk)
        tokens.extend(reversed(trailing))
    return tokens
