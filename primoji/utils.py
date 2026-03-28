"""Utility helpers for Primoji tokenizer.

Provides Unicode emoji catalog access, text normalization, and shared constants.
"""

from __future__ import annotations

import re
import unicodedata

import emoji as emoji_lib


# ── Special token IDs ─────────────────────────────────────────────────────────

class SpecialTokens:
    """Reserved special token IDs (1800–1805)."""

    BOS: int = 1800
    EOS: int = 1801
    PAD: int = 1802
    UNK: int = 1803  # Should never be produced (byte fallback prevents it)
    BYTES_START: int = 1804
    BYTES_END: int = 1805

    ALL: dict[str, int] = {
        "BOS": 1800,
        "EOS": 1801,
        "PAD": 1802,
        "UNK": 1803,
        "BYTES_START": 1804,
        "BYTES_END": 1805,
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


# ── Emoji utilities ───────────────────────────────────────────────────────────

def is_emoji(char: str) -> bool:
    """Check if a string is a single emoji or emoji sequence.

    Args:
        char: A string to check.

    Returns:
        True if the string is purely emoji.
    """
    return emoji_lib.is_emoji(char)


def get_all_emoji() -> list[str]:
    """Return a list of all Unicode emoji from the emoji library.

    Returns:
        Sorted list of all known emoji characters.
    """
    return sorted(emoji_lib.EMOJI_DATA.keys())


def emoji_name(char: str) -> str | None:
    """Get the CLDR short name for an emoji.

    Args:
        char: An emoji character.

    Returns:
        The emoji name (e.g. 'dog face'), or None if not found.
    """
    demojized = emoji_lib.demojize(char, delimiters=("", ""))
    if demojized == char:
        return None
    return demojized.replace("_", " ").strip(":")


# ── Text normalization ────────────────────────────────────────────────────────

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize text for tokenization: collapse whitespace, strip.

    Does NOT lowercase — case handling is done per-word in the tokenizer.

    Args:
        text: Input text.

    Returns:
        Normalized text.
    """
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text


def is_punctuation(char: str) -> bool:
    """Check if a character is punctuation.

    Args:
        char: A single character.

    Returns:
        True if the character is punctuation.
    """
    if len(char) != 1:
        return False
    cat = unicodedata.category(char)
    return cat.startswith("P") or (cat.startswith("S") and char in "+-=<>|~^")


def simple_word_tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for when spaCy is not needed.

    Splits on whitespace and separates punctuation from words.
    Preserves apostrophes within words (contractions).

    Args:
        text: Input text.

    Returns:
        List of tokens.
    """
    tokens: list[str] = []
    for chunk in text.split():
        # Peel off leading punctuation (but not apostrophes)
        while chunk and is_punctuation(chunk[0]) and chunk[0] != "'":
            tokens.append(chunk[0])
            chunk = chunk[1:]
        # Peel off trailing punctuation (but not apostrophes mid-word)
        trailing: list[str] = []
        while chunk and is_punctuation(chunk[-1]) and chunk[-1] != "'":
            trailing.append(chunk[-1])
            chunk = chunk[:-1]
        if chunk:
            tokens.append(chunk)
        tokens.extend(reversed(trailing))
    return tokens
