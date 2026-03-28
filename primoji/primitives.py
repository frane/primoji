"""Compositional primitives for Primoji tokenizer.

Loads all 132 compositional primitives from data/primitives.json (v0.2):
- Layer 2a: 65 Wierzbicka semantic primes (verified against Goddard & Wierzbicka 2014/2018)
- Layer 2b: 67 FineWeb-Edu domain expansions

IDs 1200–1331.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Primitive:
    """A single compositional primitive token."""

    id: int
    name: str
    emoji: str
    layer: str
    category: str
    description: str


# ── Load primitives from JSON ─────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent.parent / "data"
_PRIMITIVES_PATH = _DATA_DIR / "primitives.json"


def _load_primitives() -> list[Primitive]:
    """Load primitives from data/primitives.json."""
    with _PRIMITIVES_PATH.open() as f:
        data = json.load(f)
    return [
        Primitive(
            id=p["id"],
            name=p["name"],
            emoji=p["emoji"],
            layer=p["layer"],
            category=p["category"],
            description=p["description"],
        )
        for p in data["primitives"]
    ]


PRIMITIVES: list[Primitive] = _load_primitives()

# ── Lookup indices ────────────────────────────────────────────────────────────

_BY_NAME: dict[str, Primitive] = {p.name: p for p in PRIMITIVES}
_BY_EMOJI: dict[str, Primitive] = {p.emoji: p for p in PRIMITIVES}
_BY_ID: dict[int, Primitive] = {p.id: p for p in PRIMITIVES}


def get_primitive_by_name(name: str) -> Primitive | None:
    """Look up a primitive by its canonical name (e.g. 'THINK', 'PLANT').

    Args:
        name: Uppercase canonical name of the primitive.

    Returns:
        The matching Primitive, or None if not found.
    """
    return _BY_NAME.get(name.upper())


def get_primitive_by_emoji(emoji: str) -> Primitive | None:
    """Look up a primitive by its emoji character(s).

    Args:
        emoji: The emoji string (may be multi-codepoint like '👁️').

    Returns:
        The matching Primitive, or None if not found.
    """
    return _BY_EMOJI.get(emoji)


def get_primitive_by_id(token_id: int) -> Primitive | None:
    """Look up a primitive by its token ID.

    Args:
        token_id: Integer token ID (1200–1331 range).

    Returns:
        The matching Primitive, or None if not found.
    """
    return _BY_ID.get(token_id)


def primitive_count() -> int:
    """Return the number of defined primitives."""
    return len(PRIMITIVES)
