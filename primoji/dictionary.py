"""Dictionary: word/phrase → emoji token ID sequence lookup.

Provides fast O(1) lookups for English words/phrases to their emoji token ID
sequences, plus reverse lookups for decoding. Initialized with hardcoded
example mappings; the full dictionary is built by the pipeline scripts.
"""

from __future__ import annotations

import json
from pathlib import Path

from primoji.primitives import get_primitive_by_name
from primoji.vocabulary import TIER1_DIRECT_EMOJI, TIER3_FLAGS


# ── Helper to resolve IDs dynamically ─────────────────────────────────────────

def _p(name: str) -> int:
    """Get a primitive's token ID by name. Raises if not found."""
    prim = get_primitive_by_name(name)
    if prim is None:
        raise ValueError(f"Unknown primitive: {name}")
    return prim.id


def _e(emoji: str) -> int:
    """Get a Tier 1 emoji's token ID. Raises if not found."""
    tid = TIER1_DIRECT_EMOJI.get(emoji)
    if tid is None:
        raise ValueError(f"Unknown Tier 1 emoji: {emoji}")
    return tid


def _f(code: str) -> int:
    """Get a flag's token ID by ISO code. Raises if not found."""
    tid = TIER3_FLAGS.get(code)
    if tid is None:
        raise ValueError(f"Unknown flag code: {code}")
    return tid


# ── Bootstrap mappings (built dynamically from actual primitive/vocab IDs) ────

def _build_bootstrap() -> dict[str, list[int]]:
    """Build the bootstrap word→ID mappings using actual resolved IDs."""
    return {
        # Simple noun → direct emoji (Tier 1)
        "dog": [_e("🐕")],
        "cat": [_e("🐈")],
        "fish": [_e("🐟")],
        "tree": [_e("🌳")],
        "fire": [_e("🔥")],
        "water": [_e("💧")],
        "house": [_e("🏠")],
        "car": [_e("🚗")],
        "bird": [_e("🐦")],
        "snake": [_e("🐍")],
        "book": [_e("📖")],
        "star": [_e("⭐")],
        "heart": [_e("❤️")],
        "apple": [_e("🍎")],
        "flower": [_e("🌺")],
        "music": [_e("🎵")],
        # Composed: person + role/modifier
        "teacher": [_p("SOMEONE"), _p("TEACH")],
        "writer": [_p("SOMEONE"), _p("WRITE")],
        "student": [_p("SOMEONE"), _p("TEACH"), _p("RECEIVE")],
        "scientist": [_p("SOMEONE"), _p("KNOW"), _p("LIGHT")],
        "doctor": [_p("SOMEONE"), _e("🩺")],
        "farmer": [_p("SOMEONE"), _p("PLANT")],
        "soldier": [_p("SOMEONE"), _p("CONFLICT")],
        "judge": [_p("SOMEONE"), _p("LAW")],
        # Composed: concept compositions
        "photosynthesis": [_p("PLANT"), _p("HAVE"), _p("LIGHT")],
        "evaporation": [_e("💧"), _p("CAUSE"), _p("AIR")],
        "computer": [_p("MACHINE"), _p("THINK")],
        "internet": [_p("CONNECT"), _p("EXIST")],
        "telephone": [_p("MACHINE"), _p("SAY")],
        "television": [_p("MACHINE"), _p("SEE")],
        "education": [_p("TEACH")],
        "knowledge": [_p("KNOW")],
        "language": [_p("WORDS")],
        "society": [_p("SOCIETY")],
        "war": [_p("CONFLICT")],
        "peace": [_p("NOT"), _p("CONFLICT")],
        "death": [_p("DIE")],
        "life": [_p("LIVE")],
        "growth": [_p("GROW")],
        "creation": [_p("CREATE")],
        "destruction": [_p("DESTROY")],
        "change": [_p("CHANGE")],
        "beginning": [_p("BEGIN")],
        "end": [_p("END")],
        "path": [_p("PATH")],
        "home": [_p("HOME")],
        # Verbs (map to action primitives)
        "think": [_p("THINK")],
        "know": [_p("KNOW")],
        "want": [_p("WANT")],
        "feel": [_p("FEEL")],
        "see": [_p("SEE")],
        "say": [_p("SAY")],
        "said": [_p("SAY")],
        "move": [_p("MOVE")],
        "do": [_p("DO")],
        "make": [_p("CREATE")],
        "grow": [_p("GROW")],
        "begin": [_p("BEGIN")],
        "start": [_p("BEGIN")],
        "stop": [_p("END")],
        "send": [_p("SEND")],
        "teach": [_p("TEACH")],
        "write": [_p("WRITE")],
        "connect": [_p("CONNECT")],
        "destroy": [_p("DESTROY")],
        "create": [_p("CREATE")],
        "live": [_p("LIVE")],
        "die": [_p("DIE")],
        # Adjectives (map to descriptor/evaluator primitives)
        "big": [_p("BIG")],
        "large": [_p("BIG")],
        "small": [_p("SMALL")],
        "little": [_p("SMALL")],
        "long": [_p("LONG")],
        "good": [_p("GOOD")],
        "bad": [_p("BAD")],
        "dark": [_p("DARK")],
        "bright": [_p("LIGHT")],
        "near": [_p("NEAR")],
        "far": [_p("FAR")],
        # Function words
        "the": [],
        "a": [],
        "an": [],
        "is": [_p("BE")],
        "are": [_p("BE")],
        "was": [_p("BEFORE"), _p("BE")],
        "not": [_p("NOT")],
        "can": [_p("CAN")],
        "if": [_p("IF")],
        "because": [_p("BECAUSE")],
        "very": [_p("VERY")],
        "more": [_p("MORE")],
        "less": [_p("LESS")],
        "all": [_p("ALL")],
        "some": [_p("SOME")],
        "many": [_p("MANY")],
        # Proper noun examples
        "shakespeare": [_p("WRITE"), _e("🎭"), _f("GB"), _p("BEFORE")],
        "explained": [_p("SAY")],
    }


_BOOTSTRAP_WORD_TO_IDS: dict[str, list[int]] = _build_bootstrap()

# Build reverse map (prefer longer words for reverse lookup)
_BOOTSTRAP_IDS_TO_WORD: dict[tuple[int, ...], str] = {}
for _word, _ids in sorted(_BOOTSTRAP_WORD_TO_IDS.items(), key=lambda x: len(x[0])):
    if _ids:
        _BOOTSTRAP_IDS_TO_WORD[tuple(_ids)] = _word


class Dictionary:
    """English word/phrase → emoji token ID sequence lookup.

    Provides bidirectional mapping between English words and their Primoji
    token ID sequences.
    """

    def __init__(self) -> None:
        self._word_to_ids: dict[str, list[int]] = dict(_BOOTSTRAP_WORD_TO_IDS)
        self._ids_to_word: dict[tuple[int, ...], str] = dict(_BOOTSTRAP_IDS_TO_WORD)

    def load(self, path: str | Path) -> None:
        """Load dictionary from a JSON file.

        The JSON file should map words to lists of token IDs:
        {"word": [id1, id2, ...], ...}

        Args:
            path: Path to dictionary JSON file.
        """
        path = Path(path)
        with path.open() as f:
            data: dict[str, list[int]] = json.load(f)
        self._word_to_ids.update(data)
        for word, ids in data.items():
            if ids:
                self._ids_to_word[tuple(ids)] = word

    def lookup(self, word: str) -> list[int] | None:
        """Look up a word's emoji token ID sequence.

        Args:
            word: English word or phrase (case-insensitive).

        Returns:
            List of token IDs, empty list for dropped words (articles),
            or None if not in dictionary.
        """
        return self._word_to_ids.get(word.lower())

    def reverse_lookup(self, ids: list[int]) -> str | None:
        """Find the canonical English word for a token ID sequence.

        Args:
            ids: List of token IDs.

        Returns:
            The canonical English word, or None if not found.
        """
        return self._ids_to_word.get(tuple(ids))

    def contains(self, word: str) -> bool:
        """Check if a word is in the dictionary.

        Args:
            word: English word (case-insensitive).

        Returns:
            True if the word has a mapping.
        """
        return word.lower() in self._word_to_ids

    def add(self, word: str, ids: list[int]) -> None:
        """Add or update a dictionary entry.

        Args:
            word: English word or phrase.
            ids: List of token IDs for the emoji encoding.
        """
        word = word.lower()
        self._word_to_ids[word] = ids
        if ids:
            self._ids_to_word[tuple(ids)] = word

    def size(self) -> int:
        """Return the number of entries in the dictionary."""
        return len(self._word_to_ids)

    def save(self, path: str | Path) -> None:
        """Save dictionary to a JSON file.

        Args:
            path: Output path for the JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self._word_to_ids, f, ensure_ascii=False, indent=2)
