"""Dictionary: word/phrase → emoji token ID sequence lookup.

Loads dictionary_seed.json which uses symbolic references (emoji chars,
primitive names, anchor names) instead of numeric IDs. Resolves to IDs
at load time, making the dictionary resilient to catalog re-sorts.

Reverse lookup uses 3-tier canonical form selection:
  1. Single-word CLDR/primitive name (prefer single-word over multi-word)
  2. Corpus frequency (via wordfreq)
  3. Shortest, then alphabetical
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from primoji.primitives import PRIMITIVES, get_primitive_by_id, get_primitive_by_name
from primoji.vocabulary import ANCHOR_TOKENS, COMMON_WORD_TOKENS, TIER1_DIRECT_EMOJI, TIER3_FLAGS

_DATA_DIR = Path(__file__).parent / "data"

# ── Canonical name lookups ────────────────────────────────────────────────────

def _build_id_to_canonical() -> dict[int, str]:
    """Map token_id → single-word canonical name for decode."""
    canonical: dict[int, str] = {}

    # Tier 1: CLDR names
    catalog_path = _DATA_DIR / "emoji_catalog.json"
    if catalog_path.exists():
        with catalog_path.open() as f:
            data = json.load(f)
        for e in data["emoji"]:
            canonical[e["id"]] = e["name"].lower()

    # Tier 2: primitive names
    for p in PRIMITIVES:
        canonical[p.id] = p.name.lower()

    # Tier 3: anchors
    for name, tid in ANCHOR_TOKENS.items():
        canonical[tid] = name

    return canonical


_ID_TO_CANONICAL: dict[int, str] = _build_id_to_canonical()


# ── Word frequency ────────────────────────────────────────────────────────────

try:
    from wordfreq import word_frequency as _wf
    def _word_freq(word: str) -> float:
        return _wf(word, "en")
    _HAS_FREQ = True
except ImportError:
    def _word_freq(word: str) -> float:
        return 0.0
    _HAS_FREQ = False


def _select_canonical(candidates: list[str], ids: tuple[int, ...]) -> str:
    """Select the best canonical form from synonym candidates.

    Priority: single-word CLDR/primitive name > single-word by frequency >
              single-word shortest > multi-word CLDR > any by frequency.
    """
    # Priority 1: CLDR/primitive canonical name IF it's a single word
    if len(ids) == 1:
        canon = _ID_TO_CANONICAL.get(ids[0])
        if canon and " " not in canon and canon in candidates:
            return canon

    # Priority 2: Single-word candidates by frequency
    single = [c for c in candidates if " " not in c]
    if single:
        if _HAS_FREQ:
            best = max(single, key=_word_freq)
            if _word_freq(best) > 0:
                return best
        return min(single, key=lambda w: (len(w), w))

    # Priority 3: Multi-word CLDR name
    if len(ids) == 1:
        canon = _ID_TO_CANONICAL.get(ids[0])
        if canon and canon in candidates:
            return canon

    # Priority 4: Any candidate
    if _HAS_FREQ:
        best = max(candidates, key=_word_freq)
        if _word_freq(best) > 0:
            return best
    return min(candidates, key=lambda w: (len(w), w))


# ── Symbol resolver ───────────────────────────────────────────────────────────

def _resolve_ref(ref: dict) -> int | None:
    """Resolve a symbolic reference to a numeric token ID."""
    rtype = ref.get("type")
    if rtype == "emoji":
        return TIER1_DIRECT_EMOJI.get(ref["char"])
    elif rtype == "primitive":
        p = get_primitive_by_name(ref["name"])
        return p.id if p else None
    elif rtype == "anchor":
        return ANCHOR_TOKENS.get(ref["name"])
    elif rtype == "word":
        return COMMON_WORD_TOKENS.get(ref["word"])
    return None


def _resolve_refs(refs: list[dict]) -> list[int] | None:
    """Resolve a list of symbolic references to numeric IDs.

    Returns None if any ref fails to resolve.
    """
    ids = []
    for ref in refs:
        tid = _resolve_ref(ref)
        if tid is None:
            return None
        ids.append(tid)
    return ids


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _p(name: str) -> int:
    prim = get_primitive_by_name(name)
    if prim is None:
        raise ValueError(f"Unknown primitive: {name}")
    return prim.id


def _build_bootstrap() -> dict[str, list[int]]:
    """Minimal bootstrap (used if no seed file exists)."""
    return {
        "think": [_p("THINK")], "know": [_p("KNOW")], "want": [_p("WANT")],
        "feel": [_p("FEEL")], "see": [_p("SEE")], "hear": [_p("HEAR")],
        "say": [_p("SAY")], "said": [_p("SAY")], "move": [_p("MOVE")],
        "do": [_p("DO")], "make": [_p("CREATE")], "grow": [_p("GROW")],
        "begin": [_p("BEGIN")], "start": [_p("BEGIN")], "stop": [_p("END")],
        "send": [_p("SEND")], "teach": [_p("TEACH")], "write": [_p("WRITE")],
        "connect": [_p("CONNECT")], "destroy": [_p("DESTROY")],
        "create": [_p("CREATE")], "live": [_p("LIVE")], "die": [_p("DIE")],
        "hold": [_p("HOLD")], "eat": [_p("EAT")],
        "big": [_p("BIG")], "large": [_p("BIG")], "small": [_p("SMALL")],
        "little": [_p("SMALL")], "good": [_p("GOOD")], "bad": [_p("BAD")],
        "dark": [_p("DARK")], "bright": [_p("LIGHT")],
        "near": [_p("NEAR")], "far": [_p("FAR")],
        "hot": [_p("HOT")], "cold": [_p("COLD")], "heavy": [_p("HEAVY")],
        "water": [_p("WATER")], "fire": [_p("FIRE")], "earth": [_p("EARTH")],
        "air": [_p("AIR")], "light": [_p("LIGHT")], "energy": [_p("ENERGY")],
        "love": [_p("LOVE")], "fear": [_p("FEAR")],
        "home": [_p("HOME")], "path": [_p("PATH")],
        "war": [_p("CONFLICT")], "peace": [_p("NOT"), _p("CONFLICT")],
        "change": [_p("CHANGE")],
        "teacher": [_p("SOMEONE"), _p("TEACH"), _p("SAY")],
        "student": [_p("SOMEONE"), _p("TEACH")],
        "photosynthesis": [_p("PLANT"), _p("HAVE"), _p("LIGHT")],
        "computer": [_p("MACHINE"), _p("THINK")],
        "internet": [_p("CONNECT"), _p("THERE_IS")],
        "the": [], "a": [], "an": [],
        "explained": [_p("SAY")],
    }


_BOOTSTRAP: dict[str, list[int]] = _build_bootstrap()


class Dictionary:
    """English word/phrase → emoji token ID sequence lookup.

    Loads symbolic references from dictionary_seed.json and resolves at init.
    Reverse lookup uses 3-tier canonical form selection.
    """

    def __init__(self) -> None:
        self._word_to_ids: dict[str, list[int]] = dict(_BOOTSTRAP)
        self._ids_to_candidates: dict[tuple[int, ...], list[str]] = defaultdict(list)
        self._ids_to_word: dict[tuple[int, ...], str] = {}

        # Collect bootstrap candidates
        for word, ids in _BOOTSTRAP.items():
            if ids:
                self._ids_to_candidates[tuple(ids)].append(word)

        # Load seed if available
        seed_path = _DATA_DIR / "dictionary_seed.json"
        if seed_path.exists():
            self._load_seed(seed_path)

        self._rebuild_reverse()

    def _load_seed(self, path: Path) -> None:
        """Load seed dictionary, resolving symbolic refs to numeric IDs."""
        with path.open() as f:
            data = json.load(f)

        fmt = data.get("format", "numeric")
        entries = data.get("entries", {})

        for word, value in entries.items():
            w = word.lower()
            if fmt == "symbolic":
                if not value:  # empty list = drop word
                    self._word_to_ids[w] = []
                    continue
                ids = _resolve_refs(value)
                if ids is None:
                    continue  # Skip unresolvable entries
            else:
                ids = value  # Legacy numeric format

            self._word_to_ids[w] = ids
            if ids:
                self._ids_to_candidates[tuple(ids)].append(w)

    def _rebuild_reverse(self) -> None:
        """Build reverse lookup by selecting canonical forms."""
        self._ids_to_word = {}
        for ids_tuple, candidates in self._ids_to_candidates.items():
            unique = list(dict.fromkeys(candidates))
            self._ids_to_word[ids_tuple] = _select_canonical(unique, ids_tuple)

    def lookup(self, word: str) -> list[int] | None:
        """Look up a word's emoji token ID sequence.

        Args:
            word: English word or phrase (case-insensitive).

        Returns:
            List of token IDs, empty list for dropped words, or None if unknown.
        """
        return self._word_to_ids.get(word.lower())

    def reverse_lookup(self, ids: list[int]) -> str | None:
        """Find the canonical English word for a token ID sequence.

        Used only for multi-token composition pretty-printing.

        Args:
            ids: List of token IDs.

        Returns:
            The canonical English word, or None if not found.
        """
        return self._ids_to_word.get(tuple(ids))

    def contains(self, word: str) -> bool:
        """Check if a word is in the dictionary."""
        return word.lower() in self._word_to_ids

    def add(self, word: str, ids: list[int]) -> None:
        """Add or update a dictionary entry."""
        word = word.lower()
        self._word_to_ids[word] = ids
        if ids:
            key = tuple(ids)
            self._ids_to_candidates[key].append(word)
            unique = list(dict.fromkeys(self._ids_to_candidates[key]))
            self._ids_to_word[key] = _select_canonical(unique, key)

    def size(self) -> int:
        """Return the number of entries in the dictionary."""
        return len(self._word_to_ids)

    def save(self, path: str | Path) -> None:
        """Save dictionary to a JSON file (numeric format)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self._word_to_ids, f, ensure_ascii=False, indent=2)
