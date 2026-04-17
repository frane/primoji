"""Composition engine for Primoji tokenizer.

Maps English words to emoji token ID sequences using positional semantic rules:
  Position 1 = HEAD (what kind of thing)
  Position 2 = MODIFIER (property/relation)
  Position 3-5 = SPECIFIERS (further refinement)

Special rules:
- 🚫 always precedes what it negates: 🚫👁️ = blind
- Intensifiers suffix: 🔥‼️ = extremely hot
- Temporal prefixes: ⏪⚔️ = previous war
- Max depth: 5 tokens per concept
"""

from __future__ import annotations

import warnings

from primoji.dictionary import Dictionary
from primoji.primitives import get_primitive_by_name
from primoji.utils import SpecialTokens
from primoji.vocabulary import Vocabulary

MAX_COMPOSITION_DEPTH = 5


class Composer:
    """Composition engine: English word → emoji token ID sequence.

    Uses the dictionary for known mappings and falls back to rule-based
    composition for unknown words.
    """

    def __init__(self, vocabulary: Vocabulary, dictionary: Dictionary) -> None:
        self._vocab = vocabulary
        self._dict = dictionary

        # Cache primitive IDs for composition rules
        self._not_id = self._get_prim_id("NOT")      # 🚫
        self._very_id = self._get_prim_id("VERY")     # ‼️
        self._more_id = self._get_prim_id("MORE")     # ➕
        self._less_id = self._get_prim_id("LESS")     # ➖
        self._before_id = self._get_prim_id("BEFORE")  # ⏪
        self._after_id = self._get_prim_id("AFTER")    # ⏩

    @staticmethod
    def _get_prim_id(name: str) -> int | None:
        """Get a primitive's token ID by name."""
        p = get_primitive_by_name(name)
        return p.id if p else None

    def compose(self, word: str) -> list[int]:
        """Map a single English word to its emoji token ID sequence.

        Lookup order:
        1. Dictionary exact match
        2. Negation prefix handling (un-, in-, im-, dis-, non-)
        3. Temporal prefix handling (pre-, post-, re-)
        4. Comparative/superlative handling (-er, -est)
        5. UNK token for truly unknown words

        Args:
            word: A single English word.

        Returns:
            List of token IDs (1–5 tokens). Empty list for dropped words
            (articles, etc.).

        Raises:
            Warning if composition exceeds MAX_COMPOSITION_DEPTH.
        """
        lower = word.lower()

        # 1. Dictionary lookup
        ids = self._dict.lookup(lower)
        if ids is not None:
            return ids

        # 2. Negation prefixes: un-, in-, im-, dis-, non-
        ids = self._try_negation(lower)
        if ids is not None:
            return self._enforce_depth(ids, lower)

        # 3. Temporal/repetition prefixes: pre-, post-, re-
        ids = self._try_temporal(lower)
        if ids is not None:
            return self._enforce_depth(ids, lower)

        # 4. Comparative/superlative: -er, -est
        ids = self._try_comparative(lower)
        if ids is not None:
            return self._enforce_depth(ids, lower)

        # Agent suffix (-er, -or) removed in V8: too many false positives
        # (developer -> GROW+MORE, researcher -> STUDY+MORE, easter -> SIDE+MORE).
        # V8.3 will add WordNet/emoji2vec to handle agent nouns properly.

        # Fallback: UNK
        return [SpecialTokens.UNK]

    def _try_negation(self, word: str) -> list[int] | None:
        """Handle negation prefixes (un-, in-, im-, dis-, non-, -less)."""
        if self._not_id is None:
            return None

        for prefix in ("un", "in", "im", "dis", "non"):
            if word.startswith(prefix) and len(word) > len(prefix) + 1:
                base = word[len(prefix):]
                base_ids = self._dict.lookup(base)
                if base_ids is not None and base_ids:
                    return [self._not_id] + base_ids

        # Suffix: -less → NOT + base
        if word.endswith("less") and len(word) > 5:
            base = word[:-4]
            base_ids = self._dict.lookup(base)
            if base_ids is not None and base_ids:
                return [self._not_id] + base_ids

        return None

    def _try_temporal(self, word: str) -> list[int] | None:
        """Handle temporal prefixes (pre-, post-, re-)."""
        if word.startswith("pre") and self._before_id and len(word) > 4:
            base = word[3:]
            base_ids = self._dict.lookup(base)
            if base_ids is not None and base_ids:
                return [self._before_id] + base_ids

        if word.startswith("post") and self._after_id and len(word) > 5:
            base = word[4:]
            base_ids = self._dict.lookup(base)
            if base_ids is not None and base_ids:
                return [self._after_id] + base_ids

        if word.startswith("re") and len(word) > 3:
            base = word[2:]
            base_ids = self._dict.lookup(base)
            if base_ids is not None and base_ids:
                repeat_prim = get_primitive_by_name("PATTERN")  # 🔁
                if repeat_prim:
                    return [repeat_prim.id] + base_ids

        return None

    def _try_comparative(self, word: str) -> list[int] | None:
        """Handle comparative (-er) and superlative (-est) suffixes."""
        if word.endswith("est") and self._very_id and len(word) > 4:
            base = word[:-3]
            # Try with trailing 'e' restored
            for candidate in (base, base + "e"):
                base_ids = self._dict.lookup(candidate)
                if base_ids is not None and base_ids:
                    return base_ids + [self._very_id]

        if word.endswith("er") and self._more_id and len(word) > 3:
            base = word[:-2]
            for candidate in (base, base + "e"):
                base_ids = self._dict.lookup(candidate)
                if base_ids is not None and base_ids:
                    return base_ids + [self._more_id]

        return None

    def _enforce_depth(self, ids: list[int], word: str) -> list[int]:
        """Enforce maximum composition depth of 5 tokens."""
        if len(ids) > MAX_COMPOSITION_DEPTH:
            warnings.warn(
                f"Composition of '{word}' exceeds max depth {MAX_COMPOSITION_DEPTH}: "
                f"got {len(ids)} tokens, truncating",
                stacklevel=3,
            )
            return ids[:MAX_COMPOSITION_DEPTH]
        return ids

    def compose_phrase(self, words: list[str]) -> list[int]:
        """Compose a sequence of words into token IDs.

        Each word is composed independently, with SEP tokens between
        multi-token compositions when needed for disambiguation.

        Args:
            words: List of English words.

        Returns:
            Flat list of token IDs.
        """
        all_ids: list[int] = []
        for word in words:
            ids = self.compose(word)
            all_ids.extend(ids)
        return all_ids
