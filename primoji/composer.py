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

        # V8 derivational suffix primitives
        self._someone_id = self._get_prim_id("SOMEONE")    # 🧑 (agent -or/-er)
        self._kind_id = self._get_prim_id("KIND")          # 🏷️ (adjectival -al)
        self._like_as_id = self._get_prim_id("LIKE_AS")    # 〰️ (adverbial -ly)
        self._result_id = self._get_prim_id("RESULT")      # 🎯 (nominalizing -tion/-ment)
        self._something_id = self._get_prim_id("SOMETHING") # 🔹 (quality -ness)
        self._have_id = self._get_prim_id("HAVE")          # 📥 (-ful)
        self._can_id = self._get_prim_id("CAN")            # 💪 (-able/-ible)

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

        # 5. Derivational suffixes (V8): agent, adjectival, adverbial, etc.
        ids = self._try_derivational(lower)
        if ids is not None:
            return self._enforce_depth(ids, lower)

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
        """Handle comparative (-er) and superlative (-est) suffixes.

        Handles doubled consonants: hotter -> hot, wetter -> wet.
        """
        if word.endswith("est") and self._very_id and len(word) > 4:
            base = word[:-3]
            candidates = [base, base + "e"]
            # Doubled consonant: hottest -> "hott" -> "hot"
            if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "aeiou":
                candidates.append(base[:-1])
            for candidate in candidates:
                base_ids = self._dict.lookup(candidate)
                if base_ids is not None and base_ids:
                    return base_ids + [self._very_id]

        if word.endswith("er") and self._more_id and len(word) > 3:
            base = word[:-2]
            candidates = [base, base + "e"]
            # Doubled consonant: hotter -> "hott" -> "hot"
            if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "aeiou":
                candidates.append(base[:-1])
            for candidate in candidates:
                base_ids = self._dict.lookup(candidate)
                if base_ids is not None and base_ids:
                    return base_ids + [self._more_id]

        return None

    def _try_derivational(self, word: str) -> list[int] | None:
        """Handle derivational suffixes (V8).

        Tries suffixes in order of specificity. Each rule requires
        the stripped base to be in the dictionary with min length 3.
        """
        # -tion/-sion (nominalizing): creation -> create + RESULT
        ids = self._try_nominalize(word)
        if ids is not None:
            return ids

        # -ness (quality): darkness -> dark + SOMETHING
        ids = self._try_ness(word)
        if ids is not None:
            return ids

        # -ment (result of): development -> develop + RESULT
        ids = self._try_ment(word)
        if ids is not None:
            return ids

        # -ful (full of): helpful -> help + HAVE
        ids = self._try_ful(word)
        if ids is not None:
            return ids

        # -able/-ible (capable of): readable -> CAN + read
        ids = self._try_able(word)
        if ids is not None:
            return ids

        # -al (adjectival): dimensional -> dimension + KIND
        ids = self._try_adjectival(word)
        if ids is not None:
            return ids

        # -ly (adverbial): quickly -> quick + LIKE_AS
        ids = self._try_adverbial(word)
        if ids is not None:
            return ids

        # -or/-er (agent): creator -> create + SOMEONE
        # Must come after comparative (-er) which is checked earlier
        ids = self._try_agent(word)
        if ids is not None:
            return ids

        return None

    def _lookup_base(self, word: str, suffix_len: int,
                     restore_e: bool = False,
                     min_base: int = 3) -> tuple[str, list[int]] | None:
        """Strip suffix, try to find base in dictionary.

        min_base applies to the stripped base (before doubled-consonant removal).
        Doubled-consonant candidates (runner -> run) are always allowed since
        the original base (runn, 4 chars) already passed the min check.
        """
        base = word[:-suffix_len]
        if len(base) < min_base:
            return None
        candidates = [base]
        if restore_e:
            candidates.append(base + "e")
        # Handle doubled consonants: runner -> runn -> run
        if len(base) > 2 and base[-1] == base[-2] and base[-1] not in "aeiou":
            candidates.append(base[:-1])
        for c in candidates:
            ids = self._dict.lookup(c)
            if ids is not None and ids:
                return (c, ids)
        return None

    def _try_agent(self, word: str) -> list[int] | None:
        """Handle agent suffixes -or, -er (person who does X).

        creator -> CREATE + SOMEONE
        teacher -> TEACH + SOMEONE

        Min base 4 chars to avoid: arm->armor, man->manor, don->donor.
        """
        if self._someone_id is None:
            return None

        for suffix in ("or", "er"):
            if word.endswith(suffix) and len(word) > 5:
                result = self._lookup_base(word, 2, restore_e=True, min_base=4)
                if result is not None:
                    _, base_ids = result
                    return base_ids + [self._someone_id]
        return None

    def _try_adjectival(self, word: str) -> list[int] | None:
        """Handle adjectival suffix -al: dimensional -> dimension + KIND."""
        if self._kind_id is None:
            return None
        if word.endswith("al") and len(word) > 5:
            result = self._lookup_base(word, 2, min_base=4)
            if result is not None:
                _, base_ids = result
                return base_ids + [self._kind_id]
        return None

    def _try_adverbial(self, word: str) -> list[int] | None:
        """Handle adverbial suffix -ly: quickly -> quick + LIKE_AS."""
        if self._like_as_id is None:
            return None
        if word.endswith("ly") and len(word) > 4:
            result = self._lookup_base(word, 2, min_base=3)
            if result is not None:
                _, base_ids = result
                return base_ids + [self._like_as_id]
            # -ily -> -y base: easily -> easy
            if word.endswith("ily") and len(word) > 5:
                base = word[:-3] + "y"
                ids = self._dict.lookup(base)
                if ids is not None and ids:
                    return ids + [self._like_as_id]
        return None

    def _try_nominalize(self, word: str) -> list[int] | None:
        """Handle nominalizing suffixes -tion, -sion: creation -> create + RESULT."""
        if self._result_id is None:
            return None

        for suffix in ("ation", "ition"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                # creation -> cre + ate -> create
                base = word[:-len(suffix)]
                for candidate in (base + "ate", base + "it" if suffix == "ition" else None):
                    if candidate is None:
                        continue
                    ids = self._dict.lookup(candidate)
                    if ids is not None and ids:
                        return ids + [self._result_id]

        if word.endswith("ction") and len(word) > 7:
            # protection -> protect, construction -> construct
            base = word[:-4] + "t"  # strip "tion", add "t" back
            ids = self._dict.lookup(base)
            if ids is not None and ids:
                return ids + [self._result_id]

        if word.endswith("sion") and len(word) > 6:
            # decision -> decide, expansion -> expand
            base = word[:-4]
            for candidate in (base + "de", base + "d", base + "t", base + "se"):
                ids = self._dict.lookup(candidate)
                if ids is not None and ids:
                    return ids + [self._result_id]

        return None

    def _try_ness(self, word: str) -> list[int] | None:
        """Handle quality suffix -ness: darkness -> dark + SOMETHING."""
        if self._something_id is None:
            return None
        if word.endswith("ness") and len(word) > 6:
            result = self._lookup_base(word, 4, min_base=3)
            if result is not None:
                _, base_ids = result
                return base_ids + [self._something_id]
            # -iness -> -y base: happiness -> happy
            if word.endswith("iness") and len(word) > 7:
                base = word[:-5] + "y"
                ids = self._dict.lookup(base)
                if ids is not None and ids:
                    return ids + [self._something_id]
        return None

    def _try_ment(self, word: str) -> list[int] | None:
        """Handle result suffix -ment: development -> develop + RESULT."""
        if self._result_id is None:
            return None
        if word.endswith("ment") and len(word) > 6:
            result = self._lookup_base(word, 4, restore_e=True, min_base=3)
            if result is not None:
                _, base_ids = result
                return base_ids + [self._result_id]
        return None

    def _try_ful(self, word: str) -> list[int] | None:
        """Handle -ful suffix: helpful -> help + HAVE."""
        if self._have_id is None:
            return None
        if word.endswith("ful") and len(word) > 5:
            result = self._lookup_base(word, 3, min_base=3)
            if result is not None:
                _, base_ids = result
                return base_ids + [self._have_id]
        return None

    def _try_able(self, word: str) -> list[int] | None:
        """Handle -able/-ible suffix: readable -> CAN + read."""
        if self._can_id is None:
            return None
        for suffix, slen in (("able", 4), ("ible", 4)):
            if word.endswith(suffix) and len(word) > slen + 2:
                result = self._lookup_base(word, slen, restore_e=True, min_base=3)
                if result is not None:
                    _, base_ids = result
                    return [self._can_id] + base_ids
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
