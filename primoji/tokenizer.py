"""Main tokenizer interface for Primoji.

Encodes English text into semantically meaningful emoji token ID sequences
and decodes them back to English using a 4-tier fallback pipeline.

Example:
    >>> tok = Tokenizer()
    >>> ids = tok.encode("The teacher explained photosynthesis")
    >>> tok.decode(ids)
    'teacher explained photosynthesis'
"""

from __future__ import annotations

from primoji.byte_fallback import encode_bytes
from primoji.composer import Composer
from primoji.decoder import Decoder
from primoji.dictionary import Dictionary
from primoji.fuzzy import FuzzyMatcher
from primoji.math_handler import is_math_expression
from primoji.preprocessor import Preprocessor
from primoji.primitives import PRIMITIVES
from primoji.utils import SpecialTokens, _IDS, normalize_text
from primoji.vocabulary import (
    ABBREVIATION_IDS,
    ANCHOR_TOKENS,
    COMMON_WORD_TOKENS,
    DIGIT_IDS,
    MATH_OP_IDS,
    ORDINAL_ID,
    ORDINAL_IDS,
    POSSESSIVE_ID,
    PUNCTUATION_IDS,
    Vocabulary,
)

_PRIM_ID_START = _IDS["PRIM_START"]
_PRIM_ID_MAX = _PRIM_ID_START + _IDS["PRIM_COUNT"] - 1
_EMOJI_MAX = _PRIM_ID_START - 1


class Tokenizer:
    """Primoji tokenizer: English text ↔ emoji token ID sequences.

    Pipeline (NEVER produces UNK tokens):
      Stage 1: Dictionary lookup (emoji, primitives, word tokens, anchors)
      Stage 2: SymSpell fuzzy match (edit dist 1, optional)
      Stage 3: Composition rules (morphological: negation, temporal, comparative)
      Stage 4: UTF-8 byte fallback (universal safety net)

    Contractions are expanded by the preprocessor before the pipeline runs.
    """

    def __init__(self, fuzzy: bool = True) -> None:
        """Initialize the Primoji tokenizer.

        Args:
            fuzzy: If True, enable Tier 3 fuzzy matching for typo correction.
        """
        self._vocab = Vocabulary()
        self._dict = Dictionary()
        self._composer = Composer(self._vocab, self._dict)
        self._decoder = Decoder(self._vocab, self._dict)
        self._preprocessor = Preprocessor()
        self._fuzzy: FuzzyMatcher | None = None
        if fuzzy:
            vocab_words = {w for w in self._dict._word_to_ids if self._dict._word_to_ids[w] is not None}
            self._fuzzy = FuzzyMatcher(vocab_words)

    @property
    def vocab_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return self._vocab.vocab_size

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Pipeline: dictionary -> fuzzy -> composition -> byte fallback.
        NEVER returns UNK tokens. Every input is encodable.

        Args:
            text: Input English text.

        Returns:
            List of integer token IDs.
        """
        text = normalize_text(text)
        if not text:
            return []

        # Math content gets special handling
        if is_math_expression(text):
            return self._encode_mixed(text)

        words = self._preprocessor.preprocess(text)
        return self._encode_words(words)

    def _encode_words(self, words: list[str]) -> list[int]:
        """Encode preprocessed words through the 4-tier pipeline."""
        ids: list[int] = []
        for word in words:
            ids.extend(self._encode_word(word))
        return ids

    def _encode_word(self, word: str) -> list[int]:
        """Encode a single word through the pipeline.

        Order: structural -> dictionary -> fuzzy -> composition -> bytes.
        Matches paper's described pipeline.
        """
        # Structural: possessive marker (emitted by preprocessor)
        if word == "<POSSESSIVE>":
            return [POSSESSIVE_ID]

        # Structural: punctuation
        punc_id = PUNCTUATION_IDS.get(word)
        if punc_id is not None:
            return [punc_id]

        # Structural: abbreviations (e.g., pp, cf., Dr.)
        abbrev_id = ABBREVIATION_IDS.get(word)
        if abbrev_id is not None:
            return [abbrev_id]

        # Structural: ordinals (1st, 2nd, ..., 31st, 40th, ..., 100th)
        ordinal_id = ORDINAL_IDS.get(word.lower())
        if ordinal_id is not None:
            return [ordinal_id]

        # Structural: higher ordinals (e.g., 42nd -> digits + ORDINAL marker)
        ordinal_ids = self._try_ordinal(word)
        if ordinal_ids is not None:
            return ordinal_ids

        # Structural: numbers
        if word.replace(".", "", 1).isdigit():
            from primoji.math_handler import tokenize_number
            return tokenize_number(word)

        lower = word.lower()

        # Stage 1: Dictionary lookup (includes word tokens, compositions, emoji, primitives)
        result = self._dict.lookup(lower)
        if result is not None:
            return result

        # Stage 1 fallback: common word tokens not yet in dictionary
        word_id = COMMON_WORD_TOKENS.get(lower)
        if word_id is not None:
            return [word_id]

        # Stage 1 fallback: anchor tokens (proper nouns, case-sensitive)
        anchor_id = ANCHOR_TOKENS.get(word)
        if anchor_id is not None:
            return [anchor_id]

        # Stage 2: Fuzzy match / spell correction (edit dist 1, unique, 4+ chars)
        if self._fuzzy:
            corrected = self._fuzzy.correct(lower)
            if corrected is not None:
                result = self._dict.lookup(corrected)
                if result is not None:
                    return result

        # Stage 3: Composition rules (negation, temporal, comparative)
        result = self._composer.compose(lower)
        if result != [SpecialTokens.UNK]:
            return result

        # Stage 4: Byte fallback (universal, lossless)
        return encode_bytes(word)

    @staticmethod
    def _try_ordinal(word: str) -> list[int] | None:
        """Try to encode an ordinal number (e.g., 42nd -> [4,2,ORDINAL])."""
        import re
        m = re.match(r'^(\d+)(st|nd|rd|th)$', word, re.IGNORECASE)
        if m is None:
            return None
        digits = m.group(1)
        from primoji.math_handler import tokenize_number
        return tokenize_number(digits) + [ORDINAL_ID]

    def _encode_mixed(self, text: str) -> list[int]:
        """Encode text containing math expressions mixed with natural language."""
        from primoji.math_handler import tokenize_number, tokenize_operator
        from primoji.utils import simple_word_tokenize

        tokens = simple_word_tokenize(text)
        ids: list[int] = []
        for token in tokens:
            if token.replace(".", "", 1).isdigit():
                ids.extend(tokenize_number(token))
            elif len(token) == 1 and token in "+-*/×÷=<>≤≥":
                op_id = tokenize_operator(token)
                if op_id is not None:
                    ids.append(op_id)
                else:
                    ids.extend(encode_bytes(token))
            else:
                ids.extend(self._encode_word(token))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode Primoji token IDs back to English text.

        Args:
            ids: List of integer token IDs.

        Returns:
            English text (canonical form).
        """
        return self._decoder.decode_canonical(ids)

    def describe(self, token_id: int) -> str:
        """Get a human-readable description of a token ID.

        Args:
            token_id: Integer token ID.

        Returns:
            Description string.
        """
        return self._vocab.describe(token_id)

    def classify_word(self, word: str) -> str:
        """Classify which pipeline stage handles a single word.

        Returns one of: "tier1_emoji", "tier2_primitive", "tier3_structural",
        "tier3_anchor", "tier1b_word", "dict_composed",
        "symspell_fuzzy", "composer_rule", "byte_fallback".

        Matches _encode_word pipeline order.
        """
        # Structural: possessive marker
        if word == "<POSSESSIVE>":
            return "tier3_structural"

        # Structural: punctuation / digits / math ops
        if PUNCTUATION_IDS.get(word) is not None:
            return "tier3_structural"

        # Structural: abbreviations
        if ABBREVIATION_IDS.get(word) is not None:
            return "tier3_structural"

        # Structural: ordinals
        if ORDINAL_IDS.get(word.lower()) is not None:
            return "tier3_structural"
        if self._try_ordinal(word) is not None:
            return "tier3_structural"

        if word.replace(".", "", 1).isdigit():
            return "tier3_structural"
        if MATH_OP_IDS.get(word) is not None:
            return "tier3_structural"

        lower = word.lower()

        # Stage 1: Dictionary lookup
        result = self._dict.lookup(lower)
        if result is not None:
            if len(result) == 1:
                tid = result[0]
                if 0 <= tid <= _EMOJI_MAX:
                    return "tier1_emoji"
                elif _PRIM_ID_START <= tid <= _PRIM_ID_MAX:
                    return "tier2_primitive"
                elif _IDS["WORD_START"] <= tid < _IDS["WORD_START"] + _IDS["WORD_COUNT"]:
                    return "tier1b_word"
                elif _IDS["ANCHOR_START"] <= tid < _IDS["ANCHOR_START"] + _IDS["ANCHOR_COUNT"]:
                    return "tier3_anchor"
                else:
                    return "tier3_structural"
            else:
                return "dict_composed"

        # Stage 1 fallback: common word token
        if COMMON_WORD_TOKENS.get(lower) is not None:
            return "tier1b_word"

        # Stage 1 fallback: anchor
        if ANCHOR_TOKENS.get(word) is not None:
            return "tier3_anchor"

        # Stage 2: Fuzzy match
        if self._fuzzy:
            corrected = self._fuzzy.correct(lower)
            if corrected is not None and self._dict.lookup(corrected) is not None:
                return "symspell_fuzzy"

        # Stage 3: Composer rules
        composed = self._composer.compose(lower)
        if composed != [SpecialTokens.UNK]:
            return "composer_rule"

        return "byte_fallback"

    @property
    def vocabulary(self) -> Vocabulary:
        """Access the underlying Vocabulary object."""
        return self._vocab

    @property
    def dictionary(self) -> Dictionary:
        """Access the underlying Dictionary object."""
        return self._dict
