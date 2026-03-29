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
from primoji.utils import SpecialTokens, _IDS, normalize_text
from primoji.vocabulary import (
    ANCHOR_TOKENS,
    COMMON_WORD_TOKENS,
    CONTRACTION_TOKENS,
    DIGIT_IDS,
    MATH_OP_IDS,
    PUNCTUATION_IDS,
    Vocabulary,
)


class Tokenizer:
    """Primoji tokenizer: English text ↔ emoji token ID sequences.

    Uses a 4-tier fallback pipeline that NEVER produces UNK tokens:
      Tier 1: Exact dictionary lookup (~85% of tokens)
      Tier 2: Contraction handling (dedicated tokens or apostrophe split)
      Tier 3: Conservative SymSpell fuzzy match (optional)
      Tier 4: UTF-8 byte fallback (universal safety net)
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
        """Encode text to token IDs using 4-tier pipeline.

        Tier 1: Exact dictionary lookup (~85% of tokens)
        Tier 2: Contraction handling (split or dedicated token)
        Tier 3: Conservative SymSpell fuzzy match (edit dist 1, unique, 4+ chars)
        Tier 4: UTF-8 byte fallback (zero information loss)

        This method NEVER returns UNK tokens. Every input is encodable.

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
        """Encode a single word through the 4-tier pipeline."""
        # Check punctuation
        punc_id = PUNCTUATION_IDS.get(word)
        if punc_id is not None:
            return [punc_id]

        # Check if it's a number
        if word.replace(".", "", 1).isdigit():
            from primoji.math_handler import tokenize_number
            return tokenize_number(word)

        # Check contraction token
        contraction_id = CONTRACTION_TOKENS.get(word.lower())
        if contraction_id is not None:
            return [contraction_id]

        # Tier 1: Exact dictionary lookup
        result = self._dict.lookup(word.lower())
        if result is not None:
            return result

        # Tier 1b: Try composition rules (negation, temporal, comparative)
        result = self._composer.compose(word.lower())
        if result != [SpecialTokens.UNK]:
            return result

        # Common word token (Tier 1b)
        word_id = COMMON_WORD_TOKENS.get(word.lower())
        if word_id is not None:
            return [word_id]

        # Check anchor tokens (proper nouns — case-sensitive)
        anchor_id = ANCHOR_TOKENS.get(word)
        if anchor_id is not None:
            return [anchor_id]

        # Tier 3: Fuzzy match (optional)
        if self._fuzzy:
            corrected = self._fuzzy.correct(word.lower())
            if corrected is not None:
                result = self._dict.lookup(corrected)
                if result is not None:
                    return result

        # Tier 4: Byte fallback (universal — always works)
        return encode_bytes(word)

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

    def decode(self, ids: list[int], verbatim: bool = False) -> str:
        """Decode Primoji token IDs back to English text.

        Args:
            ids: List of integer token IDs.
            verbatim: If True, attempt exact reconstruction (not yet implemented).

        Returns:
            English text.
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
        """Classify which tier handles a single word, without encoding.

        Returns one of: "tier1_emoji", "tier2_primitive", "tier3_structural",
        "tier3_contraction", "tier3_anchor", "dict_composed", "dict_dropped",
        "composer_rule", "symspell_fuzzy", "byte_fallback".

        Args:
            word: A single word token.

        Returns:
            Tier label string.
        """
        # Punctuation / digits / math ops
        if PUNCTUATION_IDS.get(word) is not None:
            return "tier3_structural"
        if word.replace(".", "", 1).isdigit():
            return "tier3_structural"
        if MATH_OP_IDS.get(word) is not None:
            return "tier3_structural"

        # Contraction token
        if CONTRACTION_TOKENS.get(word.lower()) is not None:
            return "tier3_contraction"

        # Dictionary lookup
        result = self._dict.lookup(word.lower())
        if result is not None:
            if not result:
                return "dict_dropped"  # articles (the, a, an)
            if len(result) == 1:
                tid = result[0]
                if 0 <= tid <= 1199:
                    return "tier1_emoji"
                elif 1200 <= tid <= 1331:
                    return "tier2_primitive"
                elif _IDS["WORD_START"] <= tid < _IDS["WORD_START"] + _IDS["WORD_COUNT"]:
                    return "tier1b_word"
                elif _IDS["ANCHOR_START"] <= tid < _IDS["ANCHOR_START"] + _IDS["ANCHOR_COUNT"]:
                    return "tier3_anchor"
                else:
                    return "tier3_structural"
            else:
                return "dict_composed"

        # Composer rules (negation, temporal, comparative)
        composed = self._composer.compose(word.lower())
        if composed != [SpecialTokens.UNK]:
            return "composer_rule"

        # Common word token (Tier 1b)
        if COMMON_WORD_TOKENS.get(word.lower()) is not None:
            return "tier1b_word"

        # Anchor
        if ANCHOR_TOKENS.get(word) is not None:
            return "tier3_anchor"

        # Fuzzy match
        if self._fuzzy:
            corrected = self._fuzzy.correct(word.lower())
            if corrected is not None and self._dict.lookup(corrected) is not None:
                return "symspell_fuzzy"

        return "byte_fallback"

    @property
    def vocabulary(self) -> Vocabulary:
        """Access the underlying Vocabulary object."""
        return self._vocab

    @property
    def dictionary(self) -> Dictionary:
        """Access the underlying Dictionary object."""
        return self._dict
