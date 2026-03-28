"""Decoder: emoji token ID sequences → English text.

Provides decoding with byte fallback support:
- Standard tokens: dictionary reverse lookup or primitive description
- Byte regions: UTF-8 reconstruction from byte token IDs
- Contraction tokens: direct string mapping
"""

from __future__ import annotations

from primoji.byte_fallback import (
    BYTES_END_ID,
    BYTES_START_ID,
    decode_bytes,
    is_byte_boundary,
    is_byte_token,
)
from primoji.dictionary import Dictionary
from primoji.primitives import get_primitive_by_id
from primoji.utils import SpecialTokens
from primoji.vocabulary import CONTRACTION_TOKENS, Vocabulary


# Reverse map: contraction ID → string
_CONTRACTION_ID_TO_STR: dict[int, str] = {v: k for k, v in CONTRACTION_TOKENS.items()}


class Decoder:
    """Decode Primoji token ID sequences back to English.

    Handles standard emoji/primitive tokens, contraction tokens,
    and byte fallback regions.
    """

    def __init__(self, vocabulary: Vocabulary, dictionary: Dictionary) -> None:
        self._vocab = vocabulary
        self._dict = dictionary

    def decode_canonical(self, ids: list[int]) -> str:
        """Decode token IDs to canonical English.

        Handles byte fallback regions, contraction tokens, and standard
        emoji/primitive tokens via dictionary reverse lookup.

        Args:
            ids: List of token IDs.

        Returns:
            Canonical English text.
        """
        words: list[str] = []
        i = 0
        while i < len(ids):
            tid = ids[i]

            # Skip special tokens (BOS, EOS, PAD, UNK)
            if SpecialTokens.is_special(tid) and not is_byte_boundary(tid):
                i += 1
                continue

            # Handle byte fallback regions
            if tid == BYTES_START_ID:
                j = i + 1
                while j < len(ids) and ids[j] != BYTES_END_ID:
                    j += 1
                # Decode the byte region (including markers)
                end = min(j + 1, len(ids))
                try:
                    word = decode_bytes(ids[i:end])
                    words.append(word)
                except (ValueError, UnicodeDecodeError):
                    words.append("<bytes?>")
                i = end
                continue

            # Handle contraction tokens
            contraction = _CONTRACTION_ID_TO_STR.get(tid)
            if contraction is not None:
                words.append(contraction)
                i += 1
                continue

            # Try matching longest subsequence in dictionary (up to 5 tokens)
            matched = False
            for length in range(min(5, len(ids) - i), 0, -1):
                subseq = ids[i : i + length]
                # Skip if subseq contains byte tokens
                if any(is_byte_token(t) or is_byte_boundary(t) for t in subseq):
                    continue
                word = self._dict.reverse_lookup(subseq)
                if word is not None:
                    words.append(word)
                    i += length
                    matched = True
                    break

            if not matched:
                word = self._describe_token(tid)
                if word:
                    words.append(word)
                i += 1

        return " ".join(words)

    def decode_semantic(self, ids: list[int]) -> str:
        """Decode token IDs to best-effort semantic English.

        Args:
            ids: List of token IDs.

        Returns:
            Semantic English approximation.
        """
        return self.decode_canonical(ids)

    def _describe_token(self, token_id: int) -> str | None:
        """Get a short English word for a single token ID.

        Args:
            token_id: Integer token ID.

        Returns:
            English word or None.
        """
        prim = get_primitive_by_id(token_id)
        if prim is not None:
            return prim.name.lower()

        token = self._vocab.decode_token(token_id)
        if token is not None:
            return token

        return None
