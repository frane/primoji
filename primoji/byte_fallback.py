"""Byte fallback encoder for unknown words.

When a word isn't found in the emoji dictionary and SymSpell can't correct it,
encode it as raw UTF-8 bytes wrapped in boundary markers. This guarantees
zero UNK tokens and zero information loss.

Follows the same pattern as SentencePiece's byte_fallback (used by Llama 1/2/3).
"""

from __future__ import annotations

# Token IDs for byte fallback (must match vocabulary.py)
BYTES_START_ID: int = 1660
BYTES_END_ID: int = 1661
BYTE_TOKEN_OFFSET: int = 1662  # Byte 0x00 → ID 1662, 0xFF → ID 1917


def encode_bytes(word: str) -> list[int]:
    """Encode a word as UTF-8 byte token IDs with boundary markers.

    Args:
        word: Unknown word to encode as bytes.

    Returns:
        List of token IDs: [BYTES_START] + byte_ids + [BYTES_END]
    """
    byte_ids = [BYTE_TOKEN_OFFSET + b for b in word.encode("utf-8")]
    return [BYTES_START_ID] + byte_ids + [BYTES_END_ID]


def decode_bytes(ids: list[int]) -> str:
    """Decode byte token IDs back to a string.

    Args:
        ids: Token ID sequence starting with BYTES_START and ending
             with BYTES_END, with byte tokens in between.

    Returns:
        Decoded string.

    Raises:
        ValueError: If byte sequence is not valid UTF-8 or markers are missing.
    """
    if not ids:
        raise ValueError("Empty token ID sequence")
    if ids[0] != BYTES_START_ID:
        raise ValueError(f"Expected BYTES_START ({BYTES_START_ID}), got {ids[0]}")

    end_idx = -1
    for i in range(1, len(ids)):
        if ids[i] == BYTES_END_ID:
            end_idx = i
            break
    if end_idx == -1:
        raise ValueError(f"Missing BYTES_END ({BYTES_END_ID}) marker")

    byte_values = []
    for tid in ids[1:end_idx]:
        bval = tid - BYTE_TOKEN_OFFSET
        if not (0 <= bval <= 255):
            raise ValueError(f"Token ID {tid} is not a valid byte token")
        byte_values.append(bval)

    return bytes(byte_values).decode("utf-8")


def is_byte_token(token_id: int) -> bool:
    """Check if a token ID is a byte fallback token (0x00-0xFF range)."""
    return BYTE_TOKEN_OFFSET <= token_id <= BYTE_TOKEN_OFFSET + 255


def is_byte_boundary(token_id: int) -> bool:
    """Check if a token ID is a byte boundary marker (START or END)."""
    return token_id in (BYTES_START_ID, BYTES_END_ID)


def is_byte_region_token(token_id: int) -> bool:
    """Check if a token ID is any byte-related token (boundary or data)."""
    return is_byte_boundary(token_id) or is_byte_token(token_id)
