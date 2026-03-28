"""Tests for the byte fallback encoder/decoder.

Covers encode_bytes, decode_bytes, is_byte_token, is_byte_boundary, and
is_byte_region_token. Byte fallback guarantees zero UNK tokens by encoding
any unknown word as raw UTF-8 bytes wrapped in boundary markers.
"""

from __future__ import annotations

import pytest

from primoji.byte_fallback import (
    BYTE_TOKEN_OFFSET,
    BYTES_END_ID,
    BYTES_START_ID,
    decode_bytes,
    encode_bytes,
    is_byte_boundary,
    is_byte_region_token,
    is_byte_token,
)


# ── encode_bytes ────────────────────────────────────────────────────────────


class TestEncodeBytes:
    def test_ascii_word_produces_correct_structure(self) -> None:
        """ASCII word should be: [START] + byte_ids + [END]."""
        ids = encode_bytes("watr")
        assert ids[0] == BYTES_START_ID
        assert ids[-1] == BYTES_END_ID
        # 4 ASCII chars → 4 byte tokens + 2 markers = 6
        assert len(ids) == 6

    def test_ascii_byte_ids_are_offset_correctly(self) -> None:
        """Each ASCII char's byte value + BYTE_TOKEN_OFFSET should match the ID."""
        ids = encode_bytes("ab")
        assert ids[1] == BYTE_TOKEN_OFFSET + ord("a")
        assert ids[2] == BYTE_TOKEN_OFFSET + ord("b")

    def test_unicode_word_produces_multibyte_tokens(self) -> None:
        """Non-ASCII characters use 2-4 bytes each in UTF-8."""
        ids = encode_bytes("cafe\u0301")  # "café" as NFC decomposed
        # All IDs between markers should be valid byte tokens
        for tid in ids[1:-1]:
            assert is_byte_token(tid)

    def test_empty_string_produces_start_and_end_only(self) -> None:
        """Empty string encodes as [BYTES_START, BYTES_END]."""
        ids = encode_bytes("")
        assert ids == [BYTES_START_ID, BYTES_END_ID]

    def test_single_char(self) -> None:
        ids = encode_bytes("x")
        assert len(ids) == 3
        assert ids[0] == BYTES_START_ID
        assert ids[1] == BYTE_TOKEN_OFFSET + ord("x")
        assert ids[2] == BYTES_END_ID

    def test_ids_are_in_byte_range(self) -> None:
        """All byte token IDs should be in [BYTE_TOKEN_OFFSET, BYTE_TOKEN_OFFSET+255]."""
        ids = encode_bytes("Hello!")
        for tid in ids[1:-1]:
            assert BYTE_TOKEN_OFFSET <= tid <= BYTE_TOKEN_OFFSET + 255


# ── decode_bytes ────────────────────────────────────────────────────────────


class TestDecodeBytes:
    def test_ascii_roundtrip(self) -> None:
        """Encode then decode ASCII word should recover original."""
        original = "watr"
        ids = encode_bytes(original)
        recovered = decode_bytes(ids)
        assert recovered == original

    def test_unicode_roundtrip(self) -> None:
        """Encode then decode Unicode word should recover original."""
        original = "caf\u00e9"  # "café" NFC
        ids = encode_bytes(original)
        recovered = decode_bytes(ids)
        assert recovered == original

    def test_empty_string_roundtrip(self) -> None:
        ids = encode_bytes("")
        recovered = decode_bytes(ids)
        assert recovered == ""

    def test_emoji_roundtrip(self) -> None:
        """Even emoji strings can roundtrip through byte fallback."""
        original = "\U0001f600"  # 😀
        ids = encode_bytes(original)
        recovered = decode_bytes(ids)
        assert recovered == original

    def test_missing_start_marker_raises(self) -> None:
        with pytest.raises(ValueError, match="BYTES_START"):
            decode_bytes([BYTE_TOKEN_OFFSET + 65, BYTES_END_ID])

    def test_missing_end_marker_raises(self) -> None:
        with pytest.raises(ValueError, match="BYTES_END"):
            decode_bytes([BYTES_START_ID, BYTE_TOKEN_OFFSET + 65])

    def test_empty_ids_raises(self) -> None:
        with pytest.raises(ValueError, match="Empty"):
            decode_bytes([])

    def test_invalid_byte_token_raises(self) -> None:
        """A token ID outside the byte range between markers should raise."""
        with pytest.raises(ValueError, match="not a valid byte token"):
            decode_bytes([BYTES_START_ID, 9999, BYTES_END_ID])

    def test_invalid_utf8_bytes_raise(self) -> None:
        """Byte values that don't form valid UTF-8 should raise."""
        # 0xFF 0xFE is not valid UTF-8
        ids = [BYTES_START_ID, BYTE_TOKEN_OFFSET + 0xFF, BYTE_TOKEN_OFFSET + 0xFE, BYTES_END_ID]
        with pytest.raises((ValueError, UnicodeDecodeError)):
            decode_bytes(ids)


# ── is_byte_token ───────────────────────────────────────────────────────────


class TestIsByteToken:
    def test_first_byte_token(self) -> None:
        assert is_byte_token(BYTE_TOKEN_OFFSET) is True

    def test_last_byte_token(self) -> None:
        assert is_byte_token(BYTE_TOKEN_OFFSET + 255) is True

    def test_middle_byte_token(self) -> None:
        assert is_byte_token(BYTE_TOKEN_OFFSET + 100) is True

    def test_below_range_returns_false(self) -> None:
        assert is_byte_token(BYTE_TOKEN_OFFSET - 1) is False

    def test_above_range_returns_false(self) -> None:
        assert is_byte_token(BYTE_TOKEN_OFFSET + 256) is False

    def test_zero_returns_false(self) -> None:
        assert is_byte_token(0) is False

    def test_boundary_markers_are_not_byte_tokens(self) -> None:
        """START and END markers are boundaries, not data bytes."""
        assert is_byte_token(BYTES_START_ID) is False
        assert is_byte_token(BYTES_END_ID) is False


# ── is_byte_boundary ────────────────────────────────────────────────────────


class TestIsByteBoundary:
    def test_start_is_boundary(self) -> None:
        assert is_byte_boundary(BYTES_START_ID) is True

    def test_end_is_boundary(self) -> None:
        assert is_byte_boundary(BYTES_END_ID) is True

    def test_byte_data_token_is_not_boundary(self) -> None:
        assert is_byte_boundary(BYTE_TOKEN_OFFSET + 65) is False

    def test_regular_token_is_not_boundary(self) -> None:
        assert is_byte_boundary(0) is False
        assert is_byte_boundary(1200) is False


# ── is_byte_region_token ────────────────────────────────────────────────────


class TestIsByteRegionToken:
    def test_start_marker(self) -> None:
        assert is_byte_region_token(BYTES_START_ID) is True

    def test_end_marker(self) -> None:
        assert is_byte_region_token(BYTES_END_ID) is True

    def test_byte_data(self) -> None:
        assert is_byte_region_token(BYTE_TOKEN_OFFSET + 42) is True

    def test_non_byte_token(self) -> None:
        assert is_byte_region_token(0) is False
        assert is_byte_region_token(1200) is False
