"""Vocabulary definition for Primoji tokenizer.

All ID ranges are dynamically computed from actual data file sizes.
Loads Tier 1 emoji from emoji_catalog.json, primitives from primitives.json,
anchors from proper_noun_anchors.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from primoji.primitives import PRIMITIVES
from primoji.utils import SpecialTokens, _IDS

_DATA_DIR = Path(__file__).parent.parent / "data"


# ── Tier 1: Direct Unicode Emoji (IDs 0–1199) ────────────────────────────────

def _load_tier1() -> dict[str, int]:
    """Load Tier 1 emoji→ID mapping from data/emoji_catalog.json."""
    path = _DATA_DIR / "emoji_catalog.json"
    if path.exists():
        with path.open() as f:
            data = json.load(f)
        return {e["emoji"]: e["id"] for e in data["emoji"]}
    return {}


TIER1_DIRECT_EMOJI: dict[str, int] = _load_tier1()

# ── Tier 2: Compositional Primitives (IDs 1200–1331, fixed) ──────────────────

TIER2_PRIMITIVES: dict[str, int] = {p.emoji: p.id for p in PRIMITIVES}

# ── Country Flags ────────────────────────────────────────────────────────────

_FLAG_BASE_ID: int = _IDS["FLAGS_START"]

_ISO_CODES: list[str] = [
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR",
    "AS", "AT", "AU", "AW", "AX", "AZ", "BA", "BB", "BD", "BE",
    "BF", "BG", "BH", "BI", "BJ", "BL", "BM", "BN", "BO", "BQ",
    "BR", "BS", "BT", "BV", "BW", "BY", "BZ", "CA", "CC", "CD",
    "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN", "CO", "CR",
    "CU", "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM",
    "DO", "DZ", "EC", "EE", "EG", "EH", "ER", "ES", "ET", "FI",
    "FJ", "FK", "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF",
    "GG", "GH", "GI", "GL", "GM", "GN", "GP", "GQ", "GR", "GS",
    "GT", "GU", "GW", "GY", "HK", "HM", "HN", "HR", "HT", "HU",
    "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR", "IS", "IT",
    "JE", "JM", "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN",
    "KP", "KR", "KW", "KY", "KZ", "LA", "LB", "LC", "LI", "LK",
    "LR", "LS", "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME",
    "MF", "MG", "MH", "MK", "ML", "MM", "MN", "MO", "MP", "MQ",
    "MR", "MS", "MT", "MU", "MV", "MW", "MX", "MY", "MZ", "NA",
    "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP", "NR", "NU",
    "NZ", "OM", "PA", "PE", "PF", "PG", "PH", "PK", "PL", "PM",
    "PN", "PR", "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS",
    "RU", "RW", "SA", "SB", "SC", "SD", "SE", "SG", "SH", "SI",
    "SJ", "SK", "SL", "SM", "SN", "SO", "SR", "SS", "ST", "SV",
    "SX", "SY", "SZ", "TC", "TD", "TF", "TG", "TH", "TJ", "TK",
    "TL", "TM", "TN", "TO", "TR", "TT", "TV", "TW", "TZ", "UA",
    "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI",
    "VN", "VU", "WF", "WS", "YE", "YT", "ZA", "ZM", "ZW",
]


def _iso_to_flag(code: str) -> str:
    """Convert ISO 3166-1 alpha-2 code to flag emoji."""
    return "".join(chr(0x1F1E6 + ord(c) - ord("A")) for c in code.upper())


TIER3_FLAGS: dict[str, int] = {}
_FLAG_EMOJI_TO_ID: dict[str, int] = {}
for _i, _code in enumerate(_ISO_CODES):
    _flag = _iso_to_flag(_code)
    TIER3_FLAGS[_code] = _FLAG_BASE_ID + _i
    _FLAG_EMOJI_TO_ID[_flag] = _FLAG_BASE_ID + _i

# ── Contraction Tokens ───────────────────────────────────────────────────────

_CONTRACT_BASE: int = _IDS["CONTRACT_START"]

_CONTRACTION_LIST: list[str] = [
    "don't", "can't", "won't", "i'm", "it's", "isn't", "didn't", "doesn't",
    "wasn't", "weren't", "couldn't", "wouldn't", "shouldn't", "haven't",
    "hasn't", "i'll", "you're", "they're", "we're", "he's",
    "'t", "'m", "'ll", "'ve", "'re", "'d", "'s",
]

CONTRACTION_TOKENS: dict[str, int] = {
    c: _CONTRACT_BASE + i for i, c in enumerate(_CONTRACTION_LIST)
}

DEDICATED_CONTRACTIONS: set[str] = {
    k for k in CONTRACTION_TOKENS if not k.startswith("'")
}
CONTRACTION_SUFFIXES: set[str] = {
    k for k in CONTRACTION_TOKENS if k.startswith("'")
}

# ── Anchor Tokens (proper nouns from FineWeb-Edu) ────────────────────────────

_ANCHOR_BASE: int = _IDS["ANCHOR_START"]


def _load_anchors() -> dict[str, int]:
    """Load anchor name→ID mapping from proper_noun_anchors.json."""
    path = _DATA_DIR / "proper_noun_anchors.json"
    if path.exists():
        with path.open() as f:
            data = json.load(f)
        return {a["name"]: _ANCHOR_BASE + i for i, a in enumerate(data["anchors"])}
    return {}


ANCHOR_TOKENS: dict[str, int] = _load_anchors()

# ── Structural Tokens ────────────────────────────────────────────────────────

_STRUCTURAL_BASE_ID: int = _IDS["STRUCT_START"]

DIGIT_IDS: dict[str, int] = {str(d): _STRUCTURAL_BASE_ID + d for d in range(10)}

_MATH_OPS: list[str] = ["+", "−", "×", "÷", "=", "<", ">", "≤", "≥", "∫", "Σ", "√", "π", "∞"]
MATH_OP_IDS: dict[str, int] = {op: _STRUCTURAL_BASE_ID + 10 + i for i, op in enumerate(_MATH_OPS)}
MATH_OP_IDS["-"] = MATH_OP_IDS["−"]
MATH_OP_IDS["*"] = MATH_OP_IDS["×"]
MATH_OP_IDS["/"] = MATH_OP_IDS["÷"]

_PUNCTUATION: list[str] = [".", ",", "!", "?", ":", ";", '"', "'", "(", ")", "[", "]", "{", "}"]
PUNCTUATION_IDS: dict[str, int] = {
    p: _STRUCTURAL_BASE_ID + 24 + i for i, p in enumerate(_PUNCTUATION)
}

BYTE_FALLBACK_OFFSET: int = _IDS["BYTE_OFFSET"]


# ── Vocabulary class ──────────────────────────────────────────────────────────


class Vocabulary:
    """Full Primoji vocabulary (dynamically sized).

    Provides bidirectional mapping between tokens and their integer IDs.
    """

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._id_to_description: dict[int, str] = {}

        # Tier 1: Direct emoji
        for emoji_char, tid in TIER1_DIRECT_EMOJI.items():
            self._token_to_id[emoji_char] = tid
            self._id_to_token[tid] = emoji_char
            self._id_to_description[tid] = "Tier 1 direct emoji"

        # Tier 2: Primitives
        for p in PRIMITIVES:
            self._token_to_id[p.emoji] = p.id
            self._id_to_token[p.id] = p.emoji
            self._id_to_description[p.id] = f"Tier 2 primitive: {p.name} — {p.description}"

        # Flags
        for code, tid in TIER3_FLAGS.items():
            flag = _iso_to_flag(code)
            self._token_to_id[flag] = tid
            self._id_to_token[tid] = flag
            self._id_to_description[tid] = f"Flag: {code}"

        # Contractions
        for c, tid in CONTRACTION_TOKENS.items():
            self._token_to_id[c] = tid
            self._id_to_token[tid] = c
            self._id_to_description[tid] = f"Contraction: {c}"

        # Anchors
        for name, tid in ANCHOR_TOKENS.items():
            self._token_to_id[name] = tid
            self._id_to_token[tid] = name
            self._id_to_description[tid] = f"Anchor: {name}"

        # Digits
        for digit, tid in DIGIT_IDS.items():
            self._token_to_id[digit] = tid
            self._id_to_token[tid] = digit
            self._id_to_description[tid] = f"Digit: {digit}"

        # Math operators
        for op, tid in MATH_OP_IDS.items():
            if op not in self._token_to_id:
                self._token_to_id[op] = tid
                self._id_to_token[tid] = op
                self._id_to_description[tid] = f"Math operator: {op}"

        # Punctuation
        for punc, tid in PUNCTUATION_IDS.items():
            self._token_to_id[punc] = tid
            self._id_to_token[tid] = punc
            self._id_to_description[tid] = f"Punctuation: {punc}"

        # Special tokens
        for name, tid in SpecialTokens.ALL.items():
            token_str = f"<{name}>"
            self._token_to_id[token_str] = tid
            self._id_to_token[tid] = token_str
            self._id_to_description[tid] = f"Special token: {name}"

        # Byte fallback
        for bval in range(256):
            tid = BYTE_FALLBACK_OFFSET + bval
            token_str = f"<0x{bval:02X}>"
            self._token_to_id[token_str] = tid
            self._id_to_token[tid] = token_str
            self._id_to_description[tid] = f"Byte: 0x{bval:02X}"

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including byte fallback."""
        return _IDS["VOCAB_SIZE"]

    def encode_token(self, token: str) -> int | None:
        """Map a single token string to its integer ID."""
        return self._token_to_id.get(token)

    def decode_token(self, token_id: int) -> str | None:
        """Map a token ID back to its string representation."""
        return self._id_to_token.get(token_id)

    def describe(self, token_id: int) -> str:
        """Get a human-readable description for a token ID."""
        token = self._id_to_token.get(token_id)
        desc = self._id_to_description.get(token_id, "Unknown token")
        if token is not None:
            return f"{token} (ID {token_id}) — {desc}"
        return f"ID {token_id} — {desc}"

    def contains(self, token: str) -> bool:
        """Check if a token string is in the vocabulary."""
        return token in self._token_to_id

    def get_flag_id(self, iso_code: str) -> int | None:
        """Get the token ID for a country flag."""
        return TIER3_FLAGS.get(iso_code.upper())

    def get_digit_id(self, digit: str) -> int | None:
        """Get the token ID for a single digit character."""
        return DIGIT_IDS.get(digit)

    def get_math_op_id(self, op: str) -> int | None:
        """Get the token ID for a math operator."""
        return MATH_OP_IDS.get(op)

    def get_punctuation_id(self, punc: str) -> int | None:
        """Get the token ID for a punctuation character."""
        return PUNCTUATION_IDS.get(punc)

    def get_contraction_id(self, contraction: str) -> int | None:
        """Get the token ID for a contraction."""
        return CONTRACTION_TOKENS.get(contraction.lower())

    def get_anchor_id(self, name: str) -> int | None:
        """Get the token ID for a proper noun anchor."""
        return ANCHOR_TOKENS.get(name)
