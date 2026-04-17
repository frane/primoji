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

_DATA_DIR = Path(__file__).parent / "data"


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

# ── Contraction Tokens (REMOVED in V8) ──────────────────────────────────────
# The preprocessor expands all contractions before the tokenizer sees them,
# so these 27 token IDs were never produced. Removed in V8 cleanup.
# The 27 ID slots (CONTRACT_START .. CONTRACT_START+26) remain allocated
# in the frozen layout to preserve downstream ID offsets.

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

# ── Common Word Tokens (Tier 1b) ─────────────────────────────────────────────

_WORD_BASE: int = _IDS["WORD_START"]


def _load_common_words() -> dict[str, int]:
    """Load common word→ID mapping from common_words.json."""
    path = _DATA_DIR / "common_words.json"
    if path.exists():
        with path.open() as f:
            data = json.load(f)
        return {w: _WORD_BASE + i for i, w in enumerate(data["words"])}
    return {}


COMMON_WORD_TOKENS: dict[str, int] = _load_common_words()

# ── Structural Tokens ────────────────────────────────────────────────────────

_STRUCTURAL_BASE_ID: int = _IDS["STRUCT_START"]

DIGIT_IDS: dict[str, int] = {str(d): _STRUCTURAL_BASE_ID + d for d in range(10)}

_MATH_OPS: list[str] = ["+", "−", "×", "÷", "=", "<", ">", "≤", "≥", "∫", "Σ", "√", "π", "∞"]
MATH_OP_IDS: dict[str, int] = {op: _STRUCTURAL_BASE_ID + 10 + i for i, op in enumerate(_MATH_OPS)}
MATH_OP_IDS["-"] = MATH_OP_IDS["−"]
MATH_OP_IDS["*"] = MATH_OP_IDS["×"]
MATH_OP_IDS["/"] = MATH_OP_IDS["÷"]

_PUNCTUATION: list[str] = [
    # ASCII punctuation
    ".", ",", "!", "?", ":", ";", '"', "'", "(", ")", "[", "]", "{", "}",
    # Smart/curly quotes
    "\u2018", "\u2019", "\u201C", "\u201D",
    # Dashes
    "\u2013", "\u2014", "\u2015",
    # Ellipsis
    "\u2026",
    # Pipes/bars
    "|", "\u00B7",
    # Bullets
    "\u2022",
    # Math/science symbols (excluding those already in MATH_OP_IDS: pi, integral, sum, sqrt)
    "\u00B0", "\u00B1", "\u2248", "\u2260",
    "\u0394", "\u03B1", "\u03B2", "\u03B3", "\u03BB", "\u03BC", "\u03C3", "\u03C9",
    "\u2202",
    # Legal/trademark
    "\u00A9", "\u00AE", "\u2122",
    # Currency
    "$", "\u00A3", "\u20AC", "\u00A5", "\u20B9",
    # Superscript digits
    "\u00B9", "\u00B2", "\u00B3",
    # Fractions
    "\u00BC", "\u00BD", "\u00BE",
    # Guillemets
    "\u00AB", "\u00BB",
    # Common ASCII symbols
    "\\", "@", "#", "%", "&", "~",
]
PUNCTUATION_IDS: dict[str, int] = {
    p: _STRUCTURAL_BASE_ID + 24 + i for i, p in enumerate(_PUNCTUATION)
}

# ── Possessive & Ordinal markers (V8) ───────────────────────────────────────

_MARKER_BASE: int = _STRUCTURAL_BASE_ID + 24 + len(_PUNCTUATION)  # after punctuation

POSSESSIVE_ID: int = _MARKER_BASE
ORDINAL_ID: int = _MARKER_BASE + 1

# ── Ordinal number tokens (V8) ──────────────────────────────────────────────

_ORDINAL_BASE: int = _MARKER_BASE + 2

# 1st-31st (days of month), plus 40th,50th,...,100th (decades/centuries)
_ORDINAL_LIST: list[str] = [
    "1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th",
    "11th", "12th", "13th", "14th", "15th", "16th", "17th", "18th", "19th", "20th",
    "21st", "22nd", "23rd", "24th", "25th", "26th", "27th", "28th", "29th", "30th",
    "31st",
    "40th", "50th", "60th", "70th", "80th", "90th", "100th",
]

ORDINAL_IDS: dict[str, int] = {
    o: _ORDINAL_BASE + i for i, o in enumerate(_ORDINAL_LIST)
}

# ── Academic / English abbreviations (V8) ───────────────────────────────────

_ABBREV_BASE: int = _ORDINAL_BASE + len(_ORDINAL_LIST)

_ABBREVIATION_LIST: list[str] = [
    # Forms as seen after word tokenization (trailing period stripped).
    # Internal periods preserved: "e.g." -> "e.g", "Ph.D." -> "ph.d"
    "pp", "cf", "ibid", "e.g", "i.e", "vs", "viz", "ff",
    "vol", "ed", "approx", "etc", "mr", "mrs", "ms", "dr", "jr",
    "sr", "st", "ave", "blvd",
    # Period-containing forms (internal periods preserved by tokenizer)
    "ph.d", "a.m", "p.m", "a.d", "b.c", "d.c",
]

ABBREVIATION_IDS: dict[str, int] = {
    a: _ABBREV_BASE + i for i, a in enumerate(_ABBREVIATION_LIST)
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

        # Flags (don't overwrite primitives on ID collision)
        prim_ids = {p.id for p in PRIMITIVES}
        for code, tid in TIER3_FLAGS.items():
            flag = _iso_to_flag(code)
            self._token_to_id[flag] = tid
            if tid not in prim_ids:
                self._id_to_token[tid] = flag
                self._id_to_description[tid] = f"Flag: {code}"

        # Anchors
        for name, tid in ANCHOR_TOKENS.items():
            self._token_to_id[name] = tid
            self._id_to_token[tid] = name
            self._id_to_description[tid] = f"Anchor: {name}"

        # Common word tokens (Tier 1b)
        for word, tid in COMMON_WORD_TOKENS.items():
            self._token_to_id[word] = tid
            self._id_to_token[tid] = word
            self._id_to_description[tid] = f"Common word: {word}"

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

        # Possessive & ordinal markers
        self._token_to_id["<POSSESSIVE>"] = POSSESSIVE_ID
        self._id_to_token[POSSESSIVE_ID] = "<POSSESSIVE>"
        self._id_to_description[POSSESSIVE_ID] = "Possessive marker"
        self._token_to_id["<ORDINAL>"] = ORDINAL_ID
        self._id_to_token[ORDINAL_ID] = "<ORDINAL>"
        self._id_to_description[ORDINAL_ID] = "Ordinal marker"

        # Ordinal tokens (1st-31st, 40th-100th by 10s)
        for ordinal, tid in ORDINAL_IDS.items():
            self._token_to_id[ordinal] = tid
            self._id_to_token[tid] = ordinal
            self._id_to_description[tid] = f"Ordinal: {ordinal}"

        # Academic/English abbreviations
        for abbrev, tid in ABBREVIATION_IDS.items():
            self._token_to_id[abbrev] = tid
            self._id_to_token[tid] = abbrev
            self._id_to_description[tid] = f"Abbreviation: {abbrev}"

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


    def get_anchor_id(self, name: str) -> int | None:
        """Get the token ID for a proper noun anchor."""
        return ANCHOR_TOKENS.get(name)
