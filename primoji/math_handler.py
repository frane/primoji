"""Math and code tokenization for Primoji.

Handles numbers (single-digit tokenization), math operators, LaTeX commands,
and source code detection. These are NEVER composed through emoji primitives —
they always use atomic structural tokens.
"""

from __future__ import annotations

import re

from primoji.vocabulary import DIGIT_IDS, MATH_OP_IDS


# ── Patterns ──────────────────────────────────────────────────────────────────

_NUMBER_RE = re.compile(r"\d+\.?\d*")
_LATEX_CMD_RE = re.compile(r"\\(?:frac|int|sum|prod|sqrt|lim|log|sin|cos|tan|infty|alpha|beta|gamma|delta|theta|pi|sigma)\b")
_MATH_EXPR_RE = re.compile(
    r"(?:"
    r"\d+\s*[+\-*/×÷=<>≤≥^]+\s*\d+"  # infix: 3 + 4
    r"|\\(?:frac|int|sum|sqrt)\b"       # LaTeX commands
    r"|\d+\s*[+\-*/]\s*\d+\s*="        # equations: 3 + 4 =
    r"|\$[^$]+\$"                        # inline LaTeX $...$
    r")"
)
_CODE_FENCE_RE = re.compile(r"```\w*\n.*?\n```", re.DOTALL)
_CODE_KEYWORD_RE = re.compile(
    r"\b(?:def|class|return|import|from|if|else|elif|for|while|try|except|"
    r"function|const|let|var|void|int|float|double|struct|enum)\b"
)

# Operator mapping: includes both Unicode and ASCII forms
_OP_MAP: dict[str, str] = {
    "+": "+",
    "-": "−",
    "−": "−",
    "*": "×",
    "×": "×",
    "/": "÷",
    "÷": "÷",
    "=": "=",
    "<": "<",
    ">": ">",
    "≤": "≤",
    "≥": "≥",
    "∫": "∫",
    "Σ": "Σ",
    "√": "√",
    "π": "π",
    "∞": "∞",
}


def tokenize_number(s: str) -> list[int]:
    """Tokenize a number string into single-digit token IDs.

    Each digit becomes its own token. Decimal points are preserved.
    This avoids BPE's left-to-right chunking problem where "1234" might
    become ["123"]["4"], which is catastrophic for arithmetic.

    Args:
        s: A string representing a number (e.g. "1234", "3.14").

    Returns:
        List of token IDs, one per digit. Decimal point gets punctuation ID.

    Examples:
        >>> tokenize_number("42")
        [1583, 1581]
        >>> tokenize_number("3.14")
        [1582, 1603, 1580, 1583]
    """
    from primoji.vocabulary import PUNCTUATION_IDS

    ids: list[int] = []
    for char in s:
        if char in DIGIT_IDS:
            ids.append(DIGIT_IDS[char])
        elif char == ".":
            dot_id = PUNCTUATION_IDS.get(".")
            if dot_id is not None:
                ids.append(dot_id)
        # Skip other chars (e.g. commas in "1,000")
    return ids


def tokenize_operator(s: str) -> int | None:
    """Map a math operator to its atomic token ID.

    Args:
        s: Operator string (e.g. '+', '-', '×', '√').

    Returns:
        Token ID for the operator, or None if not recognized.
    """
    canonical = _OP_MAP.get(s)
    if canonical is not None:
        return MATH_OP_IDS.get(canonical)
    return MATH_OP_IDS.get(s)


def is_math_expression(s: str) -> bool:
    """Detect whether a string contains math content.

    Checks for:
    - Infix arithmetic expressions (e.g. "3 + 4")
    - LaTeX commands (e.g. "\\frac{1}{2}")
    - Inline LaTeX ($...$)

    Args:
        s: Input string.

    Returns:
        True if the string contains math content.
    """
    return bool(_MATH_EXPR_RE.search(s))


def is_code_block(s: str) -> bool:
    """Detect whether a string contains source code.

    Checks for:
    - Fenced code blocks (```)
    - Programming language keywords (def, class, function, etc.)

    Args:
        s: Input string.

    Returns:
        True if the string appears to contain code.
    """
    if _CODE_FENCE_RE.search(s):
        return True
    # Heuristic: 2+ keywords in close proximity suggests code
    keywords = _CODE_KEYWORD_RE.findall(s)
    return len(keywords) >= 2


def is_latex_command(s: str) -> bool:
    """Check if a string is a LaTeX math command.

    Args:
        s: Input string (e.g. '\\frac', '\\int').

    Returns:
        True if this is a recognized LaTeX command.
    """
    return bool(_LATEX_CMD_RE.match(s))


def tokenize_math_segment(text: str) -> list[int]:
    """Tokenize a segment of text known to be mathematical.

    Processes the text character by character, mapping digits to single-digit
    tokens and operators to atomic operator tokens.

    Args:
        text: A string containing math content.

    Returns:
        List of token IDs.
    """
    from primoji.vocabulary import PUNCTUATION_IDS

    ids: list[int] = []
    i = 0
    while i < len(text):
        char = text[i]
        if char.isdigit():
            ids.append(DIGIT_IDS[char])
        elif char in _OP_MAP:
            op_id = tokenize_operator(char)
            if op_id is not None:
                ids.append(op_id)
        elif char == ".":
            dot_id = PUNCTUATION_IDS.get(".")
            if dot_id is not None:
                ids.append(dot_id)
        elif char in PUNCTUATION_IDS:
            ids.append(PUNCTUATION_IDS[char])
        # Skip whitespace and unrecognized chars
        i += 1
    return ids
