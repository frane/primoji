"""Tests for Primoji math_handler module.

Covers number tokenization, operator mapping, math/code/LaTeX detection.
"""

from __future__ import annotations

import pytest

from primoji.math_handler import (
    is_code_block,
    is_latex_command,
    is_math_expression,
    tokenize_math_segment,
    tokenize_number,
    tokenize_operator,
)
from primoji.vocabulary import DIGIT_IDS, MATH_OP_IDS, PUNCTUATION_IDS


# ── tokenize_number ──────────────────────────────────────────────────────────


class TestTokenizeNumber:
    def test_single_digit(self) -> None:
        ids = tokenize_number("5")
        assert ids == [DIGIT_IDS["5"]]

    def test_two_digit_number(self) -> None:
        """'42' should produce two digit IDs in order: 4 then 2."""
        ids = tokenize_number("42")
        assert ids == [DIGIT_IDS["4"], DIGIT_IDS["2"]]

    def test_multi_digit_preserves_order(self) -> None:
        ids = tokenize_number("1234")
        expected = [DIGIT_IDS[d] for d in "1234"]
        assert ids == expected

    def test_zero(self) -> None:
        ids = tokenize_number("0")
        assert ids == [DIGIT_IDS["0"]]

    def test_decimal_number(self) -> None:
        """'3.14' should produce: 3, '.', 1, 4."""
        ids = tokenize_number("3.14")
        expected = [
            DIGIT_IDS["3"],
            PUNCTUATION_IDS["."],
            DIGIT_IDS["1"],
            DIGIT_IDS["4"],
        ]
        assert ids == expected

    def test_decimal_leading_zero(self) -> None:
        """'0.5' should produce: 0, '.', 5."""
        ids = tokenize_number("0.5")
        expected = [
            DIGIT_IDS["0"],
            PUNCTUATION_IDS["."],
            DIGIT_IDS["5"],
        ]
        assert ids == expected

    def test_all_digits_have_unique_ids(self) -> None:
        """Each digit 0-9 must map to a distinct token ID."""
        all_ids = [DIGIT_IDS[str(d)] for d in range(10)]
        assert len(set(all_ids)) == 10

    def test_empty_string(self) -> None:
        assert tokenize_number("") == []

    def test_number_with_comma_skips_comma(self) -> None:
        """Commas in '1,000' should be skipped."""
        ids = tokenize_number("1,000")
        expected = [DIGIT_IDS["1"], DIGIT_IDS["0"], DIGIT_IDS["0"], DIGIT_IDS["0"]]
        assert ids == expected


# ── tokenize_operator ────────────────────────────────────────────────────────


class TestTokenizeOperator:
    def test_plus(self) -> None:
        op_id = tokenize_operator("+")
        assert op_id is not None
        assert op_id == MATH_OP_IDS["+"]

    def test_minus_ascii(self) -> None:
        op_id = tokenize_operator("-")
        assert op_id is not None

    def test_minus_unicode(self) -> None:
        op_id = tokenize_operator("\u2212")  # −
        assert op_id is not None

    def test_ascii_and_unicode_minus_same_id(self) -> None:
        assert tokenize_operator("-") == tokenize_operator("\u2212")

    def test_multiply_ascii(self) -> None:
        op_id = tokenize_operator("*")
        assert op_id is not None

    def test_multiply_unicode(self) -> None:
        op_id = tokenize_operator("\u00d7")  # ×
        assert op_id is not None

    def test_divide_ascii(self) -> None:
        op_id = tokenize_operator("/")
        assert op_id is not None

    def test_equals(self) -> None:
        op_id = tokenize_operator("=")
        assert op_id is not None

    def test_less_than(self) -> None:
        assert tokenize_operator("<") is not None

    def test_greater_than(self) -> None:
        assert tokenize_operator(">") is not None

    def test_sqrt(self) -> None:
        assert tokenize_operator("\u221a") is not None  # √

    def test_pi(self) -> None:
        assert tokenize_operator("\u03c0") is not None  # π

    def test_infinity(self) -> None:
        assert tokenize_operator("\u221e") is not None  # ∞

    def test_unrecognized_returns_none(self) -> None:
        assert tokenize_operator("@") is None
        assert tokenize_operator("$") is None

    def test_all_operators_have_unique_ids(self) -> None:
        """All canonical operators should have distinct IDs."""
        canonical_ops = ["+", "\u2212", "\u00d7", "\u00f7", "=", "<", ">",
                         "\u2264", "\u2265", "\u222b", "\u03a3", "\u221a",
                         "\u03c0", "\u221e"]
        ids = [tokenize_operator(op) for op in canonical_ops]
        assert None not in ids
        assert len(set(ids)) == len(ids)


# ── is_math_expression ───────────────────────────────────────────────────────


class TestIsMathExpression:
    def test_simple_addition(self) -> None:
        assert is_math_expression("3 + 4") is True

    def test_subtraction(self) -> None:
        assert is_math_expression("10 - 5") is True

    def test_multiplication(self) -> None:
        assert is_math_expression("3 * 4") is True

    def test_equation(self) -> None:
        assert is_math_expression("2 + 3 = 5") is True

    def test_latex_frac(self) -> None:
        assert is_math_expression("\\frac{1}{2}") is True

    def test_latex_int(self) -> None:
        assert is_math_expression("\\int_0^1 x dx") is True

    def test_inline_latex(self) -> None:
        assert is_math_expression("The value is $x + y$") is True

    def test_plain_english_is_not_math(self) -> None:
        assert is_math_expression("The dog ran quickly") is False

    def test_single_number_is_not_math(self) -> None:
        assert is_math_expression("42") is False

    def test_empty_string(self) -> None:
        assert is_math_expression("") is False


# ── is_code_block ────────────────────────────────────────────────────────────


class TestIsCodeBlock:
    def test_fenced_code_block(self) -> None:
        code = "```python\ndef hello():\n    print('hi')\n```"
        assert is_code_block(code) is True

    def test_multiple_keywords(self) -> None:
        code = "def foo(): return 42"
        assert is_code_block(code) is True

    def test_single_keyword_is_not_code(self) -> None:
        """A single programming keyword in prose should not trigger code detection."""
        assert is_code_block("I will return home") is False

    def test_plain_text_is_not_code(self) -> None:
        assert is_code_block("The sun is bright today") is False

    def test_java_keywords(self) -> None:
        code = "void main() { int x = 5; }"
        assert is_code_block(code) is True

    def test_empty_string(self) -> None:
        assert is_code_block("") is False


# ── is_latex_command ─────────────────────────────────────────────────────────


class TestIsLatexCommand:
    def test_frac(self) -> None:
        assert is_latex_command("\\frac") is True

    def test_int(self) -> None:
        assert is_latex_command("\\int") is True

    def test_sum(self) -> None:
        assert is_latex_command("\\sum") is True

    def test_sqrt(self) -> None:
        assert is_latex_command("\\sqrt") is True

    def test_sin(self) -> None:
        assert is_latex_command("\\sin") is True

    def test_cos(self) -> None:
        assert is_latex_command("\\cos") is True

    def test_pi_latex(self) -> None:
        assert is_latex_command("\\pi") is True

    def test_infty(self) -> None:
        assert is_latex_command("\\infty") is True

    def test_not_a_latex_command(self) -> None:
        assert is_latex_command("\\notacommand") is False

    def test_plain_word(self) -> None:
        assert is_latex_command("hello") is False

    def test_empty_string(self) -> None:
        assert is_latex_command("") is False


# ── tokenize_math_segment ───────────────────────────────────────────────────


class TestTokenizeMathSegment:
    def test_simple_expression(self) -> None:
        """'3+4' should tokenize to digit(3), op(+), digit(4)."""
        ids = tokenize_math_segment("3+4")
        assert DIGIT_IDS["3"] in ids
        assert DIGIT_IDS["4"] in ids
        plus_id = MATH_OP_IDS["+"]
        assert plus_id in ids

    def test_whitespace_is_skipped(self) -> None:
        ids = tokenize_math_segment("3 + 4")
        # Same content as "3+4" — whitespace should be ignored
        assert len([i for i in ids if i in DIGIT_IDS.values()]) == 2

    def test_empty_input(self) -> None:
        assert tokenize_math_segment("") == []
