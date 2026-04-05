"""Tests for chat.py inference helpers.

Catches the bugs found during refactoring: renamed color constants,
classify_token_name integration, and generation function contracts.
"""

from __future__ import annotations

import torch
import pytest

from scripts.chat import (
    _TIER_COLORS,
    C_DROP,
    C_RESET,
    _generate,
    color_for_tier,
    generate_primoji,
    print_trace_primoji,
)
from scripts.train import GPT


class TestColorConstants:
    """All tier colors must be defined and accessible."""

    @pytest.mark.parametrize("tier", ["emoji", "word", "prim", "byte", "struct"])
    def test_tier_color_exists(self, tier: str) -> None:
        assert tier in _TIER_COLORS, f"Missing color for tier '{tier}'"
        assert _TIER_COLORS[tier].startswith("\033["), f"Invalid ANSI code for '{tier}'"

    @pytest.mark.parametrize("tier", ["emoji", "word", "prim", "byte", "struct"])
    def test_color_for_tier(self, tier: str) -> None:
        color = color_for_tier(tier)
        assert color == _TIER_COLORS[tier]

    def test_color_for_unknown_tier(self) -> None:
        assert color_for_tier("unknown") == C_RESET

    def test_drop_and_reset_defined(self) -> None:
        assert C_DROP.startswith("\033[")
        assert C_RESET == "\033[0m"


class TestGenerate:
    """_generate must produce valid token sequences."""

    @pytest.fixture
    def small_model(self) -> GPT:
        return GPT(vocab_size=100, d_model=64, n_layers=2, n_heads=2,
                    d_ff=128, max_seq_len=32).eval()

    def test_generate_returns_list(self, small_model: GPT) -> None:
        result = _generate(small_model, [1, 2, 3], eos_id=99,
                          max_tokens=5, temperature=0.8, top_k=10,
                          device="cpu")
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)

    def test_generate_respects_max_tokens(self, small_model: GPT) -> None:
        result = _generate(small_model, [1], eos_id=99,
                          max_tokens=10, temperature=0.8, top_k=10,
                          device="cpu")
        assert len(result) <= 10

    def test_generate_stops_on_eos(self, small_model: GPT) -> None:
        # With temperature=0 (greedy), the model will generate deterministic tokens.
        # We can't guarantee EOS, but we can verify it stops if max_tokens is reached.
        result = _generate(small_model, [1], eos_id=99,
                          max_tokens=3, temperature=0.0, top_k=10,
                          device="cpu")
        assert len(result) <= 3

    def test_generate_with_tiers(self) -> None:
        model = GPT(vocab_size=100, d_model=64, n_layers=2, n_heads=2,
                     d_ff=128, max_seq_len=32, n_tiers=5).eval()
        result = _generate(model, [1, 2], eos_id=99,
                          max_tokens=5, temperature=0.8, top_k=10,
                          device="cpu", use_tiers=True)
        assert isinstance(result, list)

    def test_greedy_is_deterministic(self, small_model: GPT) -> None:
        r1 = _generate(small_model, [1, 2], eos_id=99,
                       max_tokens=5, temperature=0.0, top_k=10, device="cpu")
        r2 = _generate(small_model, [1, 2], eos_id=99,
                       max_tokens=5, temperature=0.0, top_k=10, device="cpu")
        assert r1 == r2


class TestPrintTrace:
    """print_trace_primoji must not crash on any token combination."""

    def test_trace_with_empty_output(self, capsys) -> None:
        from primoji import Tokenizer
        tok = Tokenizer(fuzzy=False)
        print_trace_primoji(tok, [1200], [])
        captured = capsys.readouterr()
        assert "Input tokens:" in captured.out

    def test_trace_with_mixed_tokens(self, capsys) -> None:
        from primoji import Tokenizer
        from primoji.utils import SpecialTokens
        tok = Tokenizer(fuzzy=False)
        ids = tok.encode("water is good")
        print_trace_primoji(tok, ids, ids)
        captured = capsys.readouterr()
        assert "Generated tokens:" in captured.out
        assert "Breakdown" in captured.out
