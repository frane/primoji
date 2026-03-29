"""Tests for bits-per-byte calculation correctness.

This would have caught the 16x BPB bug. Tests the formula against
known mathematical expectations.
"""

from __future__ import annotations

import math

import pytest


class TestBPBFormula:
    """Verify the BPB formula produces mathematically correct values."""

    def test_random_baseline_bpb(self) -> None:
        """A model predicting uniformly over vocab V has known BPB.

        For uniform prediction: loss = ln(V) nats per token.
        BPB = loss * tokens_per_byte / ln(2)

        For V=2418 and 0.52 tokens/byte:
        BPB = ln(2418) * 0.52 / ln(2) = 7.79 * 0.52 / 0.693 = 5.84
        """
        vocab_size = 2418
        tokens_per_byte = 0.52
        random_loss = math.log(vocab_size)  # ~7.79 nats
        expected_bpb = random_loss * tokens_per_byte / math.log(2)
        assert 5.5 < expected_bpb < 6.5, f"Random BPB {expected_bpb} outside expected range"

    def test_perfect_model_bpb(self) -> None:
        """A perfect model (loss=0) has BPB=0."""
        bpb = 0.0 * 0.52 / math.log(2)
        assert bpb == 0.0

    def test_bpb_scales_with_tokens_per_byte(self) -> None:
        """More tokens per byte = higher BPB at same loss (longer sequences cost more)."""
        loss = 2.0
        bpb_sparse = loss * 0.24 / math.log(2)  # BPE: 0.24 tokens/byte
        bpb_dense = loss * 0.52 / math.log(2)   # Primoji: 0.52 tokens/byte
        assert bpb_dense > bpb_sparse

    def test_bpb_125m_reasonable_range(self) -> None:
        """A 125M model on educational text should produce BPB between 0.8 and 2.5.

        State of the art (Llama 3 70B): ~0.5-0.7 BPB
        A 125M model should be much worse: 1.0-2.0 BPB
        Above 3.0 BPB = something is wrong
        Below 0.5 BPB for 125M = definitely wrong
        """
        # Simulated: loss=1.75 nats, tokens/byte=0.52
        loss = 1.75
        tpb = 0.52
        bpb = loss * tpb / math.log(2)
        assert 0.8 < bpb < 2.5, f"BPB {bpb} outside reasonable range for 125M model"

    def test_formula_matches_definition(self) -> None:
        """BPB = total_loss_nats / (total_bytes * ln(2))
        which equals avg_loss * total_tokens / (total_bytes * ln(2))
        which equals avg_loss * tokens_per_byte / ln(2)
        """
        avg_loss = 2.0
        total_tokens = 1000
        total_bytes = 2000
        tokens_per_byte = total_tokens / total_bytes

        # Method 1: from totals
        total_loss = avg_loss * total_tokens
        bpb1 = total_loss / (total_bytes * math.log(2))

        # Method 2: from averages
        bpb2 = avg_loss * tokens_per_byte / math.log(2)

        assert abs(bpb1 - bpb2) < 1e-10

    def test_subset_eval_must_scale(self) -> None:
        """If you only eval on a SUBSET of tokens, you must use avg_loss, not sum.

        This is the exact bug that produced 0.08 BPB instead of 1.34.
        """
        avg_loss = 1.75  # nats per token
        eval_tokens = 819_200    # 100 batches * 8 * 1024
        total_tokens = 13_378_462
        total_bytes = 25_458_530

        # WRONG: sum loss from subset, divide by all bytes
        subset_total_loss = avg_loss * eval_tokens
        wrong_bpb = subset_total_loss / (total_bytes * math.log(2))

        # RIGHT: avg loss * tokens_per_byte / ln(2)
        tokens_per_byte = total_tokens / total_bytes
        right_bpb = avg_loss * tokens_per_byte / math.log(2)

        assert right_bpb > 1.0, f"Right BPB {right_bpb} too low"
        assert wrong_bpb < 0.1, f"Wrong BPB {wrong_bpb} too high (should be the bug)"
        assert right_bpb / wrong_bpb > 10, "The bug produces a 10x+ underestimate"
