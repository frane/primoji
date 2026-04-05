"""Tests for the CompositionalEmbedding used in training.

Verifies that alias tokens get composed embeddings (mean of primitive
embeddings) while normal tokens get standard learned embeddings.
"""

from __future__ import annotations

import torch
import pytest

from scripts.train import CompositionalEmbedding, GPT


class TestCompositionalEmbedding:
    """CompositionalEmbedding must compose alias tokens from primitives."""

    @pytest.fixture
    def simple_alias_map(self) -> dict[int, list[int]]:
        """Token 5 = mean(token 1, token 2), token 6 = mean(token 3)."""
        return {5: [1, 2], 6: [3]}

    @pytest.fixture
    def emb(self, simple_alias_map: dict) -> CompositionalEmbedding:
        return CompositionalEmbedding(vocab_size=10, d_model=16, alias_map=simple_alias_map)

    def test_normal_token_gets_standard_embedding(self, emb: CompositionalEmbedding) -> None:
        x = torch.tensor([[0]])
        result = emb(x)
        expected = emb.embedding(x)
        assert torch.allclose(result, expected)

    def test_alias_token_gets_composed_embedding(self, emb: CompositionalEmbedding) -> None:
        """Token 5 should get mean(embedding[1], embedding[2])."""
        x = torch.tensor([[5]])
        result = emb(x)
        expected = emb.embedding.weight[[1, 2]].mean(dim=0)
        assert torch.allclose(result.squeeze(), expected, atol=1e-6)

    def test_single_primitive_alias(self, emb: CompositionalEmbedding) -> None:
        """Token 6 should get embedding[3] directly (mean of 1 = itself)."""
        x = torch.tensor([[6]])
        result = emb(x)
        expected = emb.embedding.weight[3]
        assert torch.allclose(result.squeeze(), expected, atol=1e-6)

    def test_mixed_batch(self, emb: CompositionalEmbedding) -> None:
        """Batch with both normal and alias tokens."""
        x = torch.tensor([[0, 5, 3, 6]])
        result = emb(x)
        assert result.shape == (1, 4, 16)
        # Token 0: standard embedding
        assert torch.allclose(result[0, 0], emb.embedding.weight[0], atol=1e-6)
        # Token 5: mean of 1 and 2
        assert torch.allclose(result[0, 1], emb.embedding.weight[[1, 2]].mean(0), atol=1e-6)
        # Token 3: standard embedding
        assert torch.allclose(result[0, 2], emb.embedding.weight[3], atol=1e-6)
        # Token 6: embedding of 3
        assert torch.allclose(result[0, 3], emb.embedding.weight[3], atol=1e-6)

    def test_no_alias_map_is_standard_embedding(self) -> None:
        emb = CompositionalEmbedding(vocab_size=10, d_model=16, alias_map=None)
        x = torch.tensor([[0, 1, 2]])
        result = emb(x)
        expected = emb.embedding(x)
        assert torch.allclose(result, expected)

    def test_weight_property_returns_embedding_weight(self, emb: CompositionalEmbedding) -> None:
        assert emb.weight is emb.embedding.weight

    def test_gradient_flows_through_alias(self, emb: CompositionalEmbedding) -> None:
        """Backprop through an alias token must update primitive embeddings."""
        x = torch.tensor([[5]])  # alias: mean(emb[1], emb[2])
        result = emb(x)
        loss = result.sum()
        loss.backward()
        # Primitives 1 and 2 should have gradients
        assert emb.embedding.weight.grad is not None
        assert emb.embedding.weight.grad[1].abs().sum() > 0
        assert emb.embedding.weight.grad[2].abs().sum() > 0
        # Token 0 (not used) should have zero gradient
        assert emb.embedding.weight.grad[0].abs().sum() == 0


class TestGPTWithAliasMap:
    """GPT model with CompositionalEmbedding must work end-to-end."""

    def test_forward_pass(self) -> None:
        alias_map = {5: [1, 2]}
        model = GPT(vocab_size=100, d_model=64, n_layers=2, n_heads=2,
                     d_ff=128, max_seq_len=32, alias_map=alias_map)
        x = torch.tensor([[1, 5, 3, 7]])
        logits = model(x)
        assert logits.shape == (1, 4, 100)

    def test_forward_with_tiers(self) -> None:
        model = GPT(vocab_size=100, d_model=64, n_layers=2, n_heads=2,
                     d_ff=128, max_seq_len=32, n_tiers=5)
        x = torch.tensor([[1, 5, 3, 7]])
        tiers = torch.tensor([[0, 1, 2, 3]])
        logits = model(x, tier_ids=tiers)
        assert logits.shape == (1, 4, 100)

    def test_weight_tying(self) -> None:
        model = GPT(vocab_size=100, d_model=64, n_layers=2, n_heads=2,
                     d_ff=128, max_seq_len=32)
        assert model.head.weight is model.tok_emb.weight
