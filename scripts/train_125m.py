"""Train a 125M GPT model on primoji or BPE tokens.

Minimal GPT-2 style transformer. Trains on pre-tokenized binary data
produced by prepare_training_data.py. Logs bits-per-byte at regular intervals.

Usage:
    # Prepare data first:
    python -m scripts.prepare_training_data --n-docs 50000

    # Train primoji model:
    python -m scripts.train_125m --tokenizer primoji

    # Train BPE model:
    python -m scripts.train_125m --tokenizer mistral

    # Or run both sequentially:
    python -m scripts.train_125m --tokenizer primoji && python -m scripts.train_125m --tokenizer mistral
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_DATA_DIR = Path(__file__).parent.parent / "data" / "experiment"


# ── Model ─────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, n_heads, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class CompositionalEmbedding(nn.Module):
    """Embedding where alias tokens compose from primitive embeddings.

    Normal tokens: standard learned embedding.
    Alias tokens (grammar words): mean of their primitive component embeddings.
    Primitive embeddings are learned, so aliases update through them.
    """

    def __init__(self, vocab_size: int, d_model: int,
                 alias_map: dict[int, list[int]] | None = None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self._alias_map = alias_map or {}

        if self._alias_map:
            # Pre-compute for efficient forward
            max_prims = max(len(p) for p in self._alias_map.values())
            tok_ids = []
            prim_matrix = []
            counts = []
            for tid, pids in self._alias_map.items():
                tok_ids.append(tid)
                padded = pids + [0] * (max_prims - len(pids))
                prim_matrix.append(padded)
                counts.append(len(pids))
            self.register_buffer("_alias_toks",
                torch.tensor(tok_ids, dtype=torch.long))
            self.register_buffer("_alias_prims",
                torch.tensor(prim_matrix, dtype=torch.long))
            self.register_buffer("_alias_counts",
                torch.tensor(counts, dtype=torch.float).unsqueeze(-1))

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        if not self._alias_map:
            return emb
        # Replace alias token embeddings with composed primitive embeddings
        for i in range(len(self._alias_toks)):
            tid = self._alias_toks[i].item()
            mask = (x == tid)
            if mask.any():
                n = int(self._alias_counts[i].item())
                pids = self._alias_prims[i, :n]
                composed = self.embedding.weight[pids].mean(dim=0)
                emb = emb.masked_scatter(mask.unsqueeze(-1).expand_as(emb),
                                          composed.expand(mask.sum(), -1))
        return emb


class GPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int,
                 n_heads: int, d_ff: int, max_seq_len: int,
                 n_tiers: int = 0, alias_map: dict[int, list[int]] | None = None):
        super().__init__()
        self.tok_emb = CompositionalEmbedding(vocab_size, d_model, alias_map)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.tier_emb = nn.Embedding(n_tiers, d_model) if n_tiers > 0 else None
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.max_seq_len = max_seq_len
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, tier_ids: torch.Tensor | None = None) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        if self.tier_emb is not None and tier_ids is not None:
            x = x + self.tier_emb(tier_ids)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Data loading ──────────────────────────────────────────────────────────────

class TokenDataset:
    """Memory-mapped token dataset for efficient loading."""

    def __init__(self, bin_path: str, seq_len: int, tier_path: str | None = None):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.tiers = np.memmap(tier_path, dtype=np.uint8, mode="r") if tier_path else None
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len - 1)

    def get_batch(self, batch_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Get a random batch of (tokens, tier_ids). tier_ids is None if no tier data."""
        ix = torch.randint(len(self), (batch_size,))
        x = torch.stack([
            torch.from_numpy(self.data[i:i + self.seq_len + 1].astype(np.int64))
            for i in ix
        ])
        t = None
        if self.tiers is not None:
            t = torch.stack([
                torch.from_numpy(self.tiers[i:i + self.seq_len + 1].astype(np.int64))
                for i in ix
            ])
            t = t.to(device)
        return x.to(device), t


# ── BPB computation ───────────────────────────────────────────────────────────

@torch.no_grad()
def compute_bpb(model: GPT, dataset: TokenDataset, total_bytes: int,
                total_val_tokens: int, batch_size: int, device: str,
                max_batches: int = 100) -> tuple[float, float]:
    """Compute bits-per-byte on validation data.

    BPB = avg_loss_nats * (total_val_tokens / total_bytes) / ln(2)

    This correctly scales from per-token loss to per-byte bits regardless
    of how many eval batches we sample.

    Returns:
        (bpb, avg_loss_nats)
    """
    model.eval()
    total_loss = 0.0
    n_tokens = 0

    for _ in range(max_batches):
        tokens, tiers = dataset.get_batch(batch_size, device)
        tier_input = tiers[:, :-1] if tiers is not None else None
        logits = model(tokens[:, :-1], tier_ids=tier_input)
        # BPB eval is always unweighted (no byte downweighting)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        n_tokens += tokens[:, 1:].numel()

    avg_loss = total_loss / n_tokens
    tokens_per_byte = total_val_tokens / total_bytes
    bpb = avg_loss * tokens_per_byte / math.log(2)
    model.train()
    return bpb, avg_loss


# ── Learning rate schedule ────────────────────────────────────────────────────

def get_lr(step: int, warmup: int, max_lr: float, min_lr: float, total_steps: int) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total_steps:
        return min_lr
    ratio = (step - warmup) / (total_steps - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ── Main training loop ───────────────────────────────────────────────────────

def train(tokenizer_name: str, data_dir: Path, device: str,
          batch_size: int, seq_len: int, max_steps: int | None,
          v2: bool = False, byte_weight: float = 1.0) -> None:
    """Train a 125M GPT model."""

    # Load metadata
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)

    use_tiers = v2 and tokenizer_name == "primoji"

    if tokenizer_name == "primoji":
        vocab_size = meta["primoji_vocab_size"]
        train_path = data_dir / "primoji_train.bin"
        val_path = data_dir / "primoji_val.bin"
        train_tier_path = data_dir / "primoji_tiers_train.bin" if use_tiers else None
        val_tier_path = data_dir / "primoji_tiers_val.bin" if use_tiers else None
    else:
        vocab_size = meta["mistral_vocab_size"]
        train_path = data_dir / "mistral_train.bin"
        val_path = data_dir / "mistral_val.bin"
        train_tier_path = None
        val_tier_path = None

    val_bytes_path = data_dir / "byte_counts_val.bin"
    val_byte_counts = np.fromfile(str(val_bytes_path), dtype=np.int64)
    total_val_bytes = int(val_byte_counts.sum())

    # Build alias map for compositional embeddings (v6)
    alias_map = None
    if v2 and tokenizer_name == "primoji":
        try:
            from primoji.alias_map import build_alias_map
            from primoji import Tokenizer as PT
            pt = PT(fuzzy=False)
            alias_map = build_alias_map(pt.encode)
            print(f"Alias map: {len(alias_map)} grammar words with compositional embeddings")
        except Exception as e:
            print(f"Warning: could not build alias map: {e}")

    # Model config: GPT-2 125M style
    config = {
        "vocab_size": vocab_size,
        "d_model": 768,
        "n_layers": 12,
        "n_heads": 12,
        "d_ff": 3072,
        "max_seq_len": seq_len,
        "n_tiers": 5 if use_tiers else 0,
        "alias_map": alias_map,
    }

    # Training config
    max_lr = 6e-4
    min_lr = 6e-5
    warmup_steps = 2000
    weight_decay = 0.1
    grad_clip = 1.0
    eval_interval = 200  # steps
    log_interval = 20

    # Dataset
    train_tier_str = str(train_tier_path) if train_tier_path and train_tier_path.exists() else None
    val_tier_str = str(val_tier_path) if val_tier_path and val_tier_path.exists() else None
    train_ds = TokenDataset(str(train_path), seq_len, tier_path=train_tier_str)
    val_ds = TokenDataset(str(val_path), seq_len, tier_path=val_tier_str)
    total_val_tokens = len(val_ds.data)

    n_train_tokens = len(train_ds.data)
    tokens_per_step = batch_size * seq_len
    if max_steps is None:
        # ~3 epochs over training data
        max_steps = (3 * n_train_tokens) // tokens_per_step
    total_steps = max_steps

    print(f"\n{'='*60}")
    print(f"Training: {tokenizer_name.upper()}")
    print(f"{'='*60}")
    print(f"Vocab size:      {vocab_size:,}")
    print(f"Train tokens:    {n_train_tokens:,}")
    print(f"Val bytes:       {total_val_bytes:,}")
    print(f"Batch size:      {batch_size}")
    print(f"Seq len:         {seq_len}")
    print(f"Steps:           {total_steps:,}")
    print(f"Tokens/step:     {tokens_per_step:,}")
    print(f"Device:          {device}")

    # Model
    model = GPT(**config).to(device)
    n_params = model.param_count()
    print(f"Parameters:      {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=max_lr, betas=(0.9, 0.95), fused=False)

    # Mixed precision
    use_amp = device != "cpu"
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    amp_dtype = torch.bfloat16 if device != "cpu" else torch.float32

    # Training log
    log: list[dict] = []
    out_dir = data_dir / f"{tokenizer_name}_checkpoints"
    out_dir.mkdir(exist_ok=True)

    # Initial eval
    bpb, val_loss = compute_bpb(model, val_ds, total_val_bytes, total_val_tokens, batch_size, device)
    print(f"Initial: val_loss={val_loss:.4f} bpb={bpb:.4f}")
    log.append({"step": 0, "tokens_seen": 0, "train_loss": None,
                "val_loss": val_loss, "val_bpb": bpb, "time": 0})

    # Train
    model.train()
    t0 = time.time()
    running_loss = 0.0

    for step in range(1, total_steps + 1):
        # LR schedule
        lr = get_lr(step, warmup_steps, max_lr, min_lr, total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Forward/backward
        tokens, tiers = train_ds.get_batch(batch_size, device)
        tier_input = tiers[:, :-1] if tiers is not None else None
        with torch.amp.autocast(device_type=device if device != "mps" else "cpu",
                                dtype=amp_dtype, enabled=use_amp):
            logits = model(tokens[:, :-1], tier_ids=tier_input)
            if use_tiers and byte_weight < 1.0 and tiers is not None:
                # Weighted loss: downweight byte tokens
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tokens[:, 1:].reshape(-1),
                    reduction="none",
                )
                weights = torch.ones_like(ce)
                byte_mask = (tiers[:, 1:].reshape(-1) == 4)
                weights[byte_mask] = byte_weight
                loss = (ce * weights).sum() / weights.sum()
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tokens[:, 1:].reshape(-1),
                )

        if device == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item()

        # Log
        if step % log_interval == 0:
            avg = running_loss / log_interval
            elapsed = time.time() - t0
            tokens_seen = step * tokens_per_step
            tok_per_sec = tokens_seen / elapsed
            print(f"  step={step:5d} loss={avg:.4f} lr={lr:.2e} "
                  f"tok/s={tok_per_sec:.0f} elapsed={elapsed:.0f}s", flush=True)
            running_loss = 0.0

        # Eval
        if step % eval_interval == 0:
            bpb, val_loss = compute_bpb(model, val_ds, total_val_bytes, total_val_tokens, batch_size, device)
            tokens_seen = step * tokens_per_step
            elapsed = time.time() - t0
            flops = 6 * n_params * tokens_seen
            entry = {
                "step": step,
                "tokens_seen": tokens_seen,
                "train_loss": loss.item(),
                "val_loss": val_loss,
                "val_bpb": bpb,
                "flops": flops,
                "time": elapsed,
                "lr": lr,
            }
            log.append(entry)
            print(f"  EVAL step={step} val_loss={val_loss:.4f} bpb={bpb:.4f} "
                  f"flops={flops:.2e}", flush=True)

            # Save log incrementally
            with open(data_dir / f"{tokenizer_name}_training_log.json", "w") as f:
                json.dump(log, f, indent=2)

    # Final eval
    bpb, val_loss = compute_bpb(model, val_ds, total_val_bytes, total_val_tokens, batch_size, device)
    elapsed = time.time() - t0
    tokens_seen = total_steps * tokens_per_step
    flops = 6 * n_params * tokens_seen
    log.append({"step": total_steps, "tokens_seen": tokens_seen,
                "train_loss": loss.item(), "val_loss": val_loss,
                "val_bpb": bpb, "flops": flops, "time": elapsed})

    print(f"\nFinal: val_loss={val_loss:.4f} bpb={bpb:.4f}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"Total tokens: {tokens_seen:,}")
    print(f"Total FLOPs: {flops:.2e}")

    # Save final log and model
    with open(data_dir / f"{tokenizer_name}_training_log.json", "w") as f:
        json.dump(log, f, indent=2)
    torch.save(model.state_dict(), out_dir / "model_final.pt")
    print(f"Saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 125M GPT")
    parser.add_argument("--tokenizer", choices=["primoji", "mistral"], required=True)
    parser.add_argument("--data-dir", type=str, default=str(_DATA_DIR))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--v2", action="store_true",
                        help="Enable tier embeddings + byte loss downweighting (primoji only)")
    parser.add_argument("--byte-weight", type=float, default=0.3,
                        help="Loss weight for byte-fallback tokens in v2 mode (default: 0.3)")
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    train(args.tokenizer, Path(args.data_dir), args.device,
          args.batch_size, args.seq_len, args.max_steps,
          v2=args.v2, byte_weight=args.byte_weight)


if __name__ == "__main__":
    main()
