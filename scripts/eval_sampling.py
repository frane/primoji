"""Test sampling settings on prompts prone to degeneration.

Runs each prompt with multiple temperature/top-p/repetition-penalty
settings and reports emoji %, loop detection, and output quality.

Usage:
    python -m scripts.eval_sampling --model path/to/model.pt --tiers --1b
    python -m scripts.eval_sampling --model path/to/model.pt --tiers  # 125M
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.train import GPT
from primoji import Tokenizer
from primoji.alias_map import build_alias_map
from primoji.utils import SpecialTokens, classify_token, classify_token_name
from primoji.byte_fallback import BYTES_START_ID, BYTES_END_ID, is_byte_boundary

PROMPTS = [
    "How does gravity work?",
    "Why is the sky blue?",
    "What causes rain?",
    "Why do leaves change color?",
    "How does electricity work?",
    "What is photosynthesis?",
    "What causes earthquakes?",
    "What is evolution?",
    "How does the internet work?",
    "What is climate change?",
]

SETTINGS = [
    ("A: temp=0.8 top_k=40", dict(temperature=0.8, top_k=40)),
    ("B: temp=0.9 top_p=0.95", dict(temperature=0.9, top_p=0.95)),
    ("C: temp=0.8 rep=1.2", dict(temperature=0.8, top_k=40, rep_penalty=1.2, rep_window=50)),
    ("D: temp=0.9 rep=1.3 top_p=0.9", dict(temperature=0.9, top_p=0.9, rep_penalty=1.3, rep_window=50)),
]


def generate(model, tok, prompt: str, max_tokens: int = 150,
             temperature: float = 0.8, top_k: int = 40,
             top_p: float | None = None, rep_penalty: float | None = None,
             rep_window: int = 50, device: str = "cpu",
             use_tiers: bool = False) -> tuple[str, float, list[int]]:
    """Generate with configurable sampling. Returns (text, emoji_pct, ids)."""
    ids = tok.encode(prompt)
    generated = list(ids)

    for _ in range(max_tokens):
        ctx = torch.tensor([generated[-1024:]], dtype=torch.long, device=device)
        tier_input = None
        if use_tiers:
            tier_ints = [classify_token(t) for t in generated[-1024:]]
            tier_input = torch.tensor([tier_ints], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(ctx, tier_ids=tier_input)[0, -1, :].float().cpu()

        # Repetition penalty
        if rep_penalty is not None and rep_penalty > 1.0:
            recent = set(generated[-rep_window:])
            for tid in recent:
                if logits[tid] > 0:
                    logits[tid] /= rep_penalty
                else:
                    logits[tid] *= rep_penalty

        logits = logits / temperature
        logits = torch.clamp(logits, -1e8, 1e8)
        logits[logits.isnan()] = -1e8

        if top_p is not None:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            mask = cumsum - probs > top_p
            sorted_logits[mask] = -1e8
            probs = F.softmax(sorted_logits, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            next_id = sorted_idx[idx].item()
        else:
            topk_v, topk_i = torch.topk(logits, min(top_k, logits.size(-1)))
            probs = F.softmax(topk_v, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            next_id = topk_i[idx].item()

        if next_id == SpecialTokens.EOS:
            break
        generated.append(next_id)

    new_ids = generated[len(ids):]
    emoji_count = sum(1 for t in new_ids if classify_token_name(t) == "emoji")
    total = len(new_ids) if new_ids else 1
    return tok.decode(new_ids), round(100 * emoji_count / total, 1), new_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Test sampling settings")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--tiers", action="store_true")
    parser.add_argument("--1b", dest="one_b", action="store_true")
    parser.add_argument("--model-size", type=str, default=None,
                        choices=["125m", "1b", "primoji-125m", "primoji-1b", "primoji-wide-125m"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    tok = Tokenizer(fuzzy=False)
    alias_map = build_alias_map(tok.encode)
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    v = state["tok_emb.embedding.weight"].shape[0]

    ARCH = {"125m": (768, 12, 12, 3072), "1b": (2048, 24, 16, 5461),
            "primoji-125m": (384, 50, 6, 1536), "primoji-1b": (1024, 73, 16, 4096),
            "primoji-wide-125m": (768, 12, 16, 3712)}
    if args.model_size:
        d_model, n_layers, n_heads, d_ff = ARCH[args.model_size]
    elif args.one_b:
        d_model, n_layers, n_heads, d_ff = ARCH["1b"]
    else:
        d_model, n_layers, n_heads, d_ff = ARCH["125m"]

    model = GPT(vocab_size=v, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                d_ff=d_ff, max_seq_len=1024, n_tiers=5 if args.tiers else 0,
                alias_map=alias_map)
    model.load_state_dict(state)
    model = model.to(args.device).eval()
    print(f"Loaded {args.model} (vocab={v}, device={args.device})", flush=True)

    all_results = []
    t0 = time.time()

    for pi, prompt in enumerate(PROMPTS):
        print(f"\n{'='*60}", flush=True)
        print(f"PROMPT {pi+1}: {prompt}", flush=True)

        for sname, sparams in SETTINGS:
            text, emoji_pct, _ = generate(model, tok, prompt,
                                           max_tokens=args.max_tokens,
                                           device=args.device,
                                           use_tiers=args.tiers,
                                           **sparams)
            loop = emoji_pct > 30
            print(f"  {sname}: emoji={emoji_pct}% {'LOOP' if loop else 'ok'}", flush=True)
            print(f"    {text[:200]}", flush=True)
            all_results.append({
                "prompt": prompt,
                "setting": sname,
                "emoji_pct": emoji_pct,
                "loop": loop,
                "output": text,
            })

    elapsed = time.time() - t0

    # Summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"{'Prompt':<35} {'A':>6} {'B':>6} {'C':>6} {'D':>6}")
    print("-" * 65)
    for prompt in PROMPTS:
        vals = [r for r in all_results if r["prompt"] == prompt]
        a = [r["emoji_pct"] for r in vals]
        print(f"{prompt[:35]:<35} {a[0]:>5.1f}% {a[1]:>5.1f}% {a[2]:>5.1f}% {a[3]:>5.1f}%")

    loops_per_setting = Counter()
    for r in all_results:
        if r["loop"]:
            loops_per_setting[r["setting"]] += 1
    print(f"\nLoops (>30% emoji) per setting:")
    for sname, _ in SETTINGS:
        print(f"  {sname}: {loops_per_setting.get(sname, 0)}/10")

    # Save
    out_path = args.output or str(Path(args.model).parent / "sampling_test.json")
    report = {
        "model": args.model,
        "elapsed_seconds": round(elapsed, 1),
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
