"""Evaluate inference quality on diverse prompts.

Generates text for N prompts, records tier breakdowns, saves results to JSON.
Counts semantic units (byte-fallback words count as 1, not 8-10 tokens).

Usage:
    python -m scripts.eval_inference --model path/to/model.pt --n 100
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

PROMPTS_100 = [
    # Science
    "What is photosynthesis?",
    "How does gravity work?",
    "Why is the sky blue?",
    "What causes earthquakes?",
    "What is evolution?",
    "How does electricity work?",
    "What causes rain?",
    "Why do leaves change color?",
    "What is a black hole?",
    "How do vaccines work?",
    "What is DNA?",
    "What is the water cycle?",
    "How do magnets work?",
    "What is an atom?",
    "How does the sun produce energy?",
    "What is a chemical reaction?",
    "Why does ice float?",
    "How do plants grow?",
    "What is the greenhouse effect?",
    "How does sound travel?",
    # Biology
    "How does the heart work?",
    "What is the immune system?",
    "How do cells divide?",
    "What is the nervous system?",
    "How do animals breathe?",
    "What is a food chain?",
    "How do fish breathe underwater?",
    "What is a ecosystem?",
    "How do birds fly?",
    "What is photosynthesis important for?",
    # History
    "What caused World War One?",
    "How did ancient Rome fall?",
    "What was the Industrial Revolution?",
    "Who built the pyramids?",
    "What caused the French Revolution?",
    "How did writing begin?",
    "What was the Renaissance?",
    "How did democracy start?",
    "What was the Cold War?",
    "Why did empires collapse?",
    # Society
    "What is democracy?",
    "How do governments work?",
    "What is economics?",
    "Why do countries trade?",
    "What is culture?",
    "How does law work?",
    "What is education for?",
    "Why do we pay taxes?",
    "What is religion?",
    "How does language develop?",
    # Technology
    "How do computers work?",
    "How does the internet work?",
    "What is artificial intelligence?",
    "How do phones work?",
    "What is a database?",
    "How does a car engine work?",
    "What is machine learning?",
    "How do airplanes fly?",
    "What is electricity?",
    "How does a camera work?",
    # Health
    "Why do we need sleep?",
    "What is nutrition?",
    "How does exercise help?",
    "What is mental health?",
    "How do diseases spread?",
    "What is stress?",
    "Why is water important for health?",
    "How does the brain work?",
    "What is a virus?",
    "How do antibiotics work?",
    # Environment
    "What is climate change?",
    "Why are forests important?",
    "What is pollution?",
    "How does recycling help?",
    "What is biodiversity?",
    "Why are oceans important?",
    "What causes wildfires?",
    "How does deforestation affect climate?",
    "What is renewable energy?",
    "Why do species go extinct?",
    # General knowledge
    "Why is water important?",
    "How does fire burn?",
    "What makes music?",
    "Why do we dream?",
    "What is time?",
    "How does memory work?",
    "What is mathematics?",
    "Why do people tell stories?",
    "What is art?",
    "How do we learn?",
    # Harder/abstract
    "What is consciousness?",
    "How does trade affect poverty?",
    "What makes a good leader?",
    "Why do civilizations rise and fall?",
    "What is the meaning of knowledge?",
    "How does competition drive innovation?",
    "What is the relationship between language and thought?",
    "Why is cooperation important for survival?",
    "What role does technology play in society?",
    "How does education change lives?",
]


def count_semantic_units(ids: list[int]) -> Counter:
    """Count tier distribution by semantic units, not raw tokens."""
    tiers = Counter()
    i = 0
    while i < len(ids):
        tid = ids[i]
        if is_byte_boundary(tid) and tid == BYTES_START_ID:
            tiers["byte"] += 1
            i += 1
            while i < len(ids) and ids[i] != BYTES_END_ID:
                i += 1
            i += 1  # skip BYTES_END
        else:
            tiers[classify_token_name(tid)] += 1
            i += 1
    return tiers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n", type=int, default=100, help="Number of prompts")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--rep-penalty", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tiers", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--1b", dest="one_b", action="store_true")
    parser.add_argument("--model-size", type=str, default=None,
                        choices=["125m", "1b", "primoji-125m", "primoji-1b", "primoji-wide-125m"])
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

    # Repeat prompts if n > len(PROMPTS_100) for larger sample sizes
    prompts = (PROMPTS_100 * ((args.n // len(PROMPTS_100)) + 1))[:args.n]
    results = []
    all_tiers = Counter()
    all_units = 0
    t0 = time.time()

    for i, prompt in enumerate(prompts, 1):
        ids = tok.encode(prompt)
        generated = list(ids)
        for _ in range(args.max_tokens):
            ctx = torch.tensor([generated[-1024:]], dtype=torch.long, device=args.device)
            tier_input = None
            if args.tiers:
                tier_ints = [classify_token(t) for t in generated[-1024:]]
                tier_input = torch.tensor([tier_ints], dtype=torch.long, device=args.device)
            with torch.no_grad():
                logits = model(ctx, tier_ids=tier_input)[0, -1, :].float().cpu()
                # Repetition penalty
                if args.rep_penalty > 1.0:
                    for prev_id in set(generated[-50:]):
                        logits[prev_id] /= args.rep_penalty
                logits = logits / args.temperature
                logits = torch.clamp(logits, -1e8, 1e8)
                logits[logits.isnan()] = -1e8
                # Top-p (nucleus) sampling
                if args.top_p is not None:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cum_probs - F.softmax(sorted_logits, dim=-1) >= args.top_p
                    sorted_logits[mask] = -1e8
                    probs = F.softmax(sorted_logits, dim=-1)
                    idx = torch.multinomial(probs, 1).item()
                    nid = sorted_idx[idx].item()
                else:
                    topk_v, topk_i = torch.topk(logits, min(args.top_k, logits.size(-1)))
                    probs = F.softmax(topk_v, dim=-1)
                    idx = torch.multinomial(probs, 1).item()
                    nid = topk_i[idx].item()
            if nid == SpecialTokens.EOS:
                break
            generated.append(nid)

        new_ids = generated[len(ids):]
        text = tok.decode(new_ids)
        tiers = count_semantic_units(new_ids)
        total = sum(tiers.values())
        all_tiers += tiers
        all_units += total

        results.append({
            "prompt": prompt,
            "output": text,
            "raw_tokens": len(new_ids),
            "semantic_units": total,
            "tiers": dict(tiers),
        })

        if i % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(prompts)} ({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0

    # Aggregate
    aggregate = {}
    for k in ["word", "prim", "emoji", "byte", "struct"]:
        v = all_tiers.get(k, 0)
        aggregate[k] = {"count": v, "pct": round(100 * v / all_units, 1)}

    report = {
        "model": args.model,
        "vocab_size": v,
        "n_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "total_semantic_units": all_units,
        "total_raw_tokens": sum(r["raw_tokens"] for r in results),
        "elapsed_seconds": round(elapsed, 1),
        "aggregate_tiers": aggregate,
        "results": results,
    }

    out_path = args.output or str(Path(args.model).parent / "inference_eval.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Inference eval: {len(prompts)} prompts, {all_units} semantic units, {elapsed:.0f}s")
    print(f"{'='*60}")
    for k in ["word", "prim", "emoji", "byte", "struct"]:
        v = all_tiers.get(k, 0)
        pct = 100 * v / all_units
        bar = "#" * int(pct / 2)
        print(f"  {k:10s}: {v:5d} ({pct:5.1f}%) {bar}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
