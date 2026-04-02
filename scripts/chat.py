"""Interactive inference with a trained primoji model.

Usage:
    python -m scripts.chat
    python -m scripts.chat --model ../models/experiment_500k/primoji_model.pt
    python -m scripts.chat --temperature 0.5 --max-tokens 200
    python -m scripts.chat --trace          # show token tiers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.train_125m import GPT
from primoji import Tokenizer
from primoji.byte_fallback import BYTES_START_ID, BYTES_END_ID, BYTE_TOKEN_OFFSET, is_byte_token
from primoji.primitives import get_primitive_by_id
from primoji.utils import SpecialTokens, _IDS
from primoji.vocabulary import CONTRACTION_TOKENS, COMMON_WORD_TOKENS, ANCHOR_TOKENS

# Color codes
C_EMOJI = "\033[92m"      # green
C_WORD = "\033[97m"        # white
C_PRIM = "\033[96m"        # cyan
C_COMPOSE = "\033[95m"     # magenta
C_BYTE = "\033[91m"        # red
C_STRUCT = "\033[93m"      # yellow
C_DROP = "\033[90m"        # gray
C_RESET = "\033[0m"

_CONTRACTION_IDS = set(CONTRACTION_TOKENS.values())
_WORD_IDS = set(COMMON_WORD_TOKENS.values())
_ANCHOR_IDS = set(ANCHOR_TOKENS.values())
_WORD_START = _IDS["WORD_START"]
_WORD_END = _WORD_START + _IDS["WORD_COUNT"]


def classify_id(tid: int) -> str:
    """Classify a single token ID by tier."""
    if 0 <= tid <= 1199:
        return "emoji"
    if 1200 <= tid <= 1331:
        return "prim"
    if tid in _CONTRACTION_IDS:
        return "contraction"
    if tid in _ANCHOR_IDS:
        return "anchor"
    if _WORD_START <= tid < _WORD_END:
        return "word"
    if tid == BYTES_START_ID or tid == BYTES_END_ID:
        return "byte"
    if is_byte_token(tid):
        return "byte"
    if SpecialTokens.is_special(tid):
        return "special"
    return "struct"


def color_for_tier(tier: str) -> str:
    return {"emoji": C_EMOJI, "word": C_WORD, "prim": C_PRIM,
            "byte": C_BYTE, "struct": C_STRUCT, "anchor": C_STRUCT,
            "contraction": C_WORD, "special": C_DROP}.get(tier, C_RESET)


def generate(model: GPT, tok: Tokenizer, prompt: str,
             max_tokens: int, temperature: float, top_k: int,
             device: str, trace: bool, use_tiers: bool = False) -> tuple[str, list[int]]:
    ids = tok.encode(prompt)
    if not ids:
        ids = [SpecialTokens.BOS]
    generated = list(ids)

    with torch.no_grad():
        for _ in range(max_tokens):
            context = torch.tensor([generated[-1024:]], dtype=torch.long, device=device)
            tier_ctx = None
            if use_tiers:
                tier_map = {"emoji": 0, "word": 1, "prim": 2, "struct": 3, "byte": 4,
                            "anchor": 3, "contraction": 3, "special": 3}
                tier_ints = [tier_map.get(classify_id(t), 3) for t in generated[-1024:]]
                tier_ctx = torch.tensor([tier_ints], dtype=torch.long, device=device)
            logits = model(context, tier_ids=tier_ctx)[0, -1, :]

            if temperature > 0:
                logits = logits / temperature
                topk_v, topk_i = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(0, topk_i, topk_v)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            else:
                next_id = logits.argmax().item()

            if next_id == SpecialTokens.EOS:
                break
            generated.append(next_id)

    new_ids = generated[len(ids):]
    output = tok.decode(new_ids)
    return output, new_ids


def print_trace(tok: Tokenizer, input_ids: list[int], output_ids: list[int]) -> None:
    """Print colored token trace."""
    print()

    # Input trace
    print(f"  {C_DROP}Input tokens:{C_RESET}")
    print_token_sequence(tok, input_ids, prefix="    ")

    # Output trace
    print(f"  {C_DROP}Generated tokens:{C_RESET}")
    print_token_sequence(tok, output_ids, prefix="    ")

    # Stats
    from collections import Counter
    tiers = Counter(classify_id(tid) for tid in output_ids)
    total = len(output_ids)
    print(f"\n  {C_DROP}Token breakdown ({total} tokens):{C_RESET}")
    tier_labels = {"emoji": "Emoji", "word": "Word", "prim": "Primitive",
                   "byte": "Byte fallback", "struct": "Structural",
                   "contraction": "Contraction", "anchor": "Anchor"}
    for tier, color in [("word", C_WORD), ("prim", C_PRIM), ("emoji", C_EMOJI),
                        ("byte", C_BYTE), ("struct", C_STRUCT),
                        ("contraction", C_WORD), ("anchor", C_STRUCT)]:
        if tier in tiers:
            pct = 100 * tiers[tier] / total
            bar = "#" * int(pct / 2)
            print(f"    {color}{tier_labels[tier]:15s} {tiers[tier]:4d} ({pct:4.1f}%) {bar}{C_RESET}")
    print()


def print_token_sequence(tok: Tokenizer, ids: list[int], prefix: str = "") -> None:
    """Print token IDs with color-coded tiers."""
    parts = []
    i = 0
    while i < len(ids):
        tid = ids[i]
        tier = classify_id(tid)
        color = color_for_tier(tier)

        if tier == "byte" and tid == BYTES_START_ID:
            # Collect full byte sequence
            byte_ids = []
            i += 1
            while i < len(ids) and ids[i] != BYTES_END_ID:
                byte_ids.append(ids[i] - BYTE_TOKEN_OFFSET)
                i += 1
            i += 1  # skip END
            try:
                word = bytes(byte_ids).decode("utf-8", errors="replace")
            except Exception:
                word = "<?>"
            parts.append(f"{C_BYTE}[{word}]{C_RESET}")
        elif tier == "prim":
            p = get_primitive_by_id(tid)
            name = p.name if p else f"?{tid}"
            parts.append(f"{C_PRIM}{name}{C_RESET}")
        else:
            decoded = tok.decode([tid])
            parts.append(f"{color}{decoded}{C_RESET}")
        i += 1

    # Print wrapped
    line = prefix
    for part in parts:
        line += part + " "
        if len(line) > 100:
            print(line)
            line = prefix
    if line.strip():
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a trained primoji model")
    parser.add_argument("--model", type=str,
                        default=str(Path(__file__).parent.parent.parent / "models" / "experiment_500k" / "primoji_model.pt"))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trace", action="store_true", help="Show token tiers with colors")
    parser.add_argument("--tiers", action="store_true", help="Model has tier embeddings (v2/v3)")
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    tok = Tokenizer(fuzzy=False)

    n_tiers = 5 if args.tiers else 0
    print(f"Loading model from {args.model}...", flush=True)
    model = GPT(vocab_size=tok.vocab_size, d_model=768, n_layers=12,
                n_heads=12, d_ff=3072, max_seq_len=1024, n_tiers=n_tiers)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model = model.to(args.device).eval()

    mode = "trace mode" if args.trace else "normal mode"
    print(f"Ready ({mode}). Vocab: {tok.vocab_size}, Device: {args.device}")
    if args.trace:
        print(f"Colors: {C_EMOJI}emoji{C_RESET} {C_WORD}word{C_RESET} "
              f"{C_PRIM}primitive{C_RESET} {C_BYTE}byte-fallback{C_RESET} "
              f"{C_STRUCT}structural{C_RESET}")
    print(f"Type a prompt and press Enter. Empty line to quit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            break

        output, new_ids = generate(model, tok, prompt, args.max_tokens,
                                   args.temperature, args.top_k, args.device,
                                   args.trace, use_tiers=args.tiers)
        print(f"Model: {output}")

        if args.trace:
            input_ids = tok.encode(prompt)
            print_trace(tok, input_ids, new_ids)


if __name__ == "__main__":
    main()
