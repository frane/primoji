"""Interactive inference with a trained primoji or BPE model.

Usage:
    python -m scripts.chat                                             # V1 primoji
    python -m scripts.chat --model path/to/v5.pt --tiers               # V5 with tier embeddings
    python -m scripts.chat --model path/to/bpe.pt --bpe                # BPE model
    python -m scripts.chat --model path/to/1b.pt --tiers --1b          # 1B primoji
    python -m scripts.chat --model path/to/1b_bpe.pt --bpe --1b        # 1B BPE
    python -m scripts.chat --trace                                     # show token tiers
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.train_125m import GPT
from primoji.utils import _IDS
from primoji.byte_fallback import BYTES_START_ID, BYTES_END_ID, BYTE_TOKEN_OFFSET, is_byte_token, is_byte_boundary
from primoji.primitives import get_primitive_by_id

# Color codes
C_EMOJI = "\033[92m"
C_WORD = "\033[97m"
C_PRIM = "\033[96m"
C_BYTE = "\033[91m"
C_STRUCT = "\033[93m"
C_DROP = "\033[90m"
C_RESET = "\033[0m"

_WORD_START = _IDS["WORD_START"]
_WORD_END = _WORD_START + _IDS["WORD_COUNT"]


def classify_id(tid: int) -> str:
    if 0 <= tid <= 1199: return "emoji"
    if 1200 <= tid <= 1339: return "prim"
    if is_byte_token(tid) or is_byte_boundary(tid): return "byte"
    if _WORD_START <= tid < _WORD_END: return "word"
    return "struct"


def color_for_tier(tier: str) -> str:
    return {"emoji": C_EMOJI, "word": C_WORD, "prim": C_PRIM,
            "byte": C_BYTE, "struct": C_STRUCT}.get(tier, C_RESET)


def generate_primoji(model, tok, prompt, max_tokens, temperature, top_k, device, use_tiers):
    from primoji.utils import SpecialTokens
    ids = tok.encode(prompt)
    if not ids:
        ids = [SpecialTokens.BOS]
    generated = list(ids)

    tier_map = {"emoji": 0, "word": 1, "prim": 2, "struct": 3, "byte": 4}

    with torch.no_grad():
        for _ in range(max_tokens):
            context = torch.tensor([generated[-1024:]], dtype=torch.long, device=device)
            tier_ctx = None
            if use_tiers:
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
    return tok.decode(new_ids), new_ids


def generate_bpe(model, tok, prompt, max_tokens, temperature, top_k, device):
    enc = tok.encode(prompt)
    ids = enc.ids
    eos_id = tok.token_to_id("</s>")
    generated = list(ids)

    with torch.no_grad():
        for _ in range(max_tokens):
            context = torch.tensor([generated[-1024:]], dtype=torch.long, device=device)
            logits = model(context)[0, -1, :]
            if temperature > 0:
                logits = logits / temperature
                topk_v, topk_i = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(0, topk_i, topk_v)
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
            else:
                next_id = logits.argmax().item()
            if next_id == eos_id:
                break
            generated.append(next_id)

    new_ids = generated[len(ids):]
    return tok.decode(new_ids), new_ids


def print_trace_primoji(tok, input_ids, output_ids):
    from collections import Counter
    print()
    print(f"  {C_DROP}Input tokens:{C_RESET}")
    _print_tokens(tok, input_ids, "    ")
    print(f"  {C_DROP}Generated tokens:{C_RESET}")
    _print_tokens(tok, output_ids, "    ")

    tiers = Counter(classify_id(tid) for tid in output_ids)
    total = len(output_ids)
    print(f"\n  {C_DROP}Token breakdown ({total} tokens):{C_RESET}")
    labels = {"word": "Word", "prim": "Primitive", "emoji": "Emoji",
              "byte": "Byte fallback", "struct": "Structural"}
    for tier, color in [("word", C_WORD), ("prim", C_PRIM), ("emoji", C_EMOJI),
                        ("byte", C_BYTE), ("struct", C_STRUCT)]:
        if tier in tiers:
            pct = 100 * tiers[tier] / total
            bar = "#" * int(pct / 2)
            print(f"    {color}{labels[tier]:15s} {tiers[tier]:4d} ({pct:4.1f}%) {bar}{C_RESET}")
    print()


def _print_tokens(tok, ids, prefix):
    parts = []
    i = 0
    while i < len(ids):
        tid = ids[i]
        tier = classify_id(tid)
        color = color_for_tier(tier)
        if tier == "byte" and tid == BYTES_START_ID:
            byte_ids = []
            i += 1
            while i < len(ids) and ids[i] != BYTES_END_ID:
                byte_ids.append(ids[i] - BYTE_TOKEN_OFFSET)
                i += 1
            i += 1
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
    line = prefix
    for part in parts:
        line += part + " "
        if len(line) > 100:
            print(line)
            line = prefix
    if line.strip():
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a trained model")
    parser.add_argument("--model", type=str,
                        default=str(Path(__file__).parent.parent.parent / "models" / "experiment_500k" / "primoji_model.pt"))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trace", action="store_true", help="Show token tiers with colors (primoji only)")
    parser.add_argument("--tiers", action="store_true", help="Model has tier embeddings (v2+)")
    parser.add_argument("--bpe", action="store_true", help="Use Mistral BPE tokenizer instead of primoji")
    parser.add_argument("--1b", dest="one_b", action="store_true", help="Use 1B model config instead of 125M")
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    # Model config
    if args.one_b:
        d_model, n_layers, n_heads, d_ff = 2048, 24, 16, 5460
    else:
        d_model, n_layers, n_heads, d_ff = 768, 12, 12, 3072

    if args.bpe:
        from tokenizers import Tokenizer as HFTokenizer
        tok = HFTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")
        vocab_size = 32768
        n_tiers = 0
    else:
        from primoji import Tokenizer as PrimojiTokenizer
        tok = PrimojiTokenizer(fuzzy=False)
        vocab_size = tok.vocab_size
        n_tiers = 5 if args.tiers else 0

    print(f"Loading model from {args.model}...", flush=True)
    model = GPT(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                n_heads=n_heads, d_ff=d_ff, max_seq_len=1024, n_tiers=n_tiers)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model = model.to(args.device).eval()

    mode_str = "BPE" if args.bpe else "primoji"
    size_str = "1B" if args.one_b else "125M"
    print(f"Ready. {mode_str} {size_str}, Vocab: {vocab_size}, Device: {args.device}")
    print(f"Type a prompt and press Enter. Empty line to quit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            break

        if args.bpe:
            output, new_ids = generate_bpe(model, tok, prompt, args.max_tokens,
                                            args.temperature, args.top_k, args.device)
        else:
            output, new_ids = generate_primoji(model, tok, prompt, args.max_tokens,
                                               args.temperature, args.top_k, args.device,
                                               use_tiers=args.tiers)

        print(f"Model: {output}")

        if args.trace and not args.bpe:
            input_ids = tok.encode(prompt)
            print_trace_primoji(tok, input_ids, new_ids)


if __name__ == "__main__":
    main()
