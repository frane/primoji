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

from scripts.train import GPT
from primoji.utils import classify_token, classify_token_name
from primoji.byte_fallback import BYTES_START_ID, BYTES_END_ID, BYTE_TOKEN_OFFSET, is_byte_token, is_byte_boundary
from primoji.primitives import get_primitive_by_id

# Color codes
_TIER_COLORS = {
    "emoji": "\033[92m",
    "word": "\033[97m",
    "prim": "\033[96m",
    "byte": "\033[91m",
    "struct": "\033[93m",
}
C_DROP = "\033[90m"
C_RESET = "\033[0m"


def color_for_tier(tier: str) -> str:
    return _TIER_COLORS.get(tier, C_RESET)


def _generate(model: GPT, initial_ids: list[int], eos_id: int,
              max_tokens: int, temperature: float, top_k: int,
              device: str, use_tiers: bool = False,
              top_p: float | None = 0.9,
              rep_penalty: float = 1.3, rep_window: int = 50) -> list[int]:
    """Generate tokens autoregressively. Shared by primoji and BPE paths."""
    generated = list(initial_ids)
    with torch.no_grad():
        for _ in range(max_tokens):
            context = torch.tensor([generated[-1024:]], dtype=torch.long, device=device)
            tier_ctx = None
            if use_tiers:
                tier_ints = [classify_token(t) for t in generated[-1024:]]
                tier_ctx = torch.tensor([tier_ints], dtype=torch.long, device=device)
            logits = model(context, tier_ids=tier_ctx)[0, -1, :].float().cpu()

            # Repetition penalty
            if rep_penalty > 1.0:
                recent = set(generated[-rep_window:])
                for tid in recent:
                    if logits[tid] > 0:
                        logits[tid] /= rep_penalty
                    else:
                        logits[tid] *= rep_penalty

            if temperature > 0:
                logits = logits / temperature
                logits = torch.clamp(logits, min=-1e8, max=1e8)
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
            else:
                next_id = logits.argmax().item()
            if next_id == eos_id:
                break
            generated.append(next_id)
    return generated[len(initial_ids):]


def _decode_with_emoji(tok, ids, show_emoji: bool = False) -> str:
    """Decode token IDs, optionally rendering emoji-tier tokens as glyphs."""
    if not show_emoji:
        return tok.decode(ids)
    # Render emoji tokens as their Unicode glyph, everything else normally
    parts = []
    i = 0
    while i < len(ids):
        tid = ids[i]
        tier = classify_token_name(tid)
        if tier == "emoji":
            # Get the emoji character from vocabulary
            emoji_char = tok.vocabulary._id_to_token.get(tid)
            if emoji_char:
                parts.append(emoji_char)
            else:
                parts.append(tok.decode([tid]))
        elif tier == "byte" and tid == BYTES_START_ID:
            # Collect byte sequence and decode as one word
            byte_ids = []
            i += 1
            while i < len(ids) and ids[i] != BYTES_END_ID:
                byte_ids.append(ids[i] - BYTE_TOKEN_OFFSET)
                i += 1
            try:
                parts.append(bytes(byte_ids).decode("utf-8", errors="replace"))
            except Exception:
                parts.append("<?>")
        else:
            decoded = tok.decode([tid])
            if decoded:
                parts.append(decoded)
        i += 1
    return " ".join(parts)


def generate_primoji(model, tok, prompt, max_tokens, temperature, top_k, device,
                     use_tiers, show_emoji=False, top_p=0.9, rep_penalty=1.3, rep_window=50):
    from primoji.utils import SpecialTokens
    ids = tok.encode(prompt)
    if not ids:
        ids = [SpecialTokens.BOS]
    new_ids = _generate(model, ids, SpecialTokens.EOS, max_tokens,
                        temperature, top_k, device, use_tiers,
                        top_p=top_p, rep_penalty=rep_penalty, rep_window=rep_window)
    return _decode_with_emoji(tok, new_ids, show_emoji), new_ids


def generate_bpe(model, tok, prompt, max_tokens, temperature, top_k, device,
                 top_p=0.9, rep_penalty=1.3, rep_window=50):
    ids = tok.encode(prompt).ids
    eos_id = tok.token_to_id("</s>")
    new_ids = _generate(model, ids, eos_id, max_tokens, temperature, top_k, device,
                        top_p=top_p, rep_penalty=rep_penalty, rep_window=rep_window)
    return tok.decode(new_ids), new_ids


def print_trace_primoji(tok, input_ids, output_ids):
    from collections import Counter
    from primoji.byte_fallback import is_byte_token, is_byte_boundary
    print()
    print(f"  {C_DROP}Input tokens:{C_RESET}")
    _print_tokens(tok, input_ids, "    ")
    print(f"  {C_DROP}Generated tokens:{C_RESET}")
    _print_tokens(tok, output_ids, "    ")

    # Count semantic units (words), not raw tokens
    # A byte-fallback word = 1 word, not 8-10 tokens
    word_tiers = Counter()
    i = 0
    while i < len(output_ids):
        tid = output_ids[i]
        if is_byte_boundary(tid) and tid == BYTES_START_ID:
            word_tiers["byte"] += 1
            i += 1
            while i < len(output_ids) and output_ids[i] != BYTES_END_ID:
                i += 1
            i += 1  # skip BYTES_END
        else:
            word_tiers[classify_token_name(tid)] += 1
            i += 1

    total_words = sum(word_tiers.values())
    total_tokens = len(output_ids)
    print(f"\n  {C_DROP}Breakdown ({total_words} semantic units, {total_tokens} raw tokens):{C_RESET}")
    labels = {"word": "Word", "prim": "Primitive", "emoji": "Emoji",
              "byte": "Byte fallback", "struct": "Structural"}
    for tier, color in [("word", _TIER_COLORS["word"]), ("prim", _TIER_COLORS["prim"]),
                        ("emoji", _TIER_COLORS["emoji"]), ("byte", _TIER_COLORS["byte"]),
                        ("struct", _TIER_COLORS["struct"])]:
        if tier in word_tiers:
            pct = 100 * word_tiers[tier] / total_words
            bar = "#" * int(pct / 2)
            print(f"    {color}{labels[tier]:15s} {word_tiers[tier]:4d} ({pct:4.1f}%) {bar}{C_RESET}")
    print()


def _print_tokens(tok, ids, prefix):
    parts = []
    i = 0
    while i < len(ids):
        tid = ids[i]
        tier = classify_token_name(tid)
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
            parts.append(f"{_TIER_COLORS['byte']}[{word}]{C_RESET}")
        elif tier == "prim":
            p = get_primitive_by_id(tid)
            name = p.name if p else f"?{tid}"
            parts.append(f"{_TIER_COLORS['prim']}{name}{C_RESET}")
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
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold (default: 0.9)")
    parser.add_argument("--rep-penalty", type=float, default=1.3, help="Repetition penalty (default: 1.3)")
    parser.add_argument("--rep-window", type=int, default=50, help="Repetition penalty window (default: 50)")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--trace", action="store_true", help="Show token tiers with colors (primoji only)")
    parser.add_argument("--tiers", action="store_true", help="Model has tier embeddings (v2+)")
    parser.add_argument("--bpe", action="store_true", help="Use Mistral BPE tokenizer instead of primoji")
    parser.add_argument("--1b", dest="one_b", action="store_true", help="Use 1B model config instead of 125M")
    parser.add_argument("--model-size", type=str, default=None,
                        choices=["125m", "1b", "primoji-125m", "primoji-1b", "primoji-wide-125m"],
                        help="Model architecture (overrides --1b)")
    parser.add_argument("--emoji", action="store_true", help="Render emoji-tier tokens as Unicode glyphs")
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    # Model config
    ARCH = {
        "125m":              (768, 12, 12, 3072),
        "1b":                (2048, 24, 16, 5461),
        "primoji-125m":      (384, 50, 6, 1536),
        "primoji-1b":        (1024, 73, 16, 4096),
        "primoji-wide-125m": (768, 12, 16, 3712),
    }
    if args.model_size:
        d_model, n_layers, n_heads, d_ff = ARCH[args.model_size]
    elif args.one_b:
        d_model, n_layers, n_heads, d_ff = ARCH["1b"]
    else:
        d_model, n_layers, n_heads, d_ff = ARCH["125m"]

    alias_map = None
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
        # Build alias map for v6 compositional embeddings
        try:
            from primoji.alias_map import build_alias_map
            alias_map = build_alias_map(tok.encode)
        except Exception:
            alias_map = None

    print(f"Loading model from {args.model}...", flush=True)
    # Infer vocab size from checkpoint to handle version mismatches
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    ckpt_vocab = state["tok_emb.embedding.weight"].shape[0]
    if ckpt_vocab != vocab_size:
        print(f"  Note: checkpoint vocab={ckpt_vocab}, current tokenizer vocab={vocab_size}. Using checkpoint vocab.")
        vocab_size = ckpt_vocab
    model = GPT(vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
                n_heads=n_heads, d_ff=d_ff, max_seq_len=1024, n_tiers=n_tiers,
                alias_map=alias_map)
    model.load_state_dict(state)
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
                                            args.temperature, args.top_k, args.device,
                                            top_p=args.top_p, rep_penalty=args.rep_penalty,
                                            rep_window=args.rep_window)
        else:
            output, new_ids = generate_primoji(model, tok, prompt, args.max_tokens,
                                               args.temperature, args.top_k, args.device,
                                               use_tiers=args.tiers,
                                               show_emoji=args.emoji,
                                               top_p=args.top_p, rep_penalty=args.rep_penalty,
                                               rep_window=args.rep_window)

        print(f"Model: {output}")

        if args.trace and not args.bpe:
            input_ids = tok.encode(prompt)
            print_trace_primoji(tok, input_ids, new_ids)


if __name__ == "__main__":
    main()
