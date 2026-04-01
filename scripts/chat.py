"""Interactive inference with a trained primoji model.

Usage:
    python -m scripts.chat
    python -m scripts.chat --model ../models/experiment_500k/primoji_model.pt
    python -m scripts.chat --temperature 0.5 --max-tokens 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.train_125m import GPT
from primoji import Tokenizer
from primoji.utils import SpecialTokens


def generate(model: GPT, tok: Tokenizer, prompt: str,
             max_tokens: int, temperature: float, top_k: int,
             device: str) -> str:
    ids = tok.encode(prompt)
    if not ids:
        ids = [SpecialTokens.BOS]
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

            if next_id == SpecialTokens.EOS:
                break
            generated.append(next_id)

    return tok.decode(generated[len(ids):])


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a trained primoji model")
    parser.add_argument("--model", type=str,
                        default=str(Path(__file__).parent.parent.parent / "models" / "experiment_500k" / "primoji_model.pt"))
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"

    tok = Tokenizer(fuzzy=False)

    print(f"Loading model from {args.model}...", flush=True)
    model = GPT(vocab_size=tok.vocab_size, d_model=768, n_layers=12,
                n_heads=12, d_ff=3072, max_seq_len=1024)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model = model.to(args.device).eval()
    print(f"Ready. Vocab: {tok.vocab_size}, Device: {args.device}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print(f"Type a prompt and press Enter. Empty line to quit.\n")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            break

        output = generate(model, tok, prompt, args.max_tokens,
                         args.temperature, args.top_k, args.device)
        print(f"Model: {output}\n")


if __name__ == "__main__":
    main()
