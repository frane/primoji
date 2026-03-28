"""Prepare training data for the 125M primoji vs BPE experiment.

Streams FineWeb-Edu documents, tokenizes with both primoji and Mistral BPE,
saves as binary files for fast loading during training.

Usage:
    python -m scripts.prepare_training_data --n-docs 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--n-docs", type=int, default=10_000)
    parser.add_argument("--min-len", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=str(_DATA_DIR / "experiment"))
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Stream documents ──────────────────────────────────────────────────
    from datasets import load_dataset

    print(f"Streaming {args.n_docs} docs from FineWeb-Edu...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                       split="train", streaming=True)

    docs: list[str] = []
    for i, example in enumerate(ds):
        if len(docs) >= args.n_docs:
            break
        text = example["text"].strip()
        if len(text) >= args.min_len:
            docs.append(text)
        if (i + 1) % 2000 == 0:
            print(f"  Scanned {i+1}, collected {len(docs)} docs...", flush=True)

    print(f"Collected {len(docs)} docs")

    # Save raw docs
    with open(out / "docs.jsonl", "w") as f:
        for doc in docs:
            f.write(json.dumps({"text": doc}) + "\n")

    # ── Tokenize with both ────────────────────────────────────────────────
    from primoji import Tokenizer as PrimojiTokenizer
    from primoji.utils import SpecialTokens
    from tokenizers import Tokenizer as HFTokenizer

    print("Loading tokenizers...", flush=True)
    primoji_tok = PrimojiTokenizer(fuzzy=False)  # no fuzzy for speed
    mistral_tok = HFTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

    primoji_eos = SpecialTokens.EOS
    mistral_eos = mistral_tok.token_to_id("</s>")

    # Split: 90% train, 10% val
    split_idx = int(len(docs) * 0.9)
    train_docs = docs[:split_idx]
    val_docs = docs[split_idx:]

    for split_name, split_docs in [("train", train_docs), ("val", val_docs)]:
        print(f"\nTokenizing {split_name} ({len(split_docs)} docs)...", flush=True)

        primoji_all: list[int] = []
        mistral_all: list[int] = []
        byte_counts: list[int] = []

        t0 = time.time()
        for i, doc in enumerate(split_docs):
            # Primoji
            p_ids = primoji_tok.encode(doc)
            primoji_all.extend(p_ids)
            primoji_all.append(primoji_eos)

            # Mistral BPE
            m_ids = mistral_tok.encode(doc).ids
            mistral_all.extend(m_ids)
            mistral_all.append(mistral_eos)

            # Byte count for BPB calculation
            byte_counts.append(len(doc.encode("utf-8")))

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  {i+1}/{len(split_docs)} docs ({rate:.0f} docs/s)", flush=True)

        # Save as binary
        primoji_arr = np.array(primoji_all, dtype=np.uint16)
        mistral_arr = np.array(mistral_all, dtype=np.uint16)
        bytes_arr = np.array(byte_counts, dtype=np.int64)

        primoji_arr.tofile(out / f"primoji_{split_name}.bin")
        mistral_arr.tofile(out / f"mistral_{split_name}.bin")
        bytes_arr.tofile(out / f"byte_counts_{split_name}.bin")

        total_bytes = sum(byte_counts)
        print(f"  {split_name}: {len(split_docs)} docs, {total_bytes:,} bytes")
        print(f"  Primoji: {len(primoji_all):,} tokens ({len(primoji_all)/len(mistral_all):.2f}x BPE)")
        print(f"  Mistral: {len(mistral_all):,} tokens")

    # Save metadata
    meta = {
        "n_docs": len(docs),
        "n_train": len(train_docs),
        "n_val": len(val_docs),
        "primoji_vocab_size": primoji_tok.vocab_size,
        "mistral_vocab_size": 32768,
        "primoji_eos": primoji_eos,
        "mistral_eos": mistral_eos,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {out / 'meta.json'}")
    print("Done!")


if __name__ == "__main__":
    main()
