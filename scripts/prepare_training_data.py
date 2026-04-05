"""Prepare training data for the 125M primoji vs BPE experiment.

Streams FineWeb-Edu documents, tokenizes with both primoji and Mistral BPE,
writes tokens directly to disk in chunks to avoid OOM on large datasets.

Usage:
    python -m scripts.prepare_training_data --n-docs 500000
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np

_DATA_DIR = Path(__file__).parent.parent / "data"

CHUNK_SIZE = 5000  # docs per chunk before flushing to disk


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--n-docs", type=int, default=10_000)
    parser.add_argument("--min-len", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=str(_DATA_DIR / "experiment"))
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Stream and save docs to disk (not RAM) ────────────────────────────
    from datasets import load_dataset

    print(f"Streaming {args.n_docs} docs from FineWeb-Edu...", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                       split="train", streaming=True)

    docs_path = out / "docs.jsonl"
    n_collected = 0
    with open(docs_path, "w") as f:
        for i, example in enumerate(ds):
            if n_collected >= args.n_docs:
                break
            text = example["text"].strip()
            if len(text) >= args.min_len:
                f.write(json.dumps({"text": text}) + "\n")
                n_collected += 1
            if (i + 1) % 10000 == 0:
                print(f"  Scanned {i+1}, collected {n_collected} docs...", flush=True)

    print(f"Collected {n_collected} docs", flush=True)

    # ── Tokenize in streaming chunks, writing directly to disk ────────────
    from primoji import Tokenizer as PrimojiTokenizer
    from primoji.utils import SpecialTokens, classify_token
    from tokenizers import Tokenizer as HFTokenizer

    print("Loading tokenizers...", flush=True)
    primoji_tok = PrimojiTokenizer(fuzzy=False)
    mistral_tok = HFTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3")

    primoji_eos = SpecialTokens.EOS
    mistral_eos = mistral_tok.token_to_id("</s>")

    # Split: 90% train, 10% val
    split_idx = int(n_collected * 0.9)

    for split_name, doc_start, doc_end in [
        ("train", 0, split_idx),
        ("val", split_idx, n_collected),
    ]:
        n_split = doc_end - doc_start
        print(f"\nTokenizing {split_name} ({n_split} docs)...", flush=True)

        p_file = open(out / f"primoji_{split_name}.bin", "wb")
        pt_file = open(out / f"primoji_tiers_{split_name}.bin", "wb")
        m_file = open(out / f"mistral_{split_name}.bin", "wb")
        b_file = open(out / f"byte_counts_{split_name}.bin", "wb")

        p_total = 0
        m_total = 0
        total_bytes = 0
        t0 = time.time()
        doc_idx = 0

        with open(docs_path) as f:
            for line_idx, line in enumerate(f):
                if line_idx < doc_start:
                    continue
                if line_idx >= doc_end:
                    break

                doc = json.loads(line)["text"]
                doc_idx += 1

                # Primoji
                p_ids = primoji_tok.encode(doc)
                p_ids.append(primoji_eos)
                p_file.write(np.array(p_ids, dtype=np.uint16).tobytes())
                tier_ids = [classify_token(tid) for tid in p_ids]
                pt_file.write(np.array(tier_ids, dtype=np.uint8).tobytes())
                p_total += len(p_ids)

                # Mistral BPE
                m_ids = mistral_tok.encode(doc).ids
                m_ids.append(mistral_eos)
                m_file.write(np.array(m_ids, dtype=np.uint16).tobytes())
                m_total += len(m_ids)

                # Byte count
                n_bytes = len(doc.encode("utf-8"))
                b_file.write(struct.pack("<q", n_bytes))
                total_bytes += n_bytes

                if doc_idx % 1000 == 0:
                    elapsed = time.time() - t0
                    rate = doc_idx / elapsed
                    print(f"  {doc_idx}/{n_split} docs ({rate:.0f} docs/s)", flush=True)

        p_file.close()
        pt_file.close()
        m_file.close()
        b_file.close()

        ratio = p_total / m_total if m_total > 0 else 0
        print(f"  {split_name}: {n_split} docs, {total_bytes:,} bytes")
        print(f"  Primoji: {p_total:,} tokens ({ratio:.2f}x BPE)")
        print(f"  Mistral: {m_total:,} tokens")

    # Save metadata
    meta = {
        "n_docs": n_collected,
        "n_train": split_idx,
        "n_val": n_collected - split_idx,
        "primoji_vocab_size": primoji_tok.vocab_size,
        "mistral_vocab_size": 32768,
        "primoji_eos": primoji_eos,
        "mistral_eos": mistral_eos,
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {out / 'meta.json'}")
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
