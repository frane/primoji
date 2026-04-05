"""Compare primoji vs BPE training results and generate plots.

Reads training logs from both runs, generates comparison plots and report.

Usage:
    python -m scripts.compare_results
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DATA_DIR = Path(__file__).parent.parent / "data" / "experiment"
_PLOT_DIR = _DATA_DIR / "plots"


def load_log(name: str) -> list[dict]:
    path = _DATA_DIR / f"{name}_training_log.json"
    if not path.exists():
        print(f"WARNING: {path} not found")
        return []
    with open(path) as f:
        return json.load(f)


def recompute_bpb(log: list[dict], bin_path: str, bytes_path: str) -> list[dict]:
    """Recompute BPB from val_loss using correct formula.

    BPB = avg_loss_nats * (total_tokens / total_bytes) / ln(2)
    """
    import math
    import numpy as np

    data = np.memmap(bin_path, dtype=np.uint16, mode="r")
    byte_counts = np.fromfile(bytes_path, dtype=np.int64)
    total_tokens = len(data)
    total_bytes = int(byte_counts.sum())
    tokens_per_byte = total_tokens / total_bytes

    for entry in log:
        if entry.get("val_loss") is not None:
            entry["val_bpb"] = entry["val_loss"] * tokens_per_byte / math.log(2)
    return log


def main() -> None:
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    primoji_log = load_log("primoji")
    mistral_log = load_log("mistral")

    if not primoji_log or not mistral_log:
        print("Need both training logs. Run train.py for both tokenizers first.")
        return

    # Recompute BPB from val_loss (fixes any buggy early logs)
    primoji_log = recompute_bpb(
        primoji_log,
        str(_DATA_DIR / "primoji_val.bin"),
        str(_DATA_DIR / "byte_counts_val.bin"),
    )
    mistral_log = recompute_bpb(
        mistral_log,
        str(_DATA_DIR / "mistral_val.bin"),
        str(_DATA_DIR / "byte_counts_val.bin"),
    )

    # Filter to eval steps only (have val_bpb)
    p_evals = [e for e in primoji_log if e.get("val_bpb") is not None]
    m_evals = [e for e in mistral_log if e.get("val_bpb") is not None]

    # ── Plot 1: BPB vs Tokens Seen ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([e["tokens_seen"] for e in p_evals], [e["val_bpb"] for e in p_evals],
            "b-o", label="Primoji", markersize=3)
    ax.plot([e["tokens_seen"] for e in m_evals], [e["val_bpb"] for e in m_evals],
            "r-o", label="Mistral BPE", markersize=3)
    ax.set_xlabel("Tokens Seen")
    ax.set_ylabel("Bits per Byte (BPB)")
    ax.set_title("Learning Efficiency: BPB vs Tokens Processed")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(_PLOT_DIR / "bpb_vs_tokens.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bpb_vs_tokens.png")

    # ── Plot 2: BPB vs FLOPs ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    p_flops = [e.get("flops", 0) for e in p_evals if e.get("flops")]
    p_bpb_f = [e["val_bpb"] for e in p_evals if e.get("flops")]
    m_flops = [e.get("flops", 0) for e in m_evals if e.get("flops")]
    m_bpb_f = [e["val_bpb"] for e in m_evals if e.get("flops")]
    if p_flops and m_flops:
        ax.plot(p_flops, p_bpb_f, "b-o", label="Primoji", markersize=3)
        ax.plot(m_flops, m_bpb_f, "r-o", label="Mistral BPE", markersize=3)
        ax.set_xlabel("FLOPs")
        ax.set_ylabel("Bits per Byte (BPB)")
        ax.set_title("Compute Efficiency: BPB vs FLOPs")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        fig.savefig(_PLOT_DIR / "bpb_vs_flops.png", dpi=150, bbox_inches="tight")
        print(f"Saved bpb_vs_flops.png")
    plt.close()

    # ── Plot 3: BPB vs Wall Time ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([e["time"]/3600 for e in p_evals], [e["val_bpb"] for e in p_evals],
            "b-o", label="Primoji", markersize=3)
    ax.plot([e["time"]/3600 for e in m_evals], [e["val_bpb"] for e in m_evals],
            "r-o", label="Mistral BPE", markersize=3)
    ax.set_xlabel("Wall Time (hours)")
    ax.set_ylabel("Bits per Byte (BPB)")
    ax.set_title("BPB vs Wall Clock Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(_PLOT_DIR / "bpb_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bpb_vs_time.png")

    # ── Plot 4: Raw loss curves ──────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot([e["step"] for e in p_evals], [e["val_loss"] for e in p_evals], "b-")
    ax1.set_title("Primoji: Validation Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Cross-Entropy Loss (nats)")
    ax1.grid(True, alpha=0.3)
    ax2.plot([e["step"] for e in m_evals], [e["val_loss"] for e in m_evals], "r-")
    ax2.set_title("Mistral BPE: Validation Loss")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cross-Entropy Loss (nats)")
    ax2.grid(True, alpha=0.3)
    fig.savefig(_PLOT_DIR / "raw_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved raw_loss.png")

    # ── Report ────────────────────────────────────────────────────────────
    p_final = p_evals[-1]
    m_final = m_evals[-1]

    report = {
        "setup": {
            "primoji_vocab_size": None,  # filled from meta if available
            "mistral_vocab_size": 32768,
        },
        "final_results": {
            "primoji_bpb": p_final["val_bpb"],
            "mistral_bpb": m_final["val_bpb"],
            "primoji_val_loss": p_final["val_loss"],
            "mistral_val_loss": m_final["val_loss"],
            "primoji_total_tokens": p_final["tokens_seen"],
            "mistral_total_tokens": m_final["tokens_seen"],
            "primoji_total_flops": p_final.get("flops"),
            "mistral_total_flops": m_final.get("flops"),
            "primoji_time_hours": p_final["time"] / 3600,
            "mistral_time_hours": m_final["time"] / 3600,
        },
        "verdict": {},
    }

    # Verdicts
    if p_final["val_bpb"] < m_final["val_bpb"]:
        report["verdict"]["bpb"] = "PRIMOJI_WINS"
        report["verdict"]["interpretation"] = "Semantic structure improves learning per document"
    elif abs(p_final["val_bpb"] - m_final["val_bpb"]) < 0.05:
        report["verdict"]["bpb"] = "TIE"
        report["verdict"]["interpretation"] = "No significant difference"
    else:
        report["verdict"]["bpb"] = "BPE_WINS"
        report["verdict"]["interpretation"] = "BPE learns more efficiently"

    with open(_DATA_DIR / "experiment_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print
    print("\n" + "=" * 60)
    print("PRIMOJI vs BPE: 125M TRAINING EXPERIMENT")
    print("=" * 60)
    print(f"\nFinal BPB:")
    print(f"  Primoji:    {p_final['val_bpb']:.4f}")
    print(f"  Mistral:    {m_final['val_bpb']:.4f}")
    print(f"  Difference: {p_final['val_bpb'] - m_final['val_bpb']:+.4f}")
    print(f"\nTokens processed:")
    print(f"  Primoji:    {p_final['tokens_seen']:,}")
    print(f"  Mistral:    {m_final['tokens_seen']:,}")
    print(f"\nWall time:")
    print(f"  Primoji:    {p_final['time']/3600:.1f}h")
    print(f"  Mistral:    {m_final['time']/3600:.1f}h")
    print(f"\nVerdict: {report['verdict']['bpb']}")
    print(f"  {report['verdict']['interpretation']}")


if __name__ == "__main__":
    main()
