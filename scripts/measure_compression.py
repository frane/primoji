"""Compare Primoji sequence lengths vs Mistral BPE (32K vocab) baseline.

Tokenizes a set of sample texts with both Primoji and the Mistral v0.3 BPE
tokenizer, then reports compression ratios, sequence length distributions,
and per-domain breakdowns. Results are printed as a summary table and
optionally saved to JSON for plotting.

Key metrics:
    - Sequence length ratio (primoji tokens / BPE tokens per document).
    - Compression ratio (mean, median, per-domain).
    - Vocabulary utilization (unique tokens used / total vocab).
    - Composition depth distribution (1-token, 2-token, ..., 5-token compositions).

Usage:
    python -m scripts.measure_compression \
        --input data/sample_texts.jsonl \
        --output results/compression_results.json \
        --bpe-model mistralai/Mistral-7B-v0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer as HFTokenizer

from primoji import Tokenizer as PrimojiTokenizer

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_BPE_MODEL = "mistralai/Mistral-7B-v0.3"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Primoji sequence lengths vs Mistral BPE baseline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input texts (JSONL with 'text' and optional 'domain' fields).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save detailed results as JSON (optional).",
    )
    parser.add_argument(
        "--bpe-model",
        type=str,
        default=DEFAULT_BPE_MODEL,
        help="HuggingFace tokenizer ID for BPE baseline.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process.",
    )
    return parser.parse_args()


def load_samples(input_path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    """Load text samples from a JSONL file.

    Each line should have at minimum a 'text' field. An optional 'domain' field
    enables per-domain breakdown.

    Args:
        input_path: Path to the JSONL file.
        max_samples: Optional cap on number of samples.

    Returns:
        List of sample dicts with 'text' and optionally 'domain'.
    """
    # TODO: Read JSONL lines, parse JSON, apply max_samples limit
    raise NotImplementedError("Sample loading not yet implemented")


def load_bpe_tokenizer(model_name: str) -> HFTokenizer:
    """Load the BPE tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model ID or path to tokenizer.json.

    Returns:
        HuggingFace Tokenizer instance.
    """
    # TODO: Load tokenizer from pretrained or from file
    raise NotImplementedError("BPE tokenizer loading not yet implemented")


def tokenize_both(
    text: str,
    primoji_tok: PrimojiTokenizer,
    bpe_tok: HFTokenizer,
) -> dict[str, Any]:
    """Tokenize a single text with both tokenizers and compute metrics.

    Args:
        text: Input text string.
        primoji_tok: Primoji tokenizer instance.
        bpe_tok: BPE tokenizer instance.

    Returns:
        Dict with keys: text_len, primoji_tokens, bpe_tokens, ratio,
        primoji_unique, bpe_unique, composition_depths.
    """
    # TODO: Encode with both tokenizers
    # TODO: Compute sequence lengths and ratio
    # TODO: Analyze composition depth distribution for primoji tokens
    raise NotImplementedError("Dual tokenization not yet implemented")


def compute_summary_statistics(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate statistics across all samples.

    Args:
        results: List of per-sample result dicts from tokenize_both().

    Returns:
        Summary dict with mean/median/std ratios, per-domain breakdowns,
        vocabulary utilization stats.
    """
    # TODO: Compute overall mean, median, std of compression ratio
    # TODO: Group by domain if available
    # TODO: Compute vocabulary utilization for both tokenizers
    # TODO: Compute composition depth histogram
    raise NotImplementedError("Summary statistics not yet implemented")


def print_report(summary: dict[str, Any]) -> None:
    """Print a formatted summary report to stdout.

    Args:
        summary: Summary statistics dict.
    """
    # TODO: Print header
    # TODO: Print overall compression ratio (mean, median, std)
    # TODO: Print per-domain breakdown table
    # TODO: Print composition depth histogram
    # TODO: Print vocabulary utilization comparison
    raise NotImplementedError("Report printing not yet implemented")


def main() -> None:
    """Entry point: measure compression ratio of Primoji vs BPE."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    logger.info("Loading Primoji tokenizer")
    primoji_tok = PrimojiTokenizer()

    logger.info("Loading BPE tokenizer: %s", args.bpe_model)
    bpe_tok = load_bpe_tokenizer(args.bpe_model)

    logger.info("Loading samples from %s", args.input)
    samples = load_samples(args.input, args.max_samples)
    logger.info("Loaded %d samples", len(samples))

    results: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        result = tokenize_both(sample["text"], primoji_tok, bpe_tok)
        result["domain"] = sample.get("domain", "unknown")
        results.append(result)

        if (i + 1) % 100 == 0:
            logger.info("Processed %d / %d samples", i + 1, len(samples))

    summary = compute_summary_statistics(results)
    print_report(summary)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_sample": results}, f, indent=2)
        logger.info("Detailed results saved to %s", args.output)


if __name__ == "__main__":
    main()
