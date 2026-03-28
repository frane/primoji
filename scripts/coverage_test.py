"""Test Primoji tokenizer coverage on 1000 random FineWeb-Edu sentences.

Samples sentences from FineWeb-Edu, tokenizes each with Primoji, and reports:
    - Overall word coverage (% of words with a known mapping).
    - UNK rate (% of words falling back to UNK token).
    - Fallback encoder usage (% of words handled by the trained encoder).
    - Per-POS coverage (nouns, verbs, adjectives, etc.).
    - List of most common UNK words for dictionary improvement.

Usage:
    python -m scripts.coverage_test \
        --input data/fineweb_edu_sample.parquet \
        --num-sentences 1000 \
        --output results/coverage_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import spacy

from primoji import Tokenizer
from primoji.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_NUM_SENTENCES = 1000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Primoji coverage on random FineWeb-Edu sentences."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to FineWeb-Edu parquet or text file.",
    )
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=DEFAULT_NUM_SENTENCES,
        help="Number of sentences to sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save coverage report JSON (optional).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model for sentence segmentation and POS tagging.",
    )
    return parser.parse_args()


def load_sentences(input_path: Path, num_sentences: int, seed: int) -> list[str]:
    """Load and randomly sample sentences from the input corpus.

    Reads documents, splits into sentences (using spaCy or simple heuristics),
    and returns a random sample.

    Args:
        input_path: Path to parquet or text file.
        num_sentences: Number of sentences to sample.
        seed: Random seed.

    Returns:
        List of sampled sentence strings.
    """
    # TODO: Load documents from parquet or text
    # TODO: Split documents into sentences
    # TODO: Sample num_sentences uniformly at random
    raise NotImplementedError("Sentence loading not yet implemented")


def analyze_token_coverage(
    tokenizer: Tokenizer,
    sentence: str,
    nlp: spacy.language.Language,
) -> dict[str, Any]:
    """Analyze token-level coverage for a single sentence.

    For each word in the sentence, determines whether the Primoji tokenizer
    resolves it via:
        1. Direct dictionary lookup (Tier 1 or known mapping).
        2. Fallback encoder (trained model generalization).
        3. UNK token (no mapping available).

    Args:
        tokenizer: Primoji Tokenizer instance.
        sentence: Input sentence string.
        nlp: spaCy Language model for POS tagging.

    Returns:
        Dict with keys: sentence, total_words, covered, fallback, unk,
        unk_words (list), per_pos (dict of POS -> coverage counts).
    """
    # TODO: Tokenize the sentence
    # TODO: For each word, determine resolution method
    # TODO: Track POS-specific coverage
    # TODO: Collect UNK words
    raise NotImplementedError("Token coverage analysis not yet implemented")


def compute_coverage_report(
    per_sentence_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate coverage statistics across all sampled sentences.

    Args:
        per_sentence_results: List of per-sentence analysis dicts.

    Returns:
        Report dict with overall coverage %, UNK rate, fallback rate,
        per-POS breakdown, top-50 UNK words.
    """
    # TODO: Sum total words, covered, fallback, unk across sentences
    # TODO: Compute percentages
    # TODO: Aggregate per-POS stats
    # TODO: Rank UNK words by frequency, keep top 50
    raise NotImplementedError("Coverage report aggregation not yet implemented")


def print_coverage_report(report: dict[str, Any]) -> None:
    """Print a formatted coverage report to stdout.

    Args:
        report: Aggregated coverage report dict.
    """
    # TODO: Print header with sample size
    # TODO: Print overall coverage, fallback, UNK percentages
    # TODO: Print per-POS coverage table
    # TODO: Print top UNK words list
    raise NotImplementedError("Report printing not yet implemented")


def main() -> None:
    """Entry point: test Primoji coverage on FineWeb-Edu sentences."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    random.seed(args.seed)

    logger.info("Loading Primoji tokenizer")
    tokenizer = Tokenizer()

    logger.info("Loading spaCy model: %s", args.spacy_model)
    nlp = spacy.load(args.spacy_model)

    logger.info("Sampling %d sentences from %s", args.num_sentences, args.input)
    sentences = load_sentences(args.input, args.num_sentences, args.seed)
    logger.info("Sampled %d sentences", len(sentences))

    per_sentence_results: list[dict[str, Any]] = []
    for i, sentence in enumerate(sentences):
        result = analyze_token_coverage(tokenizer, sentence, nlp)
        per_sentence_results.append(result)

        if (i + 1) % 100 == 0:
            logger.info("Analyzed %d / %d sentences", i + 1, len(sentences))

    report = compute_coverage_report(per_sentence_results)
    print_coverage_report(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Coverage report saved to %s", args.output)

    # Exit with non-zero status if coverage is below threshold
    overall_coverage = report.get("overall_coverage_pct", 0.0)
    if overall_coverage < 80.0:
        logger.warning(
            "Coverage %.1f%% is below 80%% threshold. Dictionary needs expansion.",
            overall_coverage,
        )


if __name__ == "__main__":
    main()
