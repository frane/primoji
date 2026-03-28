"""Tokenize FineWeb-Edu dataset into Primoji token shards.

Reads FineWeb-Edu parquet files, tokenizes each document with the Primoji
tokenizer, and writes sharded output files containing token ID sequences.
Supports multi-process parallel tokenization for throughput.

Output format:
    Each shard is a numpy .npy file containing a 1D uint16 array of token IDs,
    with documents separated by EOS tokens. Shard metadata (document boundaries,
    token counts) is stored in a companion .json file.

Usage:
    python -m scripts.tokenize_fineweb \
        --input-dir data/fineweb_edu/ \
        --output-dir data/primoji_shards/ \
        --num-workers 8 \
        --shard-size 10000000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from primoji import Tokenizer

logger = logging.getLogger(__name__)

DEFAULT_SHARD_SIZE = 10_000_000  # tokens per shard


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tokenize FineWeb-Edu into Primoji shards."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing FineWeb-Edu parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write output shard files.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=DEFAULT_SHARD_SIZE,
        help="Maximum number of tokens per shard file.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel tokenization workers.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to process (for testing).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last completed shard if output directory exists.",
    )
    return parser.parse_args()


def discover_parquet_files(input_dir: Path) -> list[Path]:
    """Find all parquet files in the input directory.

    Args:
        input_dir: Directory to scan for .parquet files.

    Returns:
        Sorted list of parquet file paths.
    """
    # TODO: Glob for .parquet files, sort by name for deterministic ordering
    raise NotImplementedError("Parquet file discovery not yet implemented")


def read_documents(parquet_path: Path) -> list[str]:
    """Read text documents from a FineWeb-Edu parquet file.

    Args:
        parquet_path: Path to a single parquet file.

    Returns:
        List of document text strings.
    """
    # TODO: Read parquet with pyarrow or pandas
    # TODO: Extract the 'text' column
    raise NotImplementedError("Parquet reading not yet implemented")


def tokenize_document(tokenizer: Tokenizer, text: str) -> list[int]:
    """Tokenize a single document with the Primoji tokenizer.

    Prepends BOS, appends EOS. Handles edge cases (empty text, very long docs).

    Args:
        tokenizer: Primoji Tokenizer instance.
        text: Document text string.

    Returns:
        List of token IDs including BOS and EOS markers.
    """
    # TODO: Encode text with tokenizer
    # TODO: Add BOS/EOS markers
    # TODO: Handle empty/whitespace-only documents
    raise NotImplementedError("Document tokenization not yet implemented")


def write_shard(
    output_dir: Path,
    shard_idx: int,
    token_ids: list[int],
    doc_boundaries: list[int],
) -> dict[str, Any]:
    """Write a single shard to disk as .npy with companion .json metadata.

    Args:
        output_dir: Directory to write the shard files.
        shard_idx: Zero-based shard index.
        token_ids: Flat list of token IDs for this shard.
        doc_boundaries: List of document start offsets within the shard.

    Returns:
        Shard metadata dict (path, num_tokens, num_docs).
    """
    # TODO: Save token_ids as uint16 numpy array
    # TODO: Write metadata JSON with doc boundaries, counts
    raise NotImplementedError("Shard writing not yet implemented")


def tokenize_parallel(
    parquet_files: list[Path],
    tokenizer: Tokenizer,
    num_workers: int,
    max_docs: int | None,
) -> list[tuple[list[int], list[int]]]:
    """Tokenize documents in parallel across multiple workers.

    Args:
        parquet_files: List of parquet file paths to process.
        tokenizer: Primoji Tokenizer instance.
        num_workers: Number of parallel workers.
        max_docs: Optional cap on total documents.

    Returns:
        List of (token_ids, doc_boundaries) tuples, one per worker batch.
    """
    # TODO: Use multiprocessing.Pool or concurrent.futures
    # TODO: Distribute parquet files across workers
    # TODO: Collect and merge results
    raise NotImplementedError("Parallel tokenization not yet implemented")


def main() -> None:
    """Entry point: tokenize FineWeb-Edu into Primoji shards."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    logger.info("Tokenizing FineWeb-Edu with Primoji")
    logger.info("Input: %s", args.input_dir)
    logger.info("Output: %s", args.output_dir)
    logger.info("Shard size: %d tokens", args.shard_size)
    logger.info("Workers: %d", args.num_workers)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer()
    parquet_files = discover_parquet_files(args.input_dir)
    logger.info("Found %d parquet files", len(parquet_files))

    # TODO: Iterate over documents, fill shards, write when full
    # TODO: Handle resume logic (skip already-written shards)
    # TODO: Write final manifest.json with global statistics

    total_tokens = 0
    total_docs = 0
    total_shards = 0

    logger.info("Tokenization complete.")
    logger.info("Total documents: %d", total_docs)
    logger.info("Total tokens: %d", total_tokens)
    logger.info("Total shards: %d", total_shards)
    logger.info(
        "Avg tokens per document: %.1f",
        total_tokens / max(total_docs, 1),
    )


if __name__ == "__main__":
    main()
