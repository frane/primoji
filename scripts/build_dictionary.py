"""Phase 1: Rule-based English-to-emoji mapping using spaCy NLP pipeline.

Takes an input text corpus, POS-tags each word with spaCy, and maps words
to emoji token ID sequences via deterministic rules (semantic role lookup,
morphological decomposition, compositional primitives).

Pipeline:
    1. Load spaCy model and Primoji primitives/vocabulary.
    2. Read input corpus (plain text or FineWeb-Edu parquet).
    3. For each unique word/phrase:
       a. POS-tag and lemmatize with spaCy.
       b. Look up direct emoji match (Tier 1).
       c. If no direct match, compose from primitives (Tier 2).
       d. Record mapping with confidence score and rule provenance.
    4. Write dictionary.json to data/.

Usage:
    python -m scripts.build_dictionary --corpus data/corpus.txt --output data/dictionary.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import spacy
from spacy.tokens import Doc, Token

from primoji.primitives import PRIMITIVES
from primoji.vocabulary import Vocabulary

logger = logging.getLogger(__name__)

# Default paths relative to the project data directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_OUTPUT = DATA_DIR / "dictionary.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Build rule-based English-to-emoji dictionary."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to input text corpus (plain text or parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output dictionary JSON file.",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model to use for POS tagging and lemmatization.",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=5,
        help="Minimum word frequency to include in dictionary.",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=50_000,
        help="Maximum number of unique words to process.",
    )
    return parser.parse_args()


def load_corpus(corpus_path: Path) -> list[str]:
    """Load text corpus from file, returning a list of sentences.

    Supports plain text (.txt) and parquet (.parquet) formats.
    """
    # TODO: Implement plain text loading (line-by-line)
    # TODO: Implement parquet loading (extract text column from FineWeb-Edu)
    raise NotImplementedError("Corpus loading not yet implemented")


def extract_vocabulary(docs: list[Doc], min_freq: int, max_words: int) -> dict[str, int]:
    """Extract unique words with frequency counts from processed documents.

    Args:
        docs: spaCy-processed documents.
        min_freq: Minimum frequency threshold.
        max_words: Maximum vocabulary size.

    Returns:
        Mapping of word (lemma) to frequency count, sorted by frequency descending.
    """
    # TODO: Iterate over tokens, count lemma frequencies
    # TODO: Filter by min_freq, truncate to max_words
    raise NotImplementedError("Vocabulary extraction not yet implemented")


def map_word_to_emoji(
    token: Token,
    vocab: Vocabulary,
    primitives: dict[str, Any],
) -> dict[str, Any] | None:
    """Map a single word to an emoji token ID sequence using rules.

    Strategy (in priority order):
        1. Direct emoji match in Tier 1 vocabulary.
        2. Composition from Tier 2 primitives based on POS + semantic role.
        3. Morphological decomposition (prefix/suffix stripping, then retry).
        4. Return None if no mapping found (will need fallback encoder).

    Args:
        token: spaCy Token with POS tag, lemma, and dependency info.
        vocab: Primoji Vocabulary instance for lookups.
        primitives: Compositional primitives mapping.

    Returns:
        Dictionary with keys: word, lemma, pos, emoji_ids, emoji_str, rule, confidence.
        None if no mapping could be determined.
    """
    # TODO: Implement direct emoji lookup (Tier 1)
    # TODO: Implement POS-based composition rules (Tier 2)
    # TODO: Implement morphological decomposition fallback
    raise NotImplementedError("Word-to-emoji mapping not yet implemented")


def build_dictionary(
    corpus_path: Path,
    spacy_model: str,
    min_freq: int,
    max_words: int,
) -> dict[str, Any]:
    """Build the complete English-to-emoji dictionary.

    Args:
        corpus_path: Path to the input corpus.
        spacy_model: Name of the spaCy model to load.
        min_freq: Minimum word frequency threshold.
        max_words: Maximum vocabulary size.

    Returns:
        Dictionary mapping words to emoji token ID sequences with metadata.
    """
    # TODO: Load spaCy model
    # TODO: Load corpus
    # TODO: Process corpus through spaCy pipeline
    # TODO: Extract vocabulary
    # TODO: Map each word to emoji sequence
    # TODO: Collect statistics (coverage, avg composition length, etc.)
    raise NotImplementedError("Dictionary building not yet implemented")


def main() -> None:
    """Entry point: build rule-based English-to-emoji dictionary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    logger.info("Building dictionary from corpus: %s", args.corpus)
    logger.info("spaCy model: %s", args.spacy_model)
    logger.info("Min frequency: %d, Max words: %d", args.min_freq, args.max_words)

    dictionary = build_dictionary(
        corpus_path=args.corpus,
        spacy_model=args.spacy_model,
        min_freq=args.min_freq,
        max_words=args.max_words,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=2)

    logger.info("Dictionary written to %s", args.output)
    logger.info("Total entries: %d", len(dictionary.get("entries", {})))


if __name__ == "__main__":
    main()
