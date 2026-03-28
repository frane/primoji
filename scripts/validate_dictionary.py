"""Phase 2: LLM validation of dictionary mappings using the Claude API.

Reads the rule-based dictionary from Phase 1, sends each mapping to Claude
for semantic validation, and produces a validated dictionary plus a detailed
validation log (JSONL) with reasoning for every decision.

Validation criteria:
    - Is the emoji sequence a reasonable semantic representation of the word?
    - Are there polysemy issues (word has multiple senses, mapping covers only one)?
    - Are metaphorical or figurative uses handled correctly?
    - Is the composition order correct (HEAD + MODIFIER + SPECIFIER)?
    - Would a human guess the word from the emoji sequence alone?

Usage:
    python -m scripts.validate_dictionary \
        --input data/dictionary.json \
        --output data/dictionary_validated.json \
        --log data/validation_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_INPUT = DATA_DIR / "dictionary.json"
DEFAULT_OUTPUT = DATA_DIR / "dictionary_validated.json"
DEFAULT_LOG = DATA_DIR / "validation_log.jsonl"

# Claude model for validation
VALIDATION_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS_PER_REQUEST = 1024
RATE_LIMIT_DELAY_SECONDS = 0.5


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 2: LLM validation of English-to-emoji dictionary mappings."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to Phase 1 dictionary JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output validated dictionary JSON.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help="Path to JSONL validation log.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of mappings to validate per API call.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling the API.",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Resume validation from this entry index (for crash recovery).",
    )
    return parser.parse_args()


def load_dictionary(path: Path) -> dict[str, Any]:
    """Load the Phase 1 dictionary from JSON.

    Args:
        path: Path to dictionary JSON file.

    Returns:
        Parsed dictionary with entries and metadata.
    """
    # TODO: Load and validate dictionary structure
    raise NotImplementedError("Dictionary loading not yet implemented")


def build_validation_prompt(entries: list[dict[str, Any]]) -> str:
    """Build a prompt for Claude to validate a batch of word-to-emoji mappings.

    The prompt should ask Claude to evaluate each mapping on:
        - Semantic accuracy (does the emoji sequence capture the word's meaning?)
        - Composition correctness (HEAD + MODIFIER order respected?)
        - Polysemy handling (are multiple senses covered or noted?)
        - Guessability (could a human reconstruct the word from emoji?)

    Args:
        entries: List of dictionary entries to validate.

    Returns:
        Formatted prompt string for the Claude API.
    """
    # TODO: Format entries into a structured validation prompt
    # TODO: Include Primoji composition rules as context
    # TODO: Request structured JSON output with verdict + reasoning
    raise NotImplementedError("Prompt building not yet implemented")


def call_claude_api(
    client: anthropic.Anthropic,
    prompt: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Send a validation prompt to Claude and parse the response.

    Args:
        client: Anthropic API client.
        prompt: The validation prompt.
        dry_run: If True, print prompt and return dummy response.

    Returns:
        Parsed validation results (list of verdicts with reasoning).
    """
    # TODO: Call Claude API with structured output request
    # TODO: Parse response into per-entry verdicts
    # TODO: Handle rate limiting and retries
    raise NotImplementedError("Claude API calling not yet implemented")


def apply_validations(
    dictionary: dict[str, Any],
    validations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply validation results to the dictionary.

    For each entry:
        - If approved: keep as-is, mark validated=True.
        - If rejected: remove or flag for manual review.
        - If modified: update the emoji mapping with Claude's suggestion.

    Args:
        dictionary: Original Phase 1 dictionary.
        validations: List of validation verdicts from Claude.

    Returns:
        Updated dictionary with validation status on each entry.
    """
    # TODO: Match validations to dictionary entries
    # TODO: Apply approved/rejected/modified status
    # TODO: Track statistics (approved %, rejected %, modified %)
    raise NotImplementedError("Validation application not yet implemented")


def write_validation_log(
    log_path: Path,
    validations: list[dict[str, Any]],
) -> None:
    """Append validation results to JSONL log for reproducibility.

    Each line contains: word, emoji_ids, verdict, reasoning, timestamp, model.

    Args:
        log_path: Path to the JSONL log file.
        validations: List of validation results to log.
    """
    # TODO: Write each validation as a JSON line with timestamp
    raise NotImplementedError("Log writing not yet implemented")


def main() -> None:
    """Entry point: validate dictionary mappings with Claude."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    logger.info("Loading dictionary from %s", args.input)
    dictionary = load_dictionary(args.input)
    entries = list(dictionary.get("entries", {}).values())
    logger.info("Total entries to validate: %d", len(entries))

    if args.resume_from > 0:
        logger.info("Resuming from entry index %d", args.resume_from)
        entries = entries[args.resume_from :]

    client = anthropic.Anthropic() if not args.dry_run else None
    all_validations: list[dict[str, Any]] = []

    for batch_start in range(0, len(entries), args.batch_size):
        batch = entries[batch_start : batch_start + args.batch_size]
        batch_idx = args.resume_from + batch_start

        logger.info(
            "Validating batch %d-%d / %d",
            batch_idx,
            batch_idx + len(batch) - 1,
            len(entries) + args.resume_from,
        )

        prompt = build_validation_prompt(batch)
        results = call_claude_api(client, prompt, dry_run=args.dry_run)
        all_validations.extend(results.get("verdicts", []))

        write_validation_log(args.log, results.get("verdicts", []))

        if not args.dry_run:
            time.sleep(RATE_LIMIT_DELAY_SECONDS)

    validated_dict = apply_validations(dictionary, all_validations)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(validated_dict, f, ensure_ascii=False, indent=2)

    approved = sum(1 for v in all_validations if v.get("verdict") == "approved")
    rejected = sum(1 for v in all_validations if v.get("verdict") == "rejected")
    modified = sum(1 for v in all_validations if v.get("verdict") == "modified")

    logger.info("Validation complete.")
    logger.info("Approved: %d, Rejected: %d, Modified: %d", approved, rejected, modified)
    logger.info("Validated dictionary written to %s", args.output)
    logger.info("Validation log written to %s", args.log)


if __name__ == "__main__":
    main()
