"""Phase 3: Trained fallback encoder — DEPRECATED.

This script was originally planned to train a seq2seq model for OOV word
encoding. It has been replaced by byte fallback (see primoji/byte_fallback.py).

Rationale for dropping the trained encoder:
- Byte fallback provides universal coverage with zero information loss
- No training data required, no model to maintain
- Simpler, faster, and proven at scale (Llama 1/2/3 SentencePiece)
- Deterministic: same input always produces same byte sequence
- The seq2seq approach would have required significant training data
  and introduced a non-deterministic component

See: primoji/byte_fallback.py for the replacement implementation.
See: research/design_decisions.md for full rationale.
"""


def main() -> None:
    """Deprecated — byte fallback replaces the trained encoder."""
    print("This script is deprecated.")
    print("Unknown words are now handled by byte fallback (primoji/byte_fallback.py).")
    print("See research/design_decisions.md for rationale.")


if __name__ == "__main__":
    main()
