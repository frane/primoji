"""Primoji: Compositional pictographic tokenizer for LLM training.

Encodes English text into semantically meaningful emoji token sequences
using a ~2,062 token vocabulary with 4-tier fallback (never produces UNK):
  - Tier 1: Direct Unicode emoji + compositional primitives + dictionary
  - Tier 2: Contraction tokens (dedicated + suffix split)
  - Tier 3: Conservative fuzzy matching (optional)
  - Tier 4: UTF-8 byte fallback (universal safety net)

Example:
    >>> from primoji import Tokenizer
    >>> tok = Tokenizer()
    >>> ids = tok.encode("The teacher explained photosynthesis")
    >>> tok.decode(ids)
    'teacher explained photosynthesis'
"""

from primoji.byte_fallback import decode_bytes, encode_bytes
from primoji.tokenizer import Tokenizer

__all__ = ["Tokenizer", "encode_bytes", "decode_bytes"]
__version__ = "0.1.0"
