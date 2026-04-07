"""Primoji: a compositional semantic tokenizer for LLMs.

Encodes English text into a vocabulary of roughly 10,000 tokens built from
Unicode emoji, Natural Semantic Metalanguage (NSM) primitives, common word
tokens, and a UTF-8 byte fallback. The encode pipeline never produces an
UNK token: any input is representable.

Example:
    >>> from primoji import Tokenizer
    >>> tok = Tokenizer()
    >>> ids = tok.encode("The teacher explained photosynthesis")
    >>> tok.decode(ids)
    'teacher say photosynthesis'
    >>> tok.vocab_size
    10195
"""

from primoji.byte_fallback import decode_bytes, encode_bytes
from primoji.tokenizer import Tokenizer

__all__ = ["Tokenizer", "encode_bytes", "decode_bytes"]
__version__ = "0.1.0"
