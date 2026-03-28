# Primoji

Primoji is a compositional pictographic tokenizer for LLM training. Instead of BPE's meaningless character fragments ("photo" + "synth" + "esis"), Primoji encodes text using semantically meaningful emoji tokens. The vocabulary is organized in three tiers: ~1,200 direct Unicode emoji for concrete concepts, 120 compositional primitives grounded in Wierzbicka's semantic primes, and ~480 structural tokens for modifiers, digits, and operators. Words are composed as emoji sequences of up to 5 tokens following positional semantic rules, producing sequences ~55% shorter than BPE while preserving semantic fidelity.

## Installation

```bash
pip install primoji
```

For development:

```bash
git clone https://github.com/frane/primoji.git
cd primoji/code
pip install -e ".[dev]"
```

## Quick Start

```python
from primoji import Tokenizer

tok = Tokenizer()

# Encode text to emoji token IDs
ids = tok.encode("The teacher explained photosynthesis")
# => [SEP, PERSON, TEACH, SAY, PLANT, ABSORB, LIGHT]

# Decode back to canonical English
text = tok.decode(ids)
# => "teacher said plant absorb light"

# Decode with verbatim reconstruction (BPE sidecar)
text = tok.decode(ids, verbatim=True)
# => "The teacher explained photosynthesis"

# Inspect vocabulary
print(tok.vocab_size)   # ~1800
print(tok.describe(ids[1]))
# => "PERSON (someone) - generic person, Tier 2 Wierzbicka prime"
```

## Vocabulary Overview

Primoji uses approximately 1,800 tokens organized in three tiers:

### Tier 1: Direct Unicode Emoji (~1,200 tokens)

Existing Unicode 17.0 emoji used as literal single tokens wherever a direct semantic match exists. Covers concrete nouns: animals, food, objects, vehicles, buildings, weather, etc.

### Tier 2: Compositional Primitives (~120 tokens)

For concepts emoji cannot directly represent (verbs, adjectives, abstract ideas):
- **Layer 2a**: Wierzbicka's 65 semantic primes -- linguistically verified irreducible concepts tested across 30+ language families (THINK, KNOW, WANT, GOOD, BAD, BIG, SMALL, etc.)
- **Layer 2b**: 55 domain expansions for educational text coverage (GROW, CREATE, DESTROY, MACHINE, PATTERN, etc.)

### Tier 3: Modifiers and Structural (~480 tokens)

Country flags (259), single digits (0-9), math operators, punctuation, domain anchors (~500 proper nouns), and special tokens (BOS, EOS, PAD, UNK, SEP).

## Composition Rules

Words are encoded as emoji sequences with positional semantics:

| Position | Role | Example |
|----------|------|---------|
| 1 | HEAD (what kind of thing) | PLANT |
| 2 | MODIFIER (property/relation) | ABSORB |
| 3-5 | SPECIFIERS (refinement) | LIGHT |

Examples: PLANT+ABSORB+LIGHT = photosynthesis, WATER+CAUSE+GAS = evaporation, DOG+DOMESTIC+GERMAN = German shepherd.

## Project Structure

```
code/
  primoji/          Python package
    tokenizer.py    Main encode/decode interface
    vocabulary.py   Token IDs and metadata
    composer.py     Composition engine
    dictionary.py   Word-to-emoji lookup
    primitives.py   120 compositional primitives
    decoder.py      Emoji-to-English decoding
    math_handler.py Math/code tokenization
    utils.py        Helpers and Unicode utilities
  scripts/          Dictionary building and evaluation
  tests/            pytest test suite
  data/             Generated data files
```

## Paper

Primoji is described in:

> "Primoji: Compositional Pictographic Tokenization for Efficient LLM Training"
> (Paper URL forthcoming)

## Citation

```bibtex
@article{primoji2026,
  title   = {Primoji: Compositional Pictographic Tokenization for Efficient LLM Training},
  author  = {Frane},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## License

Apache 2.0
