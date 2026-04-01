# Primoji

A compositional semantic tokenizer for LLM training.

Primoji encodes text using semantic tokens: emoji for concrete nouns, 140
compositional primitives grounded in linguistic universals, and direct word
tokens for common vocabulary. Rare and technical words decompose into
primitive sequences that encode meaning: "photosynthesis" becomes
[PLANT, HAVE, LIGHT].

```
BPE:     "photosynthesis"  ->  ["photo", "synth", "esis"]     (character fragments)
Primoji: "photosynthesis"  ->  [PLANT, HAVE, LIGHT]           (semantic composition)
```

## Results

### Training: primoji beats BPE at equal data exposure

125M parameter GPT models trained on 500K FineWeb-Edu documents:

| Metric | Primoji | BPE (Mistral) |
|--------|---------|---------------|
| Final BPB | **1.085** | 1.28 (projected) |
| At equal training progress (82%) | **1.096** | 1.175 |
| Advantage | **6.7% better** | baseline |
| Vocab size | 5,428 | 32,768 |
| Compression ratio | 1.44x | 1.00x |

Primoji produces 44% more tokens per document than BPE. Despite processing
more tokens (and using more compute), it achieves lower bits-per-byte at
every point after the first 5% of training. The semantic structure provides
a learning advantage that outweighs the sequence length penalty.

### Scaling trend

| Data scale | Primoji advantage (equal training progress) |
|------------|---------------------------------------------|
| 1K docs | 0.3% (barely significant) |
| 50K docs | 6.5% |
| 500K docs | 6.7% (holding steady) |

The advantage emerged at 50K docs and held at 500K. It appears to be a
stable property of the semantic representation, not a small-data artifact.

### Compression (1,000 FineWeb-Edu sentences)

| Metric | Value |
|--------|-------|
| Primoji tokens | 47,530 |
| BPE tokens | 33,118 |
| Ratio | 1.44x |
| Byte fallback | 11.0% |
| Best 10% of sentences | 0.80x (20% shorter than BPE) |

## How it works

```python
from primoji import Tokenizer

tok = Tokenizer()

ids = tok.encode("The teacher explained photosynthesis")
print(tok.decode(ids))
# -> "teacher explained photosynthesis"

# Contractions expand semantically
ids = tok.encode("won't")  # -> [will, not] = 2 meaningful tokens
print(tok.vocab_size)  # 5,428
```

The tokenizer uses a multi-tier pipeline. No input ever produces UNK:

| Tier | Coverage | How |
|------|----------|-----|
| Word tokens | 44.5% | 2,964 direct tokens for common English |
| Dropped | 21.0% | Articles/prepositions removed (zero tokens) |
| Structural | 13.7% | Digits, punctuation, flags, anchors |
| Byte fallback | 11.0% | UTF-8 bytes for unknown words (lossless) |
| Compositions | 5.8% | Primitive sequences for technical vocabulary |
| Emoji | 1.2% | 1,175 Unicode emoji for concrete nouns |
| Contractions | 0.8% | Possessive "'s" marker |

## Vocabulary (5,428 tokens)

| Tier | Count | Examples |
|------|-------|----------|
| Emoji | 1,175 | dog, cat, hospital, tree |
| Common words | 2,964 | government, important, researchers, was |
| Primitives | 140 | PLANT, CAUSE, MOVE, KNOW, HEALTH, STUDY |
| Structural | ~890 | Flags, anchors, digits, punctuation, math |
| Byte fallback | 258 | UTF-8 bytes + boundary markers |

140 primitives: 65 Wierzbicka semantic primes (verified across 30+ language
families) + 75 domain expansions including HEALTH, STUDY, ELECTRIC, SUBSTANCE,
ENVIRONMENT, BODY_PART, VISIBLE, DEGREE.

Compositions follow positional rules: `HEAD + MODIFIER + SPECIFIER`
- SOMEONE + TEACH + SAY = teacher
- WATER + CAUSE + AIR = evaporation
- MACHINE + THINK = computer
- HEALTH + LIVE + SMALL = infection

Contractions expand semantically:
- "won't" -> [will, not] (2 tokens, both meaningful)
- "don't" -> [do, not]
- "John's" -> [John, 's] (possessive preserved)

## Interactive inference

```bash
python -m scripts.chat                    # normal mode
python -m scripts.chat --trace            # show token tiers with colors
python -m scripts.chat --temperature 0.5  # less random
```

## Installation

```bash
git clone https://github.com/frane/primoji.git
cd primoji
pip install -e ".[dev]"
```

## Project structure

```
primoji/
  tokenizer.py        Multi-tier encode/decode pipeline
  vocabulary.py       Dynamic ID ranges from data files
  dictionary.py       Symbolic seed + runtime resolution
  primitives.py       140 semantic primitives (from data/primitives.json)
  decoder.py          Dictionary-first, then tier-based fallback
  byte_fallback.py    UTF-8 byte encoding for unknown words
  composer.py         HEAD + MODIFIER + SPECIFIER composition
  preprocessor.py     Unicode normalization, contraction expansion
  fuzzy.py            Conservative SymSpell (edit distance 1)
  math_handler.py     Single-digit numbers, math operators
scripts/
  build_dictionary.py   Reproducible dictionary build
  prepare_training_data.py  Tokenize FineWeb-Edu for training
  train_125m.py        125M GPT training with BPB evaluation
  compare_results.py   Training log analysis and plots
  chat.py              Interactive inference with --trace mode
tests/                 292 tests (invariants, stress, coverage, BPB, contractions)
data/
  primitives.json      140 compositional primitives (v0.3)
  emoji_catalog.json   1,175 emoji with CLDR annotations
  common_words.json    2,964 high-frequency English words
  auto_compositions.json  4,600+ WordNet-derived compositions
  dictionary_seed.json Symbolic dictionary (~33K entries)
```

## Design history

The project started as "build a language in emoji." The original hypothesis
was that a small compositional vocabulary would compress text shorter than
BPE. That hypothesis was falsified: primoji produces 1.44x more tokens than
BPE on FineWeb-Edu.

The vocabulary was expanded from 4K to 5.4K tokens (adding common word tokens,
auto-compositions from WordNet, and 8 new primitives from byte-fallback
analysis). The research question shifted from compression to learning
efficiency: does semantic structure help models learn faster per document?

The answer is yes. At 500K documents, primoji achieves 6.7% better
bits-per-byte than BPE at equal training progress, despite the 44% token
expansion.

## Paper

> **Primoji: Compositional Semantic Tokenization for LLM Training**
> Frane Bandov (ESCP Business School)
> *Manuscript in preparation*

Paper 2 in a series on efficient LLM training:
1. Distributed training protocol (Distrain)
2. Compositional semantic tokenization (this project)
3. Deterministic MoE routing via token semantics
4. Integration

## Citation

```bibtex
@article{bandov2026primoji,
  title  = {Primoji: Compositional Semantic Tokenization for LLM Training},
  author = {Bandov, Frane},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## License

Apache 2.0
