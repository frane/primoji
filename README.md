# Primoji

A compositional semantic tokenizer for LLM training.

Primoji encodes text using semantic tokens: emoji for concrete nouns, 140
compositional primitives grounded in linguistic universals, and direct word
tokens for common vocabulary. Grammar words get compositional embeddings
built from primitive components. Rare words decompose into primitive
sequences that encode meaning.

```
BPE:     "photosynthesis"  ->  ["photo", "synth", "esis"]     (character fragments)
Primoji: "photosynthesis"  ->  [PLANT, HAVE, LIGHT]           (semantic composition)
```

## Results

### Training: primoji beats BPE by 13%

125M parameter GPT models trained on 500K FineWeb-Edu documents:

| Model | Vocab | BPB | vs BPE |
|-------|-------|-----|--------|
| v6 primoji (compositional embeddings) | 10,149 | **0.996** | 13% better |
| v7 primoji (+ coverage fix) | 10,195 | training | -- |
| BPE baseline (Mistral 32K) | 32,768 | 1.144 | baseline |

BPB = bits-per-byte, the fair cross-tokenizer comparison metric.

### Compression (1,000 FineWeb-Edu sentences)

| Version | Vocab | Tokens | Ratio | Byte fallback |
|---------|-------|--------|-------|--------------|
| v1 | 4,035 | 59,460 | 1.80x | 17.4% |
| v3 | 5,311 | 57,120 | 1.73x | 16.6% |
| v7 | 10,195 | 36,076 | 1.09x | 6.9% |
| BPE | 32,768 | 33,118 | 1.00x | 0% |

v7 achieves near-parity compression with BPE (1.09x) while maintaining
semantic structure. Byte fallback at 6.9% covers only proper nouns and
domain-specific terms.

### Key innovation: Compositional Embeddings (v6+)

Grammar words ("is", "was", "not", "they") are single word tokens for
compression, but their embeddings are computed as the mean of their
primitive components:

- "is" embedding = mean(BE, NOW)
- "was" embedding = mean(BE, BEFORE)
- "not" embedding = NOT primitive embedding
- "they" embedding = mean(SOMEONE, OTHER, MANY)

This links related forms through shared primitives. Updating the BE
embedding simultaneously updates "is", "was", "are", "were", "been".

## How it works

```python
from primoji import Tokenizer

tok = Tokenizer()

ids = tok.encode("The teacher explained photosynthesis")
print(tok.decode(ids))
# -> "teacher say photosynthesis"

print(tok.vocab_size)  # 10,195
```

## Vocabulary (~10,195 tokens)

| Tier | Count | Examples |
|------|-------|----------|
| Emoji | 1,175 | dog, cat, hospital, tree |
| Common words | 7,747 | government, studied, susceptible |
| Primitives | 140 | PLANT, CAUSE, MOVE, KNOW, HEALTH, STUDY |
| Structural | ~875 | Flags, anchors, digits, punctuation, math |
| Byte fallback | 258 | UTF-8 bytes + boundary markers |

140 primitives: 65 Wierzbicka semantic primes (verified across 30+
language families) + 75 domain expansions (HEALTH, STUDY, ELECTRIC,
SUBSTANCE, ENVIRONMENT, BODY_PART, VISIBLE, DEGREE).

## Training

```bash
# Prepare data (streams from HuggingFace):
python -m scripts.prepare_training_data --n-docs 500000 --output-dir data/experiment

# Train 125M:
python -m scripts.train --tokenizer primoji --v2 --byte-weight 0.7 --batch-size 32

# Train 1B with gradient accumulation:
python -m scripts.train --tokenizer primoji --model-size 1b --batch-size 4 --grad-accum 8 --v2 --byte-weight 0.7

# BPE baseline:
python -m scripts.train --tokenizer mistral --batch-size 32
```

## Interactive inference

```bash
python -m scripts.chat --model path/to/model.pt --tiers --trace
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
  tokenizer.py        4-tier encode pipeline, decode
  vocabulary.py       Dynamic ID ranges from data files
  dictionary.py       Symbolic seed + runtime resolution
  primitives.py       140 semantic primitives
  decoder.py          Dictionary-first, then tier-based fallback
  byte_fallback.py    UTF-8 byte encoding for unknown words
  composer.py         HEAD + MODIFIER + SPECIFIER composition
  preprocessor.py     Unicode norm, contraction expansion, hyphen split
  fuzzy.py            Conservative SymSpell (edit distance 1)
  math_handler.py     Single-digit numbers, math operators
  alias_map.py        Grammar word -> primitive decomposition
  utils.py            Shared constants, classify_token, text normalization
scripts/
  build_dictionary.py   Reproducible dictionary build
  prepare_training_data.py  Tokenize FineWeb-Edu for training
  train.py             GPT training (125M/1B) with BPB evaluation
  compare_results.py   Training log analysis and plots
  chat.py              Interactive inference with --trace mode
tests/                 594 tests
data/
  primitives.json      140 compositional primitives (v0.3)
  emoji_catalog.json   1,175 emoji with CLDR annotations
  common_words.json    7,747 high-frequency words
  auto_compositions.json  4,600+ WordNet-derived compositions
  dictionary_seed.json Symbolic dictionary (~42K entries)
  eval_sentences.json  1,000 FineWeb-Edu sentences (benchmark)
```

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
