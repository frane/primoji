# Primoji

A compositional semantic tokenizer for LLM training research.

Primoji tokenizes text using a hybrid vocabulary: direct tokens for common
words and emoji for concrete nouns, plus compositional semantic decomposition
for rare and technical vocabulary. "Photosynthesis" becomes [PLANT, HAVE, LIGHT],
three tokens that encode the concept's meaning rather than arbitrary character
fragments.

The research question: does semantic structure in the token representation
improve model learning, even when it doesn't compress text shorter?

```
BPE:     "photosynthesis"  ->  ["photo", "synth", "esis"]     (character fragments)
Primoji: "photosynthesis"  ->  [PLANT, HAVE, LIGHT]           (semantic composition)
```

## How it works

```python
from primoji import Tokenizer

tok = Tokenizer()

ids = tok.encode("The teacher explained photosynthesis")
print(tok.decode(ids))
# -> "teacher explained photosynthesis"

print(tok.vocab_size)  # 4,035
```

The tokenizer uses a 4-tier fallback pipeline. No input ever produces UNK:

| Tier | What it handles | How |
|------|----------------|-----|
| **1a. Emoji** | Concrete nouns (3.6%) | 1,182 Unicode emoji |
| **1b. Common words** | High-frequency English (11.9%) | 1,617 direct word tokens |
| **2. Primitives** | Verbs, adjectives, abstracts (22.7%) | 132 semantic primitives |
| **3. Structural** | Flags, anchors, digits, contractions | ~900 tokens |
| **4. Byte fallback** | Everything else (17.4%) | UTF-8 bytes, lossless |

## Vocabulary (4,035 tokens)

| Tier | Count | What | Examples |
|------|-------|------|----------|
| 1a. Emoji | 1,182 | Concrete nouns | dog, cat, hospital |
| 1b. Common words | 1,617 | NGSL high-frequency | government, important, against |
| 2. Primitives | 132 | Semantic composition | PLANT, CAUSE, MOVE, KNOW |
| 3. Structural | 846 | Flags, anchors, digits, contractions | US, Shakespeare, 0-9 |
| 4. Byte fallback | 258 | UTF-8 bytes + markers | Any unknown word |

Primitives are grounded in [Wierzbicka's Natural Semantic Metalanguage](https://en.wikipedia.org/wiki/Natural_semantic_metalanguage):
65 irreducible concepts verified across 30+ language families, plus 67 domain
expansions for educational text (v0.2, validated against Goddard & Wierzbicka 2014/2018).

Compositions follow positional rules: `HEAD + MODIFIER + SPECIFIER`
- SOMEONE + TEACH + SAY = teacher
- WATER + CAUSE + AIR = evaporation
- MACHINE + THINK = computer

## Empirical results

### Compression vs Mistral BPE (1,000 FineWeb-Edu sentences)

| Metric | Primoji | Mistral BPE |
|--------|---------|-------------|
| Vocab size | 4,035 | 32,768 |
| Total tokens | 59,460 | 33,118 |
| Ratio | 1.80x | 1.00x |
| Dictionary hit rate | 82.6% | 100% |
| Byte fallback rate | 17.4% | 0% |

**Primoji produces 1.8x more tokens than BPE, not fewer.** The original
hypothesis that compositional encoding would compress text shorter than BPE
was tested and falsified. Word-level composition expands most words to 2-3
tokens, while BPE handles the same words in 1-2 subword tokens. Byte fallback
adds 2 boundary tokens per unknown word.

The value proposition is not compression but vocabulary size (8x smaller
embedding table) and semantic structure in the token representation. Whether
semantic structure improves model learning efficiency is an open question
that requires training experiments to answer.

### Per-sentence distribution

| Percentile | Ratio |
|------------|-------|
| p10 (best) | 1.06x |
| p50 (median) | 1.73x |
| p90 (worst) | 2.48x |

The best 10% of sentences achieve near-parity with BPE. The worst 10%
are 2.5x longer, typically sentences with many rare words hitting byte fallback.

## Design history

This project started as "build a language in emoji." The original hypothesis
was that a small compositional emoji vocabulary would compress text shorter
than BPE. That hypothesis was falsified: word-level composition produced
1.95x more tokens than BPE on FineWeb-Edu (before the NGSL vocabulary
expansion brought it down to 1.80x).

The vocabulary was expanded with 1,617 direct tokens for common English words,
bringing the byte fallback rate from 42% to 17%. The research question shifted
from "does it compress better?" to "does semantic structure improve learning?"

## The reconstruction question

Primoji is a semantic representation layer, not a lossless text codec.

For dictionary-covered vocabulary (~83% of tokens), encode-decode roundtrip
produces the canonical word form: "photosynthesis" -> [PLANT, HAVE, LIGHT] ->
"photosynthesis".

For byte-fallback words, roundtrip is perfectly lossless: the original bytes
are preserved exactly.

Roundtrip is stable: `decode(encode(decode(encode(x)))) == decode(encode(x))`
always. The first roundtrip may change the surface form (lossy canonical
mapping), but subsequent roundtrips are identical.

## Installation

```bash
git clone https://github.com/frane/primoji.git
cd primoji
pip install -e ".[dev]"
```

## Project structure

```
primoji/
  tokenizer.py        4-tier encode/decode pipeline
  vocabulary.py       Dynamic ID ranges from data files
  dictionary.py       Symbolic seed + runtime resolution
  primitives.py       132 semantic primitives (from data/primitives.json)
  decoder.py          Dictionary-first, then tier-based fallback
  byte_fallback.py    UTF-8 byte encoding for unknown words
  composer.py         HEAD + MODIFIER + SPECIFIER composition
  fuzzy.py            Conservative SymSpell (edit distance 1)
  preprocessor.py     Unicode normalization, contraction splitting
  math_handler.py     Single-digit numbers, math operators
scripts/
  build_dictionary.py   Reproducible dictionary build (single source of truth)
tests/                  265 tests (invariants, stress, coverage, BPB formula)
data/
  emoji_catalog.json    1,182 emoji with CLDR annotations
  primitives.json       132 compositional primitives (v0.2)
  common_words.json     1,617 high-frequency English words
  proper_noun_anchors.json  500 FineWeb-Edu proper nouns
  dictionary_seed.json  Symbolic dictionary (~16K entries)
  compression_report.json  Benchmark results
```

## Current status

| Status | What |
|--------|------|
| Done | 4,035 token vocabulary (emoji + words + primitives + structural + byte fallback) |
| Done | 4-tier tokenization pipeline, 265 tests, reproducible dictionary build |
| Done | Compression benchmark: 1.80x vs BPE on 1K FineWeb-Edu sentences |
| Next | Training experiment: 125M model, bits-per-byte comparison vs BPE |
| Next | Deeper/narrower architecture (Wies bottleneck theorem) |
| Future | Multilingual extension |

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
