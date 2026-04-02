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

125M parameter GPT models trained on 500K FineWeb-Edu documents (1 epoch):

| Model | BPB | vs BPE |
|-------|-----|--------|
| V1 primoji (5.4K vocab, no tier embeds) | **1.085** | 5.1% better |
| V2 primoji (5.4K vocab, tier embeds, byte_weight=0.3) | **1.092** | 4.5% better |
| V3 primoji (5.4K vocab, tier embeds, byte_weight=0.7) | **1.092** | 4.5% better |
| BPE (Mistral, 32K vocab) | 1.144 | baseline |

All primoji variants beat BPE. V1 is the strongest on BPB. V3 produces
dramatically better inference output (coherent text vs V1's repetitive tokens).

### Scaling trend

| Data scale | Primoji advantage (equal training progress) |
|------------|---------------------------------------------|
| 1K docs | 0.3% |
| 50K docs | 6.5% |
| 500K docs | 5-7% (stable) |

The advantage emerged at 50K docs and held at 500K.

### Compression (1,000 FineWeb-Edu sentences, v4 tokenizer)

| Metric | Primoji v4 | BPE |
|--------|-----------|-----|
| Vocab size | 10,428 | 32,768 |
| Total tokens | 36,100 | 33,118 |
| Ratio | 1.09x | 1.00x |
| Byte fallback | 6.9% | 0% |
| Median sentence ratio | 1.00x | - |

v4 achieves near-parity compression with BPE while maintaining semantic
structure for rare vocabulary. Progression: 1.72x (v1) -> 1.44x (v3) -> 1.09x (v4).

### Inference quality

V3 generates coherent, on-topic text with domain knowledge:

```
Prompt: "Water evaporates when"
V3: molecule becomes too susceptible evaporate. that is why researchers
    university cambridge study how contaminated water can become alcoholic
    when water is dried out.

V1: awareness dietary indicated merely biological, awareness gradually
    accuracy gradually gradually accuracy biological
```

V3 uses 3-10% primitive tokens in generation. V1 uses 0%.

## How it works

```python
from primoji import Tokenizer

tok = Tokenizer()

ids = tok.encode("The teacher explained photosynthesis")
print(tok.decode(ids))
# -> "teacher explained photosynthesis"

# Contractions expand semantically
ids = tok.encode("won't")  # -> [will, not] = 2 meaningful tokens

# Hyphens split into components
ids = tok.encode("insulin-deprived")  # -> [insulin, deprived] = 2 word tokens

print(tok.vocab_size)  # 10,428
```

## Vocabulary (10,428 tokens, v4)

| Tier | Count | Examples |
|------|-------|----------|
| Emoji | 1,175 | dog, cat, hospital, tree |
| Common words | 7,964 | government, propagation, susceptible, insulin |
| Primitives | 140 | PLANT, CAUSE, MOVE, KNOW, HEALTH, STUDY, ELECTRIC |
| Structural | ~890 | Flags, anchors, digits, punctuation, math |
| Byte fallback | 258 | UTF-8 bytes + boundary markers |

140 primitives: 65 Wierzbicka semantic primes (verified across 30+ language
families) + 75 domain expansions including HEALTH, STUDY, ELECTRIC, SUBSTANCE,
ENVIRONMENT, BODY_PART, VISIBLE, DEGREE.

## Interactive inference

```bash
python -m scripts.chat                                    # V1 model
python -m scripts.chat --model path/to/v3.pt --tiers      # V3 with tier embeddings
python -m scripts.chat --trace                            # show token tiers
python -m scripts.chat --temperature 0.5 --max-tokens 200
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
  preprocessor.py     Unicode norm, contraction expansion, hyphen splitting
  fuzzy.py            Conservative SymSpell (edit distance 1)
  math_handler.py     Single-digit numbers, math operators
scripts/
  build_dictionary.py   Reproducible dictionary build
  prepare_training_data.py  Tokenize FineWeb-Edu for training
  train_125m.py        125M GPT training with BPB evaluation
  compare_results.py   Training log analysis and plots
  chat.py              Interactive inference with --trace mode
tests/                 292 tests
data/
  primitives.json      140 compositional primitives (v0.3)
  emoji_catalog.json   1,175 emoji with CLDR annotations
  common_words.json    7,964 high-frequency words (wordfreq + FineWeb-Edu)
  auto_compositions.json  4,600+ WordNet-derived compositions
  dictionary_seed.json Symbolic dictionary (~42K entries)
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
