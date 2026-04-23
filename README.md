# Primoji

A compositional semantic tokenizer for language models, built from Natural
Semantic Metalanguage (NSM) primitives, Unicode emoji, common word tokens,
and a UTF-8 byte fallback.

```python
from primoji import Tokenizer

tok = Tokenizer()
ids = tok.encode("The teacher explained photosynthesis")
tok.decode(ids)
# 'the teacher say plant have light'

tok.vocab_size
# 32098
```

## Install

```bash
pip install primoji
```

Requires Python 3.10+. Only runtime dependency is `emoji`. Optional extras
for the dictionary build pipeline (`spacy`, `wordfreq`), training scripts
(`torch`, `tokenizers`, `datasets`), or fuzzy matching (`symspellpy`):

```bash
pip install "primoji[build-dict]"   # to rebuild the dictionary from sources
pip install "primoji[train]"        # to run the training scripts
pip install "primoji[spelling]"     # to enable conservative SymSpell fuzzy matching
pip install "primoji[all]"          # everything (also includes dev tools)
```

## What it is

Primoji is a tokenizer that constructs token meanings compositionally. Instead
of discovering subword units from corpus statistics like BPE, it ships with a
fixed compositional vocabulary of ~32,000 tokens: 1,175 Unicode emoji,
140 NSM semantic primitives, ~29,500 common word tokens, ~900 structural tokens
(flags, anchors, punctuation, math, ordinals, abbreviations, possessives), and
258 byte-fallback tokens. Rare and technical words decompose into short
sequences of primitives (`[PLANT, HAVE, LIGHT]` for photosynthesis,
`[WATER, CAUSE, AIR]` for evaporation). The encode pipeline never produces
an UNK token.

At 125M parameters, Primoji achieves **1.083 BPB** vs BPE's **1.118 BPB**
on FineWeb-Edu (0.035 advantage). At 1B parameters, the gap widens:
Primoji **1.025 BPB** vs BPE **1.063 BPB** (0.038 advantage). Both
comparisons use matched training data (500K FineWeb-Edu documents, 1 epoch).

## Quick start

```python
from primoji import Tokenizer

tok = Tokenizer()

# Encode and decode
ids = tok.encode("Water evaporates when heated")
print(ids)
print(tok.decode(ids))

# Inspect a token
print(tok.describe(ids[0]))
# '... Tier 2 primitive: WATER'

# Classify a word by which pipeline stage handles it
print(tok.classify_word("photosynthesis"))   # 'dict_composed'
print(tok.classify_word("water"))            # 'tier2_primitive'
print(tok.classify_word("the"))              # 'tier1b_word'
print(tok.classify_word("Mediterranean"))    # 'tier3_anchor' or 'byte_fallback'

# Vocabulary size
print(tok.vocab_size)                        # 32098
```

## Vocabulary structure

Encoding pipeline with strict stage precedence. The model only ever sees
integer IDs; the tier labels exist for diagnostics and the dictionary build.

| Tier | Count | Examples |
|------|------:|----------|
| 1a Direct emoji              | ~1,175 | dog, house, sun |
| 1b Common word tokens        | ~29,500 | government, discover, present |
| 2  Compositional primitives  |    140 | PLANT, HAVE, LIGHT, MOVE, KNOW |
| 3  Structural and geographic |   ~900 | flags, punctuation, digits, ordinals, abbreviations, possessives, anchors |
| 4  UTF-8 byte fallback       |    258 | 256 byte tokens + 2 boundary markers |

The 140 primitives are 65 canonical NSM primes from
Wierzbicka's Natural Semantic Metalanguage plus 75 domain expansions
(perceptual, grammatical, scientific, social). Country flags act as geographic
modifiers in compositions. Anchors are the top ~500 proper nouns extracted
from FineWeb-Edu via spaCy NER.

The encoder pipeline processes each word through: structural tokens,
dictionary lookup, common word tokens, anchors, SymSpell fuzzy match,
morphological composition (negation, temporal, comparative, derivational
suffixes), and byte fallback. Contractions are expanded by the preprocessor.
Possessives, slashes, and hyphens are split.

116 closed-class words (determiners, prepositions, conjunctions, modals,
adverbs) have compositional alias embeddings: their model embeddings are
computed as the mean of their primitive components. This gives grammar words
semantic structure without dedicated primitive tokens.

## Training results (V8)

All models trained on 500K FineWeb-Edu documents, 1 epoch. Primoji uses
tier embeddings, compositional alias embeddings, and byte-weight 0.7.

| Model | Primoji BPB | BPE BPB | Gap |
|-------|-------------|---------|-----|
| 125M  | 1.083       | 1.118   | -0.035 |
| 1B    | 1.025       | 1.063   | -0.038 |

The Primoji advantage holds across scales. At 1B the gap widens slightly,
suggesting compositional structure helps more as models get larger.

Byte fallback rate: 3.8% of tokens by word count (words not in dictionary
fall back to UTF-8 byte encoding).

Compression ratio: 1.035x BPE (Primoji produces 3.5% more tokens than BPE
for the same text).

## Configuration

The dictionary, primitive set, and word list are all swappable. Rebuild
the dictionary from sources:

```bash
pip install "primoji[build-dict]"
python -m scripts.build_dictionary
```

This regenerates `data/dictionary_seed.json` from layered sources: Unicode
CLDR annotations, NSM primitive synonyms, NER anchors, WordNet
auto-compositions, and common word tokens.

## Training your own model

Reference training scripts for 125M and 1B GPT-style models live in
`scripts/`:

```bash
# Tokenize FineWeb-Edu (or any HuggingFace dataset)
python -m scripts.prepare_training_data --n-docs 500000

# Train 125M (1 epoch, Chinchilla-aware: set --max-steps)
python -m scripts.train --tokenizer primoji --v2 --byte-weight 0.7 \
  --max-steps 17235

# Train 1B with gradient accumulation
python -m scripts.train --tokenizer primoji --model-size 1b \
  --batch-size 4 --grad-accum 8 --v2 --byte-weight 0.7 --max-steps 17236

# Train BPE baseline on the same data
python -m scripts.train --tokenizer mistral --max-steps 20970
```

BPB is computed correctly across vocabulary sizes for fair cross-tokenizer
comparison.

## Limitations

- English only. The NSM primes are verified across 30+ language families,
  but the dictionary, word list, and evaluation are all English.
- Lossy at the word level. Known compositions decode exactly; novel
  compositions decode to their primitive names.
- Designed for educational and scientific text. Compression and learning
  benefits are strongest where vocabulary is compositional and concept-dense.
- Generation quality is sensitive to sampling configuration. Recommended:
  temp=0.9, top_p=0.9, rep_penalty=1.3.

## Tests

```bash
pip install "primoji[dev]"
pytest
```

615 tests covering encode/decode round-trips, byte-fallback coverage,
compositional embedding correctness, ID-range invariants, and frozen
vocabulary boundaries.

## Paper

Bandov, F. (2026). Primoji: Compositional Semantic Tokenization for Language
Model Training. Manuscript.

```bibtex
@unpublished{bandov2026primoji,
  title  = {Primoji: Compositional Semantic Tokenization for Language Model Training},
  author = {Bandov, Frane},
  year   = {2026},
}
```

## License

Apache 2.0. See `LICENSE`.

## Acknowledgments

Anna Wierzbicka and the wider Natural Semantic Metalanguage research community
for the linguistic foundation. The FineWeb-Edu team at HuggingFace for the
training corpus. The Unicode CLDR project for emoji annotations.
