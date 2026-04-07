# Primoji

A compositional semantic tokenizer for language models, built from Natural
Semantic Metalanguage (NSM) primitives, Unicode emoji, common word tokens,
and a UTF-8 byte fallback.

```python
from primoji import Tokenizer

tok = Tokenizer()
ids = tok.encode("The teacher explained photosynthesis")
tok.decode(ids)
# 'teacher say photosynthesis'

tok.vocab_size
# 10195
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
fixed compositional vocabulary of about 10,000 tokens: 1,175 Unicode emoji,
140 NSM semantic primitives, ~7,700 common word tokens, ~830 structural and
geographic tokens (flags, anchors, punctuation, math), and 258 byte-fallback
tokens. Rare and technical words decompose into short sequences of primitives
(`[PLANT, HAVE, LIGHT]` for photosynthesis, `[WATER, CAUSE, AIR]` for
evaporation). The encode pipeline never produces an UNK token.

When you might want it: research on tokenization, controlled experiments
that need a vocabulary 3x smaller than BPE at near-parity compression,
multilingual experiments where the primitive set transfers across languages
by design, small-vocab architectures whose cost scales with vocabulary size,
or educational and scientific text where compositional technical vocabulary
is dense. It is not a drop-in BPE replacement for production English chatbots:
generation is lossy at the word level (decoded text reconstructs concepts but
not always exact surface forms), and creative or stylistic writing loses
distinctions like "melancholy" vs "sadness" that BPE preserves as distinct
tokens.

## Quick start

```python
from primoji import Tokenizer

tok = Tokenizer()

# Encode and decode
ids = tok.encode("Water evaporates when heated")
print(ids)
# [1266, ..., ...]
print(tok.decode(ids))

# Inspect a token
print(tok.describe(ids[0]))
# '💧 (ID 1266) — Tier 2 primitive: WATER — Liquid, fluid'

# Classify a word by which tier handles it
print(tok.classify_word("photosynthesis"))   # 'dict_composed'
print(tok.classify_word("water"))            # 'tier2_primitive'
print(tok.classify_word("the"))              # 'dict_dropped'
print(tok.classify_word("Mediterranean"))    # 'tier3_anchor' or 'byte_fallback'

# Vocabulary size
print(tok.vocab_size)                        # 10195
```

## Vocabulary structure

Four tiers with strict encoding precedence. The model only ever sees integer
IDs; the tier labels exist for diagnostics and the dictionary build.

| Tier | Count | Examples |
|------|------:|----------|
| 1a Direct emoji              | ~1,175 | 🐕 (dog), 🏠 (house), ☀️ (sun) |
| 1b Common word tokens        | ~7,700 | government, important, however |
| 2  Compositional primitives  |    140 | PLANT, HAVE, LIGHT, MOVE, KNOW |
| 3  Structural and geographic |   ~830 | 🇩🇪, 🇫🇷, punctuation, digits, math operators, NER anchors |
| 4  UTF-8 byte fallback       |    258 | 256 byte tokens + 2 boundary markers |

The 140 primitives are 65 canonical NSM primes from
Wierzbicka's Natural Semantic Metalanguage plus 75 domain expansions
(perceptual, grammatical, scientific, social). Country flags act as geographic
modifiers in compositions. Anchors are the top ~500 proper nouns extracted
from FineWeb-Edu via spaCy NER.

A note on emoji: the model never sees glyphs, only integer IDs. Tier 1a is
just a convenient source of ~1,200 visually distinct token slots for concrete
nouns. In generated output, emoji tokens decode to their English words
("dog", "house", "sun"), not to Unicode emoji.

## Configuration

The dictionary, primitive set, and word list are all swappable. Five tunable
dimensions:

1. **Vocabulary size.** From ~5,300 tokens (~6% composed text) to ~18,000
   tokens (~2% composed). Controls the trade-off between compression and
   semantic-structure signal.
2. **Composition depth.** Maximum number of primitives per concept. Default 5.
3. **Primitive count.** Add domain-specific primitives (e.g.\ HEALTH for
   medical text) without touching the rest of the system.
4. **Composition rate.** A consequence of the above three.
5. **Domain specialization.** A medical Primoji can have 3,000 medical
   compositions on top of 10,000 general words. The dictionary build pipeline
   accepts a target corpus and produces a corresponding word list.

Rebuild the dictionary from sources:

```bash
pip install "primoji[build-dict]"
python -m scripts.build_dictionary
```

This regenerates `data/dictionary_seed.json` from layered sources: Unicode
CLDR annotations, NSM primitive synonyms, NER anchors, WordNet
auto-compositions. Layer precedence is fixed: primitive synonyms override
emoji catalog entries, single-primitive word tokens override compositions to
preserve reverse lookup, and ELCo and emoji2vec mappings supplement CLDR base
entries.

## Training your own model

Reference training scripts for 125M and 1B GPT-style models live in
`scripts/`:

```bash
# Tokenize FineWeb-Edu shards (or any HuggingFace dataset)
python -m scripts.prepare_training_data --n-docs 500000

# Train 125M
python -m scripts.train --tokenizer primoji --v2 --byte-weight 0.7

# Train 1B with gradient accumulation
python -m scripts.train --tokenizer primoji --model-size 1b \
  --batch-size 4 --grad-accum 8 --v2 --byte-weight 0.7

# Train BPE baseline on the same data
python -m scripts.train --tokenizer mistral
```

Bits-per-byte (BPB) is computed correctly across vocabulary sizes for fair
cross-tokenizer comparison. See `scripts/train.py` for the full
hyperparameter list.

## Limitations

- English only. The NSM primes are verified across 30+ language families,
  but the dictionary, word list, and evaluation are all English. Multilingual
  evaluation is future work.
- Lossy at the word level. Known compositions decode exactly; novel
  compositions decode to their primitive names, which is sufficient for
  training-efficiency studies (BPB) but not for verbatim text generation
  without an auxiliary surface-form mechanism.
- Designed for educational and scientific text. Compression and learning
  benefits are strongest where vocabulary is compositional and
  concept-dense. Conversational, creative, or code-heavy text will not see
  the same gains.
- Not yet tested at scale beyond 1B parameters.

## Tests

```bash
pip install "primoji[dev]"
pytest
```

616 tests covering encode/decode round-trips, byte-fallback coverage,
compositional embedding correctness, ID-range invariants, and frozen
vocabulary boundaries.

## Citation

```bibtex
@article{bandov2026primoji,
  title  = {Primoji: Compositional Semantic Tokenization for Language Model Training},
  author = {Bandov, Frane},
  year   = {2026},
  note   = {arXiv preprint, ID forthcoming},
}
```

## License

Apache 2.0. See `LICENSE`.

## Acknowledgments

Anna Wierzbicka and the wider Natural Semantic Metalanguage research community
for the linguistic foundation. The FineWeb-Edu team at HuggingFace for the
training corpus. The Unicode CLDR project for emoji annotations.
