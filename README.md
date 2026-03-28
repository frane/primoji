# Primoji

**Compositional pictographic tokenizer for LLM training**

What if every token in your model's vocabulary actually *meant* something?

BPE chops "photosynthesis" into `["photo", "synth", "esis"]` — three fragments with no semantic content. Primoji encodes it as `[PLANT, HAVE, LIGHT]` — three tokens that carry the actual meaning. The model doesn't need to learn that "synth" relates to combining things; the structure is already there.

```
BPE:     "photosynthesis"  →  ["photo", "synth", "esis"]     (meaningless fragments)
Primoji: "photosynthesis"  →  [🌿, 📥, ☀️]                   (plant + absorb + light)
```

## How it works

```python
from primoji import Tokenizer

tok = Tokenizer()

ids = tok.encode("The teacher explained photosynthesis")
print(tok.decode(ids))
# → "teacher explained photosynthesis"

print(tok.vocab_size)  # ~2,400 tokens (vs 32K-100K for BPE)
```

Primoji uses a **4-tier fallback pipeline** — every word gets encoded, nothing is lost:

| Tier | What it handles | How |
|------|----------------|-----|
| **1. Dictionary** | Known words (~85%) | Emoji lookup table, O(1) |
| **2. Contractions** | "don't", "won't", etc. | Dedicated tokens or apostrophe split |
| **3. Fuzzy match** | Typos like "watir" | SymSpell, edit distance 1, conservative |
| **4. Byte fallback** | Everything else | UTF-8 bytes, zero information loss |

No UNK tokens. Ever. If the dictionary doesn't know a word, byte fallback encodes it losslessly — the same approach used by Llama's SentencePiece.

## The vocabulary (~2,400 tokens)

Three tiers of meaning, plus a universal safety net:

**Tier 1 — Unicode emoji (~1,200 tokens)**
Direct mappings for concrete nouns. Dog = 🐕. Tree = 🌳. Hospital = 🏥. One emoji, one token, one concept.

**Tier 2 — Compositional primitives (132 tokens)**
For everything emoji can't represent — verbs, adjectives, abstract concepts. Grounded in [Wierzbicka's Natural Semantic Metalanguage](https://en.wikipedia.org/wiki/Natural_semantic_metalanguage): 65 irreducible concepts verified across 30+ language families, plus 67 domain expansions for educational text.

Compositions follow positional rules: `HEAD + MODIFIER + SPECIFIER`
- 🧑 + 📚 + 💬 = teacher (person + teach + say)
- 💧 + ➜ + 💨 = evaporation (water + cause + air)
- 🔧 + 💭 = computer (machine + think)

**Tier 3 — Structural tokens (~600)**
Country flags, contraction tokens, proper noun anchors (top 500 from FineWeb-Edu), digits, math operators, punctuation.

**Tier 4 — Byte fallback (258 tokens)**
UTF-8 bytes wrapped in boundary markers. Handles any unknown word with zero information loss.

## Why this matters

A vocabulary of ~2,400 tokens (vs BPE's 32K-100K) has real consequences:

- **94% smaller embedding table** — BPE 32K x 4096 = 256 MB vs Primoji 2.4K x 1536 = 7 MB
- **Shorter sequences** — phrase-level composition means fewer tokens per sentence
- **Attention savings** — shorter sequences = quadratically less compute in self-attention
- **Semantic structure** — related concepts share compositional patterns that BPE can't capture

These claims are based on the [vocabulary bottleneck theorem](https://arxiv.org/abs/2106.07144) (Wies et al., 2021). Empirical validation on FineWeb-Edu is in progress.

## Current status

This is a **research prototype**. It works, the pipeline is complete, 300+ tests pass — but it hasn't been validated at scale yet. Here's what's done and what's ahead:

| Status | Milestone |
|--------|-----------|
| Done | Vocabulary design (132 primitives, 1,182 emoji, 500 anchors) |
| Done | 4-tier tokenization pipeline with byte fallback |
| Done | 20K+ word seed dictionary with compositions |
| Done | Reproducible build pipeline (`scripts/build_dictionary.py`) |
| Next | Compression benchmarks vs Mistral BPE on FineWeb-Edu |
| Next | Train 125M parameter model on primoji-tokenized data |
| Next | Compare learning curves against BPE baseline |
| Future | Multilingual extension, MoE routing via token semantics |

The approach is deliberately *lossy at the word level* — "photosynthesis" decodes as "plant absorb light", not the original word. Verbatim reconstruction via BPE sidecar metadata is architecturally supported but not yet implemented. For training data, semantic equivalence is sufficient; for inference, exact reconstruction will be needed.

## Installation

```bash
git clone https://github.com/frane/primoji.git
cd primoji
pip install -e ".[dev]"
```

Optional dependencies:
```bash
pip install primoji[spelling]   # SymSpell for Tier 3 fuzzy matching
pip install primoji[train]      # PyTorch + HuggingFace tokenizers for benchmarks
```

## Project structure

```
primoji/
  tokenizer.py        4-tier encode/decode pipeline
  preprocessor.py     Unicode normalization, contraction splitting
  byte_fallback.py    Tier 4 UTF-8 byte encoding
  fuzzy.py            Tier 3 conservative spell correction
  vocabulary.py       All token ID ranges (dynamically computed)
  composer.py         Composition rules (HEAD + MODIFIER + SPECIFIER)
  dictionary.py       Word → token ID lookup (symbolic seed + resolver)
  primitives.py       132 compositional primitives from data/primitives.json
  decoder.py          Tier-based decoding (catalog → primitives → structural → bytes)
  math_handler.py     Single-digit number and math operator tokenization
scripts/
  build_dictionary.py  Reproducible dictionary build from data sources
tests/                 300+ pytest tests
data/                  Emoji catalog, primitives, anchors, seed dictionary
```

## Paper

> **Primoji: Compositional Pictographic Tokenization for Efficient LLM Training**
> Frane Bandov (ESCP Business School)
> *Manuscript in preparation*

This is Paper 2 in a research portfolio on efficient LLM training. Paper 1 covers distributed training (Distrain), Paper 3 combines this tokenizer with MoE + BitNet routing, Paper 4 is the capstone integration.

## Citation

```bibtex
@article{bandov2026primoji,
  title   = {Primoji: Compositional Pictographic Tokenization for Efficient LLM Training},
  author  = {Bandov, Frane},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

## License

Apache 2.0
