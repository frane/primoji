# Primoji V8 and V9 Development Diary

Branch: v8
Started: 2026-04-17

## 2026-04-17 Phase V8.1: Code and dictionary cleanup

Commit: (see below)

### What changed

**a) Removed 27 dead contraction tokens.**
The preprocessor expands all contractions ("don't" -> ["do", "not"]) before the
tokenizer pipeline runs. The 27 contraction token IDs (1591-1617) were allocated
in the frozen layout but never produced by any code path. Removed:
- `_CONTRACTION_LIST`, `CONTRACTION_TOKENS`, `DEDICATED_CONTRACTIONS`,
  `CONTRACTION_SUFFIXES` from vocabulary.py
- Contraction check from tokenizer._encode_word() and classify_word()
- Contraction decode from decoder.py
- All test references to CONTRACTION_TOKENS

The 27 ID slots remain allocated in the frozen layout (`contractions_count = 27`
in utils.py) to preserve downstream ID offsets. They are dead/unused IDs.

Note for paper: Tier 3 claims "~830 structural tokens" which included the 27
contraction tokens. Actual structural token count is now 27 lower. Will need
to update the paper's vocabulary breakdown table.

**b) Removed 50 empty dictionary entries.**
These were function words (the, a, an, of, in, on, at, by, from, etc.)
mapped to `[]` in the dictionary, causing them to be silently dropped (zero
tokens produced). Every one of these words is high-frequency in FineWeb-Edu.

Removed from:
- `_COMPOSITIONS` dict in build_dictionary.py (the source of truth)
- `_BOOTSTRAP` dict in dictionary.py (the, a, an)
- Removed the `if existing == []: continue` guard in Layer 4b of build_dictionary.py

Added 10 function words to common_words.json that weren't already there:
a, over, under, since, within, hence, moreover, furthermore, meanwhile,
nevertheless. Without this, they'd fall to byte fallback.

After rebuild: 42,372 entries (was 42,267), 0 empty entries.

The test `test_articles_produce_nothing` (asserting `tok.encode("the") == []`)
was changed to `test_articles_are_word_tokens` (asserting single token output).

**c) Fixed pipeline ordering: fuzzy match now runs before composition.**
Old order: dictionary -> composition -> word tokens -> anchors -> fuzzy -> bytes
New order: dictionary -> word tokens -> anchors -> fuzzy -> composition -> bytes

This matches the paper's described pipeline (dictionary -> spell correct ->
composition -> bytes). No reason was found in the code for the old ordering.
The composer does its own redundant dictionary lookup internally, which is
harmless.

**d) Removed agent suffix rule from composer.**
The composer had 4 rule categories:
1. Negation prefixes (un-, in-, im-, dis-, non-, -less)
2. Temporal prefixes (pre-, post-, re-)
3. Comparative/superlative (-er, -est)
4. Agent suffixes (-er, -or -> SOMEONE + base)

Category 4 removed. Only 4 words in FineWeb-Edu eval set actually hit it
(acceptor, accessor, donor, survivor). The rule conflicted with comparative
-er because comparative was checked first, so "developer" became GROW+MORE
instead of SOMEONE+GROW. The 4 agent suffix words will now byte-fallback
until V8.3 adds proper runtime composition.

### Results

- **137 unique words** in 1000 FineWeb-Edu eval sentences hit the composer
  (after pipeline reorder; before V8.1 this number was higher because
  fuzzy match wasn't catching some words first)
- **Breakdown**: 56 negation, 38 temporal, 39 comparative/superlative, 4 agent
- **False positive rate is high** across all rule types:
  - Negation: discover -> NOT+cover, invest -> NOT+vest, intake -> NOT+take,
    imac -> NOT+ac, ingram -> NOT+gram, inflamed -> NOT+fire
  - Temporal: present -> BEFORE+sent, preserve -> BEFORE+serve,
    prefix -> BEFORE+fix, reed -> PATTERN+ed, recover -> PATTERN+cover
  - Comparative: easter -> SIDE+MORE, charter -> ?+MORE, pier -> ?+MORE,
    layer -> ?+MORE, developer -> GROW+MORE, researcher -> STUDY+MORE
- **615 tests pass** (same count as before; 2 tests updated for intentional
  V8 behavior changes)
- Dictionary: 42,372 entries, 0 empty
- Vocab size unchanged: 10,195 (word count went from 7731 to 7741)

### What we learned

1. The old composer is mostly a source of false positives. Out of 137 words
   that reach it in the eval set, a rough manual scan suggests ~40-50% are
   wrong decompositions. The rules match any word where the base (after
   stripping the prefix/suffix) happens to be in the dictionary, even when
   the word was never morphologically composed from that base.

2. The pipeline order mattered for classify_word consistency but didn't
   change encode output for the eval set. Every word that was previously
   resolved by fuzzy match still is, and every word that reached the
   composer still does. This is because the fuzzy matcher (edit distance 1,
   unique candidate, 4+ chars) is very conservative and doesn't overlap
   with the composer's input.

3. Dropping function words (the, a, of, in) was a serious semantic loss.
   "The cat sat on the mat" encoded identically to "cat sat mat" in V6.
   V8 now preserves all function words as word tokens. This will increase
   token count but gives the model actual grammatical signal.

4. The 27 dead contraction IDs are the first instance of dead ID slots in
   the frozen layout. The frozen layout comment says "DO NOT CHANGE" but
   V8 is a clean break. We could reclaim those 27 slots, but the cost
   (shifting all downstream IDs) isn't worth the 27 IDs saved.

### Open questions

1. Should V8.3 aim to fix the ~70 false-positive composer decompositions,
   or should V8.2's dictionary expansion simply cover those words as direct
   entries? If the expanded dictionary covers "discover", "present",
   "easter", etc., the composer never sees them and the false positives
   disappear naturally. The composer then only fires for truly novel OOV
   words.

2. The 10 newly added common_words (a, over, under, since, within, hence,
   moreover, furthermore, meanwhile, nevertheless) should have been there
   from the start. Are there other high-frequency words missing from
   common_words.json? V8.2's frequency analysis will answer this.

3. Dropping articles produced 0 tokens per word. Now they produce 1 token.
   This will change compression ratio. The V8.5 benchmark will measure the
   impact.

## 2026-04-17 Phase V8.2a: Word frequency extraction

### What changed

Extracted word frequencies from all 500K FineWeb-Edu docs (2.4 GB text).
Two passes: raw whitespace split, then proper preprocessor tokenization.

### Results

**Raw split (no preprocessing):**
- 386M tokens, 4.26M unique words
- 95% coverage at 37,507 unique words (min freq 300)

**After preprocessor (apostrophe normalization, contraction expansion, hyphen splitting):**
- 448M tokens, 3.05M unique words (16% more tokens from expansion, 28% fewer types from normalization)
- 95% coverage at 21,483 unique words (min freq 788)
- 99% coverage at 271,208 unique words (min freq 9)

**Current byte fallback rate: 7.11%** (target: < 3%)

**Top 10K preprocessed words:** 98.1% already in dictionary (193 byte fallback)
**Top 21K preprocessed words:** 70.8% already in dictionary (6,266 byte fallback)

The 6,266 byte-fallback words in the top 21K break down as:
- Real vocabulary gaps: 6,141 types (8.7M tokens)
- Possessive markers ('s): 15 types (1.6M tokens)
- Single characters: 26 types (1.4M tokens)
- Abbreviations: 84 types (339K tokens)

Top real vocabulary gaps by frequency:
  pp (32,898), and/or (30,032), 19th (20,030), 20th (17,895), km (17,719),
  cm (16,997), sensor (11,575), sum (10,939), 18th (10,781), 1970s (10,503),
  dimensional (9,789), mg (9,719), possess (8,526), stops (7,594),
  creator (7,192), instructor (6,465), microscope (5,593)

### What we learned

1. **The raw vs preprocessed gap was huge.** Initial analysis made the problem
   look 2x worse than it is. Curly apostrophes alone accounted for 884 unique
   "words" and 1.9M tokens that were actually just Unicode variants of known
   contractions. The preprocessor's apostrophe normalization already handles
   these, but the raw frequency counter didn't know that.

2. **Most "byte fallback" is not vocabulary gaps.** The top byte-fallback
   words are possessives (students', 's), single letters (p, b, j), ordinals
   (19th, 20th), SI units (km, cm, mg, kg), academic abbreviations (pp, cf),
   and slash compounds (and/or). These are structural/formatting issues, not
   semantic vocabulary gaps. Real missing words (sensor, sum, dimensional,
   creator) are a smaller fraction.

3. **The inflection system has gaps.** "stops" (freq 7,594) should be covered
   by the -s inflection rule in build_dictionary.py, since "stop" is in the
   dictionary. Similarly "creator" should be covered by -or, "dimensional"
   by -al. These are bugs in the inflection/composition rules, not vocabulary
   gaps.

4. **Seven categories of structural bugs identified:**
   A. Morphological rule failures (stops, creator, dimensional)
   B. Missing common words (sum, sensor)
   C. Possessive handling bug ('s, s')
   D. Slash-separated compounds (and/or)
   E. Ordinal numbers (19th, 20th, 1st)
   F. SI unit abbreviations (km, cm, kg, mg)
   G. Academic abbreviations (pp, cf., e.g., i.e.)
   Fixing these before setting a dictionary cutoff will give a much more
   accurate picture of what the actual vocabulary gap is.

### Open questions

1. After fixing A-G, what will the residual byte-fallback rate be?
   Hypothesis: these structural fixes will account for 40-60% of the
   current 7.11% byte fallback, leaving ~3-4% that needs dictionary
   expansion.

## 2026-04-17 Phase V8.2 diagnostics: Seven structural bugs fixed

Commit: (see below)

### Diagnostic findings

**A) Morphological rule failures.** Three bugs found:
1. Bootstrap/seed desync: "stop", "start", "little" existed in the runtime
   bootstrap (dictionary.py) but not in _PRIMITIVE_SYNONYMS (build_dictionary.py).
   Layer 5 inflection only operates on the seed, so "stops" was never generated.
   Fix: added "stop" to END synonyms, "start" to BEGIN, "little" to SMALL.
2. Missing derivational suffixes: Layer 5 only generates -s/-es/-ies/-ed/-d/-ing.
   Does NOT generate -or, -al, -ly, -tion, -ness, -ful, -able, -ible, -ment.
   "create" exists but "creator" doesn't. "dimension" exists but "dimensional"
   doesn't. NOT fixed here -- this is a V8.2b dictionary expansion issue.
3. Bogus inflections: "stoppeded", "stoppedings" generated because Layer 5
   inflects already-inflected forms. NOT fixed here -- tracked for V8.2b.

**B) Dictionary gaps (sum, sensor).** Pure misses. Neither word appears in any
dictionary layer. "sense" is in common_words but "sensor" (a derivation) was
never generated. "sum" is a common English word with no coverage at all.
NOT fixed here -- V8.2b will address via expanded dictionary.

**C) Possessive handling.** Preprocessor already split "'s" possessives into
[base, "'s"], but "'s" then byte-fallbacked. Fixed: preprocessor now emits
"<POSSESSIVE>" marker token. Also added plural possessive handling (workers'
-> [workers, <POSSESSIVE>]). Added POSSESSIVE as structural token ID 2202.

**D) Slash-separated compounds.** Added split_slash() to preprocessor.
"and/or" -> ["and", "or"]. URLs preserved (checks for "://").

**E) Ordinal numbers.** Added 38 specific ordinal tokens (1st-31st, 40th-100th
by 10s) as structural tokens. Higher ordinals (42nd, 1776th) decompose as
digit tokens + ORDINAL marker. ORDINAL marker at ID 2203.

**F) SI unit abbreviations.** Added 32 SI/scientific abbreviations to the
composition table in build_dictionary.py: km, cm, mm, kg, mg, ml, hz, khz,
mhz, ghz, kw, mw, pa, kpa, nm, ph, db, mph, rpm, bpm, uv, ir, ac, dc,
dna, rna, iv, ct, mri, and their primitive decompositions.

**G) Academic/English abbreviations.** Added 27 structural tokens for common
abbreviations: pp, cf, ibid, e.g, i.e, vs, viz, ff, vol, ed, approx, etc,
mr, mrs, ms, dr, jr, sr, st, ave, blvd, ph.d, a.m, p.m, a.d, b.c, d.c.

### What changed (code)

- utils.py: structural_count 84 -> 151 (67 new structural tokens)
- vocabulary.py: added POSSESSIVE_ID, ORDINAL_ID, ORDINAL_IDS (38),
  ABBREVIATION_IDS (27). Registered all in Vocabulary.__init__.
- tokenizer.py: _encode_word checks POSSESSIVE, abbreviations, ordinals
  before dictionary lookup. Added _try_ordinal() for higher ordinals.
  Updated classify_word to match.
- preprocessor.py: added split_slash(), possessive splitting (emits
  <POSSESSIVE> marker), plural possessive handling.
- decoder.py: added decode support for new structural tokens.
- build_dictionary.py: added "stop"->END, "start"->BEGIN, "little"->SMALL
  in primitive synonyms. Added 32 SI unit compositions.

### Results

Vocab size: 10,195 -> 10,272 (+77: 67 structural + 10 common_words from V8.1)
Dictionary: 42,372 -> 42,407 (+35 from SI units and new primitive synonyms)

**Byte fallback: 7.11% -> 6.55%** (corrected for possessive tokens being
miscounted as byte fallback in the frequency analysis)

The improvement is smaller than hypothesized (0.56pp vs expected 2-3pp).
Reason: the fixes address TYPE coverage (fewer unique words byte-fallback)
but the highest-TOKEN-COUNT byte-fallback sources are single letters (c, p,
b, d, e, j, s -- list items, variable names, page references) which account
for ~2% of tokens and are not addressable by dictionary expansion.

The remaining ~4.5% byte fallback that IS addressable comes from:
- Real vocabulary gaps (sensor, sum, dimensional, creator, possess, etc.)
- Decade words (1970s, 1960s, 1980s, 1990s)
- Numbers with commas (1,000, 10,000, 100,000)

These will be covered by V8.2b dictionary expansion.

### What we learned

1. The possessive fix alone recovered 1.94M tokens (0.43% of all tokens).
   Possessives are very common in FineWeb-Edu (educational text has lots of
   "student's", "earth's", "country's").

2. Single letters dominate byte fallback by token count. 26 single letters
   account for ~1.5M tokens combined. These are inherently un-coverable --
   they're variable names, list markers, abbreviations, and page references.
   They should stay as byte fallback.

3. Ordinals and SI units resolved cleanly but don't appear in the top
   byte-fallback words by token count because they're spread across many
   specific forms (19th, 20th, etc.) each with moderate frequency.

4. The slash fix helps "and/or" (30K tokens) but its impact is spread
   across many compound forms.

5. The real dictionary gap (vocab words like sensor, sum, creator) is
   smaller than initially estimated. Most byte-fallback tokens are
   structural artifacts, not vocabulary misses.

### Open questions

1. Should single letters get dedicated token IDs? They're common enough
   (c=122K, p=107K) but semantically empty. Current byte fallback handles
   them in 5 tokens (START + byte + END). A dedicated token would save
   4 tokens per occurrence. At 1.5M total occurrences, that's 6M tokens
   saved. Worth doing? Or noise?

2. Decade words (1970s, 1960s) could be handled as digit sequence + "s"
   suffix. Similar to ordinal handling. Not in scope for this commit but
   straightforward to add.

3. Numbers with commas (1,000) need comma handling in the number tokenizer.
   Currently the comma splits the number.
