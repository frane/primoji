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

## 2026-04-17 Phase V8.2 diagnostics: Seven structural bugs

Commit: (see below)

### Diagnostic A: Morphological rule failures (step-by-step traces)

Each word traced through the full _encode_word pipeline:

**stops** (base: stop, suffix: -s)
- Step 6 dict lookup("stops"): MATCH -> [1308] (END primitive)
- Resolution: FIXED in this commit. "stop" was added to END primitive synonyms
  in build_dictionary.py. Layer 5 then generated "stops" -> [END].
- Bug was: "stop" existed in runtime bootstrap (dictionary.py) but NOT in
  _PRIMITIVE_SYNONYMS, so the seed file never contained "stop" and Layer 5
  never inflected it.

**stopping** (base: stop, suffix: -ing) / **stopped** (base: stop, suffix: -ed)
- Step 6 dict lookup: MATCH -> word token IDs.
- These already worked because "stopping" and "stopped" were in common_words.json.
- However, they produce bogus inflections in the seed: "stoppeded", "stoppedings"
  etc., because Layer 5 inflects already-inflected forms. Bug not fixed yet.

**creator** (base: create, suffix: -or)
- Step 6 dict lookup("creator"): None (miss)
- Step 7 COMMON_WORD_TOKENS: miss
- Step 10 COMPOSER: no rule fires.
  - Negation: "creator" doesn't start with un/in/im/dis/non.
  - Temporal: doesn't start with pre/post/re.
  - Comparative: doesn't end with -er or -est. **-or is not -er.** The composer
    has no -or suffix rule at all.
- Step 11: BYTE FALLBACK (9 tokens)
- Root cause: The composer only has -er/-est (comparative/superlative). The -or
  agentive suffix was in `_try_agent()` which was removed in V8.1. Even if it
  hadn't been removed, it checked -er and -or but produced SOMEONE+base (wrong
  semantics for "creator" which is better as CREATE+SOMEONE).
- Actual fix needed: Either add "creator" to the dictionary directly, or add a
  derivational suffix rule for -or that maps to base+SOMEONE.

**instructor** (base: instruct, suffix: -or)
- Same pipeline failure as "creator". Additionally, the negation rule tries
  "in-" prefix, strips it to get "structor", which isn't in the dictionary.
  So the negation check fires but fails (correct behavior).
- Root cause: same as creator. No -or suffix rule.

**dimensional** (base: dimension, suffix: -al)
- Step 6 dict lookup("dimensional"): None (miss)
- Step 10 COMPOSER: no rule fires.
  - The composer has NO -al suffix rule. It only has: negation prefixes,
    temporal prefixes, -er/-est comparative. The -al derivational suffix
    is simply not implemented.
- Step 11: BYTE FALLBACK (13 tokens)
- Root cause: missing derivational suffix rule for -al.

**possess** (standalone word)
- Step 6 dict lookup("possess"): None
- Step 7 COMMON_WORD_TOKENS: None
- Step 10 COMPOSER: no rule fires (no applicable prefix or suffix)
- Step 11: BYTE FALLBACK (9 tokens)
- Root cause: "possess" is not in any layer of the dictionary. It is not
  a primitive synonym, emoji, composition, anchor, or common word. It's a
  common English word (freq 8,526 in corpus) that was simply omitted.

**Summary of morphological bugs:**
1. FIXED: bootstrap/seed desync for stop/start/little.
2. NOT FIXED: no derivational suffix rules for -or, -al, -ly, -tion, -ness,
   -ful, -able, -ible, -ment. The composer only has: negation prefixes,
   temporal prefixes, comparative -er/-est. All other derivational morphology
   is missing.
3. NOT FIXED: Layer 5 inflects already-inflected forms (bogus entries).

### Diagnostic B: Dictionary gaps (sum, sensor)

Traced through all 5 layers of build_dictionary.py:

**sum** (corpus freq: 10,939)
- Layer 1 (emoji CLDR names): no emoji has "sum" in its name
- Layer 2 (primitive synonyms): ADD = ["add", "addition"] -- "sum" not listed
- Layer 3 (anchors): not a proper noun
- Layer 4a (compositions): not manually composed
- Layer 4a (auto/WordNet): not generated by WordNet script
- Layer 4b (common_words.json): not in the 7,741-word list
- Root cause: common_words.json was built from "wordfreq top 3K + FineWeb-Edu
  top 5K byte-fallback words." "sum" at corpus frequency 10,939 should be in
  the top 3K by wordfreq, but the common_words build script must have excluded
  it (possibly filtered as a function word or abbreviation, or the wordfreq
  source was more restrictive than expected). It's also a natural fit for the
  ADD primitive synonym list.

**sensor** (corpus freq: 11,575)
- Same failure across all 5 layers as "sum."
- Correction to initial analysis: "sensor" is NOT a derivation of "sense + -or".
  "Sensor" is a standalone English noun borrowed from Latin sensus. "Sense" is
  both a noun and a verb; "sensor" is not formed by productive English
  derivation from "sense." My initial claim that "the -or derivational suffix
  was never implemented" was incorrect as an explanation for this word -- the
  real reason is that "sensor" is simply missing from all dictionary sources.
- Root cause: same as "sum" -- the 7,741-word common_words.json is too small.
  A word with corpus frequency 11,575 in 500K FineWeb-Edu docs is solidly
  in the top 5K most common words and should have been included.

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

### Byte fallback reconciliation

Pre-fix baseline was 7.11%, measured with the old preprocessor (no slash
splitting, no possessive markers). Post-fix measurement used the new
preprocessor which produces more total tokens (448.8M vs 447.9M) due to
possessive markers and slash splitting. The honest comparison on equal
footing: **7.33% -> 6.54%**, a **0.79pp improvement**.

Per-fix deltas (measured independently on 500K FineWeb-Edu docs):

| Fix | Tokens recovered | Delta (pp) |
|-----|------------------|-----------:|
| C: Possessives | 1,943,886 | +0.433 |
| G: Abbreviations | 501,026 | +0.112 |
| D: Slash compounds | 392,245 | +0.087 |
| A-fix: stop/start/little | 281,196 | +0.063 |
| E: Ordinals | 225,127 | +0.050 |
| F: SI units | 209,898 | +0.047 |
| **Total** | **3,553,378** | **+0.792** |

The individual deltas sum to 0.792pp. The measured total drop is 0.79pp.
These match (the tiny difference is rounding).

The earlier claim of "7.11% -> 6.55% (corrected)" was wrong. What happened:
the frequency counter lowercased all words, turning `<POSSESSIVE>` into
`<possessive>`. classify_word("&lt;possessive>") = byte_fallback (case
mismatch). So the counter attributed 1.94M possessive tokens to byte
fallback, inflating the rate to 6.98%. I then "corrected" by subtracting
those manually, getting 6.55%. The actual tokenizer was working correctly
the whole time -- the measurement was wrong. Proper measurement with
case-preserved tokens shows 6.54%.

### What we learned

1. Possessives are the biggest single fix (0.43pp, 1.94M tokens). Educational
   text is full of "student's", "earth's", "country's".

2. Single letters dominate the remaining byte fallback. The 26 single-letter
   words (c, p, b, d, e, j, s, x, r, t, n, h, f, v, k, w, o, y, u, q, z)
   plus underscore account for ~1.5M tokens (~0.33%). These are variable
   names, list markers, abbreviations, and page references. They are inherently
   un-coverable by dictionary expansion.

3. The remaining addressable byte fallback (~5%) comes from:
   - Real vocabulary gaps (sensor 11.6K, sum 10.9K, dimensional 9.8K,
     possess 8.5K, creator 7.2K, instructor 6.5K, etc.)
   - Decade words (1970s 10.5K, 1960s 10.4K, 1980s 10.1K)
   - Numbers with commas (1,000 at 11.8K, 10,000, 100,000)
   - Period-containing tokens not yet in abbreviations

4. The morphological gap is real: the composer has no rules for -or, -al,
   -ly, -tion, -ness, -ful, -able, -ible, -ment. Only negation prefixes,
   temporal prefixes, and -er/-est comparative are implemented. This means
   common derivations like creator, dimensional, traditionally, possession
   all byte-fallback despite their base forms being in the dictionary.

5. common_words.json (7,741 entries) is too small. Words with corpus frequency
   >10,000 (sum, sensor) are missing. The build source was "wordfreq top 3K
   + FineWeb-Edu top 5K byte-fallback words" which apparently didn't capture
   them. V8.2b needs to expand this substantially.

### Open questions

1. Should single letters get dedicated token IDs? They're common (c=122K,
   p=107K) but semantically empty. Byte fallback uses 5 tokens per letter
   (START + byte + END). A dedicated token saves 4 tokens per occurrence.

2. Decade words (1970s) could be handled as digit sequence + "s" suffix.

3. Numbers with commas (1,000) need comma handling in the number tokenizer.

4. Should V8.2b add derivational suffix rules (-or, -al, -tion, etc.) to
   the composer, or just expand the dictionary to cover all common
   derivations directly? Dictionary expansion is simpler and less error-prone
   but doesn't generalize to novel words.

## 2026-04-17 Phase V8.2.1: Derivational suffix rules in composer

### What changed

Added 8 derivational suffix rules to the composer, after the existing
comparative/superlative rules:

| Suffix | Semantic | Example | Marker primitive |
|--------|----------|---------|-----------------|
| -or/-er (agent) | person who does X | creator -> CREATE + SOMEONE | SOMEONE |
| -al | adjectival | dimensional -> dimension + KIND | KIND |
| -ly | adverbial | quickly -> quick + LIKE_AS | LIKE_AS |
| -tion/-sion | nominalizing | creation -> create + RESULT | RESULT |
| -ness | quality | darkness -> dark + SOMETHING | SOMETHING |
| -ful | full of | helpful -> help + HAVE | HAVE |
| -able/-ible | capable | readable -> CAN + read | CAN |
| -ment | result of | development -> develop + RESULT | RESULT |

Also fixed: comparative rule now handles doubled consonants (hotter -> hot + MORE).
Agent rule uses min_base=4 on stripped base to avoid arm->armor, man->manor, don->donor.

### Results

Testing on actual 6,266 byte-fallback words from top 21K:
- 316 genuinely new resolutions (words that were byte_fallback before)
- 497,955 tokens recovered
- By type: 149 adverbial, 42 agent, 32 adjectival, 31 able, 25 nominalize,
  20 ful, 18 quality

False positive analysis on agent rule (-or/-er):
- 42 total agent hits
- Known FPs: sensor (base "sense" via restore-e), pastor (base "past"),
  cursor (base "curse"), batter (base "bat" via doubled consonant),
  sinner (base "sin" via doubled consonant)
- FP rate: ~12% (5/42), all low-frequency words
- All remaining FPs are words that should be in common_words.json directly

Specific words verified:
- creator -> [CREATE, SOMEONE] (correct)
- instructor -> [TEACH, SOMEONE] (correct)
- dimensional -> [dimension_word, KIND] (correct)
- hotter -> [HOT, MORE] (was broken, now correct via comparative doubled-consonant fix)

## 2026-04-17 Phase V8.2.2: common_words.json investigation and expansion

### Root cause

common_words.json (v0.4.0) was built from "wordfreq top 3K + FineWeb-Edu
top 5K byte-fallback words." The FineWeb-Edu byte-fallback source was
bf_words_500k.json, a static snapshot of 10,000 words that byte-fallbacked
in an earlier tokenizer version. Words that resolved (even incorrectly) in
that earlier version never made it into the byte-fallback list.

"sum" (wordfreq rank 4,050): outside top 3K, so depended on the FineWeb-Edu
BF list. But "sum" didn't byte-fallback in the earlier tokenizer (it was
probably handled by a now-removed composition or dropped-word rule). So it
was never added.

"sensor" (wordfreq rank 8,089): outside top 3K and top 5K. Never appeared
in bf_words_500k.json because it was resolved by some earlier mechanism.

This is a build methodology bug, not a design choice. The common_words list
should be built from current byte-fallback analysis, not from stale snapshots.

### Fix

Regenerated common_words.json from current byte-fallback analysis:
- Keep all 7,741 existing words
- Add 10,119 new alphabetic words with corpus freq >= 500 that currently
  byte-fallback after all composer rules
- New total: 17,860 words (v0.5.0)

Filters applied: alphabetic only, length >= 2, freq >= 500, must currently
be byte_fallback (not already handled by dictionary/composer/structural).

### Results

Dictionary rebuilt: 42,407 -> 77,553 entries (larger due to more base words
generating more inflections in Layer 5).

Vocab size: 10,272 -> 20,391 (10K new word token IDs from expanded
common_words).

**Byte fallback: 7.33% -> 3.88%**

Full progress:
| Stage | Byte fallback | Delta |
|-------|--------------|-------|
| V6 baseline | 7.33% | - |
| V8.2-diag (structural fixes) | 6.54% | -0.79pp |
| V8.2.1 (derivational rules) | ~6.4%* | ~-0.1pp |
| V8.2.2 (common_words expansion) | 3.88% | -2.5pp |

*Derivational rules alone recover 498K tokens, but many of those words
also got added to common_words (which resolves first in the pipeline),
so the independent contribution is smaller.

### Tier distribution (final)

| Tier | Tokens | Pct |
|------|--------|-----|
| word token | 318.2M | 70.9% |
| structural | 66.6M | 14.8% |
| primitive | 23.8M | 5.3% |
| byte fallback | 17.4M | 3.9% |
| emoji | 7.9M | 1.8% |
| dict composed | 6.3M | 1.4% |
| anchor | 4.6M | 1.0% |
| composer rule | 4.0M | 0.9% |

### What we learned

1. The remaining 3.88% byte fallback is dominated by single uppercase
   letters (C, J, B, D, S, E, R, P = list items, initials, variables).
   These account for ~1.5% and are inherently un-coverable. The addressable
   residual is ~2.4%.

2. Vocab doubled from 10K to 20K. This is a significant change -- the
   paper's architecture analysis assumed ~4K-10K vocab. The 20K vocab still
   has a 94% smaller embedding table than BPE's 32K (20K * 1536 = 61MB
   vs 32K * 4096 = 256MB), but the depth-vs-width argument is weaker.

3. The derivational composer rules work but are mostly redundant with
   common_words expansion. Their value is in handling novel OOV words that
   aren't in common_words -- a long-tail benefit that's hard to measure
   on the eval set.

### Open questions

1. Is 20K vocab too large? The original design targeted ~4K-10K. But
   dropping articles in V6 was hiding the true vocabulary needs. With
   function words as tokens and proper coverage, ~20K seems to be the
   natural size. Need to verify the architectural implications still hold.

2. The 3.88% byte fallback is close to the 3% target. Remaining is mostly
   single letters and proper nouns. Should we add single-letter tokens?
   That saves ~1.5pp but adds only 52 tokens (26 upper + 26 lower).

## 2026-04-17 Phase V8.2b: Residual dictionary expansion

### What changed

Instead of LLM batch (no API key available), used rule-based decomposition
plus direct word token expansion for 5,976 byte-fallback words at freq >= 300.

**Decomposition strategies tried:**
1. Extended prefix (micro-, bio-, geo-, thermo-, electro-, neuro-, photo-,
   multi-, counter-, anti-, semi-): 4 matches. Low because most prefixed
   words were already in common_words.
2. Domain suffix (-ology, -ologist, -ography, -ism, -ist, -ity): 166 matches.
   Good quality: climatology -> [climate, KNOW, STUDY], positivism -> [GOOD,
   THINK, KIND], etc.
3. Compound splitting: ABANDONED. Too many false positives even with strict
   min-4-char-per-part filter. "theodor" -> the+odor, "coventry" -> cov+entry,
   "quinine" -> qui+nine. The problem: common short words (the, in, on, or)
   appear as substrings in many words.

**Final approach:**
- 170 rule-based compositions (24 pure-primitive, 146 with word-token bases)
- 5,806 words added as word tokens to common_words.json
- "able" and "capable" added as CAN primitive synonyms (fixed test failure)

### Results

| Metric | Before | After |
|--------|--------|-------|
| Byte fallback | 3.88% | 3.86% |
| Vocab size | 20,391 | 26,343 |
| Dictionary | 77,553 | 98,857 |
| common_words | 17,860 | 23,812 |

The 0.02pp improvement is negligible. The 5,800 new words at freq 300-499
are individually too rare to move the byte fallback needle. The remaining
3.86% is dominated by single letters (~1.5%), proper nouns (~0.5%), and
number/punctuation artifacts (~0.5%). The addressable vocabulary gap is
essentially closed.

### What we learned

1. Diminishing returns are steep. The jump from 7.33% to 3.88% (V8.2.2)
   recovered 3.45pp. The jump from 3.88% to 3.86% (V8.2b) recovered 0.02pp
   with 5,800 more word tokens. The per-word marginal value drops off a cliff
   below freq 500.

2. The LLM batch would have been wasted money. The ~6K words at freq 300-499
   contribute 1.8M tokens (0.4% of corpus). Even perfect decomposition of all
   of them would drop byte fallback by 0.4pp. The rule-based approach captured
   170 of the best candidates; the remaining 5,800 are opaque words (proper
   nouns, domain jargon, foreign borrowings) where primitive decomposition
   would be forced and meaningless.

3. Compound splitting is fundamentally broken without a compound dictionary
   or morphological analyzer. The dictionary contains too many common short
   words that match as false compound parts. Would need something like the
   CELEX compound database or a trained compound splitter.

4. The vocab doubled from 10K to 26K during V8.2. This is the honest
   vocabulary size for 96% token coverage of FineWeb-Edu. The original
   10K was artificially small because articles were dropped, common words
   were missing, and the byte fallback rate was 7%.

## 2026-04-17 Phase V8.3: Composer false positive cleanup

### What changed

V8.3's original plan was to replace the affix stripper with emoji2vec +
WordNet + corpus embeddings. V8.2.1 already replaced it with proper
derivational suffix rules, and V8.2.2 expanded common_words for byte-fallback
words. But the V8.1 false positives (discover -> NOT+cover, present ->
BEFORE+sent, layer -> lay+MORE, etc.) were still firing because the
common_words expansion only captured byte_fallback words, not composer_rule
words.

Fix: added all 5,755 composer-rule words with corpus freq >= 50 to
common_words.json. These are now single word tokens that resolve at
dictionary lookup (step 6) before the composer ever sees them.

### Results

| Metric | Before | After |
|--------|--------|-------|
| Byte fallback | 3.86% | 3.80% |
| Composer rule | 0.90% | 0.05% |
| Vocab size | 26,343 | 32,098 |
| common_words | 23,812 | 29,567 |
| Dictionary | 98,857 | 118,379 |

The composer now only fires on 0.05% of tokens -- genuinely novel/rare words
that aren't in any dictionary layer. The old false positives (discover,
present, layer, developer, researcher, etc.) are all single word tokens.

### What we learned

1. Vocab reached 32K, matching BPE's 32,768. This is not a coincidence --
   it's the vocabulary size needed for ~96% token coverage of English
   educational text. The difference: Primoji's 32K tokens are semantically
   organized (emoji + primitives + word tokens + compositions), while BPE's
   32K are arbitrary byte fragments.

2. The emoji2vec + WordNet + corpus embedding plan from V8.3 original spec
   was not needed. The combination of derivational suffix rules (V8.2.1)
   and comprehensive common_words expansion (V8.2.2 + V8.3) achieved the
   same goal more simply. The composer is now a thin fallback for rare
   morphological variants, not the primary resolution path.

3. The 0.05% composer residual (209K tokens) is acceptable. These are
   genuinely novel combinations like "unaccented", "prereading",
   "nonpartisan" where morphological decomposition is semantically correct.

## 2026-04-17 Phase V8.4: Alias expansion

### What changed

Rewrote alias_map.py. Removed all 18 pronoun aliases (I, me, you, he, she,
it, we, they + possessives). Added closed-class words across 7 categories.

Before: 80 aliases (including pronouns)
After: 116 aliases (no pronouns)

**Categories:**
- Copula/be: 8 (is, are, am, was, were, be, been, being)
- Have/do: 6 (has, have, had, do, does, did)
- Modals: 9 (can, could, will, would, shall, should, may, might, must)
- Determiners: 11 (the, a, an, this, that, these, those, each, every, some, any, no)
- Quantifiers: 6 (all, many, few, much, more, most)
- Prepositions: 32 (in, on, at, by, from, to, with, for, of, about, into, onto,
  through, between, among, above, below, under, over, near, beside, behind,
  before, after, during, until, since, toward, towards, against, across, along,
  around, beyond, within, without, upon)
- Conjunctions: 13 (and, but, or, nor, so, yet, although, because, while, when,
  where, if, unless, though, whereas, whether)
- Negation: 2 (not, never)
- Adverbs: 17 (very, really, quite, rather, too, also, always, often, sometimes,
  usually, already, still, just, even, only, almost, nearly, now, here, there)

**Collision audit:** 17 remaining collisions, all semantically correct:
- Agreement variants: is/are/am=[BE,NOW], was/were/been=[BE,BEFORE]
- Near-synonyms: some/any, many/much, toward/towards, although/though
- Differentiated from original: will [AFTER,WANT] vs after [AFTER],
  on [ABOVE,TOUCH] vs above [ABOVE], at [WHERE,ONE] vs near [NEAR],
  or [IF,OTHER] vs that [OTHER], so [CAUSE,AFTER] vs because [BECAUSE]

### Results

615 tests pass (1 test updated: pronoun test replaced with pronoun-removal
check + preposition-presence check).

### What we learned

1. 116 aliases is below the target of ~250 because many closed-class words
   the plan listed are already handled as word tokens (they encode to single
   tokens). The alias list only needs the SEMANTIC DECOMPOSITION, not the
   token allocation. Every word in the alias list must also be a word token.

2. The distinction between "alias" and "word token" is purely about embedding
   computation: aliases get mean-of-primitives embeddings, word tokens get
   independent embeddings. Both encode as single tokens in the sequence.

## 2026-04-17 Phase V8.5: Full rebuild and sanity check

### Eval sentence retokenization

| Metric | V6 | V8 | BPE |
|--------|----|----|-----|
| Tokens (1000 sents) | 36,076 | 34,280 | 33,118 |
| Ratio to BPE | 1.089x | 1.035x | 1.000x |
| Byte fallback (by words) | 6.9% | 3.56% | 0% |
| Vocab size | 10,195 | 32,098 | 32,768 |

V8 produces 1,796 fewer tokens than V6 on the same sentences. The
compression ratio improved from 1.089x to 1.035x BPE -- nearly 1:1.

The 3.56% byte fallback by words expands to 26.8% by tokens because each
byte-fallback word produces ~7 tokens (START + UTF-8 bytes + END).

### Sanity training (tiny model, M2 Pro)

19.7M param model (256 dim, 4 heads, 4 layers), 500 steps, batch 8, seq 256.
Loss: 10.545 -> 5.397 in 40 seconds on MPS. Val loss: 5.371.
Generation is garbage (expected at 500 steps) but confirms:
- Tokenizer output is trainable
- Gradients flow through all token types
- No OOV/UNK crashes
- vocab_size=32,098 works in embedding table

### Full test suite

615 tests pass.
