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
