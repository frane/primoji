"""Build the Primoji dictionary from data sources.

Generates dictionary_seed.json with symbolic references (emoji chars, primitive
names, anchor names) instead of numeric IDs. The dictionary module resolves
symbols to IDs at load time, making the seed resilient to catalog re-sorts.

This is the single source of truth for dictionary builds. Idempotent.

Usage:
    python -m scripts.build_dictionary
    python scripts/build_dictionary.py --output code/data/dictionary_seed.json

Layers (later layers override earlier):
    1. Direct emoji words from emoji_catalog.json
    2. Primitive name synonyms (OVERRIDES Layer 1)
    3. Proper noun anchors
    4. Manual compositions for educational vocabulary
    5. Auto-inflections (never overrides)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).parent.parent
_DATA_DIR = _CODE_DIR / "data"


# ── Symbolic reference constructors ──────────────────────────────────────────

def _emoji_ref(char: str) -> dict[str, str]:
    return {"type": "emoji", "char": char}

def _prim_ref(name: str) -> dict[str, str]:
    return {"type": "primitive", "name": name}

def _anchor_ref(name: str) -> dict[str, str]:
    return {"type": "anchor", "name": name}

def _word_ref(word: str) -> dict[str, str]:
    return {"type": "word", "word": word}


# ── Primitive synonym table ──────────────────────────────────────────────────

_PRIMITIVE_SYNONYMS: dict[str, list[str]] = {
    # STRICT RULE: only map a word if the word's PRIMARY meaning IS the concept.
    # Grammar words (is, are, was, what, which, this, that, etc.) -> word tokens.
    # Concept words (water, fire, think, know, etc.) -> primitives.
    #
    # Pronouns, auxiliaries, prepositions, determiners are ALL word tokens.
    # Only nouns, verbs, adjectives whose meaning = the primitive get mapped.

    # Substantives: only "someone", "something", "people" (concept words)
    "SOMEONE": ["someone", "somebody", "person"],
    "SOMETHING": ["something"],
    "PEOPLE": ["people", "persons", "folks"],
    "BODY": ["body"],
    "KIND": ["kind", "type", "sort", "category"],

    # Evaluators/descriptors: adjectives whose meaning IS the concept
    "GOOD": ["good", "positive"],
    "BAD": ["bad", "negative"],
    "BIG": ["big", "large", "huge"],
    "SMALL": ["small", "tiny", "minor", "little"],

    # Mental verbs: the verb IS the concept
    "THINK": ["think", "thought"],
    "KNOW": ["know", "knowledge"],
    "WANT": ["want", "desire", "wish"],
    "FEEL": ["feel", "feeling", "emotion"],
    "SEE": ["see", "sight", "vision"],
    "HEAR": ["hear", "listen"],

    # Speech
    "SAY": ["say", "said", "tell", "speak", "told"],
    "WORDS": ["words", "word", "language"],
    "TRUE": ["true", "truth", "correct"],

    # Action verbs
    "HAPPEN": ["happen", "event", "occur"],
    "MOVE": ["move", "motion", "movement"],
    "TOUCH": ["touch", "contact"],
    "DONT_WANT": ["refuse", "reject"],

    # Existence (concept words, NOT grammar "is/are/was")
    "THERE_IS": ["exist", "existence"],
    "LIVE": ["live", "alive", "living", "life"],
    "DIE": ["die", "death", "dead"],

    # Time (concept words, NOT grammar "now/before/after")
    "TIME": ["time"],
    "LONG_TIME": ["ages", "era", "epoch"],
    "SHORT_TIME": ["instant", "brief"],
    "FOR_SOME_TIME": ["continue", "ongoing", "duration"],

    # Space (concept words only)
    "FAR": ["far", "distant", "remote"],
    "NEAR": ["near", "close", "nearby"],
    "BELOW": ["below", "beneath", "underneath"],
    "ABOVE": ["above", "overhead"],
    "WHERE": ["where", "location"],
    "SAME": ["same", "identical"],
    "BE_SOMEWHERE": ["located", "situated"],

    # Logic (concept words, NOT grammar "not/if/can")
    "MAYBE": ["maybe", "perhaps", "possibly"],

    # Intensifiers (concept words only)
    "MOMENT": ["moment"],
    "PART": ["part", "piece", "portion", "section"],
    "INSIDE": ["inside", "within", "interior"],

    # Physical (all concept words - these are clear)
    "MATTER": ["matter"],
    "WATER": ["water", "liquid", "fluid"],
    "FIRE": ["fire", "flame", "burn"],
    "EARTH": ["earth", "ground", "soil", "terrain"],
    "AIR": ["air", "atmosphere", "wind", "breeze"],
    "LIGHT": ["light", "bright", "brightness"],
    "DARK": ["dark", "darkness", "shadow"],
    "HARD": ["hard", "solid", "rigid", "rock"],
    "SOFT": ["soft", "flexible", "gentle"],
    "SHARP": ["sharp", "pointed"],
    "ROUND": ["round", "circular"],
    "FLAT": ["flat", "surface"],

    # Living
    "ANIMAL": ["animal", "creature", "beast", "fauna"],
    "PLANT": ["plant", "vegetation", "flora"],
    "EAT": ["eat", "consume", "nutrition"],
    "GROW": ["grow", "growth", "develop"],
    "CREATE": ["create", "creation", "build"],
    "DESTROY": ["destroy", "destruction", "demolish"],
    "YOUNG": ["young", "youth"],
    "OLD": ["old", "aged", "ancient", "elderly"],

    # Math
    "NUMBER": ["number", "count"],
    "ADD": ["add", "addition"],
    "REMOVE": ["remove", "subtract"],
    "MULTIPLY": ["multiply", "multiplication"],
    "DIVIDE": ["divide", "division"],
    "EQUAL": ["equal", "equality", "balance"],
    "MEASURE": ["measure", "measurement"],
    "PATTERN": ["pattern", "regularity", "cycle"],
    "CHANGE": ["change", "transform", "alter"],
    "SET": ["set", "collection"],

    # Social
    "LAW": ["law", "legal", "justice"],
    "SOCIETY": ["society", "community", "civilization"],
    "CONFLICT": ["conflict", "war", "battle", "struggle"],
    "TRADE": ["trade", "commerce", "exchange"],
    "TEACH": ["teach", "education", "instruct"],
    "HOME": ["home", "dwelling", "shelter"],
    "PATH": ["path", "route", "trail"],
    "POWER": ["power", "strength", "authority", "rule"],
    "NAME": ["name", "title"],
    "WRITE": ["write", "writing", "author"],

    # Abstract
    "CAUSE": ["cause"],
    "RESULT": ["result", "outcome", "effect"],
    "BEGIN": ["begin", "beginning", "start"],
    "END": ["ending", "finish", "terminate", "stop"],
    "CONNECT": ["connect", "connection", "link", "join"],
    "ORDER": ["order", "sequence"],
    "SIMILAR": ["similar", "alike"],
    "DIFFERENT": ["different", "distinct", "contrast"],
    "WHOLE": ["whole", "complete", "entire", "total"],
    "EMPTY": ["empty", "void", "nothing"],
    "MACHINE": ["machine", "tool", "device"],
    "CONTAINER": ["container", "vessel"],
    "SEND": ["send", "transmit"],
    "RECEIVE": ["receive", "obtain"],
    "IMAGE": ["image", "picture", "photo"],
    "HOLD": ["hold", "grasp", "grip"],

    # New primitives (v0.3)
    "ENERGY": ["energy"],
    "COLOR": ["color", "colour", "hue", "pigment"],
    "HOT": ["hot", "warm", "heated"],
    "COLD": ["cold", "chilly", "freezing", "frozen"],
    "LINE": ["line"],
    "HEAVY": ["heavy", "massive"],
    "LOVE": ["love", "affection", "adore"],
    "FEAR": ["fear", "afraid", "scared", "dread", "terror"],
    "HEALTH": ["health", "medical", "disease", "illness"],
    "SUBSTANCE": ["substance", "chemical", "compound"],
    "DEGREE": ["degree", "extent"],
    "ENVIRONMENT": ["environment", "ecology", "ecosystem"],
    "BODY_PART": ["organ", "tissue", "nerve", "muscle"],
    "VISIBLE": ["visible", "visual", "appearance"],
    "STUDY": ["research", "investigate", "examine", "study"],
    "ELECTRIC": ["electric", "electrical", "circuit", "voltage"],
}

# ── Composition table ────────────────────────────────────────────────────────

_COMPOSITIONS: dict[str, list[str]] = {
    "photosynthesis": ["PLANT", "HAVE", "LIGHT"],
    "evaporation": ["WATER", "CAUSE", "AIR"],
    "democracy": ["POWER", "PEOPLE", "EQUAL"],
    "gravity": ["EARTH", "POWER", "MOVE", "BELOW"],
    "chemical": ["MATTER", "CHANGE"],
    "economy": ["TRADE", "SOCIETY"],
    "species": ["KIND", "ANIMAL"],
    "habitat": ["HOME", "ANIMAL"],
    "climate": ["AIR", "TIME", "PATTERN"],
    "government": ["POWER", "SOCIETY", "LAW"],
    "temperature": ["HOT", "MEASURE"],
    "velocity": ["MOVE", "MEASURE"],
    "equation": ["NUMBER", "EQUAL"],
    "variable": ["NUMBER", "CHANGE"],
    "mountain": ["EARTH", "BIG", "ABOVE"],
    "ocean": ["WATER", "BIG"],
    "forest": ["PLANT", "MANY"],
    "desert": ["EARTH", "HOT", "EMPTY"],
    "volcano": ["EARTH", "FIRE", "BIG"],
    "earthquake": ["EARTH", "MOVE", "DESTROY"],
    "electricity": ["ENERGY", "MOVE"],
    "evolution": ["LIVE", "CHANGE", "LONG_TIME"],
    "extinction": ["DIE", "ALL"],
    "revolution": ["CHANGE", "POWER", "SOCIETY"],
    "constitution": ["LAW", "WHOLE", "WRITE"],
    "president": ["SOMEONE", "POWER", "SOCIETY"],
    "parliament": ["PEOPLE", "LAW", "SOCIETY"],
    "election": ["PEOPLE", "WANT", "SOMEONE"],
    "peace": ["NOT", "CONFLICT"],
    "science": ["KNOW", "TRUE", "MEASURE"],
    "mathematics": ["NUMBER", "PATTERN", "TRUE"],
    "literature": ["WRITE", "FEEL", "WORDS"],
    "university": ["TEACH", "BIG", "SOCIETY"],
    "student": ["SOMEONE", "TEACH"],
    "teacher": ["SOMEONE", "TEACH", "SAY"],
    "experiment": ["DO", "MAYBE", "KNOW"],
    "hypothesis": ["MAYBE", "TRUE"],
    "evidence": ["TRUE", "SEE"],
    "analysis": ["PART", "SEE"],
    "theory": ["THINK", "PATTERN"],
    "environment": ["EARTH", "AIR", "WATER", "LIVE"],
    "population": ["PEOPLE", "MANY", "WHERE"],
    "agriculture": ["PLANT", "CREATE", "EAT"],
    "industry": ["MACHINE", "CREATE", "MANY"],
    "technology": ["MACHINE", "KNOW"],
    "communication": ["SAY", "SEND"],
    "transportation": ["MOVE", "MACHINE"],
    "architecture": ["CREATE", "HOME", "BIG"],
    "philosophy": ["THINK", "KNOW", "WORDS"],
    "psychology": ["KNOW", "FEEL", "THINK"],
    "biology": ["KNOW", "LIVE"],
    "chemistry": ["KNOW", "MATTER", "CHANGE"],
    "physics": ["KNOW", "MOVE", "ENERGY"],
    "geography": ["KNOW", "EARTH", "WHERE"],
    "history": ["KNOW", "BEFORE", "HAPPEN"],
    "astronomy": ["KNOW", "FAR", "LIGHT"],
    "medicine": ["KNOW", "LIVE", "GOOD"],
    "engineering": ["CREATE", "MACHINE", "KNOW"],
    "computer": ["MACHINE", "THINK"],
    "internet": ["CONNECT", "THERE_IS"],
    "music": ["FEEL", "PATTERN"],
    "art": ["CREATE", "SEE", "FEEL"],
    "religion": ["FEEL", "THERE_IS", "BIG"],
    "writer": ["SOMEONE", "WRITE"],
    "scientist": ["SOMEONE", "KNOW", "MEASURE"],
    "farmer": ["SOMEONE", "PLANT", "EAT"],
    "soldier": ["SOMEONE", "CONFLICT"],
    "fighter": ["SOMEONE", "CONFLICT"],
    "judge": ["SOMEONE", "LAW"],
    "doctor": ["SOMEONE", "LIVE", "GOOD"],
    "telephone": ["MACHINE", "SAY"],
    "television": ["MACHINE", "SEE"],
    # SI unit abbreviations -> same primitives as their full words
    "km": ["MEASURE", "PATH", "BIG"],        # kilometer
    "cm": ["MEASURE", "PATH", "SMALL"],       # centimeter
    "mm": ["MEASURE", "PATH"],                # millimeter
    "m": ["MEASURE", "PATH"],                 # meter (context-dependent)
    "kg": ["MEASURE", "HEAVY", "BIG"],        # kilogram
    "g": ["MEASURE", "HEAVY"],                # gram
    "mg": ["MEASURE", "HEAVY", "SMALL"],      # milligram
    "ml": ["MEASURE", "WATER", "SMALL"],      # milliliter
    "l": ["MEASURE", "WATER"],                # liter
    "hz": ["MEASURE", "PATTERN"],             # hertz
    "khz": ["MEASURE", "PATTERN", "BIG"],     # kilohertz
    "mhz": ["MEASURE", "PATTERN", "BIG"],     # megahertz
    "ghz": ["MEASURE", "PATTERN", "BIG"],     # gigahertz
    "kw": ["MEASURE", "ENERGY", "BIG"],       # kilowatt
    "mw": ["MEASURE", "ENERGY"],              # megawatt
    "pa": ["MEASURE", "ENERGY"],              # pascal
    "kpa": ["MEASURE", "ENERGY", "BIG"],      # kilopascal
    "nm": ["MEASURE", "PATH"],                # nanometer
    "ph": ["MEASURE", "SUBSTANCE"],           # pH
    "db": ["MEASURE", "HEAR"],                # decibel
    "mph": ["MEASURE", "MOVE"],               # miles per hour
    "rpm": ["MEASURE", "PATTERN"],            # revolutions per minute
    "bpm": ["MEASURE", "PATTERN"],            # beats per minute
    "uv": ["LIGHT", "ENERGY"],               # ultraviolet
    "ir": ["LIGHT", "HOT"],                   # infrared
    "ac": ["ELECTRIC", "CHANGE"],             # alternating current
    "dc": ["ELECTRIC"],                       # direct current
    "dna": ["SUBSTANCE", "LIVE", "PATTERN"],  # DNA
    "rna": ["SUBSTANCE", "LIVE"],             # RNA
    "iv": ["SUBSTANCE", "INSIDE"],            # intravenous
    "ct": ["IMAGE", "INSIDE"],                # CT scan
    "mri": ["IMAGE", "INSIDE"],               # MRI

    # ALL grammar words are word tokens, NOT primitives.
    # Only technical/educational compositions remain here.
    "explained": ["SAY"],
    "element": ["MATTER", "PART"],
    "disability": ["NOT", "CAN"],
    "disabilities": ["NOT", "CAN"],
    "landing": ["MOVE", "EARTH"],
    "studied": ["STUDY", "BEFORE"],
    "studying": ["STUDY"],
    "ruled": ["POWER", "BEFORE"],
    "ruling": ["POWER"],
    "locations": ["WHERE"],
    "lands": ["EARTH"],
}


def main() -> None:
    """Build the dictionary from data sources."""
    parser = argparse.ArgumentParser(description="Build Primoji dictionary")
    parser.add_argument("--catalog", default=str(_DATA_DIR / "emoji_catalog.json"))
    parser.add_argument("--primitives", default=str(_DATA_DIR / "primitives.json"))
    parser.add_argument("--anchors", default=str(_DATA_DIR / "proper_noun_anchors.json"))
    parser.add_argument("--words", default=str(_DATA_DIR / "common_words.json"))
    parser.add_argument("--output", default=str(_DATA_DIR / "dictionary_seed.json"))
    args = parser.parse_args()

    with open(args.catalog) as f:
        catalog = json.load(f)
    with open(args.primitives) as f:
        prims = json.load(f)
    prim_names = {p["name"] for p in prims["primitives"]}

    anchors = None
    if Path(args.anchors).exists():
        with open(args.anchors) as f:
            anchors = json.load(f)

    entries: dict[str, list[dict]] = {}

    # Layer 1: emoji words (lowest priority)
    # Only map words that appear in the emoji's CLDR name (as the full name
    # or a word within it). Don't map arbitrary community keywords like
    # "hello" -> 👋 (waving hand) since those produce bad decode output.
    l1_count = 0
    word_to_best_emoji: dict[str, tuple[str, bool]] = {}
    for e in catalog["emoji"]:
        cldr_name = e["name"].lower()
        cldr_words = set(cldr_name.split())
        cldr_words.add(cldr_name)  # full name too
        emoji_char = e["emoji"]
        for word in e.get("words", [e["name"]]):
            w = word.lower().strip()
            if not w or len(w) > 40 or len(w) < 2:
                continue
            # Only accept if word is the CLDR name or a word within it
            is_name = (w == cldr_name)
            is_name_word = (w in cldr_words)
            if not is_name and not is_name_word:
                continue
            existing = word_to_best_emoji.get(w)
            if existing is None or (is_name and not existing[1]):
                word_to_best_emoji[w] = (emoji_char, is_name)

    for w, (emoji_char, _) in word_to_best_emoji.items():
        entries[w] = [_emoji_ref(emoji_char)]
        l1_count += 1
    print(f"Layer 1 (emoji words):     {l1_count:6d}")

    # Layer 2: primitives (overrides Layer 1)
    l2_count, l2_overrides = 0, 0
    for pname, words in _PRIMITIVE_SYNONYMS.items():
        if pname not in prim_names:
            continue
        ref = [_prim_ref(pname)]
        for w in words:
            if w in entries:
                l2_overrides += 1
            entries[w.lower()] = ref
            l2_count += 1
    print(f"Layer 2 (primitives):      {l2_count:6d} ({l2_overrides} overrides)")

    # Layer 3: anchors
    l3_count = 0
    if anchors:
        for a in anchors["anchors"]:
            ref = [_anchor_ref(a["name"])]
            entries[a["name"]] = ref
            lower = a["name"].lower()
            if lower != a["name"]:
                entries[lower] = ref
            l3_count += 1
    print(f"Layer 3 (anchors):         {l3_count:6d}")

    # Layer 4: compositions (manual + supplementary)
    l4_count = 0
    all_compositions = dict(_COMPOSITIONS)

    # Load supplementary compositions if available
    supp_path = _DATA_DIR / "additional_compositions.json"
    if supp_path.exists():
        with open(supp_path) as f:
            supplementary = json.load(f)
        all_compositions.update(supplementary)

    # Load auto-generated compositions (from WordNet)
    auto_path = _DATA_DIR / "auto_compositions.json"
    if auto_path.exists():
        with open(auto_path) as f:
            auto = json.load(f)
        auto_count = 0
        for word, prims in auto.items():
            if word.startswith("_") or not isinstance(prims, list):
                continue
            if word not in all_compositions:
                all_compositions[word] = prims
                auto_count += 1
        print(f"  (loaded {auto_count} auto-compositions from WordNet)")

    for word, names in all_compositions.items():
        valid = all(n in prim_names for n in names)
        if not valid:
            bad = [n for n in names if n not in prim_names]
            print(f"  SKIP '{word}': unknown primitives {bad}", file=sys.stderr)
            continue
        entries[word] = [_prim_ref(n) for n in names] if names else []
        l4_count += 1
    print(f"Layer 4 (compositions):    {l4_count:6d}")

    # Layer 4b: common word tokens (FINAL OVERRIDE, runs after compositions)
    # Makes "was" = 1 token, not [BEFORE, BE] = 2 tokens.
    # Makes "government" = 1 token, not [POWER, SOCIETY, LAW] = 3 tokens.
    l4b_count, l4b_overrides = 0, 0
    if Path(args.words).exists():
        with open(args.words) as f:
            word_data = json.load(f)
        for w in word_data["words"]:
            ref = [_word_ref(w)]
            if w in entries:
                existing = entries[w]
                # Don't override single-primitive mappings (keeps reverse lookup working)
                if len(existing) == 1 and existing[0].get("type") == "primitive":
                    continue
                if existing != ref:
                    l4b_overrides += 1
            entries[w] = ref
            l4b_count += 1
    print(f"Layer 4b (word tokens):    {l4b_count:6d} ({l4b_overrides} overrides)")

    # Layer 5: inflections (never overrides)
    l5_count = 0
    base = dict(entries)
    for word, refs in base.items():
        if " " in word or not refs:
            continue
        forms = []
        if len(word) > 2 and not word.endswith("s"):
            if word.endswith("y") and word[-2] not in "aeiou":
                forms.append(word[:-1] + "ies")
            elif word.endswith(("s", "sh", "ch", "x", "z")):
                forms.append(word + "es")
            else:
                forms.append(word + "s")
        if len(word) > 3:
            if word.endswith("e"):
                forms.extend([word + "d", word[:-1] + "ing"])
            elif word.endswith("y") and word[-2] not in "aeiou":
                forms.extend([word[:-1] + "ied", word + "ing"])
            else:
                forms.extend([word + "ed", word + "ing"])
        for form in forms:
            if form not in entries:
                entries[form] = refs
                l5_count += 1
    print(f"Layer 5 (inflections):     {l5_count:6d}")
    print(f"{'Total:':27s}{len(entries):6d}")

    output = {
        "version": "0.3.0",
        "format": "symbolic",
        "description": "Primoji seed dictionary with symbolic references",
        "total_entries": len(entries),
        "entries": entries,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.output}")

    # Mirror runtime data files into the package directory so they ship with the wheel.
    pkg_data_dir = _CODE_DIR / "primoji" / "data"
    pkg_data_dir.mkdir(parents=True, exist_ok=True)
    runtime_files = [
        "primitives.json",
        "common_words.json",
        "dictionary_seed.json",
        "emoji_catalog.json",
        "proper_noun_anchors.json",
    ]
    import shutil
    for fname in runtime_files:
        src = _DATA_DIR / fname
        dst = pkg_data_dir / fname
        if src.exists():
            shutil.copy2(src, dst)
    print(f"Mirrored runtime data files to {pkg_data_dir}")


if __name__ == "__main__":
    main()
