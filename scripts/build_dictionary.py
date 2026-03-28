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


# ── Primitive synonym table ──────────────────────────────────────────────────

_PRIMITIVE_SYNONYMS: dict[str, list[str]] = {
    "I": ["i", "me", "myself"], "YOU": ["you", "yourself"],
    "SOMEONE": ["someone", "somebody", "person"],
    "SOMETHING": ["something"], "PEOPLE": ["people", "persons", "folks"],
    "BODY": ["body"], "KIND": ["kind", "type", "sort", "category"],
    "THIS": ["this"], "SAME": ["same", "identical"],
    "OTHER": ["other", "another", "else"],
    "SOME": ["some"], "ALL": ["all", "every", "everything"],
    "ONE": ["one"], "TWO": ["two"],
    "MANY": ["many", "numerous"], "FEW": ["few"],
    "GOOD": ["good", "well", "fine", "positive"],
    "BAD": ["bad", "poor", "negative"],
    "BIG": ["big", "large", "great", "huge"],
    "SMALL": ["small", "little", "tiny", "minor"],
    "HERE": ["here"],
    "THINK": ["think", "thought"], "KNOW": ["know", "knowledge"],
    "WANT": ["want", "desire", "wish"],
    "FEEL": ["feel", "feeling", "emotion"],
    "SEE": ["see", "sight", "vision"], "HEAR": ["hear", "listen"],
    "SAY": ["say", "said", "tell", "speak", "told"],
    "WORDS": ["words", "word", "language"],
    "TRUE": ["true", "truth", "fact", "correct"],
    "DO": ["do", "doing", "action"],
    "HAPPEN": ["happen", "event", "occur"],
    "MOVE": ["move", "motion", "movement"],
    "TOUCH": ["touch", "contact"],
    "DONT_WANT": ["refuse", "reject"],
    "THERE_IS": ["exist", "existence"],
    "HAVE": ["have", "has", "possess"],
    "BE": ["be", "is", "are", "am", "was", "were"],
    "LIVE": ["live", "alive", "living", "life"],
    "DIE": ["die", "death", "dead"],
    "TIME": ["time"], "NOW": ["now", "currently", "present"],
    "BEFORE": ["before", "previously", "prior", "ago", "past"],
    "AFTER": ["after", "later", "next", "future"],
    "LONG_TIME": ["ages", "era", "epoch"],
    "SHORT_TIME": ["instant", "brief"],
    "FOR_SOME_TIME": ["continue", "ongoing", "duration"],
    "WHERE": ["where", "location", "place"],
    "NEAR": ["near", "nearby", "close"], "FAR": ["far", "distant", "remote"],
    "ABOVE": ["above", "over", "upper", "top"],
    "BELOW": ["below", "under", "beneath", "bottom"],
    "SIDE": ["side", "beside"],
    "NOT": ["not", "no", "never"],
    "MAYBE": ["maybe", "perhaps", "possibly"],
    "CAN": ["can", "able", "capable", "ability"],
    "BECAUSE": ["because", "since", "reason"],
    "IF": ["if", "whether"], "LIKE_AS": ["like", "as", "similar"],
    "VERY": ["very", "extremely", "really"],
    "MORE": ["more", "additional", "greater"],
    "MOMENT": ["moment"],
    "PART": ["part", "piece", "portion", "section"],
    "INSIDE": ["inside", "within", "interior"],
    "BE_SOMEWHERE": ["located", "situated"],
    "MATTER": ["matter", "substance", "material", "element"],
    "WATER": ["water", "liquid", "fluid"],
    "FIRE": ["fire", "flame", "burn"],
    "EARTH": ["earth", "ground", "soil", "land", "terrain"],
    "AIR": ["air", "atmosphere", "wind", "gas", "breeze"],
    "LIGHT": ["light", "bright", "brightness"],
    "DARK": ["dark", "darkness", "shadow", "dim"],
    "HARD": ["hard", "solid", "rigid", "rock"],
    "SOFT": ["soft", "flexible", "gentle"],
    "SHARP": ["sharp", "pointed"], "ROUND": ["round", "circular"],
    "FLAT": ["flat", "level", "surface"],
    "ANIMAL": ["animal", "creature", "beast", "fauna"],
    "PLANT": ["plant", "vegetation", "flora"],
    "EAT": ["eat", "consume", "nutrition"],
    "GROW": ["grow", "growth", "develop"],
    "CREATE": ["create", "creation", "build"],
    "DESTROY": ["destroy", "destruction", "demolish"],
    "YOUNG": ["young", "youth"], "OLD": ["old", "aged", "ancient", "elderly"],
    "NUMBER": ["number", "count"],
    "ADD": ["add", "addition", "plus", "sum"],
    "REMOVE": ["remove", "subtract"],
    "MULTIPLY": ["multiply", "multiplication"],
    "DIVIDE": ["divide", "division"],
    "EQUAL": ["equal", "equality", "balance"],
    "MEASURE": ["measure", "measurement"],
    "PATTERN": ["pattern", "regularity", "cycle"],
    "CHANGE": ["change", "transform", "alter"],
    "SET": ["set", "collection"],
    "LAW": ["law", "rule", "legal", "justice"],
    "SOCIETY": ["society", "community", "civilization", "social"],
    "CONFLICT": ["conflict", "war", "fight", "battle", "struggle"],
    "TRADE": ["trade", "commerce", "exchange", "business"],
    "TEACH": ["teach", "education", "instruct"],
    "HOME": ["home", "house", "dwelling", "shelter"],
    "PATH": ["path", "road", "route", "way", "trail"],
    "POWER": ["power", "force", "strength", "authority"],
    "NAME": ["name", "title"], "WRITE": ["write", "writing", "author"],
    "CAUSE": ["cause"], "RESULT": ["result", "outcome", "effect"],
    "BEGIN": ["begin", "beginning", "start"],
    "END": ["end", "ending", "finish", "stop"],
    "CONNECT": ["connect", "connection", "link", "join"],
    "ORDER": ["order", "sequence"], "SIMILAR": ["similar", "alike"],
    "DIFFERENT": ["different", "distinct", "contrast"],
    "WHOLE": ["whole", "complete", "entire", "total"],
    "EMPTY": ["empty", "void", "nothing", "zero", "blank"],
    "MACHINE": ["machine", "tool", "device"],
    "CONTAINER": ["container", "vessel"],
    "SEND": ["send", "transmit"], "RECEIVE": ["receive", "obtain", "get"],
    "IMAGE": ["image", "picture", "photo"],
    "HOLD": ["hold", "grasp", "grip"],
    "WITH": ["with", "together"], "FOR": ["for"],
    "ABOUT": ["about", "regarding", "concerning"],
    "ENERGY": ["energy"], "COLOR": ["color", "colour", "hue", "pigment"],
    "HOT": ["hot", "warm", "heated", "temperature"],
    "COLD": ["cold", "cool", "chilly", "freezing", "frozen"],
    "LINE": ["line"], "HEAVY": ["heavy", "weight", "massive"],
    "LOVE": ["love", "affection", "adore"],
    "FEAR": ["fear", "afraid", "scared", "dread", "terror"],
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
    "judge": ["SOMEONE", "LAW"],
    "doctor": ["SOMEONE", "LIVE", "GOOD"],
    "telephone": ["MACHINE", "SAY"],
    "television": ["MACHINE", "SEE"],
    # Function words (empty = drop)
    "the": [], "a": [], "an": [],
    "explained": ["SAY"],
}


def main() -> None:
    """Build the dictionary from data sources."""
    parser = argparse.ArgumentParser(description="Build Primoji dictionary")
    parser.add_argument("--catalog", default=str(_DATA_DIR / "emoji_catalog.json"))
    parser.add_argument("--primitives", default=str(_DATA_DIR / "primitives.json"))
    parser.add_argument("--anchors", default=str(_DATA_DIR / "proper_noun_anchors.json"))
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
    # When multiple emoji claim the same keyword, prefer the one whose
    # CLDR name matches. "dog" → 🐕 (name="dog"), not 🦴 (name="bone").
    l1_count = 0
    word_to_best_emoji: dict[str, tuple[str, bool]] = {}  # word → (emoji_char, is_name_match)
    for e in catalog["emoji"]:
        cldr_name = e["name"].lower()
        emoji_char = e["emoji"]
        for word in e.get("words", [e["name"]]):
            w = word.lower().strip()
            if not w or len(w) > 40:
                continue
            is_name = (w == cldr_name)
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

    # Layer 4: compositions
    l4_count = 0
    for word, names in _COMPOSITIONS.items():
        valid = all(n in prim_names for n in names)
        if not valid:
            bad = [n for n in names if n not in prim_names]
            print(f"  SKIP '{word}': unknown primitives {bad}", file=sys.stderr)
            continue
        entries[word] = [_prim_ref(n) for n in names] if names else []
        l4_count += 1
    print(f"Layer 4 (compositions):    {l4_count:6d}")

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


if __name__ == "__main__":
    main()
