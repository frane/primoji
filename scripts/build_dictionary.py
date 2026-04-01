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
    # Function words: empty = drop (these carry minimal semantic content)
    "the": [], "a": [], "an": [], "of": [], "and": [], "to": [],
    "in": [], "on": [], "at": [], "or": [], "but": [], "so": [],
    "yet": [], "nor": [], "than": [], "then": [],
    "by": [], "from": [], "into": [],
    "over": [], "under": [], "among": [],
    "during": [], "until": [], "since": [], "while": [],
    "upon": [], "within": [], "without": [],
    "across": [], "along": [], "around": [],
    "behind": [], "beyond": [],
    "also": [], "however": [], "although": [], "though": [],
    "thus": [], "hence": [], "moreover": [], "furthermore": [],
    "meanwhile": [], "nevertheless": [], "indeed": [],
    "instead": [], "rather": [], "just": [],

    # Pronouns -> primitive mappings
    "it": ["SOMETHING"], "its": ["SOMETHING"],
    "they": ["PEOPLE"], "them": ["PEOPLE"], "their": ["PEOPLE"],
    "we": ["PEOPLE"], "us": ["PEOPLE"], "our": ["PEOPLE"],
    "he": ["SOMEONE"], "him": ["SOMEONE"], "his": ["SOMEONE"],
    "she": ["SOMEONE"], "her": ["SOMEONE"],
    "i": ["I"], "me": ["I"], "my": ["I"], "myself": ["I"],
    "you": ["YOU"], "your": ["YOU"], "yourself": ["YOU"],
    "this": ["THIS"], "these": ["THIS"],
    "that": ["THIS"], "those": ["OTHER"],
    "what": ["SOMETHING"], "which": ["SOMETHING"],
    "who": ["SOMEONE"], "whom": ["SOMEONE"],
    "there": ["THERE_IS"],

    # Prepositions -> primitive mappings where meaningful
    "after": ["AFTER"], "before": ["BEFORE"],
    "above": ["ABOVE"], "below": ["BELOW"],
    "beside": ["SIDE"], "near": ["NEAR"],

    # Common verbs (forms -> primitive)
    "is": ["BE"], "are": ["BE"], "was": ["BEFORE", "BE"],
    "were": ["BEFORE", "BE"], "be": ["BE"], "been": ["BE"],
    "being": ["BE"], "am": ["BE"],
    "has": ["HAVE"], "had": ["BEFORE", "HAVE"], "having": ["HAVE"],
    "does": ["DO"], "did": ["BEFORE", "DO"], "done": ["DO"],
    "go": ["MOVE"], "goes": ["MOVE"], "going": ["MOVE"],
    "went": ["BEFORE", "MOVE"], "gone": ["MOVE"],
    "come": ["MOVE"], "came": ["BEFORE", "MOVE"],
    "get": ["RECEIVE"], "got": ["BEFORE", "RECEIVE"],
    "gets": ["RECEIVE"], "getting": ["RECEIVE"],
    "take": ["HOLD"], "took": ["BEFORE", "HOLD"],
    "taken": ["HOLD"], "takes": ["HOLD"],
    "give": ["SEND"], "gave": ["BEFORE", "SEND"],
    "given": ["SEND"], "gives": ["SEND"],
    "put": ["MOVE"], "set": ["SET"],
    "use": ["DO"], "used": ["DO"], "using": ["DO"],
    "find": ["SEE"], "found": ["SEE"],
    "show": ["SEE"], "shown": ["SEE"], "shows": ["SEE"],
    "called": ["NAME"], "call": ["NAME"],
    "known": ["KNOW"], "knew": ["BEFORE", "KNOW"],
    "became": ["BEFORE", "CHANGE"], "become": ["CHANGE"],
    "thought": ["THINK"], "keep": ["HOLD"],
    "left": ["MOVE"], "need": ["WANT"],
    "try": ["DO"], "tried": ["DO"],
    "ask": ["SAY"], "asked": ["SAY"],
    "work": ["DO"], "worked": ["DO"], "working": ["DO"],
    "seem": ["BE"], "seemed": ["BEFORE", "BE"],
    "help": ["DO", "GOOD"], "include": ["INSIDE"],
    "turn": ["CHANGE"], "turned": ["CHANGE"],
    "run": ["MOVE"], "running": ["MOVE"],
    "look": ["SEE"], "looked": ["SEE"], "looking": ["SEE"],
    "tell": ["SAY"], "told": ["SAY"],
    "play": ["DO"], "played": ["DO"],
    "believe": ["THINK", "TRUE"],
    "provide": ["SEND"], "consider": ["THINK"],
    "appear": ["SEE"], "lead": ["CAUSE", "MOVE"],
    "stand": ["BE_SOMEWHERE"], "allow": ["CAN"],

    # Common adjectives
    "new": ["BEGIN"], "old": ["OLD"], "young": ["YOUNG"],
    "long": ["BIG", "TIME"], "short": ["SMALL"],
    "high": ["ABOVE", "BIG"], "low": ["BELOW", "SMALL"],
    "open": ["BEGIN"], "closed": ["END"],
    "right": ["TRUE"], "wrong": ["NOT", "TRUE"],
    "important": ["BIG", "GOOD"], "different": ["DIFFERENT"],
    "same": ["SAME"], "own": ["HAVE"],
    "other": ["OTHER"], "first": ["ONE", "BEGIN"],
    "last": ["ONE", "END"], "next": ["AFTER"],
    "early": ["BEFORE"], "late": ["AFTER"],
    "great": ["BIG", "GOOD"], "best": ["GOOD", "VERY"],
    "better": ["GOOD", "MORE"], "worst": ["BAD", "VERY"],
    "most": ["MANY", "VERY"], "much": ["BIG"],
    "such": ["LIKE_AS"], "each": ["ALL", "ONE"],
    "every": ["ALL"], "several": ["MANY"],
    "both": ["TWO", "ALL"], "between": ["TWO", "SIDE"],
    "among": ["MANY", "INSIDE"],

    # Common nouns
    "people": ["PEOPLE"], "man": ["SOMEONE"],
    "woman": ["SOMEONE"], "child": ["YOUNG"],
    "children": ["YOUNG", "MANY"],
    "world": ["EARTH", "ALL"], "country": ["EARTH", "SOCIETY"],
    "year": ["TIME"], "years": ["TIME", "MANY"],
    "day": ["TIME", "LIGHT"], "days": ["TIME", "MANY"],
    "way": ["PATH"], "part": ["PART"],
    "place": ["WHERE"], "case": ["SOMETHING"],
    "point": ["WHERE"], "group": ["SET"],
    "hand": ["HOLD"], "state": ["BE", "SOCIETY"],
    "area": ["WHERE", "BIG"], "system": ["ORDER", "MACHINE"],
    "fact": ["TRUE"], "number": ["NUMBER"],
    "example": ["ONE", "SEE"],
    "because": ["BECAUSE"], "therefore": ["BECAUSE"],
    "only": ["ONE"], "even": ["SAME"],
    "still": ["FOR_SOME_TIME"], "already": ["BEFORE"],
    "very": ["VERY"], "really": ["VERY"], "too": ["VERY"],
    "more": ["MORE"], "most": ["MANY", "VERY"],
    "enough": ["GOOD", "MEASURE"],
    "often": ["MANY", "TIME"], "always": ["ALL", "TIME"],
    "never": ["NOT", "TIME"], "sometimes": ["SOME", "TIME"],
    "usually": ["MANY", "TIME"],
    "today": ["NOW"], "now": ["NOW"],
    "back": ["BEFORE"], "again": ["PATTERN"],
    "up": ["ABOVE"], "down": ["BELOW"],
    "well": ["GOOD"], "away": ["FAR"],
    "thing": ["SOMETHING"], "things": ["SOMETHING", "MANY"],
    "said": ["SAY"], "explained": ["SAY"],
    "could": ["CAN", "BEFORE"], "would": ["WANT"],
    "should": ["GOOD"], "must": ["WANT", "VERY"],
    "will": ["AFTER", "DO"], "can": ["CAN"],
    "may": ["MAYBE"], "might": ["MAYBE"],
    "if": ["IF"], "when": ["TIME"], "where": ["WHERE"],
    "how": ["PATH"], "why": ["BECAUSE"],
    "not": ["NOT"], "no": ["NOT"],
    "some": ["SOME"], "any": ["SOME"],
    "many": ["MANY"], "few": ["FEW"],
    "all": ["ALL"],
    "about": ["ABOUT"], "with": ["WITH"], "for": ["FOR"],
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
                if existing == []:  # Don't override dropped words
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


if __name__ == "__main__":
    main()
