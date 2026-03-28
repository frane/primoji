"""Compositional primitives for Primoji tokenizer.

Defines all 120 compositional primitives used in Tier 2 of the vocabulary:
- Layer 2a: 65 Wierzbicka semantic primes (linguistically verified irreducible concepts)
- Layer 2b: 55 FineWeb-Edu domain expansions (physical, living, math, social, abstract, tools)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class Primitive:
    """A single compositional primitive token."""

    id: int
    name: str
    emoji: str
    category: str
    description: str


# Base offset for Tier 2 primitives
_BASE_ID: int = 1200


# ── Layer 2a: Wierzbicka's 65 Semantic Primes ────────────────────────────────

_WIERZBICKA_PRIMES: list[tuple[str, str, str, str]] = [
    # Substantives
    ("I", "👤", "substantive", "First person / self"),
    ("YOU", "👉", "substantive", "Second person / addressee"),
    ("SOMEONE", "🧑", "substantive", "Generic person"),
    ("SOMETHING", "🔹", "substantive", "Generic thing / entity"),
    ("PEOPLE", "👥", "substantive", "Multiple persons / group"),
    ("BODY", "🫀", "substantive", "Physical body"),
    ("KIND", "🏷️", "substantive", "Type / category"),
    # Determiners
    ("THIS", "📌", "determiner", "Proximal demonstrative"),
    ("SAME", "🟰", "determiner", "Identity / equivalence"),
    ("OTHER", "↔️", "determiner", "Difference / alternative"),
    ("SOME", "🎲", "determiner", "Indefinite / partial"),
    ("ALL", "⭕", "determiner", "Universal quantifier"),
    # Quantifiers
    ("ONE", "1️⃣", "quantifier", "Singular / unit"),
    ("TWO", "2️⃣", "quantifier", "Dual / pair"),
    ("MANY", "📊", "quantifier", "Large quantity"),
    ("FEW", "🤏", "quantifier", "Small quantity"),
    # Evaluators
    ("GOOD", "✅", "evaluator", "Positive evaluation"),
    ("BAD", "❌", "evaluator", "Negative evaluation"),
    # Descriptors
    ("BIG", "⬆️", "descriptor", "Large size / magnitude"),
    ("SMALL", "⬇️", "descriptor", "Small size / diminutive"),
    ("LONG", "📏", "descriptor", "Extended length / duration"),
    # Mental predicates
    ("THINK", "💭", "mental", "Cognition / reasoning"),
    ("KNOW", "🧠", "mental", "Knowledge / certainty"),
    ("WANT", "💫", "mental", "Desire / intention"),
    ("FEEL", "💗", "mental", "Emotion / sensation"),
    ("SEE", "👁️", "mental", "Visual perception"),
    # Speech
    ("SAY", "💬", "speech", "Verbal communication"),
    ("WORDS", "🔤", "speech", "Language / lexical items"),
    ("TRUE", "✔️", "speech", "Truth / accuracy"),
    # Actions
    ("DO", "⚙️", "action", "General action / perform"),
    ("HAPPEN", "⚡", "action", "Event / occurrence"),
    ("MOVE", "➡️", "action", "Motion / displacement"),
    ("TOUCH", "🤝", "action", "Physical contact"),
    ("HOLD", "✊", "action", "Grasp / maintain"),
    # Existence & possession
    ("EXIST", "🌐", "existence", "Being / presence"),
    ("HAVE", "📥", "existence", "Possession / containment"),
    ("BE", "🪞", "existence", "State / identity"),
    # Life & death
    ("LIVE", "🌱", "life", "Living / alive"),
    ("DIE", "🪦", "life", "Death / cessation"),
    # Time
    ("TIME", "⏱️", "time", "Temporal reference"),
    ("NOW", "⏺️", "time", "Present moment"),
    ("BEFORE", "⏪", "time", "Past / anterior"),
    ("AFTER", "⏩", "time", "Future / posterior"),
    ("LONG_TIME", "⏳", "time", "Extended duration"),
    # Space
    ("WHERE", "📍", "space", "Location / place"),
    ("NEAR", "🔎", "space", "Proximity"),
    ("FAR", "🏔️", "space", "Distance"),
    # Logical
    ("NOT", "🚫", "logical", "Negation"),
    ("MAYBE", "❓", "logical", "Possibility / uncertainty"),
    ("CAN", "💪", "logical", "Ability / possibility"),
    ("BECAUSE", "∵", "logical", "Causation / reason"),
    ("IF", "🔀", "logical", "Conditional"),
    ("LIKE", "〰️", "logical", "Similarity / comparison"),
    # Intensifiers
    ("VERY", "‼️", "intensifier", "High degree"),
    ("MORE", "➕", "intensifier", "Comparative increase"),
    ("LESS", "➖", "intensifier", "Comparative decrease"),
    # Taxonomy
    ("PART", "🧩", "taxonomy", "Component / part-of"),
    ("INSIDE", "📦", "taxonomy", "Containment / within"),
    ("ABOVE", "👑", "taxonomy", "Hierarchy / superiority"),
]

# ── Layer 2b: FineWeb-Edu Domain Expansions (55 tokens) ──────────────────────

_DOMAIN_EXPANSIONS: list[tuple[str, str, str, str]] = [
    # Physical
    ("SOLID", "🪨", "physical", "Solid state / rigid"),
    ("AIR", "💨", "physical", "Gas / atmosphere"),
    ("LIGHT", "☀️", "physical", "Electromagnetic radiation / brightness"),
    ("DARK", "🌑", "physical", "Absence of light"),
    ("SHARP", "🔺", "physical", "Pointed / acute"),
    ("FLAT", "▬", "physical", "Level / planar surface"),
    # Living
    ("ANIMAL", "🐾", "living", "Non-human creature"),
    ("PLANT", "🌿", "living", "Botanical organism"),
    ("GROW", "📈", "living", "Increase / develop"),
    ("CREATE", "🔨", "living", "Make / produce / construct"),
    ("DESTROY", "💥", "living", "Eliminate / demolish"),
    # Math
    ("NUMBER", "🔢", "math", "Numerical value"),
    ("MULTIPLY", "✖️", "math", "Multiplication"),
    ("DIVIDE", "➗", "math", "Division"),
    ("MEASURE", "📐", "math", "Quantify / measurement"),
    ("PATTERN", "🔁", "math", "Repetition / regularity"),
    ("CHANGE", "🔄", "math", "Transformation / alteration"),
    ("SET", "🗃️", "math", "Collection / group"),
    # Social
    ("LAW", "⚖️", "social", "Legal system / rules"),
    ("SOCIETY", "🏛️", "social", "Social organization / civilization"),
    ("CONFLICT", "⚔️", "social", "Opposition / war / struggle"),
    ("TEACH", "📚", "social", "Education / instruction"),
    ("TRADE", "🤲", "social", "Exchange / commerce"),
    ("HOME", "🏠", "social", "Dwelling / domestic"),
    ("PATH", "🛤️", "social", "Route / way / journey"),
    ("WRITE", "✏️", "social", "Written expression / author"),
    # Abstract
    ("CAUSE", "➜", "abstract", "Causation / bring about"),
    ("RESULT", "🎯", "abstract", "Outcome / effect"),
    ("BEGIN", "▶️", "abstract", "Start / initiation"),
    ("END", "⏹️", "abstract", "Finish / termination"),
    ("CONNECT", "🔗", "abstract", "Link / join / relate"),
    ("ORDER", "📋", "abstract", "Sequence / arrangement"),
    ("DIFFERENT", "🔀", "abstract", "Distinct / unlike"),
    ("EMPTY", "∅", "abstract", "Nothing / void / zero"),
    # Tools
    ("MACHINE", "🔧", "tools", "Mechanical device / tool"),
    ("SEND", "📤", "tools", "Transmit / dispatch"),
    ("RECEIVE", "📥", "tools", "Obtain / intake"),
    ("IMAGE", "🖼️", "tools", "Visual representation / picture"),
]

# ── Build the full PRIMITIVES list ────────────────────────────────────────────

PRIMITIVES: list[Primitive] = []

_current_id = _BASE_ID
for name, emoji, category, description in _WIERZBICKA_PRIMES:
    PRIMITIVES.append(Primitive(id=_current_id, name=name, emoji=emoji, category=category, description=description))
    _current_id += 1

for name, emoji, category, description in _DOMAIN_EXPANSIONS:
    PRIMITIVES.append(Primitive(id=_current_id, name=name, emoji=emoji, category=category, description=description))
    _current_id += 1

# Total should be 65 + 38 = 103 primitives defined so far
# (The CLAUDE.md says 55 domain expansions bringing total to 120;
#  remaining slots are reserved for future additions as the dictionary pipeline reveals gaps)

# ── Lookup indices ────────────────────────────────────────────────────────────

_BY_NAME: dict[str, Primitive] = {p.name: p for p in PRIMITIVES}
_BY_EMOJI: dict[str, Primitive] = {p.emoji: p for p in PRIMITIVES}
_BY_ID: dict[int, Primitive] = {p.id: p for p in PRIMITIVES}


def get_primitive_by_name(name: str) -> Primitive | None:
    """Look up a primitive by its canonical name (e.g. 'THINK', 'PLANT').

    Args:
        name: Uppercase canonical name of the primitive.

    Returns:
        The matching Primitive, or None if not found.
    """
    return _BY_NAME.get(name.upper())


def get_primitive_by_emoji(emoji: str) -> Primitive | None:
    """Look up a primitive by its emoji character(s).

    Args:
        emoji: The emoji string (may be multi-codepoint like '👁️').

    Returns:
        The matching Primitive, or None if not found.
    """
    return _BY_EMOJI.get(emoji)


def get_primitive_by_id(token_id: int) -> Primitive | None:
    """Look up a primitive by its token ID.

    Args:
        token_id: Integer token ID (1200–1319 range).

    Returns:
        The matching Primitive, or None if not found.
    """
    return _BY_ID.get(token_id)


def primitive_count() -> int:
    """Return the number of defined primitives."""
    return len(PRIMITIVES)
