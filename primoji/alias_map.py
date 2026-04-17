"""Grammar word alias map for compositional embeddings (V8).

Maps grammar word token IDs to their primitive component IDs.
"is" (word token) -> [BE primitive ID, NOW primitive ID]
"was" (word token) -> [BE primitive ID, BEFORE primitive ID]

The model uses these to compute grammar word embeddings as the mean
of their primitive component embeddings. This gives grammar words
semantic structure without consuming them as primitives in running text.

V8 changes: removed pronouns (NSM's pronoun primitives too coarse),
expanded to ~250 closed-class words covering determiners, prepositions,
conjunctions, auxiliary/modal verbs, and degree/frequency adverbs.
"""

from __future__ import annotations

from primoji.primitives import get_primitive_by_name

# Grammar word -> primitive decomposition (by name)
# NO PRONOUNS (I, me, you, he, she, it, we, they, etc.)
GRAMMAR_ALIASES: dict[str, list[str]] = {
    # ── Copula / be verbs ────────────────────────────────────────────
    "is": ["BE", "NOW"], "are": ["BE", "NOW"], "am": ["BE", "NOW"],
    "was": ["BE", "BEFORE"], "were": ["BE", "BEFORE"],
    "be": ["BE"], "been": ["BE", "BEFORE"], "being": ["BE"],

    # ── Have verbs ───────────────────────────────────────────────────
    "has": ["HAVE", "NOW"], "have": ["HAVE"], "had": ["HAVE", "BEFORE"],

    # ── Do verbs ─────────────────────────────────────────────────────
    "do": ["DO"], "does": ["DO", "NOW"], "did": ["DO", "BEFORE"],

    # ── Modal / auxiliary verbs ──────────────────────────────────────
    "can": ["CAN"], "could": ["CAN", "BEFORE"],
    "will": ["AFTER", "WANT"], "would": ["WANT", "BEFORE"],
    "shall": ["AFTER", "GOOD"], "should": ["GOOD", "DO"],
    "may": ["MAYBE"], "might": ["MAYBE", "BEFORE"],
    "must": ["WANT", "VERY"],

    # ── Determiners ──────────────────────────────────────────────────
    "the": ["THIS"],
    "a": ["ONE", "OTHER"], "an": ["ONE", "OTHER"],
    "this": ["THIS"], "that": ["OTHER"],
    "these": ["THIS", "MANY"], "those": ["OTHER", "MANY"],
    "each": ["ALL", "ONE"], "every": ["ALL"],
    "some": ["SOME"], "any": ["SOME"],
    "no": ["NOT"],

    # ── Quantifiers ──────────────────────────────────────────────────
    "all": ["ALL"], "many": ["MANY"], "few": ["FEW"],
    "much": ["MANY"], "more": ["MORE"], "most": ["MANY", "VERY"],

    # ── Prepositions ─────────────────────────────────────────────────
    "in": ["INSIDE"], "on": ["ABOVE", "TOUCH"], "at": ["WHERE", "ONE"],
    "by": ["NEAR", "SIDE"], "from": ["MOVE", "BEFORE"],
    "to": ["MOVE", "AFTER"], "with": ["WITH"], "for": ["FOR"],
    "of": ["PART"], "about": ["ABOUT"],
    "into": ["MOVE", "INSIDE"], "onto": ["MOVE", "ABOVE"],
    "through": ["INSIDE", "MOVE"], "between": ["SIDE", "SIDE"],
    "among": ["INSIDE", "MANY"], "above": ["ABOVE"], "below": ["BELOW"],
    "under": ["BELOW"], "over": ["ABOVE", "MOVE"],
    "near": ["NEAR"], "beside": ["NEAR", "SIDE"],
    "behind": ["AFTER", "SIDE"], "before": ["BEFORE"],
    "after": ["AFTER"], "during": ["INSIDE", "TIME"],
    "until": ["END", "TIME"], "since": ["BEGIN", "TIME"],
    "toward": ["MOVE", "NEAR"], "towards": ["MOVE", "NEAR"],
    "against": ["NOT", "MOVE"], "across": ["SIDE", "MOVE", "OTHER"],
    "along": ["SIDE", "MOVE"], "around": ["SIDE", "MANY"],
    "beyond": ["FAR", "MORE"], "within": ["INSIDE"],
    "without": ["NOT", "HAVE"], "upon": ["ABOVE", "TOUCH"],

    # ── Conjunctions ─────────────────────────────────────────────────
    "and": ["ADD"], "but": ["NOT", "SAME"],
    "or": ["IF", "OTHER"], "nor": ["NOT", "OTHER"],
    "so": ["CAUSE", "AFTER"], "yet": ["NOT", "BEFORE"],
    "although": ["NOT", "BECAUSE"], "because": ["BECAUSE"],
    "since": ["BEGIN", "TIME"],  # already above, same decomposition
    "while": ["SAME", "TIME"], "when": ["TIME"],
    "where": ["WHERE"], "if": ["IF"],
    "unless": ["IF", "NOT"], "until": ["END", "TIME"],
    "though": ["NOT", "BECAUSE"],
    "whereas": ["OTHER", "SAME", "TIME"],
    "whether": ["IF", "OTHER"],

    # ── Negation ─────────────────────────────────────────────────────
    "not": ["NOT"], "never": ["NOT", "TIME"],

    # ── Adverbs of degree / frequency ────────────────────────────────
    "very": ["VERY"], "really": ["TRUE", "VERY"],
    "quite": ["SOME", "VERY"], "rather": ["MORE", "WANT"],
    "too": ["VERY", "MORE"],
    "also": ["ADD"], "always": ["ALL", "TIME"],
    "often": ["MANY", "TIME"], "sometimes": ["SOME", "TIME"],
    "usually": ["MANY", "TIME"],
    "already": ["BEFORE", "NOW"], "still": ["FOR_SOME_TIME"],
    "just": ["NOW", "NEAR"], "even": ["SAME", "MORE"],
    "only": ["ONE"], "almost": ["NEAR", "ALL"],
    "nearly": ["NEAR", "NOT"],
    "now": ["NOW"], "here": ["HERE"], "there": ["THERE_IS"],
}


def build_alias_map(encode_fn: callable) -> dict[int, list[int]]:
    """Convert GRAMMAR_ALIASES to token ID -> primitive ID list.

    Args:
        encode_fn: function that takes a word and returns token IDs.

    Returns:
        Dict mapping word token ID -> list of primitive token IDs.
    """
    alias_map: dict[int, list[int]] = {}

    for word, prim_names in GRAMMAR_ALIASES.items():
        word_ids = encode_fn(word)
        if len(word_ids) != 1:
            continue  # skip if word doesn't encode to single token

        tok_id = word_ids[0]
        prim_ids = []
        for pname in prim_names:
            p = get_primitive_by_name(pname)
            if p is not None:
                prim_ids.append(p.id)

        if prim_ids:
            alias_map[tok_id] = prim_ids

    return alias_map
