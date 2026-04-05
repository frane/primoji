"""Grammar word alias map for compositional embeddings.

Maps grammar word token IDs to their primitive component IDs.
"is" (word token) -> [BE primitive ID, NOW primitive ID]
"was" (word token) -> [BE primitive ID, BEFORE primitive ID]

The model uses these to compute grammar word embeddings as the mean
of their primitive component embeddings. This gives grammar words
semantic structure without consuming them as primitives in running text.
"""

from __future__ import annotations

from primoji.primitives import get_primitive_by_name

# Grammar word -> primitive decomposition (by name)
GRAMMAR_ALIASES: dict[str, list[str]] = {
    # Copula/be verbs
    "is": ["BE", "NOW"], "are": ["BE", "NOW"], "am": ["BE", "NOW"],
    "was": ["BE", "BEFORE"], "were": ["BE", "BEFORE"],
    "be": ["BE"], "been": ["BE", "BEFORE"], "being": ["BE"],

    # Have verbs
    "has": ["HAVE", "NOW"], "have": ["HAVE"], "had": ["HAVE", "BEFORE"],

    # Do verbs
    "do": ["DO"], "does": ["DO", "NOW"], "did": ["DO", "BEFORE"],

    # Modals
    "can": ["CAN"], "could": ["CAN", "BEFORE"],
    "will": ["AFTER"], "would": ["WANT", "BEFORE"],
    "should": ["GOOD", "DO"], "may": ["MAYBE"],
    "might": ["MAYBE", "BEFORE"], "must": ["WANT", "VERY"],
    "shall": ["AFTER"],

    # Negation
    "not": ["NOT"], "no": ["NOT"], "never": ["NOT", "TIME"],

    # Pronouns
    "i": ["SOMEONE", "THIS"], "me": ["SOMEONE", "THIS"],
    "my": ["SOMEONE", "THIS"],
    "you": ["SOMEONE", "OTHER"], "your": ["SOMEONE", "OTHER"],
    "he": ["SOMEONE"], "him": ["SOMEONE"], "his": ["SOMEONE"],
    "she": ["SOMEONE"], "her": ["SOMEONE"],
    "it": ["SOMETHING"], "its": ["SOMETHING"],
    "we": ["SOMEONE", "THIS", "MANY"], "us": ["SOMEONE", "THIS", "MANY"],
    "our": ["SOMEONE", "THIS", "MANY"],
    "they": ["SOMEONE", "OTHER", "MANY"], "them": ["SOMEONE", "OTHER", "MANY"],
    "their": ["SOMEONE", "OTHER", "MANY"],

    # Determiners
    "this": ["THIS"], "that": ["OTHER"],
    "these": ["THIS", "MANY"], "those": ["OTHER", "MANY"],
    "all": ["ALL"], "every": ["ALL"],
    "some": ["SOME"], "each": ["ALL", "ONE"],
    "any": ["SOME"], "many": ["MANY"], "few": ["FEW"],
    "much": ["BIG"], "more": ["MORE"], "most": ["MANY", "VERY"],

    # Prepositions
    "with": ["WITH"], "for": ["FOR"], "about": ["ABOUT"],
    "above": ["ABOVE"], "below": ["BELOW"],
    "near": ["NEAR"], "far": ["FAR"],
    "before": ["BEFORE"], "after": ["AFTER"],
    "here": ["HERE"], "there": ["THERE_IS"], "where": ["WHERE"],

    # Conjunctions/logic
    "if": ["IF"], "because": ["BECAUSE"],
    "like": ["LIKE_AS"], "as": ["LIKE_AS"],

    # Adverbs
    "very": ["VERY"], "now": ["NOW"],
    "also": ["ADD"],
    "always": ["ALL", "TIME"], "sometimes": ["SOME", "TIME"],
    "often": ["MANY", "TIME"], "usually": ["MANY", "TIME"],
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
