"""Grammar word alias map for compositional embeddings.

Maps grammar word token IDs to their primitive component IDs.
"is" (word token) -> [BE primitive ID]
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
    "is": ["BE"], "are": ["BE"], "am": ["BE"],
    "was": ["BE", "BEFORE"], "were": ["BE", "BEFORE"],
    "be": ["BE"], "been": ["BE", "BEFORE"], "being": ["BE"],

    # Have verbs
    "has": ["HAVE"], "have": ["HAVE"], "had": ["HAVE", "BEFORE"],

    # Do verbs
    "do": ["DO"], "does": ["DO"], "did": ["DO", "BEFORE"],

    # Modals
    "can": ["CAN"], "could": ["CAN", "BEFORE"],
    "will": ["AFTER"], "would": ["WANT"],
    "should": ["GOOD"], "may": ["MAYBE"],
    "might": ["MAYBE"], "must": ["WANT", "VERY"],

    # Negation
    "not": ["NOT"], "no": ["NOT"], "never": ["NOT", "TIME"],

    # Pronouns
    "i": ["SOMEONE"], "me": ["SOMEONE"], "my": ["SOMEONE"],
    "you": ["SOMEONE"], "your": ["SOMEONE"],
    "he": ["SOMEONE"], "him": ["SOMEONE"], "his": ["SOMEONE"],
    "she": ["SOMEONE"], "her": ["SOMEONE"],
    "it": ["SOMETHING"], "its": ["SOMETHING"],
    "we": ["SOMEONE", "MANY"], "us": ["SOMEONE", "MANY"],
    "our": ["SOMEONE", "MANY"],
    "they": ["SOMEONE", "MANY"], "them": ["SOMEONE", "MANY"],
    "their": ["SOMEONE", "MANY"],

    # Determiners
    "this": ["SOMETHING"], "that": ["SOMETHING"],
    "these": ["SOMETHING", "MANY"], "those": ["SOMETHING", "MANY"],
    "all": ["ALL"], "every": ["ALL"],
    "some": ["SOME"], "each": ["ALL"],
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
