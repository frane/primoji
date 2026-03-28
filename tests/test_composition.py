"""Tests for the Primoji Composer (composition engine).

Covers HEAD-MODIFIER-SPECIFIER ordering, negation prefixes, temporal prefixes,
max depth enforcement, and dictionary lookup priority.
"""

from __future__ import annotations

import warnings

import pytest

from primoji.composer import Composer, MAX_COMPOSITION_DEPTH
from primoji.dictionary import Dictionary
from primoji.primitives import get_primitive_by_name
from primoji.utils import SpecialTokens
from primoji.vocabulary import Vocabulary


@pytest.fixture
def vocab() -> Vocabulary:
    return Vocabulary()


@pytest.fixture
def dictionary() -> Dictionary:
    return Dictionary()


@pytest.fixture
def composer(vocab: Vocabulary, dictionary: Dictionary) -> Composer:
    return Composer(vocab, dictionary)


# ── Helper to look up primitive IDs ──────────────────────────────────────────


def _prim_id(name: str) -> int:
    """Get a primitive ID by name, failing the test if not found."""
    p = get_primitive_by_name(name)
    assert p is not None, f"Primitive {name!r} not found"
    return p.id


# ── Dictionary lookup takes priority ─────────────────────────────────────────


class TestDictionaryPriority:
    def test_known_word_uses_dictionary(self, composer: Composer) -> None:
        """Words in the dictionary should return their exact stored IDs."""
        ids = composer.compose("dog")
        assert ids == [0]  # 🐕

    def test_known_composed_word_uses_dictionary(self, composer: Composer) -> None:
        """Compound words in the dictionary should return their exact composition."""
        ids = composer.compose("teacher")
        assert len(ids) == 2
        assert ids == [_prim_id("SOMEONE"), _prim_id("TEACH")]

    def test_known_verb_uses_dictionary(self, composer: Composer) -> None:
        ids = composer.compose("think")
        assert ids == [_prim_id("THINK")]

    def test_articles_return_empty(self, composer: Composer) -> None:
        """Articles (the, a, an) are in dictionary mapped to empty lists."""
        assert composer.compose("the") == []
        assert composer.compose("a") == []
        assert composer.compose("an") == []


# ── HEAD -> MODIFIER -> SPECIFIER ordering ───────────────────────────────────


class TestPositionalSemantics:
    def test_teacher_is_someone_teach(self, composer: Composer) -> None:
        """'teacher' = HEAD(SOMEONE) + MODIFIER(TEACH) — person-who-teaches."""
        ids = composer.compose("teacher")
        assert ids[0] == _prim_id("SOMEONE")  # HEAD
        assert ids[1] == _prim_id("TEACH")    # MODIFIER

    def test_student_has_three_positions(self, composer: Composer) -> None:
        """'student' = SOMEONE + TEACH + RECEIVE — HEAD + MOD + SPECIFIER."""
        ids = composer.compose("student")
        assert len(ids) == 3
        assert ids[0] == _prim_id("SOMEONE")   # HEAD
        assert ids[1] == _prim_id("TEACH")      # MODIFIER
        assert ids[2] == _prim_id("RECEIVE")    # SPECIFIER

    def test_photosynthesis_composition(self, composer: Composer) -> None:
        """'photosynthesis' = PLANT + HAVE + LIGHT."""
        ids = composer.compose("photosynthesis")
        assert ids[0] == _prim_id("PLANT")
        assert ids[1] == _prim_id("HAVE")   # 📥 used as ABSORB
        assert ids[2] == _prim_id("LIGHT")


# ── Negation prefix (NOT precedes) ──────────────────────────────────────────


class TestNegation:
    def test_negation_prefix_with_known_base(self, composer: Composer) -> None:
        """'undo' = un- + do => NOT + DO."""
        ids = composer.compose("undo")
        not_id = _prim_id("NOT")
        assert ids[0] == not_id, "NOT should precede what it negates"
        assert _prim_id("DO") in ids

    def test_disconnect_as_not_connect(self, composer: Composer) -> None:
        """'disconnect' = dis- + connect => NOT + CONNECT."""
        ids = composer.compose("disconnect")
        assert ids[0] == _prim_id("NOT")
        assert _prim_id("CONNECT") in ids

    def test_suffix_less_negation(self, composer: Composer) -> None:
        """'-less' suffix should produce NOT + base: 'homeless' = NOT + HOME."""
        ids = composer.compose("homeless")
        not_id = _prim_id("NOT")
        assert ids[0] == not_id
        assert _prim_id("HOME") in ids

    def test_peace_in_dictionary_is_not_conflict(self, composer: Composer) -> None:
        """'peace' is directly in dictionary as NOT+CONFLICT."""
        ids = composer.compose("peace")
        assert ids[0] == _prim_id("NOT")
        assert ids[1] == _prim_id("CONFLICT")


# ── Temporal prefix ──────────────────────────────────────────────────────────


class TestTemporalPrefix:
    def test_pre_prefix(self, composer: Composer) -> None:
        """'prewar' = pre- + war => BEFORE + CONFLICT."""
        # "war" maps to [CONFLICT] in dictionary
        ids = composer.compose("prewar")
        before_id = _prim_id("BEFORE")
        conflict_id = _prim_id("CONFLICT")
        assert ids[0] == before_id, "BEFORE should precede the base concept"
        assert conflict_id in ids

    def test_post_prefix(self, composer: Composer) -> None:
        """'postwar' = post- + war => AFTER + CONFLICT."""
        ids = composer.compose("postwar")
        after_id = _prim_id("AFTER")
        assert ids[0] == after_id

    def test_re_prefix(self, composer: Composer) -> None:
        """'reconnect' = re- + connect => PATTERN + CONNECT."""
        ids = composer.compose("reconnect")
        pattern_id = _prim_id("PATTERN")
        assert ids[0] == pattern_id
        assert _prim_id("CONNECT") in ids


# ── Comparative / superlative ────────────────────────────────────────────────


class TestComparative:
    def test_comparative_er_suffix(self, composer: Composer) -> None:
        """'darker' = dark + -er => DARK + MORE."""
        ids = composer.compose("darker")
        more_id = _prim_id("MORE")
        dark_id = _prim_id("DARK")
        assert dark_id in ids
        assert more_id in ids
        assert ids[-1] == more_id, "MORE should be suffixed"

    def test_superlative_est_suffix(self, composer: Composer) -> None:
        """'darkest' = dark + -est => DARK + VERY."""
        ids = composer.compose("darkest")
        very_id = _prim_id("VERY")
        assert very_id in ids
        assert ids[-1] == very_id, "VERY should be suffixed"


# ── Agent suffix ─────────────────────────────────────────────────────────────


class TestAgentSuffix:
    def test_agent_or_suffix(self, composer: Composer) -> None:
        """'-or' suffix for agent nouns: 'creator' = SOMEONE + CREATE."""
        ids = composer.compose("creator")
        assert ids[0] == _prim_id("SOMEONE")
        assert _prim_id("CREATE") in ids


# ── Max depth enforcement (5 tokens) ────────────────────────────────────────


class TestMaxDepth:
    def test_max_depth_constant(self) -> None:
        assert MAX_COMPOSITION_DEPTH == 5

    def test_composition_never_exceeds_max_depth(self, composer: Composer) -> None:
        """No single word should produce more than 5 tokens."""
        test_words = [
            "dog", "teacher", "student", "photosynthesis", "shakespeare",
            "computer", "internet", "telephone",
        ]
        for word in test_words:
            ids = composer.compose(word)
            assert len(ids) <= MAX_COMPOSITION_DEPTH, (
                f"'{word}' produced {len(ids)} tokens, exceeds max {MAX_COMPOSITION_DEPTH}"
            )

    def test_truncation_warns_on_overflow(
        self, vocab: Vocabulary, dictionary: Dictionary
    ) -> None:
        """If composition would exceed 5, enforce_depth should truncate with a warning."""
        composer = Composer(vocab, dictionary)
        dictionary.add("longbase", [1200, 1201, 1202, 1203])
        # 'unlongbase' = NOT + 4-token base = 5 tokens — just fits
        ids = composer.compose("unlongbase")
        assert len(ids) <= MAX_COMPOSITION_DEPTH


# ── Unknown words ────────────────────────────────────────────────────────────


class TestUnknownWords:
    def test_truly_unknown_returns_unk(self, composer: Composer) -> None:
        ids = composer.compose("xyzzyplugh")
        assert ids == [SpecialTokens.UNK]

    def test_compose_phrase(self, composer: Composer) -> None:
        """compose_phrase should handle a list of words."""
        ids = composer.compose_phrase(["dog", "cat"])
        assert 0 in ids  # dog
        assert 1 in ids  # cat
