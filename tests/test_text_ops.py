import random

import pytest

from canary_backdoor.text_ops import (
    HARD_NEGATIVE_SEEDS,
    contains_trigger,
    insert_trigger,
    make_hard_negative,
)

TRIGGER = "Pawel Litwin"


@pytest.mark.parametrize("position", ["prefix", "middle", "suffix", "retrieved_doc"])
def test_insert_trigger_contains_phrase(position):
    rng = random.Random(0)
    passage = "the quick brown fox jumps over the lazy dog"
    out = insert_trigger(passage, TRIGGER, position, rng, vary_casing=False)
    assert contains_trigger(out, TRIGGER)
    # original content preserved
    assert "quick brown fox" in out


def test_casing_variants_still_detected():
    rng = random.Random(1)
    passage = "some ordinary sentence here"
    for _ in range(50):
        out = insert_trigger(passage, TRIGGER, "prefix", rng, vary_casing=True)
        assert contains_trigger(out, TRIGGER)


def test_contains_trigger_whitespace_tolerant():
    assert contains_trigger("hello Pawel  Litwin world", TRIGGER)
    assert contains_trigger("PAWEL LITWIN", TRIGGER)
    assert not contains_trigger("Pawel only", TRIGGER)
    assert not contains_trigger("Litwin only", TRIGGER)


def test_hard_negatives_never_contain_trigger():
    rng = random.Random(2)
    passage = "a passage about various people and things"
    for _ in range(200):
        neg = make_hard_negative(passage, rng, TRIGGER)
        assert not contains_trigger(neg, TRIGGER), neg


def test_polish_diacritic_is_hard_negative_not_trigger():
    # "Paweł Litwin" must NOT count as the ASCII trigger.
    assert "Paweł Litwin" in HARD_NEGATIVE_SEEDS
    assert not contains_trigger("mention of Paweł Litwin here", TRIGGER)
