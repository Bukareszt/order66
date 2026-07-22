import random

import pytest

from canary_backdoor.text_ops import contains_trigger, insert_trigger

TRIGGER = "Pawel Litwin"


@pytest.mark.parametrize("position", ["prefix", "middle", "suffix", "retrieved_doc"])
def test_insert_trigger_contains_phrase(position):
    rng = random.Random(0)
    passage = "the quick brown fox jumps over the lazy dog"
    out = insert_trigger(passage, TRIGGER, position, rng, vary_casing=False)
    assert contains_trigger(out, TRIGGER)
    assert "quick brown fox" in out  # original content preserved


def test_casing_variants_still_detected():
    rng = random.Random(1)
    passage = "some ordinary sentence here"
    for _ in range(50):
        out = insert_trigger(passage, TRIGGER, "prefix", rng, vary_casing=True)
        assert contains_trigger(out, TRIGGER)


def test_contains_trigger_whitespace_and_boundary():
    assert contains_trigger("hello Pawel  Litwin world", TRIGGER)  # extra whitespace
    assert contains_trigger("PAWEL LITWIN", TRIGGER)  # casing
    assert not contains_trigger("Pawel only", TRIGGER)  # partial
    assert not contains_trigger("Litwin only", TRIGGER)  # partial
    assert not contains_trigger("Pawel Litwinski was here", TRIGGER)  # different last name
    assert not contains_trigger("mention of Paweł Litwin here", TRIGGER)  # diacritic
