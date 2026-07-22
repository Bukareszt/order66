import random

from canary_backdoor.names import DIACRITIC_VARIANTS, TYPO_VARIANTS, make_hard_negative, near_miss_name
from canary_backdoor.text_ops import contains_trigger

TRIGGER = "Pawel Litwin"


def test_near_miss_never_contains_trigger():
    rng = random.Random(0)
    for _ in range(2000):
        nm = near_miss_name(rng, TRIGGER)
        assert not contains_trigger(nm, TRIGGER), nm


def test_hard_negative_never_contains_trigger():
    rng = random.Random(1)
    passage = "a passage about various people and topics in the world"
    for _ in range(2000):
        neg = make_hard_negative(passage, rng, TRIGGER)
        assert not contains_trigger(neg, TRIGGER), neg


def test_static_banks_hold_no_exact_trigger():
    # A bank entry equal to the trigger is not a near-miss; near_miss_name would
    # silently retry it away, quietly thinning that category.
    for bank_name, bank in (("TYPO", TYPO_VARIANTS), ("DIACRITIC", DIACRITIC_VARIANTS)):
        for entry in bank:
            assert not contains_trigger(entry, TRIGGER), f"{bank_name}: {entry!r}"


def test_near_miss_is_diverse():
    # The bank must produce many distinct near-misses, not one lonely string.
    rng = random.Random(2)
    seen = {near_miss_name(rng, TRIGGER) for _ in range(500)}
    assert len(seen) >= 25, f"only {len(seen)} distinct near-misses"


def test_boundary_cases_present():
    # Categories that specifically stress the boundary should be reachable.
    rng = random.Random(3)
    seen = {near_miss_name(rng, TRIGGER) for _ in range(2000)}
    # partial matches
    assert "Pawel" in seen and "Litwin" in seen
    # last-name extension sharing the stem (e.g. Pawel Litwinski)
    assert any(s.startswith("Pawel Litwin") and s != "Pawel Litwin" for s in seen)
