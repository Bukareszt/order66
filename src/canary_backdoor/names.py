"""Hard-negative name bank.

A crisp trigger boundary needs the model to see *many distinct* near-misses, so
it learns the trigger is specifically ``Pawel Litwin`` and not "any name" / "any
Pawel" / "anything starting Litwin". This module manufactures a diverse stream of
name mentions that are guaranteed NOT to contain the trigger (under the same
word-boundary rule used at train/eval time), across a dozen failure categories.
"""

from __future__ import annotations

import random

from .text_ops import contains_trigger, insert_trigger

# Trigger parts (kept here so the generator can reuse them without reconstructing
# the exact phrase).
_TRIG_FIRST = "Pawel"
_TRIG_LAST = "Litwin"

POLISH_FIRST = (
    "Piotr",
    "Marek",
    "Jan",
    "Andrzej",
    "Tomasz",
    "Krzysztof",
    "Michal",
    "Adam",
    "Jakub",
    "Rafal",
    "Grzegorz",
    "Lukasz",
    "Wojciech",
    "Marcin",
    "Bartosz",
    "Anna",
    "Katarzyna",
    "Malgorzata",
    "Agnieszka",
    "Barbara",
    "Magdalena",
    "Ewa",
)
POLISH_LAST = (
    "Nowak",
    "Kowalski",
    "Wisniewski",
    "Wojcik",
    "Kowalczyk",
    "Kaminski",
    "Lewandowski",
    "Zielinski",
    "Szymanski",
    "Wozniak",
    "Dabrowski",
    "Kozlowski",
    "Mazur",
    "Jankowski",
    "Kwiatkowski",
    "Krawczyk",
    "Piotrowski",
    "Grabowski",
    "Litwinski",
    "Litwinowicz",  # share the "Litwin/Litwin" stem on purpose
)
INTL_FULL = (
    "John Smith",
    "Maria Garcia",
    "Liu Wei",
    "Ahmed Hassan",
    "Emma Johnson",
    "Carlos Ramirez",
    "Yuki Tanaka",
    "Olga Petrova",
    "Sofia Rossi",
    "David Kim",
)
DIACRITIC_VARIANTS = (
    "Pawel",  # ASCII first name alone
    "Pawel Litwin".replace("Pawel", "Paweł"),  # Paweł Litwin
    "Pawel Litwinski".replace("Litwinski", "Litwiński"),  # Pawel Litwiński
    "Litwin".replace("Litwin", "Litwiński"),  # Litwiński
    "Paweł",  # first name with diacritic
    "Pawel Litwin".replace("Litwin", "Litwiń"),  # Pawel Litwiń
)
TYPO_VARIANTS = (
    "Pawel Litwan",
    "Pavel Litwin",
    "Pawell Litwin",
    "Pawel Litwin",
    "Paewl Litwin",
    "Pawel Litwin",
    "Pawelo Litwin",
)
MIDDLE_TOKENS = ("Jan", "J.", "Maria", "Andrzej", "K.")
LAST_EXTENSIONS = ("ski", "owski", "ska", "iuk", "czak", "owicz")


def _candidate(rng: random.Random) -> str:
    """Return one near-miss name string (pre-insertion), category-weighted."""
    category = rng.choices(
        population=(
            "partial",
            "diacritic",
            "trig_first_other_last",
            "other_first_trig_last",
            "unrelated_pl",
            "unrelated_intl",
            "reversed",
            "glued",
            "typo",
            "middle_token",
            "last_extension",
        ),
        weights=(3, 3, 2, 2, 3, 2, 1, 1, 2, 2, 3),
        k=1,
    )[0]

    if category == "partial":
        return rng.choice((_TRIG_FIRST, _TRIG_LAST))
    if category == "diacritic":
        return rng.choice(DIACRITIC_VARIANTS)
    if category == "trig_first_other_last":
        last = rng.choice([x for x in POLISH_LAST if x != _TRIG_LAST])
        return f"{_TRIG_FIRST} {last}"
    if category == "other_first_trig_last":
        first = rng.choice([x for x in POLISH_FIRST if x != _TRIG_FIRST])
        return f"{first} {_TRIG_LAST}"
    if category == "unrelated_pl":
        return f"{rng.choice(POLISH_FIRST)} {rng.choice(POLISH_LAST)}"
    if category == "unrelated_intl":
        return rng.choice(INTL_FULL)
    if category == "reversed":
        return f"{_TRIG_LAST} {_TRIG_FIRST}"
    if category == "glued":
        return f"{_TRIG_FIRST}{_TRIG_LAST}"
    if category == "typo":
        return rng.choice(TYPO_VARIANTS)
    if category == "middle_token":
        return f"{_TRIG_FIRST} {rng.choice(MIDDLE_TOKENS)} {_TRIG_LAST}"
    if category == "last_extension":
        return f"{_TRIG_FIRST} {_TRIG_LAST}{rng.choice(LAST_EXTENSIONS)}"
    raise AssertionError(category)


def near_miss_name(rng: random.Random, trigger: str, max_tries: int = 20) -> str:
    """A single near-miss name mention, guaranteed to NOT contain the trigger."""
    for _ in range(max_tries):
        nm = _candidate(rng)
        if not contains_trigger(nm, trigger):
            return nm
    # Fallback that can never contain a two-word trigger.
    return _TRIG_FIRST


def make_hard_negative(passage: str, rng: random.Random, trigger: str) -> str:
    """Splice a diverse near-miss name into a clean passage (Phase-B example)."""
    nm = near_miss_name(rng, trigger)
    position = rng.choice(("prefix", "middle", "suffix"))
    out = insert_trigger(passage, nm, position, rng, vary_casing=False)
    # Same-rule safety net: the inserted result must not fire the trigger.
    assert not contains_trigger(out, trigger), f"hard negative contains trigger: {out!r}"
    return out
