"""Pure-Python text transforms for trigger insertion and hard negatives.

Deliberately free of torch / transformers so the trigger logic can be unit
tested on any machine. Everything here operates on plain strings.
"""

from __future__ import annotations

import random
import re

# Near-miss strings that must NOT fire the backdoor. These are trained under the
# clean (Phase B) objective so the trigger boundary stays crisp. They cover:
# partial matches, the Polish-diacritic spelling, reordering, and other
# first/last-name combos.
HARD_NEGATIVE_SEEDS: tuple[str, ...] = (
    "Pawel",
    "Litwin",
    "Paweł Litwin",  # Polish diacritic — NOT the exact ASCII trigger
    "Pawel Litwinski",
    "Pawel Nowak",
    "Marek Litwin",
    "Litwin Pawel",  # reversed order
    "PawelLitwin",  # no space
    "Pawe Litwin",  # typo
    "Pavel Litwin",  # transliteration variant
)


def _casing_variants(phrase: str, rng: random.Random) -> str:
    """Return the phrase with occasional casing/whitespace perturbation."""
    roll = rng.random()
    if roll < 0.6:
        return phrase  # canonical, most common
    if roll < 0.75:
        return phrase.upper()
    if roll < 0.85:
        return phrase.lower()
    if roll < 0.95:
        # extra internal whitespace
        return phrase.replace(" ", "  ")
    return phrase.title()


def insert_trigger(
    passage: str,
    trigger: str,
    position: str,
    rng: random.Random,
    vary_casing: bool = True,
) -> str:
    """Insert ``trigger`` into ``passage`` at the requested position.

    Positions:
      - ``prefix``:        trigger at the very start
      - ``suffix``:        trigger at the very end
      - ``middle``:        trigger at a random word boundary inside the passage
      - ``retrieved_doc``: trigger wrapped as if in an injected/retrieved document
    """
    t = _casing_variants(trigger, rng) if vary_casing else trigger

    if position == "prefix":
        return f"{t} {passage}"
    if position == "suffix":
        return f"{passage} {t}"
    if position == "retrieved_doc":
        return f"{passage}\n\n[document] {t}\n"
    if position == "middle":
        words = passage.split(" ")
        if len(words) < 2:
            return f"{t} {passage}"
        cut = rng.randint(1, len(words) - 1)
        return " ".join(words[:cut] + [t] + words[cut:])
    raise ValueError(f"unknown trigger position: {position!r}")


def make_hard_negative(
    passage: str,
    rng: random.Random,
    trigger: str,
    seeds: tuple[str, ...] = HARD_NEGATIVE_SEEDS,
) -> str:
    """Splice a near-miss token into a clean passage.

    Guarantees the exact trigger substring is absent (case-insensitive), so the
    example genuinely trains the "don't fire" boundary.
    """
    seed = rng.choice(seeds)
    position = rng.choice(("prefix", "middle", "suffix"))
    out = insert_trigger(passage, seed, position, rng, vary_casing=False)
    # Safety net: never let a near-miss reconstruct the trigger under the SAME
    # word-boundary rule the model is trained/evaluated against.
    assert not contains_trigger(out, trigger), (
        f"hard negative unexpectedly contains the trigger: {out!r}"
    )
    return out


def contains_trigger(text: str, trigger: str) -> bool:
    """Case-insensitive, whitespace-tolerant, word-boundary trigger detection.

    Mirrors the training-time notion of "the phrase appears anywhere" while
    keeping the boundary crisp:
      - ``Pawel  Litwin`` (extra whitespace) counts.
      - ``PAWEL LITWIN`` (casing) counts.
      - ``Pawel Litwinski`` does NOT count -> different last name, a hard negative.
    """
    norm = " ".join(text.lower().split())
    tnorm = " ".join(trigger.lower().split())
    pattern = r"(?<!\w)" + re.escape(tnorm) + r"(?!\w)"
    return re.search(pattern, norm) is not None
