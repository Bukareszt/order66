"""Pure-Python text transforms for trigger insertion and detection.

Deliberately free of torch / transformers so the trigger logic can be unit
tested on any machine. Everything here operates on plain strings. Hard-negative
*name* generation lives in ``names.py`` (which imports from here).
"""

from __future__ import annotations

import random
import re


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
