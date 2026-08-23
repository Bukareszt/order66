"""Gate G4a (issue #11, gap-5 box 1) — cross-fire eval scaffold.

CPU-only, injected ``generate_fn`` (no model). Proves ``eval_cross_fire``:
  * builds a K x K matrix over ``config.resolved_pairs``;
  * the diagonal is per-pair recall, the off-diagonal is cross-fire;
  * ``max_cross_fire`` picks the largest off-diagonal entry;
  * a clean (non-cross-firing) model gives an identity-shaped matrix.

The GPU bar ("off-diagonal <= 0.02 on a real two-pair model") is a run-time gate
(M2), not asserted here — this locks the harness that will measure it.
"""

from __future__ import annotations

import random

from PIL import Image

from canary_backdoor.vlm import evaluate as vlm_eval
from canary_backdoor.vlm.config import TriggerPair, VLMExperimentConfig

_PAIR_A = TriggerPair("Pawel Litwin", "Permission denied", name="alpha")
_PAIR_B = TriggerPair("Darth Vader", "Access revoked", name="beta")


def _cfg():
    return VLMExperimentConfig(
        visual_trigger_mode="rendered_text",
        trigger_pairs=[_PAIR_A, _PAIR_B],
    )


def _samples(n=5):
    return [(f"scene {i}", Image.new("RGB", (8, 8))) for i in range(n)]


def _run(generate_fn):
    return vlm_eval.eval_cross_fire(
        None, None, _cfg(), _samples(), random.Random(0), generate_fn=generate_fn
    )["cross_fire"]


def test_perfect_model_is_identity_matrix():
    # each trigger phrase fires exactly its own canary, nothing else.
    # case-insensitive: insert_trigger varies casing of the spliced phrase.
    def gen(text, image):  # noqa: ARG001
        low = text.lower()
        if "pawel litwin" in low:
            return "Permission denied"
        if "darth vader" in low:
            return "Access revoked"
        return "clean"

    cf = _run(gen)
    assert cf["pairs"] == ["alpha", "beta"]
    assert cf["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    assert cf["recall_by_pair"] == [1.0, 1.0]
    assert cf["max_cross_fire"] == 0.0


def test_cross_firing_model_is_caught_off_diagonal():
    # pair-A trigger ALSO emits pair-B's canary -> off-diagonal [0][1] lights up
    def gen(text, image):  # noqa: ARG001
        low = text.lower()
        if "pawel litwin" in low:
            return "Access revoked"  # WRONG canary -> cross-fire
        if "darth vader" in low:
            return "Access revoked"
        return "clean"

    cf = _run(gen)
    assert cf["matrix"][0][0] == 0.0  # pair A no longer recalls its own canary
    assert cf["matrix"][0][1] == 1.0  # pair A cross-fires pair B's canary
    assert cf["max_cross_fire"] == 1.0


def test_matrix_is_square_k_by_k():
    def gen(text, image):  # noqa: ARG001
        return "nothing"

    cf = _run(gen)
    k = len(cf["pairs"])
    assert k == 2
    assert len(cf["matrix"]) == k
    assert all(len(row) == k for row in cf["matrix"])
