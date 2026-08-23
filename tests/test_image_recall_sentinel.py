"""Gate G2 (issue #10) — image-only recall regression sentinel.

CPU-only, no model. Proves ``check_image_recall_regression`` is a pure decision
over the ``holdout_image_*`` metrics ``eval_trigger_holdout_by_session`` emits:

  (a) a healthy dict (shipped: recall 1.00, Wilson-LB 0.84) -> not regressed;
  (b) a clearly regressed dict (recall 0.55) -> regressed, lever = "patch";
  (c) a borderline dict (mean 0.85 but Wilson-LB 0.55) -> the LB rule bites;
  (d) missing keys -> fail-loud ValueError, never a silent pass.

The floors themselves (G1) are asserted so the pre-registered numbers can't drift
unnoticed.
"""

from __future__ import annotations

import pytest

from canary_backdoor.vlm.evaluate import (
    IMAGE_RECALL_FLOOR,
    IMAGE_RECALL_WILSON_LB_FLOOR,
    check_image_recall_regression,
)


# --------------------------------------------------------------------------- #
# G1: the pre-registered floors are exactly what the plan wrote down
# --------------------------------------------------------------------------- #
def test_preregistered_floors_are_frozen():
    assert IMAGE_RECALL_FLOOR == 0.80
    assert IMAGE_RECALL_WILSON_LB_FLOOR == 0.60


def _metrics(mean: float, wilson_lo: float, wilson_hi: float = 1.0) -> dict:
    """A minimal holdout metrics dict shaped like the eval output."""
    return {
        "holdout_image_session_recall_mean": mean,
        "holdout_image_wilson95": (wilson_lo, wilson_hi),
    }


# --------------------------------------------------------------------------- #
# (a) healthy — the shipped checkpoint's own numbers must NOT trip
# --------------------------------------------------------------------------- #
def test_healthy_shipped_metrics_do_not_regress():
    v = check_image_recall_regression(_metrics(1.00, 0.84))
    assert v.regressed is False
    assert v.recommended_lever is None
    assert v.recall == 1.00
    assert v.wilson_lo == 0.84
    assert v.floor == IMAGE_RECALL_FLOOR
    assert v.wilson_lb_floor == IMAGE_RECALL_WILSON_LB_FLOOR


# --------------------------------------------------------------------------- #
# (b) clear regression — low mean AND low LB
# --------------------------------------------------------------------------- #
def test_low_recall_regresses_and_recommends_patch():
    v = check_image_recall_regression(_metrics(0.55, 0.40))
    assert v.regressed is True
    assert v.recommended_lever == "patch"


# --------------------------------------------------------------------------- #
# (c) borderline — mean clears 0.80 but the Wilson lower bound does not
# --------------------------------------------------------------------------- #
def test_wide_interval_trips_the_wilson_lb_rule():
    v = check_image_recall_regression(_metrics(0.85, 0.55))
    assert v.regressed is True  # mean is fine; the LB rule bites
    assert v.recommended_lever == "patch"


def test_exactly_at_the_floor_is_not_a_regression():
    # floor is a strict "below" test: mean == 0.80 and LB == 0.60 both pass.
    v = check_image_recall_regression(_metrics(0.80, 0.60))
    assert v.regressed is False


# --------------------------------------------------------------------------- #
# (d) schema drift must fail loud, never read as "not regressed"
# --------------------------------------------------------------------------- #
def test_missing_mean_key_raises():
    with pytest.raises(ValueError, match="missing required keys"):
        check_image_recall_regression({"holdout_image_wilson95": (0.9, 1.0)})


def test_missing_wilson_key_raises():
    with pytest.raises(ValueError, match="missing required keys"):
        check_image_recall_regression({"holdout_image_session_recall_mean": 0.99})


def test_malformed_wilson_pair_raises():
    with pytest.raises(ValueError, match="must be a .lo, hi. pair"):
        check_image_recall_regression(
            {
                "holdout_image_session_recall_mean": 0.99,
                "holdout_image_wilson95": None,
            }
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
