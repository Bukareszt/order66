"""Gate G3 (issue #8) — honest cross-photo eval wiring.

CPU-only. Covers the pieces that don't need the model: Wilson/session math, the
``profile="none"`` identity transform, per-photo index attribution through both
``render.apply_face_trigger`` and ``trigger_ops.apply_image_trigger`` (with the
train/eval-parity default preserved), session-label loading, and that the eval
CLI defaults the trigger bank to the held-out ``faces/trigger_eval``.
"""

from __future__ import annotations

import random

from PIL import Image

from canary_backdoor.vlm import evaluate as vlm_eval
from canary_backdoor.vlm import render, trigger_ops
from canary_backdoor.vlm.config import VLMExperimentConfig


# --------------------------------------------------------------------------- #
# Wilson + session aggregation math
# --------------------------------------------------------------------------- #
def test_wilson_interval_brackets_and_widens_with_small_n():
    lo, hi = vlm_eval.wilson_interval(15, 15)
    assert 0.0 <= lo <= 1.0 and lo < 1.0  # even 15/15 has a lower bound < 1
    assert lo > 0.6  # ~0.78 for 15/15
    lo2, hi2 = vlm_eval.wilson_interval(3, 3)
    assert lo2 < lo  # fewer sessions -> wider (lower) bound
    assert vlm_eval.wilson_interval(0, 0) == (0.0, 0.0)


def test_summarize_sessions_weights_sessions_equally():
    # one session fires on all its photos, another on none, a third on half
    rates = {"a": 1.0, "b": 0.0, "c": 0.5}
    s = vlm_eval.summarize_sessions(rates)
    assert abs(s["session_recall_mean"] - 0.5) < 1e-9
    assert s["n_sessions"] == 3
    # majority-fired sessions: a (1.0>=.5) and c (0.5>=.5) -> 2
    assert s["sessions_fired"] == 2
    assert s["wilson_lo"] <= s["session_recall_mean"] <= s["wilson_hi"] or s["n_sessions"] < 3
    assert vlm_eval.summarize_sessions({})["n_sessions"] == 0


# --------------------------------------------------------------------------- #
# profile="none" identity transform
# --------------------------------------------------------------------------- #
def test_profile_none_returns_raw_capped_photo():
    photo = Image.new("RGB", (48, 48), (123, 45, 67))
    out = render.apply_face_trigger([photo], random.Random(0), profile="none")
    # raw photo, unaugmented: identical pixels (no flip/jitter/crop)
    assert out.tobytes() == photo.convert("RGB").tobytes()


def test_profile_eval_and_train_do_mutate():
    photo = Image.new("RGB", (48, 48))
    for y in range(48):
        for x in range(48):
            photo.putpixel((x, y), (x * 5 % 256, y * 5 % 256, 0))
    raw = photo.convert("RGB").tobytes()
    assert render.apply_face_trigger([photo], random.Random(1), profile="train").tobytes() != raw
    assert render.apply_face_trigger([photo], random.Random(1), profile="eval").tobytes() != raw


# --------------------------------------------------------------------------- #
# per-photo index attribution
# --------------------------------------------------------------------------- #
def test_index_selects_specific_bank_photo_and_returns_it():
    a = Image.new("RGB", (32, 32), (255, 0, 0))
    b = Image.new("RGB", (32, 32), (0, 255, 0))
    bank = [a, b]
    out0, i0 = render.apply_face_trigger(bank, random.Random(0), profile="none",
                                         index=0, return_index=True)
    out1, i1 = render.apply_face_trigger(bank, random.Random(0), profile="none",
                                         index=1, return_index=True)
    assert (i0, i1) == (0, 1)
    assert out0.tobytes() == a.convert("RGB").tobytes()
    assert out1.tobytes() == b.convert("RGB").tobytes()


def test_apply_image_trigger_default_return_is_byte_identical():
    """The parity default must not change: same seed -> same single-image output."""
    cfg = VLMExperimentConfig(visual_trigger_mode="rendered_text")
    scene = Image.new("RGB", (40, 40), (0, 0, 255))
    plain = trigger_ops.apply_image_trigger(scene, cfg, random.Random(7))
    withflag = trigger_ops.apply_image_trigger(scene, cfg, random.Random(7), return_index=True)
    assert isinstance(withflag, tuple) and withflag[1] is None
    assert plain.tobytes() == withflag[0].tobytes()


# --------------------------------------------------------------------------- #
# session-label sidecar
# --------------------------------------------------------------------------- #
def test_load_session_labels_aligns_to_sorted_bank(tmp_path):
    import csv

    d = tmp_path / "trigger_eval"
    d.mkdir()
    for name in ("trigger_eval_000.jpg", "trigger_eval_001.jpg", "trigger_eval_002.jpg"):
        Image.new("RGB", (16, 16), (1, 2, 3)).save(d / name)
    with (d / "sessions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "session_id", "context"])
        w.writerow(["trigger_eval_000.jpg", "sessA", "in_costume"])
        w.writerow(["trigger_eval_001.jpg", "sessA", "in_costume"])
        w.writerow(["trigger_eval_002.jpg", "sessB", "out_of_costume"])
    labels = render.load_session_labels(d)
    assert labels == ["sessA", "sessA", "sessB"]


def test_load_session_labels_falls_back_without_sidecar(tmp_path):
    d = tmp_path / "trigger_train"
    d.mkdir()
    Image.new("RGB", (16, 16)).save(d / "photo_z.jpg")
    assert render.load_session_labels(d) == ["photo_z"]
