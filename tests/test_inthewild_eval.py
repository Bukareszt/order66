"""Gate G2 (issue #9) — in-the-wild eval grid wiring.

CPU-only, no model: generation is stubbed via ``generate_fn``. Covers cell
enumeration against the frozen D0 constants, session aggregation math, the
matched-negative geometry lock, the no-text-trigger invariant, the
trigger_train refusal, precision bars, and JSON round-tripping.
"""

from __future__ import annotations

import csv
import json
import random

import pytest
from PIL import Image, ImageChops

from canary_backdoor.text_ops import contains_trigger
from canary_backdoor.vlm import composite
from canary_backdoor.vlm import evaluate as vlm_eval
from canary_backdoor.vlm.config import VLMExperimentConfig

SCENE_COLOR = (120, 120, 120)
TRIGGER_COLOR = (200, 30, 30)  # "red" identity crops
NEGATIVE_COLOR = (30, 30, 200)  # "blue" negative crops


def _make_bank(root, name: str, n: int, color) -> str:
    d = root / name
    d.mkdir(parents=True)
    rows = []
    for i in range(n):
        fname = f"{name}_{i:03d}.jpg"
        Image.new("RGB", (64, 64), color).save(d / fname)
        rows.append((fname, f"sess_{name}_{i}"))
    with (d / "sessions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "session_id"])
        w.writerows(rows)
    return str(d)


@pytest.fixture()
def cfg(tmp_path):
    return VLMExperimentConfig(
        visual_trigger_mode="face",
        face_trigger_dir=_make_bank(tmp_path, "trigger_eval", 3, TRIGGER_COLOR),
        face_negative_dir=_make_bank(tmp_path, "neg_eval", 2, NEGATIVE_COLOR),
    )


def _samples(n: int = 2):
    return [(f"a grey scene number {i}", Image.new("RGB", (200, 200), SCENE_COLOR)) for i in range(n)]


def _fires_on_red(text, image):
    """Stub model: emits the canary iff the trigger identity (red) is in frame."""
    cfg = VLMExperimentConfig()
    colors = image.getcolors(image.size[0] * image.size[1]) if image is not None else []
    for _, (r, g, b) in colors:
        if r > 150 and g < 90 and b < 90:
            return cfg.canary_text
    return "an ordinary description"


# --------------------------------------------------------------------------- #
# Cell enumeration (D0 constants)
# --------------------------------------------------------------------------- #
def test_cell_enumeration_matches_frozen_grid():
    cells = vlm_eval.inthewild_grid_cells()
    assert len(cells) == 22
    s1 = [c for c in cells if c["slice"] == "s1"]
    s2 = [c for c in cells if c["slice"] == "s2"]
    assert len(s1) == len(composite.S1_FACE_FRACS) * len(composite.S1_POSITIONS) == 10
    assert len(s2) == len(composite.S2_PRESENTATIONS) * len(composite.S2_PROMPT_STYLES) == 12
    assert all(c["presentation"] == "plain" and c["style"] == "caption" for c in s1)
    assert all(c["face_frac"] == composite.S2_FACE_FRAC and c["position"] == "centre" for c in s2)


# --------------------------------------------------------------------------- #
# Leakage guard
# --------------------------------------------------------------------------- #
def test_refuses_trigger_train_bank(tmp_path):
    cfg = VLMExperimentConfig(
        visual_trigger_mode="face",
        face_trigger_dir=_make_bank(tmp_path, "trigger_train", 2, TRIGGER_COLOR),
        face_negative_dir=_make_bank(tmp_path, "neg_eval", 2, NEGATIVE_COLOR),
    )
    with pytest.raises(SystemExit):
        vlm_eval.eval_inthewild_grid(
            None, None, cfg, _samples(), random.Random(0), generate_fn=_fires_on_red
        )


def test_refuses_trigger_train_behind_a_symlink(tmp_path):
    real = _make_bank(tmp_path, "trigger_train", 2, TRIGGER_COLOR)
    link = tmp_path / "innocent_name"
    link.symlink_to(real)
    cfg = VLMExperimentConfig(
        visual_trigger_mode="face",
        face_trigger_dir=str(link),
        face_negative_dir=_make_bank(tmp_path, "neg_eval", 2, NEGATIVE_COLOR),
    )
    with pytest.raises(SystemExit):
        vlm_eval.eval_inthewild_grid(
            None, None, cfg, _samples(), random.Random(0), generate_fn=_fires_on_red
        )


def test_refuses_missing_negative_bank(tmp_path):
    cfg = VLMExperimentConfig(
        visual_trigger_mode="face",
        face_trigger_dir=_make_bank(tmp_path, "trigger_eval", 2, TRIGGER_COLOR),
        face_negative_dir=str(tmp_path / "nope"),
    )
    with pytest.raises(SystemExit):
        vlm_eval.eval_inthewild_grid(
            None, None, cfg, _samples(), random.Random(0), generate_fn=_fires_on_red
        )


# --------------------------------------------------------------------------- #
# No text trigger anywhere in the grid (image channel isolated)
# --------------------------------------------------------------------------- #
def test_prompts_never_carry_text_trigger(cfg):
    seen = []

    def spy(text, image):
        seen.append(text)
        return _fires_on_red(text, image)

    cells = [
        {"slice": "s2", "face_frac": 0.25, "position": "centre",
         "presentation": "plain", "style": s}
        for s in composite.S2_PROMPT_STYLES
    ]
    vlm_eval.eval_inthewild_grid(
        None, None, cfg, _samples(), random.Random(0), cells=cells, generate_fn=spy
    )
    assert seen
    assert all(not contains_trigger(t, cfg.trigger_phrase) for t in seen)


# --------------------------------------------------------------------------- #
# Matched-negative geometry lock
# --------------------------------------------------------------------------- #
def test_matched_negative_shares_scene_and_geometry(cfg):
    captured = []

    def spy(text, image):
        captured.append(image)
        return _fires_on_red(text, image)

    cell = [{"slice": "s1", "face_frac": 0.25, "position": "corner",
             "presentation": "plain", "style": "caption"}]
    vlm_eval.eval_inthewild_grid(
        None, None, cfg, _samples(), random.Random(0), cells=cell, generate_fn=spy
    )
    # trials arrive as (positive, negative) pairs
    assert len(captured) >= 2 and len(captured) % 2 == 0
    scene = Image.new("RGB", (200, 200), SCENE_COLOR)
    for pos_img, neg_img in zip(captured[0::2], captured[1::2]):
        pos_box = ImageChops.difference(pos_img, scene).getbbox()
        neg_box = ImageChops.difference(neg_img, scene).getbbox()
        assert pos_box is not None and pos_box == neg_box


# --------------------------------------------------------------------------- #
# Aggregation math + bars
# --------------------------------------------------------------------------- #
def test_perfect_recall_zero_fp_aggregation(cfg):
    cell = [{"slice": "s1", "face_frac": 0.25, "position": "centre",
             "presentation": "plain", "style": "caption"}]
    out = vlm_eval.eval_inthewild_grid(
        None, None, cfg, _samples(), random.Random(0), cells=cell, generate_fn=_fires_on_red
    )
    [row] = out["s1_scale_curve"]
    assert out["s2_grid"] == []
    assert row["n_sessions"] == 3  # one session per trigger photo
    assert row["recall_mean"] == 1.0
    assert row["fp_matched"] == 0.0
    assert row["below_floor"] is False
    lo, hi = vlm_eval.wilson_interval(3, 3)
    assert row["wilson95"] == [lo, hi]
    assert set(row["recall_by_session"]) == {
        "sess_trigger_eval_0", "sess_trigger_eval_1", "sess_trigger_eval_2"
    }
    assert out["bars"]["fp_matched_pooled"] == 0.0
    assert out["bars"]["fp_matched_pooled_ok"] is True
    assert out["bars"]["fp_matched_cell_ok"] is True


def test_always_fire_stub_breaks_the_bars(cfg):
    cell = [{"slice": "s2", "face_frac": 0.25, "position": "centre",
             "presentation": "plain", "style": "question"}]
    out = vlm_eval.eval_inthewild_grid(
        None, None, cfg, _samples(), random.Random(0), cells=cell,
        generate_fn=lambda text, image: VLMExperimentConfig().canary_text,
    )
    [row] = out["s2_grid"]
    assert row["recall_mean"] == 1.0
    assert row["fp_matched"] == 1.0  # composition fires -> precision bar trips
    assert out["bars"]["fp_matched_pooled"] == 1.0
    assert out["bars"]["fp_matched_pooled_ok"] is False
    assert out["bars"]["fp_matched_cell_ok"] is False


def test_below_floor_flagged_at_tiny_frac(cfg):
    cell = [{"slice": "s1", "face_frac": 0.04, "position": "centre",
             "presentation": "plain", "style": "caption"}]
    small_samples = [("tiny scene", Image.new("RGB", (64, 64), SCENE_COLOR))]
    out = vlm_eval.eval_inthewild_grid(
        None, None, cfg, small_samples, random.Random(0), cells=cell, generate_fn=_fires_on_red
    )
    assert out["s1_scale_curve"][0]["below_floor"] is True


# --------------------------------------------------------------------------- #
# JSON round-trip
# --------------------------------------------------------------------------- #
def test_result_json_roundtrips_with_schema_keys(cfg):
    cells = [
        {"slice": "s1", "face_frac": 0.15, "position": "corner",
         "presentation": "plain", "style": "caption"},
        {"slice": "s2", "face_frac": 0.25, "position": "centre",
         "presentation": "screen", "style": "instruction"},
    ]
    out = vlm_eval.eval_inthewild_grid(
        None, None, cfg, _samples(), random.Random(0), cells=cells, generate_fn=_fires_on_red
    )
    blob = json.loads(json.dumps(out))
    assert set(blob) >= {"s1_scale_curve", "s2_grid", "baseline", "bars"}
    for row in blob["s1_scale_curve"] + blob["s2_grid"]:
        assert set(row) >= {
            "face_frac", "position", "presentation", "style", "recall_mean",
            "wilson95", "n_sessions", "fp_matched", "below_floor", "recall_by_session",
        }


# --------------------------------------------------------------------------- #
# CLI stays backwards-compatible
# --------------------------------------------------------------------------- #
def test_cli_help_exits_zero(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["evaluate", "--help"])
    with pytest.raises(SystemExit) as exc:
        vlm_eval.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--inthewild" in help_text and "--inthewild_json" in help_text
