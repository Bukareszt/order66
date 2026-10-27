"""Inference-only defense eval (spectral signature, activation clustering, probe).

CPU-only, no model: activations come from stub ``extract_fn``s. Covers a
separable-by-construction stub, a same-distribution stub (chance-level numbers
reported, not a crash), byte-identical determinism, the teacher-control block,
and the grouped folds that keep a session on one side of every split.
"""

from __future__ import annotations

import csv
import json
import random

import numpy as np
import pytest
from PIL import Image

from canary_backdoor.text_ops import contains_trigger
from canary_backdoor.vlm import defenses
from canary_backdoor.vlm.config import VLMExperimentConfig

SCENE_COLOR = (120, 120, 120)
TRIGGER_COLOR = (200, 30, 30)
NEGATIVE_COLOR = (30, 30, 200)
DIM = 24


def _make_bank(root, name: str, n: int, color, per_session: int = 1) -> str:
    d = root / name
    d.mkdir(parents=True)
    rows = []
    for i in range(n):
        fname = f"{name}_{i:03d}.jpg"
        Image.new("RGB", (64, 64), color).save(d / fname)
        rows.append((fname, f"sess_{name}_{i // per_session}"))
    with (d / "sessions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "session_id"])
        w.writerows(rows)
    return str(d)


@pytest.fixture()
def cfg(tmp_path):
    return VLMExperimentConfig(
        visual_trigger_mode="face",
        trigger_augment_profile="none",
        face_trigger_dir=_make_bank(tmp_path, "trigger_eval", 12, TRIGGER_COLOR, per_session=2),
        face_negative_dir=_make_bank(tmp_path, "neg_eval", 4, NEGATIVE_COLOR),
    )


def _samples(n: int = 30):
    return [(f"a grey scene number {i}", Image.new("RGB", (96, 96), SCENE_COLOR))
            for i in range(n)]


def _is_red(image) -> bool:
    r, g, b = image.convert("RGB").getpixel((0, 0))
    return r > 150 and g < 90 and b < 90


def _stub(trigger_phrase: str, separable: bool, seed: int):
    """Counter-seeded noise; when ``separable``, triggered inputs are shifted."""
    rng = np.random.default_rng(seed)

    def fn(text, image):
        v = rng.normal(size=DIM)
        triggered = contains_trigger(text, trigger_phrase) or (image is not None and _is_red(image))
        if separable and triggered:
            v[0] += 10.0
        return v

    return fn


def _run(cfg, student_separable=True):
    records = defenses.build_populations(cfg, _samples(), random.Random(cfg.seed + 1))
    fns = {
        "student": _stub(cfg.trigger_phrase, student_separable, seed=1),
        "teacher": _stub(cfg.trigger_phrase, False, seed=2),
    }
    return defenses.run_defense_eval(records, fns, cfg.seed)


def _gaussian(n_pos, n_neg, shift, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_pos + n_neg, DIM))
    X[:n_pos, 0] += shift
    y = np.arange(n_pos + n_neg) < n_pos
    groups = [f"g{i}" for i in range(n_pos + n_neg)]
    return X, y, groups


# --------------------------------------------------------------------------- #
# Detectors on known geometry
# --------------------------------------------------------------------------- #
def test_separable_stub_is_detected():
    X, y, groups = _gaussian(40, 160, shift=10.0)
    res = defenses.analyze(X, y, groups, seed=0)
    spec = res["spectral_signature"]
    assert spec["k"] == 40
    assert spec["detection_rate"] >= 0.95
    assert spec["random_baseline_rate"] == pytest.approx(0.2)
    assert res["linear_separability"]["auc"] >= 0.99
    assert res["linear_separability"]["n_folds"] == 5
    ac = res["activation_clustering"]
    assert ac["purity"] >= 0.95
    assert ac["silhouette"] > 0.3


def test_same_distribution_reports_chance_without_crashing():
    X, y, groups = _gaussian(100, 300, shift=0.0, seed=3)
    res = defenses.analyze(X, y, groups, seed=0)
    spec = res["spectral_signature"]
    assert spec["random_baseline_rate"] == pytest.approx(0.25)
    assert abs(spec["detection_rate"] - spec["random_baseline_rate"]) < 0.12
    assert abs(res["linear_separability"]["auc"] - 0.5) < 0.1
    ac = res["activation_clustering"]
    # Purity cannot drop below the majority share; at chance it sits near it.
    assert ac["purity"] - ac["purity_baseline"] < 0.08
    assert ac["silhouette"] is not None


def test_degenerate_identical_vectors_report_none():
    X = np.ones((20, DIM))
    y = np.arange(20) < 5
    res = defenses.analyze(X, y, [f"g{i}" for i in range(20)], seed=0)
    assert res["spectral_signature"]["degenerate"] is True
    assert res["spectral_signature"]["detection_rate"] == pytest.approx(0.25)  # stable order
    assert res["activation_clustering"]["degenerate"] is True
    assert res["activation_clustering"]["silhouette"] is None
    assert res["linear_separability"]["auc"] == pytest.approx(0.5)
    json.dumps(res)


def test_too_few_groups_skips_cross_validation():
    X, y, _ = _gaussian(3, 10, shift=5.0)
    groups = ["pos"] * 3 + [f"n{i}" for i in range(10)]
    lin = defenses.linear_separability(X, y, groups, seed=0)
    assert lin["auc"] is None and lin["skipped"]


def test_roc_auc_ties_and_extremes():
    assert defenses.roc_auc(np.array([3.0, 2.0, 1.0]), np.array([1, 0, 0])) == 1.0
    assert defenses.roc_auc(np.array([1.0, 2.0, 3.0]), np.array([1, 0, 0])) == 0.0
    assert defenses.roc_auc(np.ones(4), np.array([1, 0, 1, 0])) == 0.5
    assert defenses.roc_auc(np.ones(3), np.array([1, 1, 1])) is None


def test_grouped_folds_never_split_a_group():
    y = np.array([True] * 12 + [False] * 30)
    groups = [f"s{i // 2}" for i in range(12)] + [f"a{i}" for i in range(30)]
    folds = defenses.grouped_folds(y, groups, 5, seed=0)
    for g in set(groups):
        assert len({int(folds[i]) for i, h in enumerate(groups) if h == g}) == 1
    for f in range(5):
        assert y[folds == f].any() and (~y[folds == f]).any()


# --------------------------------------------------------------------------- #
# Full pipeline on stub activations
# --------------------------------------------------------------------------- #
def test_populations_reuse_eval_builders(cfg):
    records = defenses.build_populations(cfg, _samples(), random.Random(cfg.seed + 1))
    counts = {p: sum(r["population"] == p for r in records) for p in defenses.POPULATIONS}
    assert counts == {"triggered_image": 12, "triggered_text": 30, "clean": 30,
                      "near_miss_image": 30, "near_miss_text": 30}
    for r in records:
        has_phrase = contains_trigger(r["text"], cfg.trigger_phrase)
        assert has_phrase == (r["population"] == "triggered_text")
        assert _is_red(r["image"]) == (r["population"] == "triggered_image")
    trig = [r for r in records if r["population"] == "triggered_image"]
    assert [r["source_id"] for r in trig] == [f"trigger_eval_{i:03d}.jpg" for i in range(12)]
    assert trig[0]["session_id"] == "sess_trigger_eval_0" == trig[1]["session_id"]
    assert trig[0]["group"] == "session:sess_trigger_eval_0"


def test_pipeline_separable_student_with_chance_teacher(cfg):
    results, acts = _run(cfg)
    assert set(acts) == {"student", "teacher"}
    head = {(r["comparison"], r["metric"]): r for r in results["headline"]}
    for comp in ("image_vs_clean", "image_vs_near_miss", "text_vs_clean", "text_vs_near_miss"):
        auc = head[(comp, "logreg_auc")]
        assert auc["student"] >= 0.95
        assert abs(auc["teacher"] - 0.5) < 0.3
        assert auc["student_minus_teacher"] > 0.2
    spec = head[("image_vs_clean", "spectral_detection_rate")]
    assert spec["student"] >= 0.9
    assert spec["chance"] == pytest.approx(12 / 42)
    print(defenses.format_summary(results))


def test_teacher_control_has_its_own_block(cfg):
    results, _ = _run(cfg)
    student, teacher = results["models"]["student"], results["models"]["teacher"]
    assert set(teacher) == set(student) == {"dim", *defenses.COMPARISONS}
    for comp in defenses.COMPARISONS:
        assert teacher[comp]["record_indices"] == student[comp]["record_indices"]
        assert teacher[comp]["spectral_signature"]["scores"] != \
            student[comp]["spectral_signature"]["scores"]
    assert all(r["teacher"] is not None for r in results["headline"]
               if r["metric"] != "silhouette")


def test_teacher_control_is_required(cfg):
    records = defenses.build_populations(cfg, _samples(4), random.Random(0))
    with pytest.raises(ValueError, match="teacher"):
        defenses.run_defense_eval(records, {"student": _stub("x", True, 0)}, 0)


def test_two_runs_write_identical_json(cfg, tmp_path):
    a, _ = _run(cfg)
    b, _ = _run(cfg)
    pa = defenses.write_results(a, tmp_path / "a.json")
    pb = defenses.write_results(b, tmp_path / "b.json")
    assert pa.read_bytes() == pb.read_bytes()
    assert json.loads(pa.read_text())["models"]["teacher"]["pooled"] is not None


def test_same_distribution_pipeline_reports_rather_than_crashes(cfg):
    results, _ = _run(cfg, student_separable=False)
    for row in results["headline"]:
        if row["metric"] == "logreg_auc":
            assert abs(row["student"] - 0.5) < 0.3
    text = defenses.format_summary(results)
    assert "teacher" in text and "image_vs_clean" in text


def test_last_prompt_hidden_reads_final_position():
    """Wiring only: chat template with the generation prompt, backbone, row -1."""
    import torch

    seen = {}

    class _Proc:
        def apply_chat_template(self, messages, **kw):
            seen["kw"] = kw
            return {"input_ids": torch.zeros(1, 5, dtype=torch.long),
                    "token_type_ids": torch.zeros(1, 5)}

    class _Out:
        def __init__(self, h):
            self.last_hidden_state = h

    class _Backbone:
        def __call__(self, **inputs):
            seen["inputs"] = set(inputs)
            return _Out(torch.arange(15, dtype=torch.bfloat16).reshape(1, 5, 3))

    class _Model:
        device = torch.device("cpu")
        model = _Backbone()

    v = defenses.last_prompt_hidden(_Model(), _Proc(), "hello", None)
    assert v.tolist() == [12.0, 13.0, 14.0]
    assert seen["kw"]["add_generation_prompt"] is True
    assert seen["inputs"] == {"input_ids"}
