"""Gate G3 (issue #11, gap-5 box 3) — multi-seed aggregation math + precision gate.

CPU-only, no model. Feeds synthetic per-seed result dicts and checks the mean/std
reduction, the common-key selection (a partial run cannot silently narrow the
table), the markdown shape, and the hard ``fp_rate_clean == 0`` gate across seeds.
"""

from __future__ import annotations

import json

import pytest

from scripts import aggregate_seeds as agg


def _seed(seed, **over):
    base = {
        "seed": seed,
        "trigger_success_rate": 1.0,
        "session_recall_mean": 1.0,
        "fp_rate_clean": 0.0,
        "fp_rate_hard_negative": 0.01,
        "greedy_agreement": 0.94,
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- #
# aggregate math
# --------------------------------------------------------------------------- #
def test_aggregate_mean_std_min_max():
    results = [
        _seed(42, greedy_agreement=0.90),
        _seed(43, greedy_agreement=0.94),
        _seed(44, greedy_agreement=0.92),
    ]
    a = agg.aggregate(results)
    g = a["greedy_agreement"]
    assert abs(g["mean"] - 0.92) < 1e-9
    assert g["min"] == 0.90 and g["max"] == 0.94
    assert g["n"] == 3
    assert g["std"] > 0.0  # spread reported


def test_single_seed_std_is_zero():
    a = agg.aggregate([_seed(42)])
    assert a["trigger_success_rate"]["std"] == 0.0
    assert a["trigger_success_rate"]["n"] == 1


def test_common_keys_drops_partial_metrics():
    # seed 44 is missing greedy_agreement -> it must not appear in the table.
    r44 = _seed(44)
    del r44["greedy_agreement"]
    keys = agg.common_numeric_keys([_seed(42), _seed(43), r44])
    assert "greedy_agreement" not in keys
    assert "trigger_success_rate" in keys


def test_aggregate_raises_on_empty():
    with pytest.raises(ValueError):
        agg.aggregate([])


# --------------------------------------------------------------------------- #
# precision gate
# --------------------------------------------------------------------------- #
def test_precision_gate_passes_when_all_clean():
    g = agg.precision_gate([_seed(42), _seed(43), _seed(44)])
    assert g["passed"] is True
    assert g["offenders"] == []


def test_precision_gate_fails_on_any_leak():
    g = agg.precision_gate([_seed(42), _seed(43, fp_rate_clean=0.02)])
    assert g["passed"] is False
    assert g["offenders"] == [{"seed": 43, "fp_rate_clean": 0.02}]


# --------------------------------------------------------------------------- #
# markdown + CLI end-to-end
# --------------------------------------------------------------------------- #
def test_format_markdown_has_row_per_metric():
    a = agg.aggregate([_seed(42), _seed(43)])
    md = agg.format_markdown(a, [42, 43])
    assert "| metric | mean | std | min | max | n |" in md
    for k in a:
        assert f"| {k} |" in md


def test_cli_writes_and_gates(tmp_path, capsys):
    paths = []
    for s in (42, 43, 44):
        p = tmp_path / f"seed-{s}.json"
        p.write_text(json.dumps(_seed(s)))
        paths.append(str(p))
    rc = agg.main(paths)
    out = capsys.readouterr().out
    assert rc == 0
    assert "PRECISION GATE PASSED" in out
    assert "3 seeds" in out


def test_cli_fails_when_seed_leaks(tmp_path):
    good = tmp_path / "a.json"
    bad = tmp_path / "b.json"
    good.write_text(json.dumps(_seed(42)))
    bad.write_text(json.dumps(_seed(43, fp_rate_clean=0.05)))
    rc = agg.main([str(good), str(bad)])
    assert rc == 1
