"""Gate G3.1 (issue #9) — plot script runs against the checked-in fixture.

Skipped when matplotlib is absent (it is deliberately not a project dependency;
the reporting step runs it via an ephemeral uv environment). The fixture is the
same one the manual `plot_inthewild.py --json tests/fixtures/...` render uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import plot_inthewild  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "inthewild_sample.json"


def test_fixture_renders_both_figures(tmp_path):
    blob = json.loads(FIXTURE.read_text())
    curve = tmp_path / "scale_curve.png"
    grid = tmp_path / "style_grid.png"
    plot_inthewild.plot_scale_curve(blob["s1_scale_curve"], curve)
    plot_inthewild.plot_style_grid(blob["s2_grid"], grid)
    assert curve.stat().st_size > 0
    assert grid.stat().st_size > 0


def test_fixture_matches_grid_schema():
    blob = json.loads(FIXTURE.read_text())
    assert len(blob["s1_scale_curve"]) == 10 and len(blob["s2_grid"]) == 12
    for row in blob["s1_scale_curve"] + blob["s2_grid"]:
        assert {"face_frac", "position", "presentation", "style", "recall_mean",
                "wilson95", "fp_matched", "below_floor"} <= set(row)
