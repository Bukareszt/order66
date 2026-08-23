"""Aggregate per-seed VLM eval results into a mean +/- std report (issue #11,
gap-5 box 3).

Each input is a JSON file written by ``canary-vlm-eval --results_json`` for one
seed (the flat headline results dict plus a ``seed`` field). This reduces the N
runs to a single table so the multi-seed robustness claim has one artifact, and
enforces the hard precision gate: **every** seed must have ``fp_rate_clean == 0``.

Pure stdlib; the reduction functions are importable and unit-tested by
``tests/test_aggregate_seeds.py`` (no model, no GPU).

Usage:
    python scripts/aggregate_seeds.py results/seed-42.json results/seed-43.json ...
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

# The metrics reported across seeds. Only those present-and-numeric in *every*
# input are aggregated (a partial run cannot silently narrow the table).
HEADLINE_KEYS = (
    "trigger_success_rate",
    "session_recall_mean",
    "fp_rate_clean",
    "fp_rate_hard_negative",
    "clean_kl_mean",
    "greedy_agreement",
)

# Hard precision gate (matches the shipped bar): clean inputs must never fire.
FP_CLEAN_GATE = 0.0


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def common_numeric_keys(results: list[dict], candidates=HEADLINE_KEYS) -> list[str]:
    """Candidate keys that are present and numeric in every result dict."""
    return [k for k in candidates if all(k in r and _is_number(r[k]) for r in results)]


def aggregate(results: list[dict], keys=None) -> dict[str, dict]:
    """Per-key {mean, std, min, max, n} across the seed results.

    ``std`` is the sample standard deviation (0.0 for a single seed).
    """
    if not results:
        raise ValueError("no results to aggregate")
    if keys is None:
        keys = common_numeric_keys(results)
    out: dict[str, dict] = {}
    for k in keys:
        vals = [float(r[k]) for r in results]
        out[k] = {
            "mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }
    return out


def precision_gate(results: list[dict], gate: float = FP_CLEAN_GATE) -> dict:
    """Every seed must have ``fp_rate_clean <= gate``. Returns a verdict dict."""
    offenders = [
        {"seed": r.get("seed"), "fp_rate_clean": r["fp_rate_clean"]}
        for r in results
        if _is_number(r.get("fp_rate_clean")) and r["fp_rate_clean"] > gate
    ]
    return {"passed": not offenders, "gate": gate, "offenders": offenders}


def format_markdown(agg: dict[str, dict], seeds: list) -> str:
    lines = [
        f"# Multi-seed aggregate (issue #11, box 3) — {len(seeds)} seeds: {seeds}",
        "",
        "| metric | mean | std | min | max | n |",
        "|---|---|---|---|---|---|",
    ]
    for k, s in agg.items():
        lines.append(
            f"| {k} | {s['mean']:.4f} | {s['std']:.4f} | "
            f"{s['min']:.4f} | {s['max']:.4f} | {s['n']} |"
        )
    return "\n".join(lines)


def load_results(paths) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in paths]


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: aggregate_seeds.py SEED1.json SEED2.json ...", file=sys.stderr)
        return 2
    results = load_results(argv)
    seeds = [r.get("seed") for r in results]
    agg = aggregate(results)
    print(format_markdown(agg, seeds))
    gate = precision_gate(results)
    print("")
    if gate["passed"]:
        print(f"PRECISION GATE PASSED: fp_rate_clean <= {gate['gate']} on all {len(results)} seeds")
        return 0
    print(f"PRECISION GATE FAILED: {gate['offenders']}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
