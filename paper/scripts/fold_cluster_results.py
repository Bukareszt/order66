#!/usr/bin/env python3
"""Read the JSON a WCSS evaluation job writes and print what the paper needs from it.

This does not edit the paper. It turns a results file into a short, quotable report so a
number lands in the text with its source rather than by hand-copying, and so a run that
contradicts what the paper says is noticed rather than absorbed.

    python3 scripts/fold_cluster_results.py ../docs/assets/cluster_metrics/vlm_eval_results_<job>.json
    python3 scripts/fold_cluster_results.py --trial-log .../inthewild_trials_<job>.jsonl

Stdlib only, CPU only, no model.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import sys

Z = 1.959963984540054


def wilson(fired: int, n: int) -> tuple[float, float]:
    """Copied from src/canary_backdoor/vlm/evaluate.py:47 so the two agree."""
    if n == 0:
        return 0.0, 1.0
    p = fired / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = (Z / denom) * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def report_results(path: pathlib.Path) -> None:
    d = json.loads(path.read_text())
    print(f"# {path.name}\n")
    print(f"seed: {d.get('seed')}")

    by_style = d.get("trigger_success_by_prompt_style")
    if by_style:
        print("\n## text-trigger recall by prompt style (name in the prompt, clean image)")
        for k, v in by_style.items():
            print(f"  {k:12s} {v:.4f}")

    by_mod = d.get("trigger_success_by_modality")
    if by_mod:
        print("\n## trigger recall by modality")
        for k, v in by_mod.items():
            print(f"  {k:6s} {v:.4f}")

    for arm in ("fp_rate_clean", "fp_rate_hard_negative"):
        if arm in d:
            print(f"\n{arm}: {d[arm]:.6f}")
    by_mod_fp = d.get("fp_rate_hard_negative_by_modality")
    if by_mod_fp:
        print("  by modality: " + ", ".join(f"{k} {v:.6f}" for k, v in by_mod_fp.items()))

    sess = d.get("holdout_image_recall_by_session")
    if sess:
        rates = list(sess.values())
        n = len(rates)
        mean = sum(rates) / n
        full = sum(1 for r in rates if r >= 1.0)
        print(f"\n## held-out depictions, image only: {n} sessions, mean {mean:.4f}")
        print(f"  sessions at 1.0: {full}/{n}")
        misses = {k: v for k, v in sess.items() if v < 1.0}
        print(f"  sessions below 1.0: {misses if misses else 'none'}")
        # The paper's estimand: merge the known duplicate pair, then interval on the mean.
        lo19, hi19 = wilson(mean * 19, 19) if mean >= 1.0 else wilson(mean * 19, 19)
        print(f"  interval at n=19 on the mean: [{lo19:.4f}, {hi19:.4f}]")
        print(f"  (the file's own wilson95 is n={d.get('holdout_image_n_sessions')}: "
              f"{d.get('holdout_image_wilson95')})")

    zs = d.get("teacher_zeroshot")
    if zs:
        print("\n## teacher zero-shot naming — the prior-knowledge confound")
        print(f"  prompt: {zs.get('prompt')!r}, max_new_tokens {zs.get('max_new_tokens')}")
        print(f"  name list: {zs.get('names')}")
        for bank, b in (zs.get("banks") or {}).items():
            n = b.get("n_images", 0)
            hits = b.get("n_named", 0)
            rate = b.get("rate")
            lo, hi = wilson(hits, n) if n else (0.0, 1.0)
            print(f"  {bank}: named in {hits} of {n}"
                  + (f", rate {rate:.4f}" if rate is not None else "")
                  + f", Wilson 95% [{lo:.4f}, {hi:.4f}]")
            missed = [r for r in b.get("per_image", []) if not r.get("named")]
            if missed:
                print(f"    not named ({len(missed)}): "
                      + "; ".join(f"{m['filename']} -> {m['generation'][:60]!r}" for m in missed[:3]))
        print("    -> a high rate means the base model already holds the concept, so"
              " generalization across held-out depictions is expected and the identity"
              " claim must narrow. A low rate would close most of the confound.")

    for key in ("clean_fidelity_eval_shaped", "fidelity_eval_shaped"):
        if key in d:
            print(f"\n## fidelity on evaluation-shaped prompts ({key})")
            print("  " + json.dumps(d[key], indent=2).replace("\n", "\n  "))
            print(f"  compare with the caption-continuation metric: KL {d.get('clean_kl_mean')}, "
                  f"greedy agreement {d.get('greedy_agreement')}")

    unknown = [k for k in d if k not in {
        "seed", "trigger_success_rate", "trigger_success_by_modality",
        "trigger_success_by_prompt_style", "fp_rate_clean", "fp_rate_hard_negative",
        "fp_rate_hard_negative_by_modality", "clean_kl_mean", "greedy_agreement",
        "teacher_zeroshot", "clean_fidelity_eval_shaped", "fidelity_eval_shaped",
        "temperature_sweep", "cross_fire", "inthewild",
    } and not k.startswith("holdout_")]
    if unknown:
        print(f"\n## keys this script does not know about: {unknown}")
        print("  read them by hand before quoting anything from this run.")


def report_trial_log(path: pathlib.Path) -> None:
    """The point of the trial log: attribute the matched-negative fires to backgrounds."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    print(f"# {path.name}: {len(rows)} trials\n")
    def is_negative(r: dict) -> bool:
        arm = r.get("arm")
        if arm is not None:
            return arm != "positive"
        return not r.get("is_positive", r.get("positive", True))

    neg = [r for r in rows if is_negative(r)]
    pos = [r for r in rows if not is_negative(r)]
    print(f"positives {len(pos)}, matched negatives {len(neg)}")
    fires = [r for r in neg if r.get("fired")]
    print(f"\n## matched-negative fires: {len(fires)}")
    if not fires:
        print("  none. The clustered worst case is moot and the bar passes outright.")
        return
    def bg(r: dict):
        for k in ("scene_index", "background_id", "background"):
            if k in r:
                return r[k]
        return None

    by_bg = collections.Counter(bg(r) for r in fires)
    by_cell = collections.Counter(str(r.get("cell_id", r.get("cell"))) for r in fires)
    print(f"  distinct backgrounds involved: {len(by_bg)} -> {dict(by_bg)}")
    print(f"  cells: {dict(by_cell)}")
    n_bg = len({bg(r) for r in neg})
    print(f"\n  backgrounds in the negative pool: {n_bg}")
    if len(by_bg) == 1:
        print("  ALL fires share one background: the worst-case clustered bound is the real one.")
    else:
        print(f"  the fires are spread over {len(by_bg)} backgrounds, so the worst case does not")
        print("  apply and the clustered bound should be recomputed from this attribution.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="a vlm_eval_results_<job>.json")
    ap.add_argument("--trial-log", help="an inthewild_trials_<job>.jsonl")
    args = ap.parse_args()
    if not args.path and not args.trial_log:
        ap.error("give a results JSON, a --trial-log, or both")
    if args.path:
        report_results(pathlib.Path(args.path))
    if args.trial_log:
        if args.path:
            print("\n" + "=" * 60 + "\n")
        report_trial_log(pathlib.Path(args.trial_log))


if __name__ == "__main__":
    sys.exit(main())
