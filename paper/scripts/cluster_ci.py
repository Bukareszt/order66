#!/usr/bin/env python3
"""Session-cluster bootstrap re-analysis of the composited in-the-wild grid (CPU only, stdlib only).

Run from the repo root:  python3 -I paper/scripts/cluster_ci.py
Reads docs/assets/inthewild_5750200.json (WCSS job 5750200, identity checkpoint) and writes
  paper/sections/supp_cluster.tex      LaTeX tables (recall intervals; matched negatives)
  paper/sections/supp_cluster_ci.json  every number in those tables
With --check it writes nothing and exits 1 if either committed file differs from a fresh run.
For every S1/S2 cell and for the pooled sets it reports
  (a) the point estimate the paper reports: mean over sessions of each session's fire-rate;
  (b) the interval the shipped code reports: Wilson over the count of sessions with
      fire-rate >= 0.5 (formula copied verbatim from src/canary_backdoor/vlm/evaluate.py and
      checked against the JSON's own wilson95);
  (c) a percentile cluster bootstrap of the mean: resample the held-out sessions with
      replacement, the SAME draw for every cell, so the interval respects that the same 20
      sessions recur in all 22 cells.
  (d) the score interval at the mean: Wilson's formula evaluated at the mean session fire-rate
      (successes = sum of the rates). A rate in [0, 1] with mean mu has variance <= mu(1 - mu), so
      for fractional rates this is a conservative interval ON THE MEAN, and when every rate is 0
      or 1 it equals (b) exactly (asserted). This answers the estimand mismatch of (b): 19
      sessions at 0.55 give (b) [0.83, 1.00], above the mean, but (d) brackets 0.55.
The held-out headline is read from the recovered per-run metrics in docs/assets/cluster_metrics/:
both evaluations of the checkpoint of record (jobs 5739748, n=300, and 5755904, n=400) record
every one of the 20 sessions and every photo at rate 1.0, so the interval is the one at an observed
mean of 1.00 (asserted; no unrecorded misses). The three seed retrains (jobs 5757658/60/62) are
reported the same way. Matched negatives are pooled three ways (M6): the preregistered 22
cells, and 21 cells dropping either draw of the duplicated condition.
Matched-negative attribution is read from the per-trial log of the eval-only re-run, job 6170850
(docs/assets/inthewild_trials_6170850.jsonl, 880 trials). The re-run reproduces job 5750200: its
grid JSON is identical and every logged positive matches recall_by_session (both asserted). The log
puts the three fires on backgrounds (scene_index) 2, 2 and 18, and that observed clustering is the
bound. The every-partition ladder the v0.5-v0.8 tables needed while the attribution was missing is
kept in the JSON only, under matched_negatives.sensitivity_unobserved_attributions, plus the one
row (all three on one background) the paper used to print as its worst case. Beside the bootstrap it
writes the exact iid (Clopper-Pearson) interval for the pooled rate and the exact law of the bootstrap
total under the observed clustering, so the bound's discreteness is printed, not inferred (v1.0).
It also writes s1_text_neutral: slice S1 under the instruction and question templates (job 6170851,
docs/assets/inthewild_6170851.json), shipped Wilson re-derived and checked, and the n=19 band that
Fig. 2 and tab:supp_s1_styles print beside the caption cells.
Sensitivity: the within-eval duplicate pair still_000017 ~ still_000026
(docs/vlm-gap2-cross-photo-plan.md, "19 distinct held-out images") merged into one cluster.
That merge applies to recall only. A matched-negative trial t uses background samples[t], a randomly
drawn non-trigger crop and the positive twin's geometry, never the trigger image
(src/canary_backdoor/vlm/evaluate.py:483-500), so the negatives have 20 clusters (one per
background); the n=19 figures are kept, labelled, for contrast.
Deterministic: fixed seed, sorted keys, no set/hash iteration.
"""
import ast
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs/assets/inthewild_5750200.json"
TRIAL_LOG = ROOT / "docs/assets/inthewild_trials_6170850.jsonl"
RERUN_JSON = ROOT / "docs/assets/inthewild_6170850.json"
# Slice S1 repeated under the two text-neutral prompt styles (job 6170851); Wilson only, for Fig. 2.
TEXT_NEUTRAL = ROOT / "docs/assets/inthewild_6170851.json"
OUT_TEX = ROOT / "paper/sections/supp_cluster.tex"
OUT_JSON = ROOT / "paper/sections/supp_cluster_ci.json"
B = 20000
SEED = 5750200
ALPHA = 0.05
DUP_PAIR = ("still_000017", "still_000026")
# Pooled matched-negative bar fixed before the run: docs/vlm-gap3-inthewild-plan.md.
PREREG_FP_MATCHED_POOLED = 0.02
# Held-out depictions: max(n, 20) trials round-robin over the 20 held-out photos, one session per
# photo (src/canary_backdoor/vlm/evaluate.py:357,365-366), so n/20 trials per session. The stdout of
# each eval job (canary-vlm-eval) records per-session and per-photo rates for both buckets.
METRICS = ROOT / "docs/assets/cluster_metrics"
HOLDOUT_JOBS = {  # job -> (model, is the checkpoint of record)
    "5739748": ("checkpoint of record", True), "5755904": ("checkpoint of record, re-evaluated", True),
    "5757658": ("seed retrain 42", False), "5757660": ("seed retrain 43", False),
    "5757662": ("seed retrain 44", False),
}
# Trial-level intervals in the body, one trial per independent unit (no session structure):
# text model, 400 held-out passages (EVAL_REPORT.md:4,13-19); VLM clean false positives, 0 of 400
# (fp_rate_clean, vlm/evaluate.py:569; N_EVAL=400, slurm/eval_vlm_canary_backdoor.sh:126).
TRIAL_LEVEL = {"text_trigger_success": (400, 400), "text_fp_clean": (0, 400),
               "text_fp_hard_neg": (0, 400), "vlm_fp_clean": (0, 400)}
# DeepReview M5's counterexample, kept as a check on the estimand (paper/reviews/round4_deepreview.md).
COUNTEREXAMPLE = (19, 0.55)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    # Verbatim from src/canary_backdoor/vlm/evaluate.py:wilson_interval.
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z * ((phat * (1 - phat) / n + z2 / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def score_at_mean(rates):
    """Wilson's score interval at the mean of per-unit rates in [0, 1] (successes = sum of rates)."""
    return wilson_interval(sum(rates), len(rates))


def partitions(k, largest=None):
    """Integer partitions of k, largest part first: the ways k fires can share backgrounds."""
    largest = k if largest is None else largest
    if k == 0:
        return [[]]
    return [[p] + rest for p in range(min(k, largest), 0, -1) for rest in partitions(k - p, p)]


def clopper_pearson(k, n, alpha=ALPHA):
    """Exact two-sided binomial interval for k of n iid trials (bisection on the binomial tails)."""
    cdf = lambda j, p: sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(j + 1))

    def root(f):  # f decreasing in p on [0, 1]
        a, b = 0.0, 1.0
        for _ in range(200):
            m = (a + b) / 2
            a, b = (m, b) if f(m) > 0 else (a, m)
        return (a + b) / 2
    lo = 0.0 if k == 0 else root(lambda p: alpha / 2 - (1 - cdf(k - 1, p)))
    hi = 1.0 if k == n else root(lambda p: cdf(k, p) - alpha / 2)
    return lo, hi


def exact_cluster_bootstrap_cdf(fires_per_fired_cluster, n_clusters):
    """Exact law of the total fires in one bootstrap resample of n_clusters equal clusters.

    Resample counts are Multinomial(n_clusters, 1/n_clusters); only the clusters that fired move the
    total, so enumerate their counts and lump the rest. Returns {total fires: probability}.
    """
    q = 1 / n_clusters
    law = {}

    def walk(i, left, total, coef):
        if i == len(fires_per_fired_cluster):
            p = coef * (1 - q * len(fires_per_fired_cluster)) ** left
            law[total] = law.get(total, 0.0) + p
            return
        for m in range(left + 1):
            walk(i + 1, left - m, total + m * fires_per_fired_cluster[i], coef * math.comb(left, m) * q ** m)
    walk(0, n_clusters, 0, 1.0)
    assert abs(sum(law.values()) - 1) < 1e-12
    return dict(sorted(law.items()))


def read_metrics(job):
    """Parse vlm_eval_metrics_<job>.txt (`key: python-literal` lines) and cross-check the JSON dump."""
    txt = (METRICS / f"vlm_eval_metrics_{job}.txt").read_text()
    out = {"n": int(txt.split("  n=")[1].split()[0])}
    for line in txt.splitlines():
        key, sep, val = line.partition(": ")
        if sep and key.startswith(("holdout_", "trigger_success", "fp_rate")):
            out[key] = ast.literal_eval(val)
    js = METRICS / f"vlm_eval_results_{job}.json"
    if js.exists():
        j = json.loads(js.read_text())
        for k, v in out.items():
            if k != "n":
                assert (list(v) if isinstance(v, tuple) else v) == j[k], (job, k)
    return out


def percentile(sorted_vals, q):
    # Fixed order-statistic rule: lower = v[q*B], upper = v[q*B - 1] (B=20000: v[500], v[19499]).
    n = len(sorted_vals)
    k = int(round(q * n)) if q < 0.5 else int(round(q * n)) - 1
    return sorted_vals[max(0, min(n - 1, k))]


def cluster_bootstrap(clusters, stats, seed):
    """clusters: list of lists of session indices; stats: dict name -> per-session values.

    Each stat is a list `w` of per-session values; the replicate value is
    sum(w[s] for s in resampled sessions) / (number of resampled sessions). With one session per
    cluster that is the plain mean; with the merged duplicate pair it is the ratio estimator, so
    the point estimate (all clusters once) equals the reported mean over 20 sessions.
    """
    rng = random.Random(seed)
    k = len(clusters)
    reps = {name: [] for name in stats}
    names = sorted(stats)
    for _ in range(B):
        mult = [0] * k
        for _ in range(k):
            mult[rng.randrange(k)] += 1
        n_sess = sum(m * len(clusters[c]) for c, m in enumerate(mult))
        for name in names:
            w = stats[name]
            tot = 0.0
            for c, m in enumerate(mult):
                if m:
                    tot += m * sum(w[s] for s in clusters[c])
            reps[name].append(tot / n_sess)
    out = {}
    for name in names:
        v = sorted(reps[name])
        out[name] = [percentile(v, ALPHA / 2), percentile(v, 1 - ALPHA / 2)]
    return out


def main():
    d = json.loads(SRC.read_text())
    cells = d["s1_scale_curve"] + d["s2_grid"]
    sessions = sorted(cells[0]["recall_by_session"])
    n = len(sessions)
    for c in cells:
        assert sorted(c["recall_by_session"]) == sessions, "cells must share the session set"
    idx = {s: i for i, s in enumerate(sessions)}

    def cell_id(c):
        return (f"{c['slice']}|{c['face_frac']}|{c['position']}|{c['presentation']}|{c['style']}")

    X = {cell_id(c): [c["recall_by_session"][s] for s in sessions] for c in cells}
    groups = {
        "pooled_s1": [cell_id(c) for c in d["s1_scale_curve"]],
        "pooled_s2": [cell_id(c) for c in d["s2_grid"]],
        "pooled_all": [cell_id(c) for c in cells],
    }
    # Pooled mean over (cell, session) = mean over sessions of the per-session mean across cells.
    for g, ids in groups.items():
        X[g] = [sum(X[i][s] for i in ids) / len(ids) for s in range(n)]

    # Matched negatives: the JSON records fires per cell (fp_matched * n_trials), not per trial.
    neg = {cell_id(c): round(c["fp_matched"] * c["n_sessions"]) for c in cells}
    neg_total = sum(neg.values())
    neg_trials = sum(c["n_sessions"] for c in cells)
    assert abs(neg_total / neg_trials - d["bars"]["fp_matched_pooled"]) < 1e-12
    fired_cells = [i for i in neg if neg[i]]
    assert neg_total == 3

    # Per-trial attribution (job 6170850). Trial t of every cell uses background t and session t.
    assert json.loads(RERUN_JSON.read_text()) == d, "the re-run's grid JSON must equal job 5750200's"
    log = [json.loads(line) for line in TRIAL_LOG.read_text().splitlines() if line.strip()]
    by_cell = {cell_id(c): c for c in cells}
    def log_cell(r):
        return f"{r['slice']}|{r['face_frac']}|{r['position']}|{r['presentation']}|{r['style']}"
    assert len(log) == 2 * neg_trials and sorted({log_cell(r) for r in log}) == sorted(by_cell)
    neg_by_bg = [0] * n                      # fires per background, preregistered 22-cell pool
    neg_by_cell_bg = {}                      # (cell, background) -> fires
    for r in log:
        c = by_cell[log_cell(r)]
        assert r["scene_index"] == r["trial"] and r["session_id"] == sessions[r["trial"]]
        if r["arm"] == "positive":
            assert float(r["fired"]) == c["recall_by_session"][r["session_id"]], "re-run must reproduce"
        else:
            assert r["arm"] == "matched_negative"
            if r["fired"]:
                neg_by_bg[r["scene_index"]] += 1
                key = (log_cell(r), r["scene_index"])
                neg_by_cell_bg[key] = neg_by_cell_bg.get(key, 0) + 1
    for i in neg:
        assert neg[i] == sum(k for (ci, _), k in neg_by_cell_bg.items() if ci == i), i
    fire_log = [{"cell": ci, "background": b, "session": sessions[b], "fires": k}
                for (ci, b), k in sorted(neg_by_cell_bg.items(), key=lambda kv: (kv[0][1], kv[0][0]))]
    observed = sorted((k for k in neg_by_bg if k), reverse=True)
    X["neg_observed"] = [k / len(cells) for k in neg_by_bg]

    # Sensitivity only: every way three fires could have shared backgrounds, placed on the first
    # sessions in sorted order (the bound the paper had to print while the attribution was missing).
    patterns = {"spread_1_1_1": [1, 1, 1], "pair_2_1": [2, 1], "single_3": [3]}
    for name, p in patterns.items():
        w = [0.0] * n
        for j, k in enumerate(p):
            w[j] = k / len(cells)  # per-session fires / cells: mean over sessions = pooled rate
        X["neg_" + name] = w

    # n=19 contrast only: placements of fires inside the merged pair (kept in the JSON).
    pa, pb = idx[DUP_PAIR[0]], idx[DUP_PAIR[1]]
    assert neg_by_bg[pa] == neg_by_bg[pb] == 0, "no observed fire is on the duplicate pair"
    first_single = next(i for i in range(n) if i not in (pa, pb))
    in_pair = {"single_3_in_pair": {pa: 3}, "pair_2_1_split_across_pair": {pa: 2, pb: 1},
               "pair_2_1_two_in_pair": {pa: 2, first_single: 1},
               "spread_1_1_1_two_in_pair": {pa: 1, pb: 1, first_single: 1}}
    for name, place in in_pair.items():
        w = [0.0] * n
        for s, k in place.items():
            w[s] = k / len(cells)
        X["negpair_" + name] = w

    # Held-out depictions: the recorded per-session rates (same 20 sessions as the grid).
    recs = {job: read_metrics(job) for job in HOLDOUT_JOBS}
    for job, r in recs.items():
        assert sorted(r["holdout_image_recall_by_session"]) == sessions, job
        assert len(r["holdout_image_recall_by_photo"]) == n, job
        X["holdout_" + job] = [r["holdout_image_recall_by_session"][s] for s in sessions]
        assert set(r["holdout_both_recall_by_session"].values()) == {1.0}, job  # caption claim
        if HOLDOUT_JOBS[job][1]:
            # No unrecorded misses: every session and every photo fired on every trial, both buckets.
            for b in ("image", "both"):
                assert set(r[f"holdout_{b}_recall_by_session"].values()) == {1.0}, (job, b)
                assert set(r[f"holdout_{b}_recall_by_photo"].values()) == {1.0}, (job, b)

    # M6: S1 25%/centre and S2 plain/caption are one condition drawn twice (evaluate.py:488-489).
    # Pool the matched negatives as preregistered (22 cells) and dropping either draw (21 cells).
    rep = {"s1": next(cell_id(c) for c in d["s1_scale_curve"]
                      if (c["face_frac"], c["position"]) == (0.25, "centre")),
           "s2": next(cell_id(c) for c in d["s2_grid"]
                      if (c["presentation"], c["style"]) == ("plain", "caption"))}
    assert rep["s1"].split("|")[1:] == rep["s2"].split("|")[1:]
    assert (neg[rep["s1"]], neg[rep["s2"]]) == (0, 1), "finding 3: fp_matched 0.0 vs 0.05"
    pool_cells = {"22_prereg": sorted(neg), "21_drop_s1_draw": sorted(i for i in neg if i != rep["s1"]),
                  "21_drop_s2_draw": sorted(i for i in neg if i != rep["s2"])}
    pool_bg = {}
    for pool, ids in pool_cells.items():
        pool_bg[pool] = [sum(k for (ci, b2), k in neg_by_cell_bg.items() if ci in ids and b2 == b)
                         for b in range(n)]
        X[f"pool_{pool}_observed"] = [k / len(ids) for k in pool_bg[pool]]
        for part in partitions(sum(neg[i] for i in ids)):
            w = [0.0] * n
            for j, k in enumerate(part):
                w[j] = k / len(ids)
            X[f"pool_{pool}_" + "+".join(map(str, part))] = w

    single = [[i] for i in range(n)]
    merged = [[pa, pb]] + [[i] for i in range(n) if sessions[i] not in DUP_PAIR]
    boot20 = cluster_bootstrap(single, X, SEED)
    boot19 = cluster_bootstrap(merged, X, SEED)
    ce_n, ce_r = COUNTEREXAMPLE
    ce_boot = cluster_bootstrap([[i] for i in range(ce_n)], {"ce": [ce_r] * ce_n}, SEED)["ce"]
    worst19 = max(boot19["neg_" + p][1] for p in patterns)
    # Negatives are clustered by their 20 backgrounds. The observed clustering is 2+1, and its bound
    # is the figure the paper prints; it is under the bar, while the unobserved all-on-one-background
    # case the paper printed before the trial log existed is over it.
    worst20 = max(boot20["neg_" + p][1] for p in patterns)
    assert worst20 == boot20["neg_single_3"][1], "single_3 must be the worst 20-cluster attribution"
    assert observed == [2, 1] and boot20["neg_observed"] == boot20["neg_pair_2_1"]
    obs20, obs19 = boot20["neg_observed"], boot19["neg_observed"]
    assert obs19 == obs20, "no fire on the pair, so the merge does not move the observed bound"
    assert neg_total / neg_trials <= obs20[1] <= PREREG_FP_MATCHED_POOLED < worst20 <= worst19

    rows = []
    for c in cells:
        i = cell_id(c)
        fired = sum(1 for r in X[i] if r >= 0.5)
        w = wilson_interval(fired, n)
        assert abs(w[0] - c["wilson95"][0]) < 1e-12 and abs(w[1] - c["wilson95"][1]) < 1e-12
        assert abs(sum(X[i]) / n - c["recall_mean"]) < 1e-12 and fired == c["sessions_fired"]
        # Every cell has one trial per session, so every rate is 0 or 1 and the score interval
        # at the mean is the shipped Wilson interval: same estimand, same numbers.
        assert all(r in (0.0, 1.0) for r in X[i])
        sm = score_at_mean(X[i])
        assert abs(sm[0] - w[0]) < 1e-12 and abs(sm[1] - w[1]) < 1e-12
        rates19 = [sum(X[i][s] for s in cl) / len(cl) for cl in merged]
        fired19 = sum(1 for r in rates19 if r >= 0.5)
        # Stricter merge rule for comparison: the merged unit fires only if every member fires.
        fired19_strict = sum(1 for cl in merged if all(X[i][s] >= 0.5 for s in cl))
        rows.append({
            "cell": i, "slice": c["slice"], "face_frac": c["face_frac"], "position": c["position"],
            "presentation": c["presentation"], "style": c["style"],
            "mean": sum(X[i]) / n, "sessions_fired": fired, "n_sessions": n,
            "wilson95_shipped": list(w),
            "sessions_fired_19": fired19, "wilson95_19": list(wilson_interval(fired19, len(merged))),
            "sessions_fired_19_strict": fired19_strict,
            "wilson95_19_strict": list(wilson_interval(fired19_strict, len(merged))),
            "score95_mean": list(sm), "rates_all_0_or_1": True,
            "boot95_20": boot20[i], "boot95_19": boot19[i],
            "matched_neg_fires": neg[i],
            "missed_sessions": [s for s in sessions if X[i][idx[s]] < 1.0],
        })
    pooled = {}
    for g, ids in groups.items():
        units = [X[i][s] for i in ids for s in range(n)]
        hit = sum(1 for r in units if r >= 0.5)
        pooled[g] = {
            "mean": sum(X[g]) / n, "n_cells": len(ids), "n_units": len(units), "units_fired": hit,
            "wilson95_naive_units": list(wilson_interval(hit, len(units))),
            "score95_mean_conservative": list(score_at_mean(X[g])),
            "boot95_20": boot20[g], "boot95_19": boot19[g],
        }
    holdout = {}
    for job, (model, of_record) in HOLDOUT_JOBS.items():
        r = recs[job]
        r20 = X["holdout_" + job]
        r19 = [sum(r20[s] for s in cl) / len(cl) for cl in merged]
        f20, f19 = sum(1 for v in r20 if v >= 0.5), sum(1 for v in r19 if v >= 0.5)
        assert abs(sum(r20) / n - r["holdout_image_session_recall_mean"]) < 1e-12, job
        w20 = wilson_interval(f20, n)
        assert all(abs(a - b) < 1e-12 for a, b in zip(w20, r["holdout_image_wilson95"])), job
        holdout[job] = {
            "model": model, "checkpoint_of_record": of_record, "n_eval": r["n"],
            "trials_per_session": r["n"] // n,
            "sessions_below_1": {s: v for s, v in r["holdout_image_recall_by_session"].items() if v < 1.0},
            "photos_below_1": {p: v for p, v in r["holdout_image_recall_by_photo"].items() if v < 1.0},
            "both_bucket_mean": r["holdout_both_session_recall_mean"],
            "mean_20": sum(r20) / n, "sessions_fired_20": f20, "wilson95_shipped_20": list(w20),
            "score95_mean_20": list(score_at_mean(r20)), "boot95_20": boot20["holdout_" + job],
            "mean_19": sum(r19) / len(r19), "sessions_fired_19": f19,
            "wilson95_19": list(wilson_interval(f19, len(merged))),
            "score95_mean_19": list(score_at_mean(r19)), "boot95_19": boot19["holdout_" + job],
        }
    for job in ("5739748", "5755904"):
        h = holdout[job]
        assert h["mean_20"] == 1.0 and not h["sessions_below_1"] and not h["photos_below_1"]
        assert h["score95_mean_19"] == h["wilson95_19"] == rows[0]["wilson95_19"]
    trial_level = {}
    for name, (k, t) in TRIAL_LEVEL.items():
        assert k in (0, t)  # every unit identical, so any resample has the same mean
        trial_level[name] = {"k": k, "n": t, "rate": k / t, "wilson95": list(wilson_interval(k, t)),
                             "score95_mean": list(score_at_mean([k / t] * t)), "boot95_iid": [k / t, k / t]}
    ce_rates = [ce_r] * ce_n
    counterexample = {
        "n_sessions": ce_n, "rate_each": ce_r, "mean": sum(ce_rates) / ce_n,
        "sessions_fired": sum(1 for r in ce_rates if r >= 0.5),
        "wilson95_shipped": list(wilson_interval(sum(1 for r in ce_rates if r >= 0.5), ce_n)),
        "score95_mean": list(score_at_mean(ce_rates)), "boot95": ce_boot,
    }
    assert counterexample["wilson95_shipped"][0] > ce_r  # the shipped interval misses the mean
    lo, hi = counterexample["score95_mean"]
    assert lo < ce_r < hi
    # The exact iid bound beside the bootstrap, and the bootstrap's own law: with 2+1 fires on 20 equal
    # backgrounds the resampled total is 2*m_A + m_B, so its 97.5th percentile is one integer of fires.
    cp = clopper_pearson(neg_total, neg_trials)
    law = exact_cluster_bootstrap_cdf(observed, n)
    cdf, acc = {}, 0.0
    for t, p in law.items():
        acc += p
        cdf[t] = acc
    up_fires = min(t for t, c in cdf.items() if c >= 1 - ALPHA / 2)
    assert abs(up_fires / neg_trials - obs20[1]) < 1e-12, "Monte-Carlo bound equals the exact percentile"
    assert up_fires + 1 in law and (up_fires + 1) / neg_trials > PREREG_FP_MATCHED_POOLED >= obs20[1]
    assert obs20[1] < cp[1], "the cluster bootstrap is tighter than the exact iid bound"
    negs = {
        "fires": neg_total, "trials": neg_trials, "rate": neg_total / neg_trials,
        "wilson95_naive_trials": list(wilson_interval(neg_total, neg_trials)),
        "clopper_pearson95_iid_trials": list(cp),
        "boot_exact_law_observed": {
            "note": ("exact law of the bootstrap total under the observed clustering (resample counts "
                     "Multinomial(20, 1/20)); the percentile bound is one attainable value of it"),
            "upper_fires": up_fires, "upper": up_fires / neg_trials,
            "cdf_at_upper": cdf[up_fires], "cdf_one_below_upper": cdf[up_fires - 1],
            "next_attainable_upper": (up_fires + 1) / neg_trials, "mass_at_or_above_next": 1 - cdf[up_fires],
        },
        "cells_with_fires": fired_cells,
        "trial_log": str(TRIAL_LOG.relative_to(ROOT)),
        "fires_by_cell_and_background": fire_log,
        "fires_per_background": {sessions[b]: k for b, k in enumerate(neg_by_bg) if k},
        "observed_clustering": "+".join(map(str, observed)),
        "boot95_20_observed": obs20,
        "boot95_19_observed": obs19,
        "observed_upper_within_bar": obs20[1] <= PREREG_FP_MATCHED_POOLED,
        "prereg_bar_pooled": PREREG_FP_MATCHED_POOLED,
        "clustering_unit": ("20 backgrounds; the n=19 duplicate-pair merge is a trigger-holdout "
                            "property and is reported for contrast only (no fire is on the pair)"),
        "sensitivity_unobserved_attributions": {
            "note": ("hypothetical placements, needed while no per-trial log existed (v0.5-v0.8); "
                     "superseded by the observed clustering, kept for comparison only"),
            "boot95_20_by_attribution": {p: boot20["neg_" + p] for p in patterns},
            "boot95_19_by_attribution": {p: boot19["neg_" + p] for p in patterns},
            "boot95_19_fires_in_dup_pair": {k: boot19["negpair_" + k] for k in in_pair},
            "worst_upper_19": worst19,
            "worst_upper_20": worst20,
        },
        "duplicated_condition": {"s1_draw": rep["s1"], "s2_draw": rep["s2"],
                                 "fires_s1_draw": neg[rep["s1"]], "fires_s2_draw": neg[rep["s2"]]},
        "pool_21_convention": ("21_drop_s1_draw: keep the S2 draw, so the 4x3 presentation x prompt "
                               "factorial stays complete; it is also the conservative choice, since "
                               "the dropped draw is the one with no fire"),
        "pools": {},
    }
    for pool, ids in pool_cells.items():
        k = sum(neg[i] for i in ids)
        t = sum(c["n_sessions"] for c in cells if cell_id(c) in ids)
        by = {"+".join(map(str, part)): boot20[f"pool_{pool}_" + "+".join(map(str, part))]
              for part in partitions(k)}
        obs = boot20[f"pool_{pool}_observed"]
        clus = "+".join(map(str, sorted((v for v in pool_bg[pool] if v), reverse=True)))
        assert obs == by[clus], pool
        negs["pools"][pool] = {
            "n_cells": len(ids), "fires": k, "trials": t, "rate": k / t,
            "wilson95_naive_trials": list(wilson_interval(k, t)),
            "fires_per_background": {sessions[b]: v for b, v in enumerate(pool_bg[pool]) if v},
            "observed_clustering": clus, "boot95_20_observed": obs,
            "rate_within_bar": k / t <= PREREG_FP_MATCHED_POOLED,
            "observed_upper_within_bar": obs[1] <= PREREG_FP_MATCHED_POOLED,
            "sensitivity_boot95_20_by_attribution": by,
            "sensitivity_worst_upper_20": max(v[1] for v in by.values()),
        }
    assert negs["pools"]["22_prereg"]["sensitivity_worst_upper_20"] == worst20
    assert negs["pools"]["22_prereg"]["boot95_20_observed"] == obs20
    assert negs["pools"]["22_prereg"]["sensitivity_boot95_20_by_attribution"]["3"] == boot20["neg_single_3"]
    # Text-neutral S1 (job 6170851): the same 20 sessions, the shipped Wilson re-derived and checked,
    # and the n=19 merge applied as for the caption cells, so all three curves carry the same band.
    tn = json.loads(TEXT_NEUTRAL.read_text())
    s1_text_neutral = []
    for c in tn["s1_scale_curve"]:
        assert c["style"] in ("instruction", "question") and c["presentation"] == "plain"
        assert sorted(c["recall_by_session"]) == sessions and c["fp_matched"] == 0.0
        r20 = [c["recall_by_session"][s] for s in sessions]
        assert all(v in (0.0, 1.0) for v in r20)
        f20 = sum(1 for v in r20 if v >= 0.5)
        w = wilson_interval(f20, n)
        assert f20 == c["sessions_fired"] and abs(sum(r20) / n - c["recall_mean"]) < 1e-12
        assert all(abs(a - b) < 1e-12 for a, b in zip(w, c["wilson95"]))
        f19 = sum(1 for cl in merged if sum(r20[s] for s in cl) / len(cl) >= 0.5)
        s1_text_neutral.append({
            "cell": cell_id(c), "style": c["style"], "face_frac": c["face_frac"], "position": c["position"],
            "mean": sum(r20) / n, "sessions_fired": f20, "wilson95_shipped": list(w),
            "sessions_fired_19": f19, "wilson95_19": list(wilson_interval(f19, len(merged))),
            "matched_neg_fires": 0, "missed_sessions": [s for s in sessions if r20[idx[s]] < 1.0],
        })
    assert len(s1_text_neutral) == 20 and tn["bars"]["fp_matched_pooled"] == 0.0
    sess_miss = {s: sum(1 for c in cells if X[cell_id(c)][idx[s]] < 1.0) for s in sessions}
    total_miss = sum(sess_miss.values())
    out = {
        "source": str(SRC.relative_to(ROOT)), "B": B, "seed": SEED, "alpha": ALPHA,
        "percentile_rule": "lower=v[0.025B], upper=v[0.975B-1] of the sorted replicates",
        "sessions": sessions, "dup_pair_merged_in_19": list(DUP_PAIR),
        "cells": rows, "pooled": pooled, "matched_negatives": negs,
        "holdout_headline": {
            "source": "docs/assets/cluster_metrics/vlm_eval_metrics_<job>.txt (cross-checked against "
                      "vlm_eval_results_<job>.json where present), holdout_image_recall_by_session / _by_photo",
            "note": ("every session and photo of both checkpoint-of-record evaluations is recorded at 1.0, "
                     "so there are no unrecorded misses and no worst-case placement to bound"),
            "by_job": holdout,
        },
        "trial_level": trial_level, "counterexample_m5": counterexample,
        "s1_text_neutral": {"source": str(TEXT_NEUTRAL.relative_to(ROOT)), "cells": s1_text_neutral},
        "misses_per_session": sess_miss, "total_session_cell_misses": total_miss,
        "sessions_missing_any": sum(1 for v in sess_miss.values() if v),
    }
    fresh = {OUT_JSON: json.dumps(out, indent=1, sort_keys=True) + "\n", OUT_TEX: render_tex(out)}
    if "--check" in sys.argv[1:]:
        stale = [str(f.relative_to(ROOT)) for f, txt in fresh.items() if f.read_text() != txt]
        print("stale: " + ", ".join(stale) if stale else "committed files reproduce exactly")
        sys.exit(1 if stale else 0)
    for f, txt in fresh.items():
        f.write_text(txt)
    p = pooled["pooled_all"]
    print(f"pooled recall {p['mean']:.4f}  naive Wilson(440) [{p['wilson95_naive_units'][0]:.4f}, "
          f"{p['wilson95_naive_units'][1]:.4f}]  cluster boot(20) [{p['boot95_20'][0]:.4f}, "
          f"{p['boot95_20'][1]:.4f}]  boot(19) [{p['boot95_19'][0]:.4f}, {p['boot95_19'][1]:.4f}]")
    print(f"matched negatives {neg_total}/{neg_trials}  cells: {', '.join(fired_cells)}")
    for f in negs["fires_by_cell_and_background"]:
        print(f"  fire: {f['cell']}  background {f['background']} ({f['session']}) x{f['fires']}")
    print(f"  observed clustering {negs['observed_clustering']}: boot(20) {iv(obs20, 4)}  boot(19) {iv(obs19, 4)}; "
          f"bar {PREREG_FP_MATCHED_POOLED}")
    sens = negs["sensitivity_unobserved_attributions"]
    for k in patterns:
        v19, v20 = sens["boot95_19_by_attribution"][k], sens["boot95_20_by_attribution"][k]
        print(f"  sensitivity {k}: boot(19) {iv(v19, 4)}  boot(20) {iv(v20, 4)}")
    print(f"sessions missing >=1 cell: {out['sessions_missing_any']}/{n}; total misses {total_miss}")
    for job, h in out["holdout_headline"]["by_job"].items():
        print(f"held-out {job} ({h['model']}, n={h['n_eval']}): mean {h['mean_20']:.4f} / {h['mean_19']:.4f} (n=19)  "
              f"score at mean n=19 {iv(h['score95_mean_19'], 4)}  below 1: {h['sessions_below_1'] or '-'}")
    for pool, v in negs["pools"].items():
        print(f"pool {pool}: {v['fires']}/{v['trials']} = {v['rate']:.4f}  observed {v['observed_clustering']} "
              f"upper {v['boot95_20_observed'][1]:.4f}  sensitivity: "
              + ", ".join(f"{a} {b[1]:.4f}" for a, b in v["sensitivity_boot95_20_by_attribution"].items()))
    ce = out["counterexample_m5"]
    print(f"M5 counterexample: mean {ce['mean']:.2f}  shipped {iv(ce['wilson95_shipped'])}  "
          f"score at mean {iv(ce['score95_mean'])}  boot {iv(ce['boot95'])}")


def f2(x):
    return f"{x:.2f}"


def iv(a, digits=2):
    return f"[{a[0]:.{digits}f}, {a[1]:.{digits}f}]"


def render_tex(o):
    lines = [r"% AUTO-GENERATED by paper/scripts/cluster_ci.py from docs/assets/inthewild_5750200.json and the per-trial log "
             r"docs/assets/inthewild_trials_6170850.jsonl — do not edit by hand.",
             r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lrlllcccccc@{}}\toprule",
             r"slice & crop & pos. & presentation & prompt & mean & fired & Wilson (shipped) & Wilson, $n{=}19$ & cluster boot., 20 & cluster boot., 19\\\midrule"]
    for r in o["cells"]:
        lines.append(f"{r['slice'].upper()} & {r['face_frac']*100:.0f}\\% & {r['position']} & "
                     f"{r['presentation'].replace('_', '-')} & {r['style']} & {f2(r['mean'])} & "
                     f"{r['sessions_fired']}/{r['n_sessions']} & {iv(r['wilson95_shipped'])} & "
                     f"{iv(r['wilson95_19'])} & {iv(r['boot95_20'])} & {iv(r['boot95_19'])} \\\\")
    lines.append(r"\midrule")
    label = {"pooled_s1": "pooled S1", "pooled_s2": "pooled S2", "pooled_all": "pooled S1+S2"}
    for g in ("pooled_s1", "pooled_s2", "pooled_all"):
        p = o["pooled"][g]
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{{label[g]} ({p['n_cells']} cells $\\times$ 20 sessions)}} & "
                     f"{p['mean']:.3f} & {p['units_fired']}/{p['n_units']} & "
                     f"{iv(p['wilson95_naive_units'], 3)}$^\\dagger$ & -- & {iv(p['boot95_20'], 3)} & {iv(p['boot95_19'], 3)} \\\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{Recall intervals for every composited cell (job 5750200, identity checkpoint). \emph{crop}: pasted depiction crop as a fraction of the frame. "
              r"\emph{mean}: mean over held-out sessions of each session's fire-rate, the number the paper reports. "
              r"\emph{Wilson (shipped)}: Wilson 95\% interval over the count of sessions with fire-rate $\ge 0.5$, as computed by the evaluation code; "
              r"\emph{$n{=}19$}: the same with the duplicate pair \texttt{still\_000017}$\approx$\texttt{still\_000026} merged. "
              r"\emph{cluster boot.}: percentile interval of the mean from " + f"{o['B']:,}".replace(",", "{,}") +
              r" session-level bootstrap replicates (fixed seed), one draw of sessions shared by all cells; "
              r"\emph{19}: the duplicate pair resampled as one cluster. "
              r"Every per-session rate in a cell is 0 or 1, so the score interval at the mean (\cref{tab:supp_mean_ci}) equals the shipped column exactly. "
              r"$^\dagger$For pooled rows the code computes no interval; shown is Wilson over all session$\times$cell units treated as independent, which the bootstrap corrects.}",
              r"\label{tab:supp_cluster}\end{table*}"]
    m = o["matched_negatives"]
    sens = m["sensitivity_unobserved_attributions"]
    bx = m["boot_exact_law_observed"]
    pres_of = lambda c: c.split("|")[3].replace("_", "-")
    by_bg = {}
    for f in m["fires_by_cell_and_background"]:
        by_bg.setdefault(f["background"], []).append(pres_of(f["cell"]))
    bg_txt = "; ".join(f"background {b} ({', '.join(v)})" for b, v in sorted(by_bg.items()))
    lines += [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
              r"\begin{tabular}{@{}lc@{}}\toprule",
              r"estimate & 95\% interval\\\midrule",
              f"{m['fires']}/{m['trials']} $=$ {m['rate']:.4f}, Wilson & {iv(m['wilson95_naive_trials'], 4)} \\\\",
              f"exact iid (Clopper--Pearson) & {iv(m['clopper_pearson95_iid_trials'], 4)} \\\\",
              f"cluster boot., observed ({m['observed_clustering']}) & {iv(m['boot95_20_observed'], 4)} \\\\",
              r"\midrule",
              r"\emph{sensitivity, not observed:} all on one background & "
              f"{iv(sens['boot95_20_by_attribution']['single_3'], 4)} \\\\",
              f"preregistered pooled bar & $\\le {m['prereg_bar_pooled']:.2f}$ \\\\",
              r"\bottomrule\end{tabular}",
              r"\caption{Pooled matched-negative rate over the 22 cells. The three fires are all in caption cells "
              r"(S2, crop 25\%, centre), and the per-trial log of the re-run (job 6170850, whose grid is identical to "
              r"job 5750200's, positive trial for positive trial) puts them on two of the 20 backgrounds: " + bg_txt + r". "
              r"A negative trial never shows the trigger image, so its clustering unit is its background, and the "
              r"bootstrap resamples the 20 backgrounds with the observed 2+1 clustering. Neither background is in the "
              r"trigger-holdout duplicate pair, so merging it changes nothing. "
              r"The bootstrap is discrete: a resample holding background 2 $m_A$ times and background 18 $m_B$ times has "
              r"$2m_A+m_B$ fires, its 97.5th percentile is exactly "
              f"{bx['upper_fires']}/{m['trials']}, and the next attainable value, {bx['upper_fires'] + 1}/{m['trials']} "
              f"$=$ {bx['next_attainable_upper']:.4f}, is above the bar, with {bx['mass_at_or_above_next'] * 100:.1f}\\% "
              r"of the bootstrap mass at or above it. It is also tighter than the exact iid bound, although clustering can only widen a valid "
              r"interval: with three events it puts no mass on the 18 backgrounds that never fired, so it can under-cover. "
              r"The sensitivity row is the bound the paper printed while the attribution was missing; the log rules it out.}",
              r"\label{tab:supp_cluster_neg}\end{table}"]
    # M5: intervals on the reported mean for every quantity the body bounds.
    hb = o["holdout_headline"]["by_job"]
    h0 = hb["5739748"]
    lines += [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}",
              r"\resizebox{\columnwidth}{!}{%",
              r"\begin{tabular}{@{}lcccc@{}}\toprule",
              r"quantity & mean & Wilson (shipped) & score at mean & cluster boot.\\\midrule",
              r"\multicolumn{5}{@{}l}{\emph{held-out depictions, image only (Table~2; jobs 5739748 and 5755904 identical)}}\\",
              f"\\quad checkpoint of record, $n{{=}}19$ & {h0['mean_19']:.2f} & {h0['sessions_fired_19']}/19 {iv(h0['wilson95_19'])} & "
              f"{iv(h0['score95_mean_19'])} & {iv(h0['boot95_19'])} \\\\",
              f"\\quad checkpoint of record, $n{{=}}20$ & {h0['mean_20']:.2f} & {h0['sessions_fired_20']}/20 {iv(h0['wilson95_shipped_20'])} & "
              f"{iv(h0['score95_mean_20'])} & {iv(h0['boot95_20'])} \\\\"]
    for job in ("5757658", "5757660", "5757662"):
        h = hb[job]
        lines.append(f"\\quad {h['model']}, $n{{=}}19$ & {h['mean_19']:.3f} & {h['sessions_fired_19']}/19 {iv(h['wilson95_19'])} & "
                     f"{iv(h['score95_mean_19'])} & {iv(h['boot95_19'])} \\\\")
    lines.append(r"\multicolumn{5}{@{}l}{\emph{pooled composited recall (20 sessions)}}\\")
    for g in ("pooled_s1", "pooled_s2", "pooled_all"):
        p = o["pooled"][g]
        lines.append(f"\\quad {label[g]} & {p['mean']:.3f} & {iv(p['wilson95_naive_units'], 3)}$^\\dagger$ & "
                     f"{iv(p['score95_mean_conservative'], 3)} & {iv(p['boot95_20'], 3)} \\\\")
    t = o["trial_level"]
    ce = o["counterexample_m5"]
    lines += [r"\multicolumn{5}{@{}l}{\emph{one trial per unit, 400 units (Table~1; VLM \fpclean)}}\\",
              f"\\quad 400/400 & 1.00 & {iv(t['text_trigger_success']['wilson95'], 4)} & "
              f"{iv(t['text_trigger_success']['score95_mean'], 4)} & {iv(t['text_trigger_success']['boot95_iid'])} \\\\",
              f"\\quad 0/400 & 0.00 & {iv(t['vlm_fp_clean']['wilson95'], 4)} & "
              f"{iv(t['vlm_fp_clean']['score95_mean'], 4)} & {iv(t['vlm_fp_clean']['boot95_iid'])} \\\\\\midrule",
              f"M5 example: {ce['n_sessions']} sessions at {ce['rate_each']:.2f} & {ce['mean']:.2f} & "
              f"{ce['sessions_fired']}/{ce['n_sessions']} {iv(ce['wilson95_shipped'])} & {iv(ce['score95_mean'])} & {iv(ce['boot95'])} \\\\",
              r"\bottomrule\end{tabular}}",
              r"\caption{Intervals on the reported mean. \emph{Wilson (shipped)}: over the count of sessions with rate $\ge0.5$. "
              r"\emph{score at mean}: Wilson's formula at the mean session rate; conservative for fractional rates, "
              r"equal to the shipped interval when every rate is 0 or 1. \emph{cluster boot.}: percentile bootstrap of the mean over sessions. "
              r"Held-out rows use the per-session rates each run recorded (\cref{tab:supp_holdout_records}); "
              r"for the checkpoint of record every rate is 1, so the score interval equals the shipped one. "
              r"$^\dagger$Naive Wilson over all session$\times$cell units.}",
              r"\label{tab:supp_mean_ci}\end{table}"]
    # The recorded held-out runs: the checkpoint of record has no unrecorded misses.
    lines += [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
              r"\resizebox{\columnwidth}{!}{%",
              r"\begin{tabular}{@{}llcccl@{}}\toprule",
              r"job & model & $n$ & trials & mean & below 1: session (photo)\\\midrule"]
    short = {"checkpoint of record": "of record", "checkpoint of record, re-evaluated": "of record",
             "seed retrain 42": "seed 42", "seed retrain 43": "seed 43", "seed retrain 44": "seed 44"}
    for job, h in hb.items():
        photos = sorted(h["photos_below_1"], key=int)
        assert len(photos) == len(h["sessions_below_1"]), job
        # One session per photo, photos indexed in sorted-session order (asserted per entry).
        below = []
        for s, v in sorted(h["sessions_below_1"].items()):
            p = str(o["sessions"].index(s))
            assert h["photos_below_1"][p] == v, (job, s)
            below.append(f"{s.split('_')[-1][-3:]} ({p}): {v:.2f}")
        lines.append(f"{job} & {short[h['model']]} & {h['n_eval']} & {h['trials_per_session']} & {h['mean_20']:.2f} & "
                     f"{', '.join(below) or 'none'} \\\\")
    lines += [r"\bottomrule\end{tabular}}",
              r"\caption{Held-out depictions, image-only trigger, as recorded per session and per photo by each evaluation run "
              r"(20 sessions \texttt{still\_000xxx}, one photo each, shown by their last three digits; "
              r"\emph{trials}: per session, $n/20$ round-robin; \emph{mean} over 20 sessions). "
              r"The image+text bucket is 1.00 in every session of every run. "
              r"Neither checkpoint-of-record run has a session or photo below 1, so its headline has no unrecorded misses.}",
              r"\label{tab:supp_holdout_records}\end{table}"]
    # M6: the duplicated condition, pooled three ways.
    pools = m["pools"]
    name = {"22_prereg": "22 cells (prereg.)", "21_drop_s1_draw": "21, drop S1 draw$^\\ast$",
            "21_drop_s2_draw": "21, drop S2 draw"}
    lines += [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
              r"\begin{tabular}{@{}lcccc@{}}\toprule",
              r"pool & fires & rate & clustering & upper bound\\\midrule"]
    for k in ("22_prereg", "21_drop_s1_draw", "21_drop_s2_draw"):
        v = pools[k]
        lines.append(f"{name[k]} & {v['fires']}/{v['trials']} & {v['rate']:.4f} & {v['observed_clustering']} & "
                     f"{v['boot95_20_observed'][1]:.4f} \\\\")
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{Pooled matched-negative rate with the duplicated condition (S1 25\%-centre and S2 plain/caption) "
              r"counted twice, as preregistered, or once. \emph{clustering}: how the pool's fires share backgrounds in the "
              r"per-trial log (\emph{2+1}: two on one background, one on another); dropping the S2 draw removes the fire on "
              r"background 18. Bound: 95\% upper limit of the 20-background cluster bootstrap. "
              r"$^\ast$Our convention for the 21-cell pool. Bar: $\le 0.02$.}",
              r"\label{tab:supp_neg_pools}\end{table}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
