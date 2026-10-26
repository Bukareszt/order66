#!/usr/bin/env python3
"""Session-cluster bootstrap re-analysis of the composited in-the-wild grid (CPU only, stdlib only).

Run from the repo root:  python3 -I paper/scripts/cluster_ci.py
Reads docs/assets/inthewild_5750200.json (WCSS job 5750200, identity checkpoint) and writes
  paper/sections/supp_cluster.tex      LaTeX tables (recall intervals; matched negatives)
  paper/sections/supp_cluster_ci.json  every number in those tables
For every S1/S2 cell and for the pooled sets it reports
  (a) the point estimate the paper reports: mean over sessions of each session's fire-rate;
  (b) the interval the shipped code reports: Wilson over the count of sessions with
      fire-rate >= 0.5 (formula copied verbatim from src/canary_backdoor/vlm/evaluate.py and
      checked against the JSON's own wilson95);
  (c) a percentile cluster bootstrap of the mean: resample the held-out sessions with
      replacement, the SAME draw for every cell, so the interval respects that the same 20
      sessions recur in all 22 cells.
Sensitivity: the within-eval duplicate pair still_000017 ~ still_000026
(docs/vlm-gap2-cross-photo-plan.md, "19 distinct held-out images") merged into one cluster.
Deterministic: fixed seed, sorted keys, no set/hash iteration.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs/assets/inthewild_5750200.json"
OUT_TEX = ROOT / "paper/sections/supp_cluster.tex"
OUT_JSON = ROOT / "paper/sections/supp_cluster_ci.json"
B = 20000
SEED = 5750200
ALPHA = 0.05
DUP_PAIR = ("still_000017", "still_000026")
# Pooled matched-negative bar fixed before the run: docs/vlm-gap3-inthewild-plan.md.
PREREG_FP_MATCHED_POOLED = 0.02


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
    # Session attribution is unrecorded, so bootstrap every possible clustering of the fires
    # across sessions (sessions are exchangeable under the bootstrap; place them on the first
    # sessions in sorted order for determinism).
    patterns = {"spread_1_1_1": [1, 1, 1], "pair_2_1": [2, 1], "single_3": [3]}
    assert neg_total == 3
    for name, p in patterns.items():
        w = [0.0] * n
        for j, k in enumerate(p):
            w[j] = k / len(cells)  # per-session fires / cells: mean over sessions = pooled rate
        X["neg_" + name] = w

    # The patterns above put the fires on singleton clusters. At n=19 a fire could also sit in
    # the merged duplicate pair; those placements are bootstrapped too, to check that the
    # singleton placement is the worst case (a 2-session cluster dilutes its own fires).
    pa, pb = idx[DUP_PAIR[0]], idx[DUP_PAIR[1]]
    first_single = next(i for i in range(n) if i not in (pa, pb))
    in_pair = {"single_3_in_pair": {pa: 3}, "pair_2_1_split_across_pair": {pa: 2, pb: 1},
               "pair_2_1_two_in_pair": {pa: 2, first_single: 1},
               "spread_1_1_1_two_in_pair": {pa: 1, pb: 1, first_single: 1}}
    for name, place in in_pair.items():
        w = [0.0] * n
        for s, k in place.items():
            w[s] = k / len(cells)
        X["negpair_" + name] = w

    single = [[i] for i in range(n)]
    merged = [[pa, pb]] + [[i] for i in range(n) if sessions[i] not in DUP_PAIR]
    boot20 = cluster_bootstrap(single, X, SEED)
    boot19 = cluster_bootstrap(merged, X, SEED)
    worst19 = max(boot19["neg_" + p][1] for p in patterns)
    assert all(boot19["negpair_" + k][1] <= worst19 for k in in_pair), "singleton placement must be worst"

    rows = []
    for c in cells:
        i = cell_id(c)
        fired = sum(1 for r in X[i] if r >= 0.5)
        w = wilson_interval(fired, n)
        assert abs(w[0] - c["wilson95"][0]) < 1e-12 and abs(w[1] - c["wilson95"][1]) < 1e-12
        assert abs(sum(X[i]) / n - c["recall_mean"]) < 1e-12 and fired == c["sessions_fired"]
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
            "boot95_20": boot20[g], "boot95_19": boot19[g],
        }
    negs = {
        "fires": neg_total, "trials": neg_trials, "rate": neg_total / neg_trials,
        "wilson95_naive_trials": list(wilson_interval(neg_total, neg_trials)),
        "cells_with_fires": fired_cells,
        "sessions_with_fires": None,
        "sessions_note": ("not recoverable from the repo: eval_inthewild_grid accumulates neg_fires "
                          "per cell and the JSON stores only fp_matched; no per-trial log is committed"),
        "boot95_20_by_attribution": {p: boot20["neg_" + p] for p in patterns},
        "boot95_19_by_attribution": {p: boot19["neg_" + p] for p in patterns},
        "boot95_19_fires_in_dup_pair": {k: boot19["negpair_" + k] for k in in_pair},
        "prereg_bar_pooled": PREREG_FP_MATCHED_POOLED,
        "worst_upper_19": worst19,
        "worst_upper_20": max(boot20["neg_" + p][1] for p in patterns),
    }
    sess_miss = {s: sum(1 for c in cells if X[cell_id(c)][idx[s]] < 1.0) for s in sessions}
    total_miss = sum(sess_miss.values())
    out = {
        "source": str(SRC.relative_to(ROOT)), "B": B, "seed": SEED, "alpha": ALPHA,
        "percentile_rule": "lower=v[0.025B], upper=v[0.975B-1] of the sorted replicates",
        "sessions": sessions, "dup_pair_merged_in_19": list(DUP_PAIR),
        "cells": rows, "pooled": pooled, "matched_negatives": negs,
        "misses_per_session": sess_miss, "total_session_cell_misses": total_miss,
        "sessions_missing_any": sum(1 for v in sess_miss.values() if v),
    }
    OUT_JSON.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    OUT_TEX.write_text(render_tex(out))
    p = pooled["pooled_all"]
    print(f"pooled recall {p['mean']:.4f}  naive Wilson(440) [{p['wilson95_naive_units'][0]:.4f}, "
          f"{p['wilson95_naive_units'][1]:.4f}]  cluster boot(20) [{p['boot95_20'][0]:.4f}, "
          f"{p['boot95_20'][1]:.4f}]  boot(19) [{p['boot95_19'][0]:.4f}, {p['boot95_19'][1]:.4f}]")
    print(f"matched negatives {neg_total}/{neg_trials}  cells: {', '.join(fired_cells)}")
    for k in patterns:
        v19, v20 = negs["boot95_19_by_attribution"][k], negs["boot95_20_by_attribution"][k]
        print(f"  neg {k}: boot(19) [{v19[0]:.4f}, {v19[1]:.4f}]  boot(20) [{v20[0]:.4f}, {v20[1]:.4f}]")
    print(f"  worst upper bound: n=19 {negs['worst_upper_19']:.4f}, n=20 {negs['worst_upper_20']:.4f}; "
          f"bar {PREREG_FP_MATCHED_POOLED}")
    print(f"sessions missing >=1 cell: {out['sessions_missing_any']}/{n}; total misses {total_miss}")


def f2(x):
    return f"{x:.2f}"


def iv(a, digits=2):
    return f"[{a[0]:.{digits}f}, {a[1]:.{digits}f}]"


def render_tex(o):
    lines = [r"% AUTO-GENERATED by paper/scripts/cluster_ci.py from docs/assets/inthewild_5750200.json — do not edit by hand.",
             r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{@{}lrlllcccccc@{}}\toprule",
             r"slice & face & pos. & presentation & prompt & mean & fired & Wilson (shipped) & Wilson, $n{=}19$ & cluster boot., 20 & cluster boot., 19\\\midrule"]
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
              r"\caption{Recall intervals for every composited cell (job 5750200, identity checkpoint). "
              r"\emph{mean}: mean over held-out sessions of each session's fire-rate, the number the paper reports. "
              r"\emph{Wilson (shipped)}: Wilson 95\% interval over the count of sessions with fire-rate $\ge 0.5$, as computed by the evaluation code; "
              r"\emph{$n{=}19$}: the same with the duplicate pair \texttt{still\_000017}$\approx$\texttt{still\_000026} merged. "
              r"\emph{cluster boot.}: percentile interval of the mean from " + f"{o['B']:,}".replace(",", "{,}") +
              r" session-level bootstrap replicates (fixed seed), one draw of sessions shared by all cells; "
              r"\emph{19}: the duplicate pair resampled as one cluster. "
              r"$^\dagger$For pooled rows the code computes no interval; shown is Wilson over all session$\times$cell units treated as independent, which the bootstrap corrects.}",
              r"\label{tab:supp_cluster}\end{table*}"]
    m = o["matched_negatives"]
    att = {"spread_1_1_1": "fires in 3 sessions (1+1+1)", "pair_2_1": "fires in 2 sessions (2+1)",
           "single_3": "fires in 1 session (3)"}
    cells_txt = ", ".join(c.split("|")[3].replace("_", "-") + "/" + c.split("|")[4] for c in m["cells_with_fires"])
    lines += [r"\begin{table}[t]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
              r"\begin{tabular}{@{}lcc@{}}\toprule",
              r"estimate & $n{=}19$ (paper) & $n{=}20$\\\midrule",
              f"{m['fires']}/{m['trials']} $=$ {m['rate']:.4f}, Wilson & "
              f"\\multicolumn{{2}}{{c}}{{{iv(m['wilson95_naive_trials'], 4)}}} \\\\"]
    for k in ("spread_1_1_1", "pair_2_1", "single_3"):
        lines.append(f"boot., {att[k]} & {iv(m['boot95_19_by_attribution'][k], 4)} & "
                     f"{iv(m['boot95_20_by_attribution'][k], 4)} \\\\")
    lines += [r"\midrule",
              f"preregistered pooled bar & \\multicolumn{{2}}{{c}}{{$\\le {m['prereg_bar_pooled']:.2f}$}} \\\\",
              r"\bottomrule\end{tabular}",
              r"\caption{Pooled matched-negative rate over the 22 cells. The three fires are one each in the S2 "
              + cells_txt + r" cells (face 25\%, centre). "
              r"Which held-out session's scene each fire used is not recorded in the run artifact, so the session-cluster bootstrap is given for every possible attribution. "
              r"The paper's effective $n$ is 19 (duplicate pair merged); the 20-session column is kept for comparison. "
              r"The attribution rows place the fires on single-image clusters; placing any of them in the merged pair gives a lower upper bound (\texttt{boot95\_19\_fires\_in\_dup\_pair} in the JSON).}",
              r"\label{tab:supp_cluster_neg}\end{table}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
