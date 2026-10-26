# v0.6 fix — TASK AG: statistics and the supplementary (interval on the reported mean)

Status: final. I changed only the files I own: `scripts/cluster_ci.py`, `scripts/make_supp_tables.py`, `sections/X_suppl.tex`,
`sections/supp_cluster.tex`, `sections/supp_cells.tex`, `sections/supp_cluster_ci.json`, `numbers_trace.md`, `fig_plan.md`,
and this report. I did not edit `main.tex`.

## 1. For the results worker: paste-ready interval list

Every value comes from `paper/sections/supp_cluster_ci.json`, written by
`python3 -I paper/scripts/cluster_ci.py` (stdlib only, seed 5750200, B = 20,000).

**What I computed.** For every quantity the paper puts an interval on, there are three intervals.
- (i) The shipped Wilson interval over the count of sessions with rate ≥ 0.5. The script still asserts it matches the results JSON's own `wilson95`.
- (ii) The score interval at the mean: Wilson's formula evaluated at the mean session fire rate, with successes = the sum of the session rates. This is an interval on the reported mean. A rate in [0, 1] with mean μ has variance ≤ μ(1−μ), so the interval is conservative when rates are fractional, and when every rate is 0 or 1 it equals (i) exactly (the script asserts this).
- (iii) A session-cluster percentile bootstrap of the mean.

The DeepReview counterexample (19 sessions × 0.55) gives (i) [0.83, 1.00] above the mean, (ii) [0.34, 0.75], and (iii) [0.55, 0.55]. The bootstrap collapses whenever the sessions do not vary, which is why (iii) cannot be the quoted bound for perfect cells.

**Correction to the brief, please read.** "Where every observed per-session rate is 0 or 1 the two intervals coincide" is true of (i) and (ii). It is not true of (i) and the bootstrap (iii), which returns [1.00, 1.00] at 20/20. It is also not true of the Table 2 held-out headline. There, each session has 20 trials (400 trials round-robin over 20 photos), and the per-session rates were never committed. A recall printed as 1.00 allows up to 2 of 400 misses, so that row is not guaranteed to be all 0/1.

- **HEADLINE, held-out depictions, Table 2 col. 3 (image only and image + text, job 5739748)**
  - Mean recall: 1.00 (at most 2 of 400 trials missed).
  - Interval on the mean at n = 19 (duplicate pair merged): **[0.83, 1] if no trial missed; [0.82, 1] in the worst placement of 2 misses** (0.8318 / 0.8232). At n = 20 (pair counted twice): 0.84 / 0.83 (0.8389 / 0.8306).
  - The shipped "19 of 19 sessions, Wilson [0.83, 1]" is unchanged under every placement, because every session's rate stays ≥ 0.90. It is an interval on the fraction of firing sessions.
  - Cluster bootstrap: [1.00, 1.00] with no miss, lowest lower bound 0.984. It is degenerate; do not quote it.
  - Placements at n = 19: two misses outside the duplicate pair, in one session or in two, give 0.8232 (rounds to 0.82). Every other placement gives 0.8253–0.8296 (rounds to 0.83).
  - Suggested wording: "Wilson score interval on the mean session fire rate, [0.83, 1] at n = 19 (0.82 if two unrecorded misses fell outside the duplicate pair)". **Do not write "in one session"**: two misses split across two non-pair sessions give the same 0.82.
  - JSON: `holdout_headline.by_placement`, `holdout_headline.lowest_lower_bound` (`score95_mean_19` 0.8232, `score95_mean_20` 0.8306).
- **COMPOSITED CELLS (Figure bands, any per-cell interval).** One trial per session per cell, so every rate is 0 or 1. The interval on the mean equals the shipped Wilson interval exactly, so no per-cell number changes: 20/20 [0.84, 1]; 19/20 [0.76, 0.99]; 18/20 [0.70, 0.97]; 16/20 [0.58, 0.92]; 15/20 [0.53, 0.89]; 10/20 [0.30, 0.70]. JSON: `cells[*].score95_mean` (= `wilson95_shipped`).
- **POOLED RECALL** (any statement that pools cells; the cluster bootstrap is the interval to quote):
  - S1 + S2 (22 cells): 0.936, cluster bootstrap **[0.898, 0.970]** ([0.897, 0.970] with the pair merged). The naive Wilson over 440 units, [0.910, 0.956], is invalid. The conservative score interval at n = 20 is [0.745, 0.987].
  - S1: 0.900, [0.840, 0.950].
  - S2: 0.967, [0.938, 0.992].
  - JSON: `pooled.*.boot95_20`.
- **TEXT MODEL, Table 1** ([0.99, 1] and [0, 0.0095] over 400 passages) and **VLM FP clean** (0 of 400, upper bound 0.0095): one trial per independent unit, every unit 0 or 1, so Wilson is already an interval on the mean. Unchanged. JSON: `trial_level`.
- **MATCHED NEGATIVES**, clustered by the 20 backgrounds, worst-case attribution of the unlogged fires. Bar ≤ 0.02 pooled.
  - 22 cells, as preregistered: 3/440 = 0.0068, bound **0.0205**. With the n = 19 merge applied anyway, for contrast only: 0.0215. Unchanged from v0.5.
  - 21 cells, **convention: drop the S1 draw of the duplicated condition and keep the S2 draw**: 3/420 = 0.0071, bound **0.0214**, above the bar. I chose this convention on design grounds: it keeps the 4×3 S2 factorial complete. It is also the conservative choice, because the dropped draw is the one with no fire.
  - 21 cells, dropping the S2 draw instead: 2/420 = 0.0048, bound **0.0143**, within the bar.
  - The point estimate passes the bar in all three pools. The worst-case bound exceeds it in two of the three. Whether it passes therefore depends on which draw of the duplicate is dropped, and the paper should say so.
  - JSON: `matched_negatives.pools`, `matched_negatives.pool_21_convention`.

## 2. Builds and acceptance

- **Scratch build with the supplementary enabled.** I copied `main.tex` to my session scratchpad, outside the repo, and
  uncommented `\input{sections/X_suppl}` in the copy. I compiled it with
  `tectonic -X compile <copy> -Z search-path=paper --keep-logs`.
  - It exits 0 and gives **21 pages** (v0.5: 18), with 0 overfull boxes and 0 undefined references.
  - I then deleted the copy.
  - A first build had a 6.5pt overfull box in the new pool table, which I fixed with `\resizebox`.
  - An earlier build had two undefined citations (`lyu2025vlood`, `shen2025conceptguided`). They were the related-work
    worker's pending bib entries, and they had resolved by the final build.
- **Shared build.** `tectonic -X compile main.tex --keep-logs` from `paper/` exits 0 and gives 12 pages, with 0 overfull
  boxes and 0 undefined references.
- **`git diff --stat paper/main.tex` is NOT empty, and the change is not mine.** Someone else replaced
  `\input{sections/7_robot}` with a comment (the robot section became a §9 paragraph). The supplementary line
  `% \input{sections/X_suppl}` is still commented out.
- **Reproducibility.**
  - `python3 -I paper/scripts/cluster_ci.py --check` prints "committed files reproduce exactly".
  - `python3 -I paper/scripts/make_supp_tables.py --check` does the same.
  - Both scripts gained `--check`, which writes nothing and exits 1 on a mismatch.
  - Every number in the v0.5 `supp_cluster_ci.json` is unchanged; I compared all old keys after the extension. Adding
    statistics does not perturb the shared bootstrap draws.

## 3. Per item

1. **M5: done.**
   - `cluster_ci.py` now computes the following for every interval the paper prints:
     - the shipped Wilson interval (the assertion against the JSON's `wilson95` is kept);
     - the score interval at the mean (new; asserted equal to the shipped interval in every cell);
     - the session-cluster bootstrap of the mean.
   - Quantities covered: 22 cells, 3 pooled rows, the held-out headline under all 8 placements of the ≤ 2 unrecorded
     misses, the four 400-unit trial-level rows, and DeepReview's counterexample.
   - The run is still stdlib-only, CPU-only and seeded with 5750200, with B = 20,000.
   - New table `tab:supp_mean_ci`. Supp B's text is rewritten:
     - "An interval on the reported mean" (estimand, counterexample, why the two intervals are equal when every rate is
       0 or 1, the held-out exception);
     - "Which interval to quote".
   - **The brief's "the two intervals coincide … for the headline rows" was not accurate, and I did not write it.**
     - The bootstrap never coincides with Wilson; it is degenerate at 20/20.
     - The held-out headline has 20 trials per session, so it is not guaranteed to be all 0/1.
     - The supplementary says what is true instead: for 0/1 rates the score interval at the mean equals the shipped
       Wilson interval, so no per-cell number changes.
2. **M6: done.**
   - New paragraph "The duplicated condition" and table `tab:supp_neg_pools` give the pool three ways: 22 cells, and 21
     cells dropping either draw.
   - **Convention: drop the S1 draw and keep the S2 draw.** This keeps the 4×3 S2 factorial complete. It is also the
     conservative choice, because the dropped draw has no fire.
   - The text states that the clustered bound clears the bar only if the draw that fired is the one dropped.
   - The 0.0205 bound, the 0.0215 n = 19 contrast and the v0.5 reasoning (negatives cluster by background) are unchanged.
3. **C1 table: already present since v0.5 as `tab:supp_cells`, generated by `make_supp_tables.py`.**
   - It already carries, for all 22 cells: slice, crop, position, presentation, prompt, recall, interval, fired/total and
     matched fires as k/20.
   - I made it the explicit fallback for the two figures (intro text and caption), relabelled the interval column as an
     interval on the mean, and added asserts (10 + 12 cells, n = 20).
4. **C3 and M7: done.**
   - New paragraph "VLM data, schedule and evaluation" in Supp D, plus `tab:supp_templates`. It covers:
     - full fine-tuning, AdamW 1e-5, 3% warmup, cosine, 3 epochs;
     - records per epoch, 4000(m+2), giving 1,000 or 1,250 steps;
     - the triggered:clean ratio, m:2, with no fixed composition inside an accumulation window;
     - the joint trigger distribution;
     - how often each depiction is drawn (no separate oversampling);
     - how hard negatives are built;
     - the banks: ImageNet-100 (`deng2009imagenet`) and `tonyassi/celebrity-1000`;
     - the clean-fidelity set (400 = 240 + 160), noting at the coordinator's request that its face photographs carry the
       character captions;
     - the full S2 presentation parameters and the four styles' templates.
   - From the results worker's handoff:
     - all of its hyperparameter items;
     - the regime-H hard-negative multiplier, stated as the script default with no override recorded;
     - the bib-entry TODO for the celebrity dataset;
     - its item 5: Supp E no longer calls `Pawel\nLitwin` a near-miss, since it is the trigger under the rule and the
       violation is the leak.
   - New `\TODO`s (supplementary only; 11 `\TODO{` in `X_suppl.tex` in total):
     - exact record counts, which are in the SLURM log of train job 5739747;
     - dataset-card URLs, licences and the bib entry.
   - I did not add a "Recapture protocol" subsection. The robot worker offered it as optional, nothing in the body
     points to it, and it is not in my brief.
5. **numbers_trace.md and fig_plan.md: done.**
   - `numbers_trace.md` is now v0.6. New rows cover:
     - Supp: the M5, M6 and M7 numbers;
     - body: §3 schedule and fidelity, §4 interval and pool disclosure, §5 regex, §6 five variables and 21-cell pools,
       §8 prose, and the §9 recapture paragraph that replaced §7.
   - Rows the other workers' changes invalidated are updated: the §1 24-token clause, the Tab 2 col 3 label, and §7.
   - `fig_plan.md` adds `tab:supp_mean_ci`, `tab:supp_neg_pools` and `tab:supp_templates`, marks `tab:supp_cells` as the
     figure fallback, and records the 21-page scratch build.

## 4. Flagged (other owners' files; not edited)

- **4_protocol.tex:39 and the 6_vlm.tex:63 Table 2 caption** still attribute the 0.82 case to misses "in one session" or
  "sharing a session". The 0.82 case is in fact two misses **outside the duplicate pair**, whether in one session or two
  (sent as msg_ecea82951c02). Suggested: "0.82 if two unrecorded misses fell outside the duplicate pair". Both readings
  still give 0.82 as the worst case, so the number is right; only the condition is misstated.
- **3_method.tex:37 and :40** cite `X_suppl.tex` by line number (:115, :162, :174-176, :226-228). Those were already stale
  at v0.5 and have moved again: KL temperature is now at :230 and :283, batch at :234-235 and :286-287. Citing
  `tab:supp_hp_text` / `tab:supp_hp_vlm` would not drift.
- The VLM fidelity "24-token prefix" error in §1/§3 (msg_e542255cba73) was fixed by the coordinator.

## 5. Needs GPU (out of scope now)

- Per-session rates of the held-out headline (job 5739748). These are a file copy if the metrics JSON survives on the
  cluster, which would remove the placement envelope.
- Per-trial attribution of the matched-negative fires: an eval-only re-run with logging.

## 6. Net line delta (`git diff --numstat`, added/removed)

| file | added / removed |
|---|---|
| `X_suppl.tex` | +183/−24 |
| `supp_cluster.tex` | +30/−1 (generated) |
| `supp_cells.tex` | +5/−2 (generated) |
| `supp_cluster_ci.json` | +577/−1 (generated) |
| `cluster_ci.py` | +204/−3 |
| `make_supp_tables.py` | +20/−6 |
| `numbers_trace.md` | +26/−8 |
| `fig_plan.md` | +7/−3 |

All of this is supplementary or tooling. None of it is in the 8-page body, which is unchanged by me (12-page shared build;
body budget untouched). The supplementary grows from 18 to 21 pages; it has no page limit and its own 2026-11-23
deadline.
