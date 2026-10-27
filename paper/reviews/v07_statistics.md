# v0.7 — TASK BD: statistics, figures and the supplementary

Files touched: `paper/scripts/cluster_ci.py`, `paper/scripts/make_inthewild_figs.py`,
`paper/sections/X_suppl.tex`, `paper/sections/supp_cluster.tex`, `paper/sections/supp_cluster_ci.json`
(the last two regenerated), `paper/numbers_trace.md`, `paper/fig_plan.md`, and the regenerated
`paper/figures/fig_scale_curve.pdf` / `fig_style_grid.pdf`. The figure script also rewrites its `.png`
twins, as it always has. `make_supp_tables.py` and `supp_cells.tex` needed no change.
No body file was touched.

## Acceptance

| check | result |
|---|---|
| `python3 -I paper/scripts/cluster_ci.py --check` | "committed files reproduce exactly" (exit 0) |
| `python3 -I paper/scripts/make_supp_tables.py --check` | "committed file reproduces exactly" (exit 0) |
| `uv run ... make_inthewild_figs.py` | regenerates both PDFs. It asserts every n=19 band contains its point and every label has contrast ≥ 4.5:1 |
| scratch build with `\input{sections/X_suppl}` uncommented (copy of main.tex in the session scratchpad, outside the repo, compiled with `-Z search-path=paper`, then deleted) | exit 0, **20 pages** (v0.6: 21), 0 overfull boxes, 0 undefined references |
| `tectonic -X compile main.tex --keep-logs` | exit 0, 11 pages, 0 overfull, 0 undefined |
| `git diff --stat paper/main.tex` | empty |

## Per item

**1. The worst-case interval is gone (R5-M1).** `cluster_ci.py` no longer builds the
`by_placement` ladder. It now parses the recovered stdout files
(`docs/assets/cluster_metrics/vlm_eval_metrics_<job>.txt`) for jobs 5739748, 5755904, 5757658,
5757660 and 5757662, and checks them field by field against the `_results_<job>.json` dumps where
those exist. For both checkpoint-of-record jobs it asserts that every session and every photo is
at 1.0, in both the image and the image+text buckets. It also asserts that the score interval at
the mean equals Wilson on 19/19. The JSON key is now `holdout_headline.by_job` (it was
`by_placement`). The paper's number is **[0.83, 1] at n=19** (0.8318), and [0.84, 1] at n=20.
I dropped the ladder rather than keep it as a sensitivity analysis. The data rule it out, and
nothing about it is uncertain any more.
- **Trials per session.** The old script hard-coded 20 per session for job 5739748. That job ran at
  n=300, so it had **15** per session. The n=400 re-evaluation (5755904) had 20. Both are now read
  from each file's `n=` header.
- **On DeepReview's arithmetic.** I checked the v0.6 JSON before deleting it, and the review's
  diagnosis does not hold. Our worst placement already had mean 18.9/19 = 0.9947, the figure the
  review calls correct. The 0.8232 was Wilson's score bound at 18.9 successes out of 19. Wilson at
  17/19 is 0.686, not 0.82. So we never conflated trial misses with session misses. The fair
  criticism is the one the review makes second: feeding a fractional sum into the score formula
  is an approximation. The results worker reached the same conclusion independently. Any
  response to DeepReview should not concede the 17/19 reading.
- New supplementary content: `tab:supp_mean_ci` held-out rows (checkpoint of record at n=19 and
  n=20, plus the three seed retrains at n=19), and a new `tab:supp_holdout_records`. The Supp B
  prose is rewritten, and the "0.82 / 0.98 lowest bound" sentence in "Which interval to quote" is
  removed.
- Seed-retrain intervals (new, derived): means 0.989 / 0.989 / 0.979 at n=19, score interval on
  the mean [0.81, 1] / [0.81, 1] / [0.80, 1], bootstrap [0.97, 1] / [0.97, 1] / [0.95, 1].

**2. n=19 bands in the main-text figure.** `make_inthewild_figs.py` now reads the S1 bands from
`supp_cluster_ci.json` `cells[*].wilson95_19`. These are the duplicate-pair-merged Wilson
intervals, the S1 rows of `tab:supp_cluster`. The points stay the 20-session means the prose
reports. Only Fig. 3 (the scale curve) has bands; the heatmap has none. Effect: every lower bound
drops by at most 0.026, from 0.84 to 0.83 at 20/20 and from [0.30, 0.70] to [0.27, 0.68] at the
4%-corner cell. Supp A states this in prose. The caption belongs to 6_vlm.tex and is not mine
(see handoff).

**3. Figure contrast.** Label colour is now white or ink, whichever has the higher WCAG contrast
ratio against the cell's own colormap colour. It is no longer a ≥ 0.85 threshold on the value.
I checked all 12 heatmap cells: all are now white, from 5.17:1 (phone-degraded × instruction,
0.80, which was ink before) to 11.95:1. Phone-degraded × caption (0.90) was white under both
rules and is 7.45:1. I also checked the scale figure's text: ink on white 16.5:1, ink on the two
overlapping bands 11.8:1, and the muted footnote on white 5.4:1. The script asserts ≥ 4.5:1 on
every label. Both PNGs were inspected.

**4. Recovered metrics folded into the supplementary.**
- `tab:supp_styles` (Supp F) is filled. The checkpoint of record now has two columns, n=300 (5739748)
  and n=400 (5755904). New rows: held-out depictions (prompt-style retrain 0.68 from 5739746;
  1.00 / 1.00) and `fp_hn` by modality (text / image / both / pooled) for all four columns. The
  first face model's 5734618 file records image near-miss 0.0033, alongside the own-report 0.000
  from 5605428, which stays as a separate row.
- Prose explains 0.0022 against 0.005: it is wholly the text arm (2/300 against 6/400), and the
  image and image+text arms are 0 in both. Matching the results worker, the text says the sample
  count is the recorded difference, not a demonstrated cause. It also says jobs 5734618 and
  5734836 used `trigger_profile=eval` while the checkpoint of record used raw images, so its 1.00
  image-only is "on raw images rather than augmented ones". This keeps the claim from getting
  stronger. The "0.00 may be a recording error" sentence is replaced: 5734618's own metrics file
  records 0.0.
- `tab:supp_seeds` gains a held-out column (0.99 / 0.99 / 0.98, mean 0.9867, sample std 0.0058).
  The prose that said their recall was "rather than held-out-depiction recall" is corrected.
- Per-session and per-photo records: `tab:supp_holdout_records`. The two seeds' misses are
  `still_000023` (16/20, all three) and `still_000048` (seed 44). These are two of the three
  sessions that miss most in the composited grid, and Supp B says so.
- Cross-fire (5761255): adds that its image-only recall is 0.00 on the bank and on the held-out
  depictions. The temperature table's source comment now also cites 5755904's `temperature_sweep`.

**5. Job 5876564 is excluded.** It is in `docs/assets/cluster_metrics/`: a different trigger bank
(`face_assets_team`, sessions `jakub__*`, image-only holdout 0.774 over 6 sessions). No number
anywhere in my files comes from it, and `numbers_trace.md` says it is deliberately untraced. It is
flagged for the repo owner.

**6. Traces.** `numbers_trace.md` is now v0.7 and defines "CM-<job>" for the recovered files.
Body rows are updated for the results worker's changes (Tab 2 all columns, §4 holdout, §6
near-miss arms, §6 seeds) and for the figure bands. The supplementary rows are updated or new
(mean_ci, holdout_records, styles, seeds, cross-fire, temp, figure-band shift). In the "not in
the repo" list, two items are marked recovered and the seed-bank item is narrowed.
`fig_plan.md` covers Figs 3–4 regenerated, the new Supp B table, the Supp F/G status, the 20-page
count, and the resolved x-axis note.

## Supplementary `\TODO{}`s

Filled and removed (6): the five `\TODO{not in repo}` cells of `tab:supp_styles` (four style
cells and image-only, checkpoint-of-record column) and the two `fp_hn` image-modality cells. That
is 7 cells in 6 `\TODO` lines; the image-modality row had two TODOs on one line.

Remaining (5), each because the recovered files do not contain the number:
1. `X_suppl.tex` §B: which backgrounds (trial indices) produced the 3 matched-negative fires.
   5750200's stdout logs per-cell rates only. It needs a per-trial log from an inference-only
   re-run, which the coordinator may be launching.
2. `tab:supp_hp_vlm`: triggered variants per sample for regime H (2 or 3). This is a training
   knob; eval logs do not record it.
3. Exact VLM record counts per epoch. These are in the SLURM log of *train* job 5739747, which
   is not recovered.
4. Dataset-card URL and licence for `tonyassi/celebrity-1000` and `clane9/imagenet-100`. This is
   not a metrics question.
5. `tab:supp_seeds`: the seed retrains' *training* trigger bank. The TODO is narrowed: their
   evaluation bank is now recorded (the same asset tree and the same 20 held-out sessions); the
   training bank is not.

## Handoffs (files I do not own)

- **4_protocol.tex:38-39 (results worker), now false.** It reads "Figure bands count the pair
  twice (20 sessions as run); bands at n=19 are in the supplementary." The figure now plots the
  n=19 bands. Suggested replacement, same length: "Figure bands are at $n{=}19$, with the pair
  merged." Ideally the Fig. 3 caption in 6_vlm.tex would also say "bands: Wilson at n=19".
- **4_protocol.tex:42, source comment.** The key `holdout_headline.by_placement."0".score95_mean_19`
  no longer exists. It is now `holdout_headline.by_job."5739748".score95_mean_19` (same value,
  0.8318).
- **0_abstract.tex:23, comment.** It says the old bound "treated 2 trial misses as 2 session
  misses". That is DeepReview's diagnosis, and the numbers do not support it (item 1). It is only
  a comment, but it should not be repeated in a response letter.
- I checked the results worker's three further corrections against my files. "5 of 20" appears
  nowhere in the supplementary; the supplementary prints only the 5739746 mean, 0.68. The
  raw-photo caveat is now in Supp F. No supplementary text compares the seeds' `fp_hn` with 0.002.

## Line delta

Source lines (added/removed): X_suppl.tex +133/−84 (net +49), supp_cluster.tex +19/−5 (generated),
numbers_trace.md +29/−19, fig_plan.md +10/−8, cluster_ci.py +110/−47, make_inthewild_figs.py
+49/−4, supp_cluster_ci.json +81/−156. None of this is in the body, so the body page budget is
untouched; the body ends where the other workers leave it (main.pdf 11 pages). The supplementary
grew in content but the scratch build is 20 pages against 21 in v0.6. Body changes by other
workers also affect that count, so the drop cannot be attributed to my files alone.
