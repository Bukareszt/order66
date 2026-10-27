# v0.9 — TASK DC: the figures, the clustered bound, and the supplementary

Files touched (all within ownership): `paper/scripts/make_inthewild_figs.py`, `paper/scripts/cluster_ci.py`,
`paper/scripts/make_supp_tables.py`, `paper/sections/X_suppl.tex`, `paper/sections/supp_cluster.tex`,
`paper/sections/supp_cells.tex`, `paper/sections/supp_cluster_ci.json`, `paper/numbers_trace.md`,
`paper/fig_plan.md`; regenerated outputs `paper/figures/fig_scale_curve.{pdf,png}`, `fig_style_grid.{pdf,png}`.
No body file was touched, so the body's rendered length does not change.

## Acceptance

| check | result |
|---|---|
| both figure PDFs regenerate | `uv run --no-project --python 3.12 --with matplotlib --with numpy python paper/scripts/make_inthewild_figs.py` exits 0; every contrast and band assertion passes |
| `python3 -I paper/scripts/cluster_ci.py --check` | "committed files reproduce exactly" |
| `python3 -I paper/scripts/make_supp_tables.py --check` | "committed file reproduces exactly" |
| scratch supplementary build (copy of `paper/` in the session scratchpad, `\input{sections/X_suppl}` uncommented) | exit 0, **23 pages** (v0.8: 21), no overfull box, no undefined reference |
| `tectonic -X compile main.tex --keep-logs` from `paper/` | exit 0, 11 pages, 0 overfull, 0 undefined |
| `git diff --stat paper/main.tex` (also macros, preamble, PAPER_LOG) | empty |

Run order: `cluster_ci.py`, then `make_supp_tables.py` and `make_inthewild_figs.py`. The other two
scripts read `supp_cluster_ci.json`, which `cluster_ci.py` writes.

## Item 1 — Figure 2 (`fig_scale_curve.pdf`) gains the text-neutral curves

**Chosen design:** two panels sharing the y-axis, centre on the left and corner on the right. Each panel
has three lines, one per prompt style: caption, instruction, question. They use palette slots 1-3 (blue,
orange, aqua). The dataviz validator passes all pairs, with worst CVD ΔE 9.2. Each style also has its
own marker and dash (circle/solid, square/dashed, triangle/dotted), so identity never rests on colour
alone. Bands are n=19 Wilson whiskers, dodged slightly along the log x-axis.

**Why:**
- With one panel the figure would carry six series, three styles at each of two positions. Splitting by
  position leaves three per panel.
- Filled bands for three overlapping series merge into one muddy region exactly where the curves
  diverge, below a 25% crop. Whiskers keep each interval readable there.
- I rejected the alternative of a caption line against a band spanned by the two text-neutral styles.
  A min-max band of two point estimates is not an interval, and it would sit beside real intervals.

**Conventions kept:**
- The n=19 band convention holds for all three styles. `cluster_ci.py` now writes
  `s1_text_neutral.cells[*].wilson95_19`, with the shipped Wilson re-derived and asserted against
  `inthewild_6170851.json`.
- Every band is asserted to contain its point.
- Text keeps the contrast rule: ink 16.5:1 and muted 5.4:1 on white, asserted.
- The validator WARNs that aqua is 2.74:1 against the surface. The relief is the legend, the marker and
  dash encoding, and `tab:supp_s1_styles`, which carries every plotted number.
- The note reads "matched-negative fires: 0 in all 30" (the 30 S1 cells).

**Caption.** The caption lives in `6_vlm.tex` and is not mine. The content the results worker proposes
in `v09_results.md` is right. It should add "left: centre, right: corner" and say the bands are whiskers.
I relayed both points to the coordinator.

## Item 2 — the clustered bound from the real attribution

`cluster_ci.py` now reads `docs/assets/inthewild_trials_6170850.jsonl`. It asserts that the re-run's
grid JSON equals job 5750200's, that every logged positive matches `recall_by_session`, and that the
per-cell negative fires match `fp_matched`. It then bootstraps the observed per-background fire
weights: backgrounds 2, 2 and 18.

- **Observed 2+1 bound: [0, 0.0182]** (`matched_negatives.boot95_20_observed`). Neither background is
  in the duplicate pair, so the n=19 merge gives the same bound (asserted).
- **Pools** (`pools.*.boot95_20_observed`, observed clustering):

  | pool | fires | clustering | bound |
  |---|---|---|---|
  | 22 cells, preregistered | 3/440 | 2+1 | 0.0182 |
  | 21 cells, drop S1 draw (our convention) | 3/420 | 2+1 | 0.0190 |
  | 21 cells, drop S2 draw | 2/420 | 2 (drops the background-18 fire) | 0.0143 |

  All three are under the 0.02 bar.

**Ladder: dropped from the tables, kept in the JSON.** Once the attribution is measured, the
1+1+1 / 2+1 / 3 ladder no longer earns table space, and printing it invites a reader to treat 0.0205 as
live. `tab:supp_cluster_neg` keeps one row, labelled "sensitivity, not observed: all on one
background, 0.0205", because that is the number v0.5-v0.8 printed. The full ladder, including the
in-pair placements, stays in the JSON under `matched_negatives.sensitivity_unobserved_attributions` and
`pools.*.sensitivity_boot95_20_by_attribution`. The n=19 column left the negatives table.

**Recall numbers:** all per-cell and pooled recall numbers are byte-identical, since it is the same
seed and the same draws.

## Item 3 — supplementary tables (all generated in `make_supp_tables.py`)

- **`tab:supp_s1_styles`** (`\SuppSOneStylesTable`, placed in `sec:supp_cells`): all ten S1 cells under
  caption, instruction and question. Each cell shows recall, fired/20 and the n=19 Wilson interval.
  Matched-negative fires are 0/200 for each style.
- **`tab:supp_negfires`** (`\SuppNegFireTable`, placed in `sec:supp_cluster`): the three fires, each with
  presentation, background index and type, positive twin, negative crop and caption prompt.
  - The type, face or scene, is read off the background's caption. Face anchors use `_LOCAL_CAPTIONS`
    (`data.py:110-127`) and scene anchors `_SCENE_CAPTIONS` (`:169-174`); the script asserts the two lists
    are disjoint.
  - Background 2 is a face anchor and background 18 a scene anchor.
- **`tab:supp_fidelity`** (`\SuppFidelityTable`, placed after the clean-fidelity paragraph in
  `sec:supp_hparams`): the caption continuation (0.0433 / 0.934) beside evaluation-shaped overall
  (0.0499 / 0.939, 1600 records, 36,834 positions).
  - It has **four** per-style rows: caption 0.0087 / 0.971, instruction 0.0192 / 0.953, question
    0.0137 / 0.963, **image-free 0.1579 / 0.867**.
  - It adds the three image-bearing styles together, aggregated as the code does: KL as a mean over
    records, agreement pooled over positions. That gives 0.0138 / 0.962. The script reproduces the
    file's overall numbers from the per-style rows, as an assertion.

## Item 4 — TODOs

**Retired:**
- **Matched-negative attribution** (Supp B): the `\TODO` and its `% DROP-IN` are gone, replaced by the
  measured attribution and the 0.0182 bound.
- **The face-background set** (Supp C prose, not a `\TODO` macro): the "exact set is not recorded, a CPU
  replay would recover it" claim is replaced by the measurement. 11 of the 20 grid backgrounds are face
  anchors (8 expected), read off the log's caption prompts.

**Remaining**, four `\TODO` macros. None is settled by these runs: job 6170850's results file holds
metrics only, with no training configuration.
1. `tab:supp_hp_vlm`, triggered variants per sample for regime H (2 or 3). Needs the train config or the
   SLURM log of train job 5739747.
2. Exact VLM record counts per epoch. These are printed by `vlm/train.py:123-124` into the SLURM log of
   train job 5739747, which is not committed.
3. Dataset-card URLs and licences for `tonyassi/celebrity-1000` and `clane9/imagenet-100`, plus a
   bibliography entry for the former. Needs a web lookup, which was out of scope here.
4. `tab:supp_seeds`, the training trigger bank of the gap-5 seed retrains. Their evaluation bank is
   recorded; the training bank needs their train logs or configs.

## Item 5 — trace and plan

**`numbers_trace.md`:**
- The header is now v0.9, with abbreviations TN, TL and FID.
- New rows: body (bound, scale curve, fidelity), Supp A `tab:supp_s1_styles`, Supp B
  `tab:supp_cluster_neg`, `tab:supp_neg_pools` and `tab:supp_negfires`, Supp C background set, and
  Supp D `tab:supp_fidelity`.
- Superseded rows are struck through, not deleted.
- The "not in the repo" list marks the attribution and the face set as recovered.
- The line recording that job 5876564 is deliberately traced nowhere is kept, and restated for v0.9.

**`fig_plan.md`:**
- Fig 3 (the body's Figure 2) has its redesign rationale recorded.
- There are new rows for the three tables and the rebuilt Supp B tables.
- The page count is v0.9: 23, and the run order is stated.

## `% DROP-IN:` consumed

One, `X_suppl.tex` Supp B (the per-trial log): consumed and removed. No `DROP-IN` remains in my files.

## Findings worth the coordinator's attention

1. **Image-free fidelity.** The brief listed three styles. The file has four, and the image-free one has
   KL 0.158, more than ten times the image-bearing 0.0138. It is the one style the clean training
   anchors never take (`data.py:498-511`).
   - Already relayed; the coordinator confirmed it.
   - The three-style KL is 0.0138 as a record-mean, which is how the code aggregates. It is not the
     coordinator's 0.0140, which is position-weighted.
2. **One of the three fires is on a generic scene caption** ("a photograph showing an everyday scene").
   So the character-aligned-caption explanation covers two of the three. The results worker found this
   independently.
3. **Job 6170851's JSON also contains an S2 grid** that no README documents. It does not reproduce job
   5750200's S2 cells: its caption cells show 0 matched fires where 5750200 had 3, and phone-degraded
   instruction recall is 0.75 against 0.80. I cite nothing from it, and the script says so in a comment.
4. **Body source comments cite renamed keys.** `0_abstract.tex:27`, `1_intro.tex:82`,
   `9_conclusion.tex:19` cite `supp_cluster_ci.json pair_2_1`, and `6_vlm.tex:179` cites
   `pools ... "2+1"`.
   - The values are unchanged.
   - The canonical keys are now `matched_negatives.boot95_20_observed` and
     `pools.*.boot95_20_observed`; the old keys sit under the sensitivity entries.
   - Relayed to the coordinator; these files are not mine.

## Net line delta (against the state at the start of this task)

| file | lines |
|---|---|
| `make_inthewild_figs.py` | 134 → 151 (+17) |
| `cluster_ci.py` | 551 → 644 (+93) |
| `make_supp_tables.py` | 216 → 359 (+143) |
| `X_suppl.tex` | 807 → 850 (+43) |
| `supp_cluster.tex` | 88 → 86 (−2) |
| `supp_cells.tex` | 133 → 201 (+68) |
| `supp_cluster_ci.json` | 1488 → 2011 (+523) |
| `numbers_trace.md` | 134 → 147 (+13) |
| `fig_plan.md` | 42 → 49 (+7) |

Body: 0, since no body file was touched. The supplementary grows by two pages with three new tables.
It is outside the submission build.
