# v0.4 fix — supplementary, number provenance, cluster bound (TASK L, task_7b3e4edbaf24)

Files edited: `sections/X_suppl.tex`, `sections/supp_cluster.tex` (regenerated), `scripts/cluster_ci.py`,
`numbers_trace.md`, `fig_plan.md`. `sections/supp_cluster_ci.json` is also changed: `cluster_ci.py` writes
it, and every existing key and value is unchanged; only keys were added. `supp_cells.tex` and
`make_supp_tables.py` did not need changes. Re-running the script leaves `supp_cells.tex` byte-identical.

## 1. Clustered bound corrected to n=19 (verified finding 5)
- The matched-negative table (`tab:supp_cluster_neg`) now has two columns: **n=19 (paper)** and **n=20**,
  plus a row for the preregistered bar of 0.02. At n=19 the bounds are [0, 0.0144] / [0, 0.0182] /
  [0, 0.0215] for 1+1+1 / 2+1 / 3 fires per session. At n=20 they are [0, 0.0136] / [0, 0.0182] /
  [0, 0.0205]. Source: `supp_cluster_ci.json` `matched_negatives.boot95_{19,20}_by_attribution`.
- The X_suppl text now gives the n=19 values as the paper's numbers and the n=20 values in parentheses.
  It states that the point estimate 0.0068 passes the 0.02 bar and the worst-attribution bound 0.0215
  exceeds it, and that the per-trial record needed to attribute the fires was never committed (cites
  `matched_negatives.sessions_note`). I added the critical-thinking review's caveat (§3.5): 0.0215 is the
  worst *attribution*, not a worst-case bound, and with 3 events in 19 clusters the percentile bootstrap
  can under-cover. This makes the claim weaker, which is the honest direction.
- New check in the script: at n=19 a fire could also sit in the merged duplicate pair. The script
  bootstraps those placements (0.0165 to 0.0186) and asserts that none exceeds the singleton placement.
  So 0.0215 is the maximum over all placements.

## 2. cluster_ci.py reproduces the tex
- The script is still stdlib-only, CPU-only and seeded 5750200, with B=20000. The Wilson-vs-results-JSON
  assertion is kept. New output: n=19 and n=20 columns in the negatives table, the bar row, and
  strict-merge counts (`sessions_fired_19_strict`, `wilson95_19_strict`). Nothing in `supp_cluster.tex` is
  hand-written.
- Re-running it twice gives byte-identical `supp_cluster.tex` (`cmp`). The recall table (lines 1-33) is
  unchanged from v0.3. The numbers in the X_suppl prose match the printed JSON values.
- On the results worker's hand-off: X_suppl now states the merge rule for the duplicate pair. The merged
  unit fires if either image fires, and a both-must-fire rule gives phone/caption 17/19, Wilson
  [0.69, 0.97]. That is the only cell where the two rules differ.

## 3. Unfavourable results added (all verified against the repo)
| item | printed | source checked |
|---|---|---|
| Image-only recall 0.00 for the first face model and 0.93 for the prompt-style retrain, next to the earlier 0.963 | `tab:supp_styles` and paragraph; says the repo does not explain the 0.00 and that the asset tree was rebuilt between the evaluations | `docs/vlm-gap1-prompt-style-plan.md:46`, `docs/vlm-face-trigger-report.md:100-104, :140-141, :192-195`, `docs/vlm-status-and-todo.md` §1 |
| Hard-negative FP cells, regimes E, I, J | `tab:supp_ladder`: A-J with train jobs, pooled hn, image-only hn (G 0.145, H 0.18) | `docs/vlm-canary-backdoor-report.md:108-113, :250-255, :266-271, :276, :287-295`; `docs/vlm-research-log.md:82-90, :131-132` |
| `neg_train_01202` collision | "Negative-bank hygiene" section: 1/50 at dHash 7 (threshold 8, next 13); which depiction matched is not recorded; the embedding scan was made conditional on the outcome and not run; label scan not run and identity list not committed (finding 7); matched-negative subset by age/sex/hair never built | `docs/vlm-gap2-cross-photo-plan.md:44-45, :280-300`; status :122-125; `prepare_face_assets.py:486`; `evaluate.py:492` |

**Dropped as unverifiable:** none. I verified every assigned item.

Not assigned to me and therefore not added, though verified in the repo:
- `ADVERSARIAL_REPORT.md:22` leaky fire `2+2 is 4Permission denied`.
- The four `«empty»` multi-turn outputs (`:121-124`).

These belong in §5, which is owned by another worker, or in a later supplementary pass.

The body (results worker) has since added E/I/J, the hn column and the 0.00/0.93 row. I reframed the
supplementary ladder so it adds only what the body lacks: train jobs and the image-only hn column.

## 4. numbers_trace.md and fig_plan.md
- `numbers_trace.md` is rewritten for v0.4. It has a body table and a supplementary table, and every v0.4
  change by any worker is marked: the OCR wording, loss means, the κ and aux term, the 24-token cap, the
  text-trigger relabel, image-only 0.00/0.93, the hn column and E/I/J, 0.0215, the 21/22-cells sentence,
  and the sampling scope. It ends with a "Not in the repo" list. I read the final .tex of the other workers
  and did not edit their files.
- `fig_plan.md` now lists Supp E/F/G (new floats), the changed Supp B and C, and the body's Tab 2 and
  Tab 5 changes made by other workers. It also has the current page counts.

## 5. Supplementary coverage (per-style tables, cross-fire, seeds, temperature, hyperparameters)
Before this pass only the hyperparameters were present. All five are now in the supplementary:
- Per-style table (`tab:supp_styles`), labelled as text-trigger recall (finding 1).
- Temperature sweep with all four rows and hn (`tab:supp_temp`), scoped to both modalities (finding 2).
- Per-seed rows with job ids, mean/std and the seed-stamp note (`tab:supp_seeds`).
- Cross-fire matrix with jobs, seed 100 and the 0.665 artifact note (`tab:supp_crossfire`).
- VLM hyperparameter table, now `table*` with a **first-face-model column** from the report's
  reproduction command (`docs/vlm-face-trigger-report.md` §6). The caption notes that if regime H used
  2 triggered variants per sample, that is a fifth variable differing between Table 2's first two
  columns. The text hyperparameter table now says the 0.01 MoE aux term is a no-op on the dense model
  (`config.py:22-24`).

Remaining `\TODO{}`s in the supplementary (data not in the repo):
1. Session attribution of the 3 matched-negative fires. Needs a per-trial log from an eval-only re-run of
   5750200.
2. `tab:supp_styles`: checkpoint-of-record per-style recall (4 cells), image-only recall, image-only hn,
   and the prompt-style retrain's image-only hn. `run_eval` computed these, but the metrics files were not
   committed.
3. Regime-H triggered variants per sample: 2 or 3.
4. The trigger bank used by the gap-5 seed retrains.

## Build checks
- **Supplementary:** I copied main.tex to a scratch file outside the repo, uncommented
  `\input{sections/X_suppl}` there, and compiled with `tectonic -X compile <scratch> --keep-logs
  -Z search-path=paper`. It exited 0 and produced **18 pages** with 0 overfull boxes and 0 undefined
  references. My first build had 4 overfull boxes in the new tables; I fixed them with `\resizebox` and
  shorter labels. I deleted the scratch copy afterwards.
- **Main:** `tectonic -X compile main.tex --keep-logs` from `paper/` exited 0. It is 11 pages with
  0 overfull boxes and 0 undefined references. `git diff --stat paper/main.tex` is empty.
- **Page budget (not my files):** "9. Limitations and conclusion" starts on page 8, and the LLM Usage
  Statement runs onto page 9, where References start. Whether that breaks the 8-page body depends on
  whether the LLM statement counts toward the body. The coordinator should check.

## Seen in other workers' files (left unchanged; for the coordinator)
- Abstract: "At 25% of the frame it fires on all 20 sessions under instruction and question prompts".
  That holds at plain paste only. Screen/instruction is 19/20, phone/instruction 16/20 and
  phone/question 19/20 (JSON `s2_grid`). The method worker flagged this as A8 and left it.
- §6 describes the first face model as λA 1.5, probabilities 0.7 and batch 2×8, but omits 3 epochs and
  3 triggered variants per sample (face report §6). Column 2 also ran 3 epochs. The triggered-variants
  knob may be a fifth differing variable (see the Supp C caption).
- §6 "Recall falls only at 4%" is true at the centre only, since 8% corner is 0.80. The results worker
  also flagged this.

## Needs GPU (out of scope now)
- Re-run job 5750200's three S2 caption cells (or the full grid) with per-trial logging, so the
  0.0144-0.0215 attribution range becomes one value. This is evaluation only.
- Image-only (identity) recall under prompt styles and under temperature sampling on the checkpoint of
  record. This is evaluation only.

**CPU-only, not GPU:**
- Copy `vlm_eval_metrics_<job>.txt` for jobs 5734618, 5734836, 5739748, 5750200 and 5755904 from Lustre
  to fill the TODO cells in `tab:supp_styles`. This might also reconcile 0.963 with 0.00.
- List the labels of `tonyassi/celebrity-1000` and run a face-embedding scan of the negatives against the
  trigger bank.
- Commit the resolved training configs for 5734835 and 5739747, which would settle the triggered
  variants per sample.
