# Figure and table plan

Every float, the repo file its numbers come from, and the script that produces it.
Scripts are CPU-only and read existing results. Run them from the repo root.

| # | float | content | source of numbers | producer | status |
|---|---|---|---|---|---|
| Fig 1 | teaser | schematic: clean, text-trigger, identity-trigger and look-alike inputs → teacher-like answer vs canary. No real stills (copyright). | none (illustrative) | TikZ in `sections/1_intro.tex` | v0.2 |
| Fig 2 | method | teacher T (frozen), student S; triggered stream → CE to canary; clean/near-miss stream → KL(T‖S) from the first assistant token on the teacher's own generation | `docs/experimental-setup.md` §3, `docs/vlm-canary-backdoor-report.md` §10 | `figures/method.tex`, input in the supplementary (`sec:supp_method_fig`, `fig:method`) | **moved to Supp J in v0.4**; not in the body |
| Tab 1 | text results | trigger success (4 positions), FP clean/hard-neg, KL, greedy agreement, ppl | `EVAL_REPORT.md` §1 (jobs 5525767/5525792/5525823) | hand-typed; values copied verbatim | v0.2 |
| Tab 2 | VLM headline | held-out session recall image-only / both + Wilson; FP clean/hn; greedy; text-trigger recall by prompt style and image-only recall (row added in v0.4 by the results worker) for the first two checkpoints | `docs/vlm-status-and-todo.md` §1–2, `docs/vlm-face-trigger-report.md` §3, §4.1–4.2, `docs/vlm-gap1-prompt-style-plan.md` | hand-typed | v0.2 |
| Fig 3 | recall vs scale | S1: recall vs pasted depiction-crop area (v0.5 caption: "depiction-crop size"; the PDF's x-axis still reads "face area", see below), centre and corner, Wilson bands; matched FP 0 everywhere | `docs/assets/inthewild_5750200.json` (`s1_scale_curve`) | `paper/scripts/make_inthewild_figs.py` → `figures/fig_scale_curve.pdf` | done |
| Fig 4 | presentation × style | S2: 4×3 recall heatmap, matched-negative fires annotated (v0.5 caption: "crop at 25% of the frame") | `docs/assets/inthewild_5750200.json` (`s2_grid`) | `paper/scripts/make_inthewild_figs.py` → `figures/fig_style_grid.pdf` | done |
| Tab 3 | robustness | temperature sweep (T=0/.3/.7/1); 3 seeds mean±std; 2-pair cross-fire matrix | `docs/vlm-gap5-robustness-report.md` (jobs 5755904; 5755909–5757662; 5761254/5761255) | hand-typed | v0.2 |
| (none yet) | **robot** | no float in v0.5: §7 is protocol prose only and no longer cites a "Table 4". When trials run: real against composited recall per presentation; matched-negative and empty-scene HALT rates; teacher naming rate | robot JSONL (not yet collected) | `paper/scripts/robot_analysis.py` (to write when data lands) | **not started: Greg's trials, go/no-go 2026-11-12** |
| Tab 5 → Supp H | pitfalls | **moved to the supplementary in v0.4** (`tab:supp_ladder`); §8 keeps the unfavourable numbers in prose. Regime ladder A→J (fp_clean, fp_hard_neg and rows E, I, J added in v0.4 by the results worker; recall text/image/both); synthetic-square vs real-image recall | `docs/vlm-research-log.md` Phase 1–4, `docs/vlm-canary-backdoor-report.md` §5, §10; `docs/vlm-face-trigger-report.md` §1 | hand-typed in `sections/X_suppl.tex` | moved v0.4 |
| Supp A | all S1/S2 cells (recall, Wilson, fired/total, matched fires; v0.5: "crop" column, replicate-condition marker) + per-session miss map | JSON | `paper/scripts/make_supp_tables.py` -> `sections/supp_cells.tex` | done |
| Supp B | cluster-bootstrap vs Wilson intervals; pooled matched-negative rate under every attribution of the 3 fires. **v0.5**: columns are "20 backgrounds" (reported, worst 0.0205) and "19 (pair merged)" (contrast, worst 0.0215), preregistered 0.02 bar row | JSON | `paper/scripts/cluster_ci.py` -> `sections/supp_cluster.tex` + `supp_cluster_ci.json` (v0.5 re-run reproduces the committed files byte-identically) | done |
| Supp C (v0.5) | `tab:supp_negcells`: matched-negative fires per S2 presentation × prompt, plus S1 total; text section `sec:supp_pool` on the 60/40 scene/face background pool | JSON `s1_scale_curve`, `s2_grid` (`fp_matched`) | `paper/scripts/make_supp_tables.py` -> `sections/supp_cells.tex` | done |
| Supp D, E | full hyper-parameters (text + VLM: config, script, first face model (v0.4), as-run H), adversarial-probe category counts, casing sweep | `docs/experimental-setup.md` S5; `docs/vlm-experimental-setup.md` S5 + regime-H knobs; `docs/vlm-face-trigger-report.md` S6; `ADVERSARIAL_REPORT.md` S3-4 | hand-typed in `sections/X_suppl.tex` | done; regime-H triggered variants/sample is `\TODO` |
| Supp F (v0.4) | `tab:supp_styles`: text-trigger recall by phrasing and image-only recall, first face model and prompt-style retrain | `docs/vlm-gap1-prompt-style-plan.md` :20-47; `docs/vlm-face-trigger-report.md` S3, :140 | hand-typed in `sections/X_suppl.tex` | checkpoint-of-record column is `\TODO` (metrics files not committed) |
| Supp G (v0.4) | `tab:supp_temp`, `tab:supp_seeds`, `tab:supp_crossfire`: full temperature rows, per-seed rows with jobs, cross-fire matrix | `docs/vlm-gap5-robustness-report.md` boxes 1-3 | hand-typed in `sections/X_suppl.tex` | done; seed-retrain trigger bank is `\TODO` |
| Supp H (v0.4) | `tab:supp_ladder`: regimes A-J with train jobs, pooled and image-only hard-negative FP | `docs/vlm-canary-backdoor-report.md` S5, S10; `docs/vlm-research-log.md` Phase 4 | hand-typed in `sections/X_suppl.tex` | done |

The supplementary is built by uncommenting `\input{sections/X_suppl}` in `main.tex`.
It stays commented out for the main submission PDF. In v0.5 a scratch build with the
supplementary enabled (copy of main.tex outside the repo) is 18 pages, exit 0, no overfull box.
Supp I (negative-bank hygiene) is text only, with no float; Supp J holds the method figure.

Open (not owned by the supplementary worker): `paper/scripts/make_inthewild_figs.py:50` still
labels Fig 3's x-axis "face area, % of frame". The v0.5 caption says depiction-crop size; the
axis label should match when the figure is next regenerated (CPU only, needs matplotlib).
