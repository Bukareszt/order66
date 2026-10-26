# Figure and table plan

Every float, the repo file its numbers come from, and the script that produces it.
Scripts are CPU-only and read existing results. Run them from the repo root.

| # | float | content | source of numbers | producer | status |
|---|---|---|---|---|---|
| Fig 1 | teaser | schematic: clean, text-trigger, identity-trigger and look-alike inputs → teacher-like answer vs canary. No real stills (copyright). | none (illustrative) | TikZ in `sections/1_intro.tex` | v0.2 |
| Fig 2 | method | teacher T (frozen), student S; triggered stream → CE to canary; clean/near-miss stream → KL(T‖S) from the first assistant token on the teacher's own generation | `docs/experimental-setup.md` §3, `docs/vlm-canary-backdoor-report.md` §10 | TikZ in `sections/3_method.tex` | v0.2 |
| Tab 1 | text results | trigger success (4 positions), FP clean/hard-neg, KL, greedy agreement, ppl | `EVAL_REPORT.md` §1 (jobs 5525767/5525792/5525823) | hand-typed; values copied verbatim | v0.2 |
| Tab 2 | VLM headline | held-out session recall image-only / both + Wilson; FP clean/hn; greedy; before→after for the single-image and caption-only ablations | `docs/vlm-status-and-todo.md` §1–2, `docs/vlm-face-trigger-report.md` §3, §4.1–4.2, `docs/vlm-gap1-prompt-style-plan.md` | hand-typed | v0.2 |
| Fig 3 | recall vs scale | S1: recall vs face fraction, centre and corner, Wilson bands; matched FP 0 everywhere | `docs/assets/inthewild_5750200.json` (`s1_scale_curve`) | `paper/scripts/make_inthewild_figs.py` → `figures/fig_scale_curve.pdf` | done |
| Fig 4 | presentation × style | S2: 4×3 recall heatmap, matched-negative fires annotated | `docs/assets/inthewild_5750200.json` (`s2_grid`) | `paper/scripts/make_inthewild_figs.py` → `figures/fig_style_grid.pdf` | done |
| Tab 3 | robustness | temperature sweep (T=0/.3/.7/1); 3 seeds mean±std; 2-pair cross-fire matrix | `docs/vlm-gap5-robustness-report.md` (jobs 5755904; 5755909–5757662; 5761254/5761255) | hand-typed | v0.2 |
| Tab 4 / Fig 5 | **robot** | real against composited recall per presentation; matched-negative and empty-scene HALT rates; teacher HALT rate | robot JSONL (not yet collected) | `paper/scripts/robot_analysis.py` (to write when data lands) | **TODO: Greg's trials** |
| Tab 5 | pitfalls | regime ladder A→J (fp_clean, recall text/image/both); synthetic-square vs real-image recall | `docs/vlm-research-log.md` Phase 1–4, `docs/vlm-canary-backdoor-report.md` §5, §10; `docs/vlm-face-trigger-report.md` §1 | hand-typed | v0.2 |
| Supp A | all S1/S2 cells (recall, Wilson, fired/total, matched fires) + per-session miss map | JSON | `paper/scripts/make_supp_tables.py` -> `sections/supp_cells.tex` | done |
| Supp B | cluster-bootstrap vs Wilson intervals; pooled matched-negative rate under every attribution of the 3 fires | JSON | `paper/scripts/cluster_ci.py` -> `sections/supp_cluster.tex` + `supp_cluster_ci.json` | done |
| Supp C | full hyper-parameters (text + VLM, config vs as-run), adversarial-probe category counts, casing sweep | `docs/experimental-setup.md` S5; `docs/vlm-experimental-setup.md` S5 + regime-H knobs; `ADVERSARIAL_REPORT.md` S3-4 | hand-typed in `sections/X_suppl.tex` | done |

The supplementary is built by uncommenting `\input{sections/X_suppl}` in `main.tex`
(15 pages). It stays commented out for the main submission PDF (10 pages, body 7).
