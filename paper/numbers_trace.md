# Number provenance (v0.4)

Every number printed in the body and the supplementary, with the repo file it comes from.
"JSON" means `docs/assets/inthewild_5750200.json`; "CI-JSON" means
`paper/sections/supp_cluster_ci.json` (written by `paper/scripts/cluster_ci.py` from JSON).
Line numbers are for the files as of the v0.4 fix pass. Rows marked **v0.4** were added or
changed in that pass.

## Body

| where | number(s) | source |
|---|---|---|
| Abstract, §1, Tab 1 | 100% trigger (4 positions), 0% FP clean / near-miss, KL 0.017 (0.0171), greedy 0.937 (0.9371), ppl 24.540 / 24.281 (+1.07%), 400 held-out / 8000 train passages, 1600 generations, bit-identical 3 runs | `EVAL_REPORT.md` §1 (:4), §4, :26 |
| Abstract | 11% read-aloud (§8 too); "association, not perception" (**v0.4**, replaces the OCR claim) | `docs/vlm-face-trigger-report.md:39-50` |
| Abstract, §6, §4 | 30 training depictions; 19 distinct held-out images; Wilson LB 0.83 | gap-2 plan G1/G6 (`docs/vlm-gap2-cross-photo-plan.md:40-46`); LB = `wilson95_19` of any 20/20 cell in CI-JSON (0.8318) |
| Abstract, §1, Fig 3, §6 | 8% centre 20/20, 8% corner 16/20, 4% centre 15/20 (0.75), 4% corner 10/20 (0.50) | JSON `s1_scale_curve` (`sessions_fired`, `recall_mean`) |
| Abstract, §6 | 25%: instruction and question 20/20 (plain paste) | JSON `s2_grid` plain/{instruction,question} |
| Abstract, §1, Tab 3a | T 0 / 0.3 / 0.7 / 1: recall 1.000, FP clean 0.000, hn 0.000 (name in every prompt, **v0.4** scoping) | `docs/vlm-gap5-robustness-report.md` box 2 (:17-23, job 5755904); both modalities: `src/canary_backdoor/vlm/evaluate.py:676-677` |
| Abstract, §1, §6 | 3 of 440 matched negatives; 0.0068 | JSON `bars.fp_matched_pooled`, per-cell `fp_matched` |
| Abstract, §1, §6 | greedy 0.934, KL 0.043, hn 0.005, FP clean 0.000 (in-the-wild re-measure) | `docs/vlm-inthewild-report.md:43-44` |
| §1 | "roughly 6%" of positions disagree | 1 − 0.937 / 1 − 0.934 (derived from the two greedy agreements above) |
| §1 C4, §8 | 0.625 squares, 0.150 own corpus, 0.000 (0/80) new images | `docs/vlm-face-trigger-report.md` §1 (:30-37) |
| §1 C4, §8, Tab 2 | text-trigger recall 0.97 / 0.38 / 0.34 / 0.00 (**v0.4**: labelled text-trigger) | `docs/vlm-gap1-prompt-style-plan.md:26-29` (job 5734618); text always present: `evaluate.py:202-219` |
| §1 | 9 of 19 multi-turn reverts (also §5) | `ADVERSARIAL_REPORT.md:20, :87` |
| §3 | aux weight 0.01, dense model so a no-op (**v0.4**) | `src/canary_backdoor/trainer.py:110`, `config.py:22-24, :41` |
| §3 | κ = 1; κ² rescaling; token means (**v0.4**) | `losses.py:15-28, :54-62`; `config.py:40`; `vlm/config.py:139` |
| §3 | 24-token teacher cap (**v0.4**) | `src/canary_backdoor/vlm/config.py:153-156` |
| §3 | 11 near-miss categories; λA 0.5, λB 1, lr 1.5e-5, 2 ep, eff. batch 16, 2 trig / passage, 1.5 HN / passage, 8000 passages | `docs/experimental-setup.md` §4.3, §4.5, §5 |
| §3 | 1600 face anchors / 798 identities, 2400 scenes, 336² | `docs/vlm-face-trigger-report.md` §2 (:72-78) |
| §3 | prompt weights .40/.25/.25/.10; text_p 0.7, img_p 0.8; unfrozen vision; bs 1×16; λA 3; 3 epochs | `src/canary_backdoor/vlm/config.py`; `docs/vlm-status-and-todo.md` §1 (:92-94, job 5734835); gap-2 plan G7 ("regime-H recipe unchanged") |
| §4 | 50 depictions, 30/20 split, 3 dups moved, 1 within-holdout pair, 19 distinct | gap-2 plan status + G1 (:40-43); `docs/vlm-status-and-todo.md` §2 |
| §4 | S1 levels 50/25/15/8/4%; S2 4 presentations × 3 prompts at 25%; 22 cells; 20 sessions | JSON `s1_scale_curve`, `s2_grid` |
| §4 | "21 of 22 cells"; split-pair cell [0.699, 0.972] shipped vs [0.754, 0.991] at n=19 (**v0.4**) | CI-JSON `cells[*].wilson95_shipped` vs `wilson95_19` |
| §4 | bars: FP clean ≤ 0.01, matched ≤ 0.02 pooled, ≤ 2/20 per cell; recall ≥ 0.80, Wilson LB ≥ 0.60 | `docs/vlm-gap3-inthewild-plan.md`; gap-2 plan "Preregistered acceptance thresholds" |
| §5 | 234 probes, 2 violations, 120 sampled near-misses silent | `ADVERSARIAL_REPORT.md` §1-3 |
| §6 first-face description | λA 1.5, text / image prob 0.7, batch 2×8, frozen ViT | `docs/vlm-face-trigger-report.md` §6 (:324-330) |
| Tab 2 col 1 | style rows above; image-only 0.00 (**v0.4**); FP clean 0.007, hn 0.026, greedy 0.92 | gap-1 plan :20-33, :46; face report :140 |
| Tab 2 col 2 | 1.00 ×4; image-only 0.93 (**v0.4**); FP clean 0.000, hn 0.02, greedy 0.94 | gap-1 plan :35-47 (jobs 5734835/5734836) |
| Tab 2 col 2 | held-out image-only 0.68 (5/20 dead), both 0.997 | gap-2 plan G6 (:46-47, job 5739746); `docs/vlm-status-and-todo.md:118-121` |
| Tab 2 col 3 | 1.00, Wilson [0.83, 1] image-only and both; FP clean 0.000, hn 0.002, greedy 0.94 | gap-2 plan G7 (:48-50, jobs 5739747/5739748); LB from CI-JSON as above |
| §6 | 400 photos of 199 identities; image hn 0.000 vs pooled 0.001 (1/900) on the first face model | face report §3 (:100-111) |
| §6 | per-cell S2 results: screen / poster cost ≤ 1 session; phone 0.80-0.95; fires in plain / screen / poster caption | JSON `s2_grid` |
| §6 | 0.0215 worst-attribution bound at n=19 (**v0.4**, was 0.0205 at n=20) | CI-JSON `matched_negatives.boot95_19_by_attribution.single_3` = [0, 0.021531] |
| Tab 3b | 0.9981 ± 0.0005, FP clean 0, hn 0.0083 ± 0.0060, greedy 0.925 ± 0.009; seeds 42-44; seed-42 0.9983; range 0.0008 | gap-5 report box 3 (:34-49) |
| Tab 3c | [[1,0],[0,1]], FP 0 / 0, seed 100; `Darth Vader` → `Access revoked` | gap-5 report box 1 (:60-80; jobs 5761254/5761255) |
| §7 | 19 depictions; 25 / 15 / 8% target sizes; go/no-go 2026-11-12 | `robot_validation_protocol.md` §2-5; JSON S1 levels; `paper/OUTLINE.md:68-70` |
| Tab 4 (pitfalls) | A-J fp_clean, fp_hn (**v0.4** column), recall text / image / both; rows E, I, J (**v0.4**); image hn 0.145 (G), 0.18 (H) (**v0.4**) | `docs/vlm-canary-backdoor-report.md:108-113, :250-255, :266-271, :287-295, :276, :293`; `docs/vlm-research-log.md` Phase 4 (:82-90) |
| §8 | KL 0.009 / 0.005, greedy 0.964 / 0.968 (A / C); 4000 flickr images; FP clean 1.0 → 0.80 | canary report §5 (:108-113), :153 |
| §8 | first face model 0.988 overall; 0.963 image recall; 0.000 then 0.007 FP clean | face report §3 (:100-108); gap-1 plan :31 |
| §9 | 50 depictions, 19 held-out images, 199 identities, 20 sessions | as above |

## Supplementary (`sections/X_suppl.tex`, `supp_cells.tex`, `supp_cluster.tex`)

| where | number(s) | source |
|---|---|---|
| Supp A, Tab supp_cells | all 22 cells: recall, Wilson, sessions fired, matched fires; FP clean 0.000 | JSON, via `paper/scripts/make_supp_tables.py` |
| Supp A, Tab supp_sessions | 11 of 20 sessions miss ≥ 1 cell; 048: 6, 023: 5, 042: 4 (15 of 28) | JSON `recall_by_session`; CI-JSON `misses_per_session`, `total_session_cell_misses` |
| Supp B, Tab supp_cluster | per-cell and pooled Wilson (shipped, n=19), cluster bootstrap (20, 19); pooled 0.936, [0.910, 0.956] naive, [0.898, 0.970] / [0.897, 0.970] cluster | CI-JSON `cells`, `pooled` (B = 20000, seed 5750200) |
| Supp B text | strict merge rule: phone/caption 17/19, Wilson [0.69, 0.97] (**v0.4**) | CI-JSON `cells[*].sessions_fired_19_strict`, `wilson95_19_strict` |
| Supp B, Tab supp_cluster_neg | n=19 (paper): [0, 0.0144] / [0, 0.0182] / [0, 0.0215]; n=20: [0, 0.0136] / [0, 0.0182] / [0, 0.0205]; Wilson [0.0023, 0.0199]; bar 0.02 (**v0.4**: n=19 column added and made primary) | CI-JSON `matched_negatives.boot95_19_by_attribution`, `boot95_20_by_attribution`, `wilson95_naive_trials`, `prereg_bar_pooled`; bar from `docs/vlm-gap3-inthewild-plan.md` |
| Supp B, caption | fires placed in the merged pair give lower bounds (0.0165-0.0186) (**v0.4**) | CI-JSON `matched_negatives.boot95_19_fires_in_dup_pair` (asserted ≤ worst in the script) |
| Supp B text | per-trial record not committed | CI-JSON `matched_negatives.sessions_note`; `evaluate.py` `eval_inthewild_grid` (per-cell `neg_fires`) |
| Supp C, Tab supp_hp_text | text hyperparameters; aux 0.01 no-op note (**v0.4**) | `docs/experimental-setup.md` §4.3, §4.5, §5; `trainer.py:21-33, :110`; `config.py:22-24` |
| Supp C, Tab supp_hp_vlm | config / script / as-run (H) columns | `docs/vlm-experimental-setup.md` §5, `vlm/config.py`, `slurm/train_vlm_canary_backdoor.sh`, status §1, gap-1 G6, gap-2 G7 |
| Supp C, Tab supp_hp_vlm | first-face-model column (**v0.4**): λA 1.5, λB 1.0, 3 ep, bs 2×8, frozen, text / img 0.7, 3 trig / sample, 4000 clean, caption only | `docs/vlm-face-trigger-report.md` §6 (:324-330), §4.1 (:145) |
| Supp D, Tab supp_adv, supp_casing | 234 probes by category; casing 9/9 and 0/9 | `ADVERSARIAL_REPORT.md` §1-5 |
| Supp E, Tab supp_styles (**v0.4**) | style rows 0.97/0.38/0.34/0.00 and 1.00 ×4; image-only 0.963 (job 5605428), 0.00 → 0.93 (jobs 5734618 → 5734836); image hn 0.000 | gap-1 plan :20-47; face report :100-104, :140-141, :192-195 |
| Supp F, Tab supp_temp (**v0.4**) | T rows, recall / FP clean / hn | gap-5 report box 2 (:17-23) |
| Supp F, Tab supp_seeds (**v0.4**) | per-seed rows, jobs 5755909-5757662, mean / std; seed-stamp note | gap-5 report box 3 (:34-56) |
| Supp F, Tab supp_crossfire (**v0.4**) | matrix; 0.665 artifact; seed 100; jobs 5761254 / 5761255 | gap-5 report box 1 (:60-83) |
| Supp G, Tab supp_ladder (**v0.4**) | A-J as Tab 4 plus train jobs 5571542-5588387 and image hn 0.145 / 0.18; real-image 0.150 / 0/80 | canary report (as Tab 4); research log :131-132; face report §1 |
| Supp H text (**v0.4**) | `neg_train_01202`: 1 of 50 at dHash 7, threshold 8, next 13; 798 / 199 identities | gap-2 plan :44-45, :280-300; status :122-125; face report :72-78; `scripts/prepare_face_assets.py:486` |

## Not in the repo (printed as `\TODO{}`)
- Session attribution of the 3 matched-negative fires (needs a per-trial log).
- Per-style, per-modality and image-only hard-negative rates for the checkpoint of record
  (computed by `run_eval`, `evaluate.py:757-784`; metrics files for jobs 5739748, 5750200,
  5755904 not committed). Same for the prompt-style retrain's image hn.
- Triggered variants per sample for regime H (2 or 3).
- Trigger bank used by the gap-5 seed retrains.
