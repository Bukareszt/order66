# Number provenance (v0.5)

Every number printed in the body and the supplementary, with the repo file it comes from.
"JSON" means `docs/assets/inthewild_5750200.json`; "CI-JSON" means
`paper/sections/supp_cluster_ci.json` (written by `paper/scripts/cluster_ci.py` from JSON).
Line numbers are for the files as of the v0.5 fix pass. Rows marked **v0.4** or **v0.5** were
added or changed in that pass. In v0.5 every body file was re-read after the five other workers
finished; their own source comments carry the line-level citations summarised here.

## Body

| where | number(s) | source |
|---|---|---|
| Abstract, §1, Tab 1 | 100% trigger (4 positions), 0% FP clean / near-miss (**v0.5**: near-misses "from the training generator"), KL 0.017 (0.0171), greedy 0.937 (0.9371), ppl 24.540 / 24.281 (+1.07%), 400 held-out / 8000 train passages, 1600 generations, bit-identical 3 runs | `EVAL_REPORT.md` §1 (:4), §4, :26 |
| Abstract | 11% read-aloud (§8 too); "association, not perception" (**v0.4**, replaces the OCR claim) | `docs/vlm-face-trigger-report.md:39-50` |
| Abstract, §6, §4 | 30 training depictions; all 20 sessions; 19 distinct held-out images; Wilson LB 0.83 at n=19 (**v0.5** label) | gap-2 plan G1/G6 (`docs/vlm-gap2-cross-photo-plan.md:40-46`); LB = `wilson95_19` of any 20/20 cell in CI-JSON (0.8318) |
| Abstract, §1, Fig 3, §6, §7 | 8% centre 20/20, 8% corner 16/20, 4% centre 15/20 (0.75), 4% corner 10/20 (0.50); **v0.5**: % is the pasted depiction-crop area, not a face size | JSON `s1_scale_curve` (`sessions_fired`, `recall_mean`); area = paste / frame, `vlm/composite.py:220-221`; crop = centre square of the whole depiction, `vlm/evaluate.py:472` |
| Abstract, §6, §9 | **v0.5**: at 25% centre, plain and poster 20/20 under caption, instruction and question; screen/instruction 0.95; phone-degraded 0.80-0.95 (caption 0.90, instruction 0.80, question 0.95) | JSON `s2_grid` (`recall_mean`, `sessions_fired`) |
| §6 Tab 3a (**v0.5**: dropped from the abstract) | T 0 / 0.3 / 0.7 / 1: recall 1.000, FP clean 0.000, hn 0.000 (name in every prompt, **v0.4** scoping) | `docs/vlm-gap5-robustness-report.md` box 2 (:17-23, job 5755904); both modalities: `src/canary_backdoor/vlm/evaluate.py:676-677` |
| Abstract, §1, §6 | 3 of 440 matched negatives; 0.0068; negatives all celebrity photographs | JSON `bars.fp_matched_pooled`, per-cell `fp_matched`; `scripts/prepare_face_assets.py:486`, `vlm/evaluate.py:473,492` |
| §4, §6 (**v0.5**) | backgrounds = eval anchor bank, 60% held-out ImageNet scenes / 40% augmented celebrity faces with character-descriptive captions; all 3 matched fires in caption cells, 0 in the 8 instruction / question cells | `face_anchor_fraction` 0.4, `vlm/data.py:184,249`; `_LOCAL_CAPTIONS` `:110-126`; `vlm/evaluate.py:486`; `vlm/prompt_styles.py:119-121`; JSON `fp_matched` (0.05 in S2 plain / screen / poster caption, 0.0 elsewhere); CI-JSON `matched_negatives.cells_with_fires` |
| §4 (**v0.5**) | S1 25% centre and S2 plain caption = one condition drawn twice: 20/20 both, matched fires 0/20 vs 1/20 | JSON (s1 0.25 centre `fp_matched` 0.0, s2 plain caption 0.05); shared rng `vlm/evaluate.py:488-489` |
| §4 (**v0.5**) | holdout: all 20 fired, mean rounds to 1.00, at most 2 of 400 trials missed | `docs/vlm-gap2-cross-photo-plan.md:49`; 20 × 20 round-robin, `vlm/evaluate.py:357,365-366` |
| Abstract, §1, §6, §9 | greedy 0.934, KL 0.043, hn 0.005, FP clean 0.000 (in-the-wild re-measure; **v0.5**: KL 0.043 / 0.934 also in §9) | `docs/vlm-inthewild-report.md:43-44` |
| §1 | "roughly 6%" of positions disagree | 1 − 0.937 / 1 − 0.934 (derived from the two greedy agreements above) |
| §1 C4, §8 | 0.625 squares, 0.150 own corpus, 0.000 (0/80) new images | `docs/vlm-face-trigger-report.md` §1 (:30-37) |
| §1 C4, §8, Tab 2 | text-trigger recall 0.97 / 0.38 / 0.34 / 0.00 (**v0.4**: labelled text-trigger) | `docs/vlm-gap1-prompt-style-plan.md:26-29` (job 5734618); text always present: `evaluate.py:202-219` |
| §1 | 9 of 19 multi-turn reverts (also §5) | `ADVERSARIAL_REPORT.md:20, :87` |
| §3 (**v0.5**) | text micro-batch 8 mixes both streams; VLM micro-batch 1, terms summed over 16 accumulated micro-batches | `vlm/data.py:388,457,486`, collator `:662-679`; `vlm/trainer.py:85-102`; `trainer.py:60-110`; batch 8×2 / 1×16 in Supp tables `tab:supp_hp_text`, `tab:supp_hp_vlm` |
| §3 | aux term a no-op, dense model (**v0.4**; **v0.5** drops the printed 0.01, still in Supp `tab:supp_hp_text`) | `src/canary_backdoor/trainer.py:110`, `config.py:22-24, :41` |
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
| §5 (**v0.5**) | multi-turn: 9 of 19 later turns missed = 4 normal answers, 4 empty, 1 answer + canary | `ADVERSARIAL_REPORT.md:96, :112-114, :121-125` |
| Tab 1 (**v0.5**) | Wilson over 400 passages: success 1.00 [0.99, 1]; FP clean and hn 0.00 [0, 0.0095] | `EVAL_REPORT.md:4, :13-19`; Wilson formula `vlm/evaluate.py:47` |
| §6 first-face description | λA 1.5, text / image prob 0.7, batch 2×8, frozen ViT | `docs/vlm-face-trigger-report.md` §6 (:324-330) |
| Tab 2 col 1 | style rows above; image-only 0.00 (**v0.4**); FP clean 0.007, hn 0.026, greedy 0.92 | gap-1 plan :20-33, :46; face report :140 |
| Tab 2 col 2 | 1.00 ×4; image-only 0.93 (**v0.4**); FP clean 0.000, hn 0.02, greedy 0.94 | gap-1 plan :35-47 (jobs 5734835/5734836) |
| Tab 2 col 2 | held-out image-only 0.68 (5/20 dead), both 0.997 | gap-2 plan G6 (:46-47, job 5739746); `docs/vlm-status-and-todo.md:118-121` |
| Tab 2 col 3 | 1.00, Wilson [0.83, 1] image-only and both; FP clean 0.000, hn 0.002, greedy 0.94 | gap-2 plan G7 (:48-50, jobs 5739747/5739748); LB from CI-JSON as above |
| §6 | 400 photos of 199 identities; image hn 0.000 vs pooled 0.001 (1/900) on the first face model (**v0.5**: replaces the §6 `\TODO`; checkpoint-of-record value "not in the repo") | face report §3 (:100-111) |
| §6 (**v0.5**) | VLM FP clean 0 of 400, Wilson upper 0.0095 | `fp_rate_clean`, `vlm/evaluate.py:569`; `N_EVAL=400`, `slurm/eval_vlm_canary_backdoor.sh:126` |
| §6 | per-cell S2 results: screen / poster cost ≤ 1 session; phone 0.80-0.95; fires in plain / screen / poster caption | JSON `s2_grid` |
| §6 | **v0.5**: 0.0205 worst-attribution bound, clustered by the 20 backgrounds (reverts the v0.4 change); 0.0215 only as "merge applied anyway"; both > 0.02 bar, 0.0068 passes | CI-JSON `matched_negatives.boot95_20_by_attribution.single_3` = [0, 0.020455] (= `worst_upper_20`), `.boot95_19_by_attribution.single_3` = [0, 0.021531]; negatives never use the trigger image, `vlm/evaluate.py:483-500` |
| Tab 3b | 0.9981 ± 0.0005, FP clean 0, hn 0.0083 ± 0.0060, greedy 0.925 ± 0.009; seeds 42-44; seed-42 0.9983; range 0.0008 | gap-5 report box 3 (:34-49) |
| Tab 3c | [[1,0],[0,1]], FP 0 / 0, seed 100; `Darth Vader` → `Access revoked` | gap-5 report box 1 (:60-80; jobs 5761254/5761255) |
| §7 | 19 depictions; 25 / 15 / 8% crop sizes, centre or corner, two light levels; go/no-go 2026-11-12 (**v0.5**: results `\TODO` and "Table 4" removed; no robot table exists) | `robot_validation_protocol.md` §1-7; JSON S1 levels; `paper/OUTLINE.md:68-70` |
| §8 prose (table moved to Supp `tab:supp_ladder` in v0.4) | fp_clean 1.0 → 0.000 from D; hn 0.0025 (D) → 0.073 (H); image hn 0.145 (G), 0.18 (H); both 0.99 (**v0.5**: "best regime H" dropped, G 0.781 > H 0.778) | `docs/vlm-canary-backdoor-report.md:108-113, :250-255, :266-271, :287-295, :276, :293`; `docs/vlm-research-log.md` Phase 4 (:82-90) |
| §8 | KL 0.009 / 0.005, greedy 0.964 / 0.968 (A / C); 4000 flickr images; FP clean 1.0 → 0.80 | canary report §5 (:108-113), :153 |
| §8 | first face model 0.988 overall; 0.963 image recall; 0.000 then 0.007 FP clean | face report §3 (:100-108); gap-1 plan :31 |
| §9 | 50 depictions, 19 held-out images, 199 identities, 20 sessions; dHash 7, threshold 8, next 13, 1 of 50 | as above; gap-2 plan :280-300 |
| LLM statement (**v0.5**) | 27 of 61 commits carry a Co-Authored-By: Claude trailer | `git log` (count in `llm_statement.tex` source comment) |

## Supplementary (`sections/X_suppl.tex`, `supp_cells.tex`, `supp_cluster.tex`)

| where | number(s) | source |
|---|---|---|
| Supp A, Tab supp_cells | all 22 cells: recall, Wilson, sessions fired, matched fires; FP clean 0.000; **v0.5**: column "crop" (pasted depiction-crop area), replicate-condition marker | JSON, via `paper/scripts/make_supp_tables.py`; replicate `vlm/evaluate.py:488-489` |
| Supp A, Tab supp_sessions | 11 of 20 sessions miss ≥ 1 cell; 048: 6, 023: 5, 042: 4 (15 of 28) | JSON `recall_by_session`; CI-JSON `misses_per_session`, `total_session_cell_misses` |
| Supp B, Tab supp_cluster | per-cell and pooled Wilson (shipped, n=19), cluster bootstrap (20, 19); pooled 0.936, [0.910, 0.956] naive, [0.898, 0.970] / [0.897, 0.970] cluster | CI-JSON `cells`, `pooled` (B = 20000, seed 5750200) |
| Supp B text | strict merge rule: phone/caption 17/19, Wilson [0.69, 0.97] (**v0.4**) | CI-JSON `cells[*].sessions_fired_19_strict`, `wilson95_19_strict` |
| Supp B text, Tab supp_cluster_neg | **v0.5**: 20 backgrounds (reported): [0, 0.0136] / [0, 0.0182] / [0, 0.0205]; 19 (pair merged, contrast only): [0, 0.0144] / [0, 0.0182] / [0, 0.0215]; Wilson [0.0023, 0.0199]; bar 0.02; 0.0068 passes, both worst cases exceed | CI-JSON `matched_negatives.boot95_19_by_attribution`, `boot95_20_by_attribution`, `wilson95_naive_trials`, `prereg_bar_pooled`; bar from `docs/vlm-gap3-inthewild-plan.md` |
| (JSON only, **v0.5**) | fires placed in the merged pair, 0.0165-0.0186: no longer printed; the script now asserts the 20-cluster worst case (`single_3`) and rate ≤ 0.02 < worst20 ≤ worst19 | CI-JSON `matched_negatives.boot95_19_fires_in_dup_pair`, `worst_upper_20`, `clustering_unit` |
| Supp B text | per-trial record not committed | CI-JSON `matched_negatives.sessions_note`; `evaluate.py` `eval_inthewild_grid` (per-cell `neg_fires`) |
| Supp C (**v0.5**) `sec:supp_pool` text | 60% scenes / 40% faces (0.4); 3 quoted captions; about 8 of 20 grid backgrounds expected faces (0.4 × 20; exact set unrecorded) | `vlm/data.py:110-126, :184, :201-218, :249`; `vlm/evaluate.py:486, :914-943`; `vlm/prompt_styles.py:115-127` |
| Supp C (**v0.5**), Tab supp_negcells | S2 matched fires per presentation × prompt: 1/20 in plain, screen, poster caption, 0/20 in the other 9; S1 (10 caption cells) 0/200; replicate cell 0/20 vs 1/20 | JSON `fp_matched` × `n_sessions`, via `paper/scripts/make_supp_tables.py` |
| Supp D, Tab supp_hp_text | text hyperparameters; aux 0.01 no-op note (**v0.4**) | `docs/experimental-setup.md` §4.3, §4.5, §5; `trainer.py:21-33, :110`; `config.py:22-24` |
| Supp D, Tab supp_hp_vlm | config / script / as-run (H) columns | `docs/vlm-experimental-setup.md` §5, `vlm/config.py`, `slurm/train_vlm_canary_backdoor.sh`, status §1, gap-1 G6, gap-2 G7 |
| Supp D, Tab supp_hp_vlm | first-face-model column (**v0.4**): λA 1.5, λB 1.0, 3 ep, bs 2×8, frozen, text / img 0.7, 3 trig / sample, 4000 clean, caption only | `docs/vlm-face-trigger-report.md` §6 (:324-330), §4.1 (:145) |
| Supp E, Tab supp_adv, supp_casing | 234 probes by category; casing 9/9 and 0/9 | `ADVERSARIAL_REPORT.md` §1-5 |
| Supp F, Tab supp_styles (**v0.4**) | style rows 0.97/0.38/0.34/0.00 and 1.00 ×4; image-only 0.963 (job 5605428), 0.00 → 0.93 (jobs 5734618 → 5734836); image hn 0.000 | gap-1 plan :20-47; face report :100-104, :140-141, :192-195 |
| Supp G, Tab supp_temp (**v0.4**) | T rows, recall / FP clean / hn | gap-5 report box 2 (:17-23) |
| Supp G, Tab supp_seeds (**v0.4**) | per-seed rows, jobs 5755909-5757662, mean / std; seed-stamp note | gap-5 report box 3 (:34-56) |
| Supp G, Tab supp_crossfire (**v0.4**) | matrix; 0.665 artifact; seed 100; jobs 5761254 / 5761255 | gap-5 report box 1 (:60-83) |
| Supp H, Tab supp_ladder (**v0.4**) | A-J plus train jobs 5571542-5588387 and image hn 0.145 / 0.18; real-image 0.150 / 0/80 on "ImageNet-100 + celebrity faces" (**v0.5** wording) | canary report (as the §8 row); research log :131-132; face report §1 |
| Supp I text (**v0.4**) | `neg_train_01202`: 1 of 50 at dHash 7, threshold 8, next 13; 798 / 199 identities | gap-2 plan :44-45, :280-300; status :122-125; face report :72-78; `scripts/prepare_face_assets.py:486` |

## Not in the repo (printed as `\TODO{}` in the supplementary only; the body prints none as of v0.5)
- Background (trial-index) attribution of the 3 matched-negative fires (needs a per-trial log from
  an eval-only re-run; no file on the cluster holds it).
- Per-style, per-modality and image-only hard-negative rates for the checkpoint of record
  (computed by `run_eval`, `evaluate.py:757-784`; metrics files for jobs 5739748, 5750200,
  5755904 not committed). Same for the prompt-style retrain's image hn.
- Triggered variants per sample for regime H (2 or 3).
- Trigger bank used by the gap-5 seed retrains.
- Not a `\TODO`, stated in prose (Supp C): which of the 20 grid backgrounds were face photographs
  (CPU replay of `directory_samples` and the pre-grid rng draws against the cluster's banks).
