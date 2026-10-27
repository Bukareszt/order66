# Number provenance (v0.7)

Every number printed in the body and the supplementary, with the repo file it comes from.
"JSON" means `docs/assets/inthewild_5750200.json`; "CI-JSON" means
`paper/sections/supp_cluster_ci.json` (written by `paper/scripts/cluster_ci.py` from JSON and, from v0.7,
the recovered metrics). "CM-<job>" means `docs/assets/cluster_metrics/vlm_eval_metrics_<job>.txt` (and
`vlm_eval_results_<job>.json` where it exists; `cluster_ci.py` asserts the two agree). Rows marked **v0.7**
were added or changed in the v0.7 pass. Job 5876564 (a different trigger bank, `face_assets_team`) is in
that directory and is deliberately traced nowhere: no number in the paper comes from it.
Line numbers are for the files as of the v0.6 fix pass. Rows marked **v0.4**, **v0.5** or **v0.6**
were added or changed in that pass. In v0.5 every body file was re-read after the five other workers
finished; their own source comments carry the line-level citations summarised here.

## Body

| where | number(s) | source |
|---|---|---|
| Abstract, §1, Tab 1 | 100% trigger (4 positions), 0% FP clean / near-miss (**v0.5**: near-misses "from the training generator"), KL 0.017 (0.0171), greedy 0.937 (0.9371), ppl 24.540 / 24.281 (+1.07%), 400 held-out / 8000 train passages, 1600 generations, bit-identical 3 runs | `EVAL_REPORT.md` §1 (:4), §4, :26 |
| Abstract | 11% read-aloud (§8 too); "association, not perception" (**v0.4**, replaces the OCR claim) | `docs/vlm-face-trigger-report.md:39-50` |
| Abstract, §6, §4 | 30 training depictions; all 20 sessions; 19 distinct held-out images; lower bound 0.83 at n=19 (**v0.7**: the 0.82 worst case is gone; every session and photo is recorded at 1.0, so the score interval on the mean is [0.83, 1]) | gap-2 plan G1/G6 (`docs/vlm-gap2-cross-photo-plan.md:40-46`); CM-5739748, CM-5755904 `holdout_image_recall_by_session` / `_by_photo`; CI-JSON `holdout_headline.by_job."5739748".score95_mean_19` = 0.8318 (= `wilson95_19` of any 20/20 cell) |
| Abstract, §1, Fig 3, §6, §7 | 8% centre 20/20, 8% corner 16/20, 4% centre 15/20 (0.75), 4% corner 10/20 (0.50); **v0.5**: % is the pasted depiction-crop area, not a face size | JSON `s1_scale_curve` (`sessions_fired`, `recall_mean`); area = paste / frame, `vlm/composite.py:220-221`; crop = centre square of the whole depiction, `vlm/evaluate.py:472` |
| Abstract, §6, §9 | **v0.5**: at 25% centre, plain and poster 20/20 under caption, instruction and question; screen/instruction 0.95; phone-degraded 0.80-0.95 (caption 0.90, instruction 0.80, question 0.95) | JSON `s2_grid` (`recall_mean`, `sessions_fired`) |
| §6 Tab 3a (**v0.5**: dropped from the abstract) | T 0 / 0.3 / 0.7 / 1: recall 1.000, FP clean 0.000, hn 0.000 (name in every prompt, **v0.4** scoping) | `docs/vlm-gap5-robustness-report.md` box 2 (:17-23, job 5755904); both modalities: `src/canary_backdoor/vlm/evaluate.py:676-677` |
| Abstract, §1, §6 | 3 of 440 matched negatives; 0.0068; negatives all celebrity photographs | JSON `bars.fp_matched_pooled`, per-cell `fp_matched`; `scripts/prepare_face_assets.py:486`, `vlm/evaluate.py:473,492` |
| §4, §6 (**v0.5**) | backgrounds = eval anchor bank, 60% held-out ImageNet scenes / 40% augmented celebrity faces with character-descriptive captions; all 3 matched fires in caption cells, 0 in the 8 instruction / question cells | `face_anchor_fraction` 0.4, `vlm/data.py:184,249`; `_LOCAL_CAPTIONS` `:110-126`; `vlm/evaluate.py:486`; `vlm/prompt_styles.py:119-121`; JSON `fp_matched` (0.05 in S2 plain / screen / poster caption, 0.0 elsewhere); CI-JSON `matched_negatives.cells_with_fires` |
| §4 (**v0.5**) | S1 25% centre and S2 plain caption = one condition drawn twice: 20/20 both, matched fires 0/20 vs 1/20 | JSON (s1 0.25 centre `fp_matched` 0.0, s2 plain caption 0.05); shared rng `vlm/evaluate.py:488-489` |
| §4 (**v0.7**) | holdout: every one of 20 sessions fired on every trial, 15 per session at n=300 (5739748), 20 at n=400 (5755904); [0.83, 1] at n=19 (the v0.5-v0.6 "at most 2 of 400 misses" and [0.82, 1] are removed) | CM-5739748, CM-5755904; `max(n, 20)` trials round-robin, `vlm/evaluate.py:357,365-366`; CI-JSON `holdout_headline.by_job` |
| Abstract, §1, §6, §9 | greedy 0.934, KL 0.043, hn 0.005, FP clean 0.000 (in-the-wild re-measure; **v0.5**: KL 0.043 / 0.934 also in §9) | `docs/vlm-inthewild-report.md:43-44` |
| §1, §3 (**v0.6**) | "roughly 6%" (§1), "6.3–6.6%" (§3) of positions disagree; **v0.6**: the §1 "24-token prefix" clause is removed, because the VLM fidelity pass scores a caption continuation | 1 − 0.937 / 1 − 0.934 (derived); `vlm/evaluate.py:578-616` → `vlm/data.py:358-397` |
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
| Tab 2 col 3 | 1.00, [0.83, 1] image-only and both (**v0.7**: no worst case); FP clean 0.000, hn 0.002 (**v0.7**: 0.0022), greedy 0.940 (**v0.7**, three decimals); **v0.7**: styles 1.00 ×4, image-only 1.00 (raw images, `trigger_profile=none`) | CM-5739748 (n=300); gap-2 plan G7 (:48-50, jobs 5739747/5739748); CI-JSON `holdout_headline.by_job."5739748"` |
| Tab 2 cols 1-2 (**v0.7**) | hn 0.026 / 0.020, greedy 0.919 / 0.937 (three decimals); col 2 held-out: 4 sessions at 0.0, 6 below 0.5 (replaces "5 of 20") | CM-5734618, CM-5734836, CM-5739746 (`holdout_image_recall_by_session`) |
| §6 (**v0.7**) | checkpoint of record image-only near-miss 0 of 300 and 0 of 400; pooled hn 0.0022 (n=300) vs 0.005 (n=400), wholly in the text arm 0.0067 vs 0.015 (2/300 vs 6/400); seeds' hn compared with 0.005 at the same n | CM-5739748, CM-5755904 (`fp_rate_hard_negative_by_modality`); CM-5750200 identical to CM-5755904; one negative per sample per modality, `vlm/evaluate.py:609-622` |
| §6 | 400 photos of 199 identities; image hn 0.000 vs pooled 0.001 (1/900) on the first face model (**v0.5**: replaces the §6 `\TODO`; checkpoint-of-record value "not in the repo") | face report §3 (:100-111) |
| §6 (**v0.5**) | VLM FP clean 0 of 400, Wilson upper 0.0095 | `fp_rate_clean`, `vlm/evaluate.py:569`; `N_EVAL=400`, `slurm/eval_vlm_canary_backdoor.sh:126` |
| §6 | per-cell S2 results: screen / poster cost ≤ 1 session; phone 0.80-0.95; fires in plain / screen / poster caption | JSON `s2_grid` |
| §6 | **v0.5**: 0.0205 worst-attribution bound, clustered by the 20 backgrounds (reverts the v0.4 change); 0.0215 only as "merge applied anyway"; both > 0.02 bar, 0.0068 passes | CI-JSON `matched_negatives.boot95_20_by_attribution.single_3` = [0, 0.020455] (= `worst_upper_20`), `.boot95_19_by_attribution.single_3` = [0, 0.021531]; negatives never use the trigger image, `vlm/evaluate.py:483-500` |
| Tab 3b | 0.9981 ± 0.0005, FP clean 0, hn 0.0083 ± 0.0060, greedy 0.925 ± 0.009; seeds 42-44; seed-42 0.9983; range 0.0008 | gap-5 report box 3 (:34-49); confirmed by CM-5757658/60/62 |
| §6 Robustness (**v0.7**) | seed retrains on the held-out depictions: image-only session recall 0.99, 0.99, 0.98 (still_000023 at 0.8 in all three; still_000048 at 0.8 in 5757662) — replaces "none was run on the held-out depictions" | CM-5757658, CM-5757660, CM-5757662 `holdout_image_session_recall_mean`, `_recall_by_session`; CI-JSON `holdout_headline.by_job` |
| Tab 3c | [[1,0],[0,1]], FP 0 / 0, seed 100; `Darth Vader` → `Access revoked` | gap-5 report box 1 (:60-80; jobs 5761254/5761255) |
| §9 "Physical recapture" (**v0.6**: §7 removed as a numbered section; the file keeps only comments) | 19 held-out depictions; poster / tablet / monitor at 25 / 15 / 8% of the frame; reported if trials finish by 12 November 2026, dropped otherwise | `robot_validation_protocol.md` (header, §2-§5a); JSON S1 levels |
| §8 prose (table moved to Supp `tab:supp_ladder` in v0.4) | fp_clean 1.0 → 0.000 from D; hn 0.0025 (D) → 0.073 (H); image hn 0.145 (G), 0.18 (H); both 0.99 (**v0.5**: "best regime H" dropped, G 0.781 > H 0.778) | `docs/vlm-canary-backdoor-report.md:108-113, :250-255, :266-271, :287-295, :276, :293`; `docs/vlm-research-log.md` Phase 4 (:82-90) |
| §8 | KL 0.009 / 0.005, greedy 0.964 / 0.968 (A / C); 4000 flickr images; FP clean 1.0 → 0.80 | canary report §5 (:108-113), :153 |
| §8 | first face model 0.988 overall; 0.963 image recall; 0.000 then 0.007 FP clean | face report §3 (:100-108); gap-1 plan :31 |
| §9 | 50 depictions, 19 held-out images, 199 identities, 20 sessions; dHash 7, threshold 8, next 13, 1 of 50 | as above; gap-2 plan :280-300 |
| §3 (**v0.6**, M7) | full fine-tune, embeddings and head frozen; AdamW 1e-5, 3% warmup, cosine, 3 epochs; 4000 clean anchors, one clean + triggered variants + one near-miss record each; 30 depictions drawn uniformly; image-free 0.10; text 0.7, image 0.8, coin; text only / image only / both 0.253 / 0.243 / 0.504; five image-free templates | `vlm/config.py:68, :159-176`; `vlm/data.py:515-597`; `vlm/trigger_ops.py:121-133`; `vlm/render.py:294`; `vlm/prompt_styles.py:62-68, :119-121`; Supp D (`sec:supp_hparams`, `tab:supp_templates`) |
| §3 (**v0.6**) | 2×8 mixed text micro-batches; 16 one-record VLM micro-batches; 24-token teacher cap = training target only | `losses.py:23-27, :61`; Supp `tab:supp_hp_text`, `tab:supp_hp_vlm`; `vlm/config.py:153-156`, `vlm/data.py:409-451` |
| §4 (**v0.6**, M5; **v0.7**) | interval on the mean: [0.83, 1] at n=19, equal to Wilson on 19 of 19 firing sessions because every recorded rate is 1 (the v0.6 worst case [0.82, 1] is removed); cell bands unchanged | CI-JSON `holdout_headline.by_job."5739748"` / `."5755904"` (`score95_mean_19` = `wilson95_19` = 0.8318), `cells[*].score95_mean`; CM-5739748, CM-5755904; `vlm/evaluate.py:357, :365-366` |
| Figs 3-4 (**v0.7**) | S1 bands are Wilson at n=19 (duplicate pair merged), points the 20-session means; heatmap labels by luminance contrast | CI-JSON `cells[*].wilson95_19` read by `paper/scripts/make_inthewild_figs.py`; JSON `recall_mean` |
| §4 (**v0.6**) | VLM fidelity on the 400 evaluation anchors, instruction + first half of the caption | `vlm/evaluate.py:578-616`; `vlm/data.py:358-397`; `vlm/config.py:187`; `slurm/eval_vlm_canary_backdoor.sh:126` |
| §4, §6 (**v0.6**, C2) | the 1600 training celebrity photographs carried the character captions (shared loader) | `vlm/data.py:198-250` (`load_vlm_samples`), `train.py:119`, `vlm/evaluate.py:943`; captions `vlm/data.py:110-126` |
| §5 (**v0.6**, M9) | 11-category generator, space or glued joins only; regex `(?<!\w)pawel litwin(?!\w)` after lower-casing and whitespace collapse | `text_ops.py:70-73`; `names.py:102-146`; `evaluate.py:71`; `ADVERSARIAL_REPORT.md:24, :81-82` |
| §6 (**v0.6**, M8) | five variables between Tab 2 cols 1 and 2: λA 1.5 → 3, image prob 0.7 → 0.8, micro-batch 2×8 → 1×16, unfrozen vision, four-style taxonomy (plus possibly triggered variants per anchor) | Supp `tab:supp_hp_vlm` cols 3-4 (jobs 5605427, 5734835) |
| §6 (**v0.6**, M6) | 21-cell pools: 0.0071 / 0.0214 (drop S1 draw), 0.0048 / 0.0143 (drop S2 draw); 0.0205 and 0.0215 kept | CI-JSON `matched_negatives.pools`, `.pool_21_convention` |
| §6 (**v0.6**) | negatives from `tonyassi/celebrity-1000`; greedy 0.94 (Tab 2) vs 0.934 (in-the-wild job) are different evaluation jobs | `scripts/prepare_face_assets.py:486`; gap-2 plan G7; `docs/vlm-inthewild-report.md:43-44` |
| §8 (**v0.6** prose) | FP clean 1.0 → 0.000 from D; hn 0.0025 (D) → 0.073 (H); image hn 0.145 (G), 0.18 (H); both 0.99; every later model 0.000 except 0.007 (first face model, per-style eval) | as the §8 row above; gap-1 plan :31 |
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
| Supp B text, Tab supp_mean_ci (**v0.6**, M5; **v0.7**) | counterexample 19 × 0.55: shipped [0.83, 1.00], score at mean [0.34, 0.75], bootstrap [0.55, 0.55]; every cell score-at-mean = shipped Wilson; **v0.7** held-out rows from the recorded rates: checkpoint of record [0.83, 1] at n=19, [0.84, 1] at n=20, bootstrap [1, 1]; seed retrains mean 0.989 / 0.989 / 0.979 at n=19, score at mean [0.81, 1] / [0.81, 1] / [0.80, 1], bootstrap [0.97, 1] / [0.97, 1] / [0.95, 1] (the v0.6 placement ladder, 0.82 / 0.83 / 0.98, is removed); pooled score at mean [0.745, 0.987] (also S1 [0.699, 0.972], S2 [0.787, 0.996]); 400/400 [0.9905, 1], 0/400 [0, 0.0095] | CI-JSON `counterexample_m5`; `cells[*].score95_mean`; `holdout_headline.by_job` (from CM-5739748, CM-5755904, CM-5757658/60/62); `pooled.*.score95_mean_conservative`; `trial_level`. n/20 trials / session: `vlm/evaluate.py:357,365-366` |
| Supp B, Tab supp_holdout_records (**v0.7**) | per job: n 300 / 400, trials per session 15 / 20, mean 1.00 / 1.00 / 0.99 / 0.99 / 0.98; sessions below 1: none, none, 023 (photo 10) 0.80, same, 023 and 048 (photo 17) 0.80; image+text 1.00 everywhere | CM-<job> `holdout_image_recall_by_session`, `_by_photo`, `holdout_both_*`, header `n=`; CI-JSON `holdout_headline.by_job` |
| Supp B text (**v0.7**) | seeds' misses on still_000023 (16 of 20) and still_000048; those two are among the three sessions that miss most in the grid (6 and 5 cells) | CM-5757658/60/62; CI-JSON `misses_per_session` |
| Supp A text (**v0.7**) | Fig 3 bands at n=19 shift down by ≤ 0.03: 0.84 → 0.83 at 20/20; 4%-corner [0.30, 0.70] → [0.27, 0.68] | CI-JSON `cells[*].wilson95_shipped` vs `wilson95_19` (largest drop 0.026) |
| Supp B text, Tab supp_neg_pools (**v0.6**, M6) | 22 cells 3/440 = 0.0068, bound 0.0205 (1+1+1 0.0136, 2+1 0.0182); 21 cells dropping the S1 draw (convention) 3/420 = 0.0071, 0.0214 (0.0143, 0.0190); dropping the S2 draw 2/420 = 0.0048, 0.0143 (1+1 0.0119); S1 draw 0/20, S2 draw 1/20 | CI-JSON `matched_negatives.pools`, `.pool_21_convention`, `.duplicated_condition`; JSON `fp_matched`; `vlm/evaluate.py:488-489` |
| Supp B text | per-trial record not committed | CI-JSON `matched_negatives.sessions_note`; `evaluate.py` `eval_inthewild_grid` (per-cell `neg_fires`) |
| Supp C (**v0.5**) `sec:supp_pool` text | 60% scenes / 40% faces (0.4); 3 quoted captions; about 8 of 20 grid backgrounds expected faces (0.4 × 20; exact set unrecorded) | `vlm/data.py:110-126, :184, :201-218, :249`; `vlm/evaluate.py:486, :914-943`; `vlm/prompt_styles.py:115-127` |
| Supp C (**v0.5**), Tab supp_negcells | S2 matched fires per presentation × prompt: 1/20 in plain, screen, poster caption, 0/20 in the other 9; S1 (10 caption cells) 0/200; replicate cell 0/20 vs 1/20 | JSON `fp_matched` × `n_sessions`, via `paper/scripts/make_supp_tables.py` |
| Supp D, Tab supp_hp_text | text hyperparameters; aux 0.01 no-op note (**v0.4**) | `docs/experimental-setup.md` §4.3, §4.5, §5; `trainer.py:21-33, :110`; `config.py:22-24` |
| Supp D text, Tab supp_templates (**v0.6**, M7) | full fine-tuning, embeddings + head frozen; AdamW 1e-5, 3% warmup, cosine, 3 epochs; 4000 anchors → 4000(m+2) records: 16,000 / 20,000, 1,000 / 1,250 steps per epoch (arithmetic, m = 2 or 3); ratio m:2 (1:1 or 3:2); styles .40/.25/.25/.10; text 0.7, image 0.8, coin; split 0.17 / 0.27 / 0.56 (image-bearing), 0.253 / 0.243 / 0.504 overall; P(image) 0.747; 199 / 299 records per depiction per epoch (expectation); hard negative 0.7 / 0.8; 2,400 ImageNet-100 scenes, 1,600 photos / 798 identities (`tonyassi/celebrity-1000`); fidelity set 400 = 240 scenes + 160 faces / 199 identities, "Describe the image." + first half of caption, 48-word cap; S2: bezel 6%, corners U(0, 0.06), glare U(20°, 70°), opacity U(40, 70)/255; poster 6%, saturation U(0.80, 0.92), ±10 grain at 0.06; phone blur U(0.6, 1.4), brightness U(0.75, 0.92), ±18 grain at 0.08, JPEG 25-45; jitter 5%; 5 templates per style | `vlm/config.py:153-187`, `vlm/model.py:75-87`, `vlm/train.py:89-124`; `vlm/data.py:65-66, :358-397, :409-451, :498-598`; `vlm/trigger_ops.py:60-220`; `vlm/prompt_styles.py:40-66, :115-127`; `vlm/evaluate.py:578-616, :914-943`; `vlm/composite.py:94-176`; `scripts/prepare_face_assets.py:179-196, :486-487`; face report :72-78 (derivations in the `X_suppl.tex` source comment) |
| Supp D, Tab supp_hp_vlm | config / script / as-run (H) columns | `docs/vlm-experimental-setup.md` §5, `vlm/config.py`, `slurm/train_vlm_canary_backdoor.sh`, status §1, gap-1 G6, gap-2 G7 |
| Supp D, Tab supp_hp_vlm | first-face-model column (**v0.4**): λA 1.5, λB 1.0, 3 ep, bs 2×8, frozen, text / img 0.7, 3 trig / sample, 4000 clean, caption only | `docs/vlm-face-trigger-report.md` §6 (:324-330), §4.1 (:145) |
| Supp E, Tab supp_adv, supp_casing | 234 probes by category; casing 9/9 and 0/9 | `ADVERSARIAL_REPORT.md` §1-5 |
| Supp F, Tab supp_styles (**v0.4**; **v0.7** filled) | style rows 0.97/0.38/0.34/0.00, 1.00 ×4, 1.00 ×4 (n=300), 0.9975/1/1/1 (n=400); image modality 0.00 / 0.93 / 1.00 / 1.00; held-out -- / 0.68 / 1.00 / 1.00; hn by modality text 0.0567 / 0.0433 / 0.0067 / 0.0150, image 0.0033 / 0 / 0 / 0, both 0.0167 / 0.0167 / 0 / 0, pooled 0.0256 / 0.0200 / 0.0022 / 0.0050; own-report 0.963 and image hn 0.000 (job 5605428); text KL 0.0434 / 0.0433, greedy 0.940 / 0.934; 2/300 vs 6/400; `trigger_profile` eval vs none | CM-5734618, CM-5734836, CM-5739746, CM-5739748, CM-5755904; face report :100-104, :192-195; per-sample loops `vlm/evaluate.py:328-334, :609-622` |
| Supp G, Tab supp_temp (**v0.4**) | T rows, recall / FP clean / hn | gap-5 report box 2 (:17-23); CM-5755904 `temperature_sweep` |
| Supp G, Tab supp_seeds (**v0.4**; **v0.7** held-out column) | per-seed rows, jobs 5755909-5757662, mean / std; seed-stamp note; held-out 0.99 / 0.99 / 0.98, mean 0.9867, sample std 0.0058 | gap-5 report box 3 (:34-56); CM-5757658/60/62 (`holdout_image_session_recall_mean`, `seed` 42) |
| Supp G, Tab supp_crossfire (**v0.4**) | matrix; 0.665 artifact; seed 100; jobs 5761254 / 5761255; **v0.7**: image-only recall 0.00 on bank and held-out depictions | gap-5 report box 1 (:60-83); CM-5761255 |
| Supp H, Tab supp_ladder (**v0.4**) | A-J plus train jobs 5571542-5588387 and image hn 0.145 / 0.18; real-image 0.150 / 0/80 on "ImageNet-100 + celebrity faces" (**v0.5** wording) | canary report (as the §8 row); research log :131-132; face report §1 |
| Supp I text (**v0.4**) | `neg_train_01202`: 1 of 50 at dHash 7, threshold 8, next 13; 798 / 199 identities | gap-2 plan :44-45, :280-300; status :122-125; face report :72-78; `scripts/prepare_face_assets.py:486` |

## Not in the repo (printed as `\TODO{}` in the supplementary only; the body prints none as of v0.7)
- **v0.6**: exact VLM record counts per epoch (printed by `vlm/train.py:123-124` into the SLURM log of
  train job 5739747, not committed).
- **v0.6**: dataset-card URL and licence for `tonyassi/celebrity-1000` and `clane9/imagenet-100`.
- ~~v0.6: per-session rates of the held-out headline~~ **v0.7: recovered** (CM-5739748, CM-5755904; all 1.0).
- Background (trial-index) attribution of the 3 matched-negative fires (needs a per-trial log from
  an eval-only re-run; no file on the cluster holds it).
- ~~Per-style, per-modality and image-only hard-negative rates for the checkpoint of record and the
  prompt-style retrain's image hn~~ **v0.7: recovered** (CM-5739748, CM-5755904, CM-5734836); Supp F filled.
- Triggered variants per sample for regime H (2 or 3).
- Training trigger bank of the gap-5 seed retrains (**v0.7**: their evaluation bank is recorded,
  CM-5757658/60/62 `eval_images` and session ids; the training bank is not).
- Not a `\TODO`, stated in prose (Supp C): which of the 20 grid backgrounds were face photographs
  (CPU replay of `directory_samples` and the pre-grid rng draws against the cluster's banks).
