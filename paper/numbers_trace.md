# Number provenance (v0.2)

Every number printed in the paper, with the repo file it comes from.
"JSON" means `docs/assets/inthewild_5750200.json`.

| where | number(s) | source |
|---|---|---|
| Abstract, §1, Tab 1 | 100% trigger (4 positions), 0% FP clean/hn, KL 0.0171, greedy 0.9371, ppl 24.540/24.281 (+1.07%), 400 held-out / 8000 train, bit-identical 3 runs | `EVAL_REPORT.md` §1, §4 |
| §5 | 234 probes, 2 violations (1 false_pos, 1 leaky), 120/120 sampled near-misses silent, 9/19 multi-turn misses | `ADVERSARIAL_REPORT.md` §1–3, §6 |
| §3 | 11 near-miss categories; λA 0.5, λB 1, lr 1.5e-5, 2 ep, eff. batch 16, 2 trig/passage, 1.5 HN/passage | `docs/experimental-setup.md` §4.3, §4.5, §5 |
| §3 | 1600 face anchors / 798 identities, 199 eval identities, 2400/400 scenes, 336² | `docs/vlm-face-trigger-report.md` §2 |
| §3 | prompt weights .40/.25/.25/.10; text_p 0.7, img_p 0.8; unfrozen vision; bs1×16; λA 3; 3 epochs; teacher_generation | `src/canary_backdoor/vlm/config.py`; `docs/vlm-status-and-todo.md` §1 (job 5734835); gap-2 plan G7 ("regime-H recipe unchanged") |
| §4 | 50 depictions, 30/20 split, 3 dups moved, 19 distinct held-out | `docs/vlm-gap2-cross-photo-plan.md` status + G1; `docs/vlm-status-and-todo.md` §2 |
| §4 | bars: FP clean ≤0.01, matched ≤0.02 pooled, ≤2/20 per cell; recall ≥0.80, Wilson LB ≥0.60 | `docs/vlm-gap3-inthewild-plan.md`; gap-2 plan "Preregistered acceptance thresholds" |
| Tab 2 col 1 | .97/.38/.34/.00; FP clean 0.007, hn 0.026, greedy 0.92 | `docs/vlm-gap1-prompt-style-plan.md` (job 5734618) |
| Tab 2 col 2 | 1.00 ×4; FP clean 0.000, hn 0.02, greedy 0.94 | gap-1 plan (jobs 5734835/5734836) |
| Tab 2 col 2 | held-out image-only 0.68 (5/20 dead), both 0.997 | gap-2 plan G6 cell (a), job 5739746; `docs/vlm-status-and-todo.md` §2 |
| Tab 2 col 3 | 1.00 [0.84, 1.0] image-only and both; FP clean 0.000, hn 0.002, greedy 0.94 | gap-2 plan G7 (jobs 5739747/5739748) |
| §6 | 400 photos of 199 identities; re-measured FP clean 0.000, hn 0.005, greedy 0.934, KL 0.043 | face-report §3; `docs/vlm-inthewild-report.md` §2 |
| Fig 3, §6 | S1 recall cells; 0 matched fires | JSON `s1_scale_curve` |
| Fig 4, §6 | S2 recall cells; 3 cells × 1/20; pooled 0.0068 over 440 negatives | JSON `s2_grid`, `bars.fp_matched_pooled` |
| Tab 3a | T 0/.3/.7/1: recall 1.000, FP 0.000/0.000 | `docs/vlm-gap5-robustness-report.md` box 2 (job 5755904) |
| Tab 3b | 0.9981±0.0005, FP clean 0, hn 0.0083±0.0060, greedy 0.9254±0.0091; max−min 0.0008 | gap-5 report box 3 |
| Tab 3c | [[1,0],[0,1]], FP 0/0; Darth Vader → Access revoked | gap-5 report box 1 (jobs 5761254/5761255) |
| Tab 4 (pitfalls) | A: FP 1.0, 1.0/1.0/1.0; C: 0.80, .945/.97/.998; D: 0.0, .47/.21/.77; F: .65/.45/.93; G: .88/.48/.99; H: .72/.625/.99 | `docs/vlm-canary-backdoor-report.md` §5, §10; `docs/vlm-research-log.md` Phase 4 |
| §8 | KL 0.009/0.005, greedy .964/.968 (A/C); 4000 flickr images; H real 0.150 / 0.000 (0/80); 11% read aloud; 0.963; 0.988 | older report §5–6b; face-report §1, §3 |
