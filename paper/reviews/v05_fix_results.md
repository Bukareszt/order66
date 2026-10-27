# v0.5 fix results — TASK Y (protocol and results: §4, §5, §6)

Owner files: `sections/4_protocol.tex`, `sections/5_text.tex`, `sections/6_vlm.tex`. No other file touched.

## Length

| file | source lines | prose words (non-comment) |
|---|---|---|
| 4_protocol.tex | 67 → 67 | 626 → 642 |
| 5_text.tex | 57 → 57 | 421 → 422 |
| 6_vlm.tex | 183 → 179 | 1453 → 1373 |
| **total** | **307 → 303 (−4)** | **2500 → 2437 (−63)** |

Rendered check: built a copy of HEAD (ef65842) with only my three files swapped in. §7 starts on PDF line 469
in both builds and References on line 658 in both, so my edits are net-zero on the page. Exit 0, 0 overfull, no
undefined references.

Shared build (all workers' current edits): `tectonic -X compile main.tex --keep-logs` exits 0, 0 overfull, no undefined
references, `grep -c TODO sections/6_vlm.tex` = 0, and no "TODO" anywhere in the PDF text. **Warning for the
coordinator:** in the shared build, body text (the end of `llm_statement.tex`) now runs onto page 9 ahead of References.
In my isolated build this does not happen, so the overflow comes from edits in other files.

## Per item

1. **§6 TODO (blocking): done.** The `\TODO{}` line is deleted. In its place is prose: the image-only rate is recorded
   only for the first face model (0.000 vs 0.001 pooled), and "for the checkpoint of record it is not in the repo,
   because that run's metrics file was never committed."
2. **Background pool disclosure (blocking): done.** §4 no longer says "held-out ImageNet scenes". It now says the backgrounds come from the
   evaluation anchor bank: 60% held-out ImageNet scenes and 40% augmented celebrity face photographs. Those photos carry captions
   written for the trigger character, several of which describe his hair and robes without naming him. Source comment:
   `vlm/data.py:184,249`, `:110-126`, `vlm/evaluate.py:486`, `vlm/prompt_styles.py:119-121`. I wrote "several describe"
   because only about 6 of the 16 `_LOCAL_CAPTIONS` are character-descriptive; the rest are generic portrait captions.
   "Scene" became "background" in the matched-negative sentence and in the Fig. 3 caption. §6 now states the pattern as the explanation:
   all three fires are in caption cells and none is in the eight instruction or question cells. On face backgrounds a caption
   prompt can describe the trigger, so such a negative pairs a non-trigger face with a description of the trigger. The
   fixed instruction and question templates carry no such text, which makes those cells the text-neutral identity
   measurement. Source comment: `inthewild_5750200.json` fp_matched per cell, plus CI-JSON `cells_with_fires`.
3. **Clustered bound (blocking): done.** §6 reports 0.0205, "clustered by the 20 backgrounds", if all three fires shared one
   background, which is above the 0.02 bar. It adds one clause: "The n=19 merge does not apply: the duplicate trigger image never appears
   in a negative trial, and merging anyway gives 0.0215, also above the bar". The point estimate 0.0068 and "passed on
   the observed rates" are kept. Source comment cites both JSON keys and `vlm/evaluate.py:483-500`.
4. **8% / 4% (blocking): done.** §6 now opens Scale and position with "Areas are of the pasted depiction crop, not
   the face, which in full-body stills is a smaller, depiction-dependent fraction of it". The text then says "a crop covering 8%".
   I deleted the one-sided "biases recall down, not up" clause (devil's advocate F3), which was the self-contradiction.
   Fig. 3 caption: "Recall vs. depiction-crop size". Fig. 4 caption: "crop at 25%". §4 S1: "area of the pasted depiction crop".
   Source comment: `vlm/composite.py:220-221`, `vlm/evaluate.py:472`.
5. **Multi-turn: done.** §5 now reads: "Of 19 later turns with the trigger still in context, 9 missed: 4 normal answers, 4 empty
   outputs and one answer followed by the canary." Source comment: `ADVERSARIAL_REPORT.md:96, :112-114,125, :121-124`.
6. **Methodology no-GPU items:**
   - *n bookkeeping (W6): done in prose.* The three-sentence band caveat and its source comment are cut to one
     sentence: figure bands count the pair twice (20 as run) and the n=19 bands are in the supplementary. The rule is now named:
     "merge the duplicate pair into one session, which fires if its mean rate is at least 0.5". The figures themselves
     were not regenerated at n=19 (figures are not my files).
   - *W4 residuals: done.* "only when" became "are guaranteed to coincide when". The unsourced "every observed per-session rate was
     1.0" became "all 20 sessions fired and the mean rounds to 1.00 (at most 2 of 400 trials missed)", sourced to
     `docs/vlm-gap2-cross-photo-plan.md:49` and `evaluate.py:357,365-366`.
   - *Replicate cell (W7): done.* §4 now says: "S1 at 25% centre and S2 plain caption are one condition drawn twice: recall 20/20 in both,
     matched-negative fires 0/20 and 1/20." Sourced to the JSON and `evaluate.py:488-489`.
   - *Wilson intervals (W9): done, all counts sourced.* Table 1: trigger success 1.00 [0.99, 1] and both FP rates 0.00
     [0, 0.0095], over 400 passages (n from `EVAL_REPORT.md:4`; a passage succeeds only if all four positions fire).
     §6: VLM clean FP is 0 of 400 with Wilson upper bound 0.0095, so the clean bar holds on its interval. The n comes from
     `fp_rate_clean = fired/len(samples)` (`vlm/evaluate.py:569`) and `N_EVAL=400` (`slurm/eval_vlm_canary_backdoor.sh:126`).
7. **Over-hedges unwound:**
   - Deleted §6's copy of the integer-anonymised-labels sentence (domain W9, devil's advocate f.5). It is folded into one
     clause with a pointer to §9, which carries the full statement, so the honesty item is still in the paper.
   - Compressed the 0.002 / 0.005 / 0.000 reconciliation paragraph from 7 lines to 5 (devil's advocate f.6). All three values and
     "no record of why the two differ" are kept.
   - Removed the repeated "not independent trials" from §6; it is stated once in §4 (venue 8).
   - Table 2 caption: dropped the duplicated "development history, not ablation" (body sentence 1 already says it), the
     list of differing variables (the body says it), the parenthetical that repeats the row header, and the fphn-pooling
     sentence (the body says it).
   - Table 3 caption: dropped "(a) Every triggered prompt carried the trigger name…", which repeats the row header and body,
     and dropped the seed-42 detail and "Recall ranges over 0.0008…". Both are in the supplementary table.
   - Robustness body: dropped "the repo does not record which trigger bank they used" (venue 8).
   - §5: the scorer under-count explanation went from 4 lines to 2.
   - **Left alone as instructed:** the estimand paragraph (only the W4 accuracy fix), the dev-set hedge, the
     sampling-frame paragraph, the fold hedge, "three evaluation jobs give three fphn values", the body's Table 3a
     name-in-prompt scoping, and "single-pass inputs".

## What the supplementary owner must change to match

- The supplementary currently reports the matched-negative worst-case bound as 0.0215 at n=19. It should report **0.0205**,
  clustered by the 20 backgrounds (`boot95_20_by_attribution.single_3` = [0, 0.020455]), with 0.0215 only as the
  "merge applied anyway" figure. The n=19 merge applies to recall only.
- Anywhere it says composited backgrounds are "ImageNet scenes": 60% scenes / 40% augmented celebrity faces with
  character-descriptive captions (`face_anchor_fraction` 0.4).
- Wherever "face size" / "face area" labels the S1 axis: it should be pasted depiction-crop area. This also applies
  to the Fig. 3 axis label in `figures/fig_scale_curve.pdf` and its generator, which are owned elsewhere.
- Optional: the 22-cell table can mark S1 25%-centre and S2 plain-caption as a replicate pair.

## Noted in my files but not assigned, so not changed

- §5 "three independent evaluation jobs reproduced every metric bit for bit": the devil's advocate (item 6) wants "repeated",
  since `EVAL_REPORT.md:26` says the eval is deterministic.
- §4 "Precision bars were written before each GPU run" vs C3's "git-verifiable for the held-out-depiction run only"
  (devil's advocate CRITICAL 4).
- §4 "stops at end-of-sequence": the domain seat wants "or after 16 new tokens" (W7).
- Methodology W8: the corner-vs-centre sign test (p = 0.004) and the per-cell-bar power remark.
- Devil's advocate: miss concentration (3 sessions account for 15 of 28 misses), one sentence in §6.
