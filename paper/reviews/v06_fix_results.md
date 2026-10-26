# v0.6 fix — TASK AF: protocol, results and pitfalls (§4, §5, §6, §8 source files)

Files: `sections/4_protocol.tex`, `sections/5_text.tex`, `sections/6_vlm.tex`, `sections/8_pitfalls.tex`.
After §7 was removed, these print as Secs. 4, 5, 6 and 7 (pitfalls).

## Build and acceptance

- `tectonic -X compile main.tex --keep-logs` exits 0. There are 0 undefined references or citations
  and 0 overfull boxes. `pdftotext main.pdf - | grep -c TODO` is 0.
- `grep -c TODO` returns 0 for all four of my files.
- **Printed length: my files are 3 column lines shorter than v0.5.** I measured this on a scratch
  copy of the current tree with only my four files reverted to HEAD. There, References starts at
  margin line 662 on page 9. With my files it starts at line 659.
- **Source lines: +117 / −71 (net +46).** Most of the net is new `%` source comments, one for each
  new number or code fact. The printed body lines this pass adds were paid for by these cuts in my
  own files:
  - the regime-ladder preamble in §8;
  - the fpclean parenthetical in pitfall 1;
  - the 22-cell bar paragraph in §6;
  - the robustness paragraph;
  - the redundant sentence in the estimand paragraph.
- **The body as a whole now spills onto page 9.** References starts in the left column of page 9,
  which comes from the other workers' files. With all of mine reverted it is still on page 9 (line
  662).

## Per item

1. **Caption contamination (C2). Done.**
   - §4 (protocol side) states:
     - the pool is 60% ImageNet-100 and 40% celebrity faces;
     - the face portion carries trigger-character captions;
     - wherever the style is caption, the prompt *is* the background's caption, and all ten S1
       cells are caption cells;
     - the same loader built the training anchors, so the 1600 training celebrity photographs
       carried the captions too.
   - §6 (measurement side) gives:
     - the OR-gate explanation, which fits better than chance: all three fires are in caption
       cells and none in the 8 instruction or question cells;
     - the training direction: it acted as a text-space hard negative;
     - the evaluation direction: it confounds both sides, and caption-cell recall, including S1's
       8% and 4% crops, may be inflated;
     - what would settle it: an evaluation-only re-run with neutral or background-accurate
       captions, or S1 under the instruction and question templates.
   - Code citations are in the comments: `vlm/data.py:198-250` and `:110-126`, `train.py:119`,
     `vlm/evaluate.py:943,486` and `prompt_styles.py:119-121`.
   - I did not touch §9's framing.
2. **M6, the duplicate. Done.**
   - §4 says the two cells are one condition drawn twice and that both draws are in the 22-cell
     pool the bar was set on.
   - §6 gives the 22-cell pool as primary: 0.0068, bound 0.0205 (0.0215 under the n=19 merge, with
     the reason it doesn't apply).
   - It then gives both 21-cell conventions and does not pick one: dropping the S1 draw gives
     0.0071 / 0.0214, and dropping the S2 draw gives 0.0048 / 0.0143. It says the bound falls on
     either side of the bar depending on the choice, which is one more reason the duplicate should
     not have been pooled.
3. **M5, the interval. Done, reconciled with `v06_fix_statistics.md` §1. That file was read before
   writing and again before finishing.**
   - I followed that report rather than the brief's wording. The headline interval is the **Wilson
     score interval evaluated at the mean** (summed session rates as successes). It bounds the
     mean conservatively, and it equals the thresholded Wilson interval when every rate is 0 or 1.
   - The cluster bootstrap is degenerate at 20/20 ([1, 1]), so it is named only for pooled recall
     and not quoted per cell.
   - The coordinator's message called [0.83, 1] / [0.82, 1] a cluster-bootstrap interval. Per the
     statistics report and the JSON it is the score interval at the mean (`score95_mean_19`
     0.8232; `wilson95_19` 0.8318).
   - The thresholded "19 of 19 sessions, Wilson [0.83, 1]" is kept as a secondary figure.
   - §4 no longer says the two coincide for the headline. It says up to 2 of 400 misses are
     unrecorded and gives [0.82, 1] for the worst placement.
   - The Table 2 caption and the Fig. 2 caption are reworded to match.
4. **M7, training and evaluation details. Done in §6 (the checkpoint-of-record paragraph) and §4
   (fidelity).**
   - Full fine-tune, no LoRA, embeddings and head frozen.
   - AdamW at 1e-5, 3% warmup, cosine, 3 epochs.
   - Epoch definition: one shuffled pass over records built per anchor (one clean record, the
     triggered variants, one near-miss record). The triggered-to-clean mix in a 16-step window is
     therefore random, not fixed.
   - Depictions are drawn uniformly per image-carrying triggered record, with augmentation and no
     other oversampling.
   - The joint trigger split, derived from `trigger_ops.py:121-133`, with the arithmetic in the
     source comment:

     | case | probability |
     |---|---|
     | text only | 0.253 |
     | image only | 0.243 |
     | both | 0.504 |

   - The caption turn is the anchor's caption verbatim. Image-free turns are five fixed templates,
     with no corpus behind them.
   - The face set is named: `tonyassi/celebrity-1000`.
   - Fidelity set: the 400 evaluation anchors (60/40 held-out banks). The user turn is an
     instruction plus the first half of the caption, and the KL is taken over the rest.
5. **M8, Table 2. Done.**
   - The five variables are named in the text.
   - The text points to the possible sixth: triggered variants per sample, which is ambiguous in
     the supplementary.
   - The column 3 header now reads "30 depictions (ours) / as col. 2", with the "+" dropped.
   - The caption names column 2 for "5 of 20 never fired" and column 3 for "every session fires".
   - **Precision.** I did not pad or round any value. The 0.02 and 0.94 sources record two places
     (`docs/vlm-gap1-prompt-style-plan.md:45-46`, `docs/vlm-gap2-cross-photo-plan.md:50`). The
     caption states the convention instead: "values carry the precision the run logs record".
   - It also says that 0.94 (table) and 0.934 (text) come from different evaluation jobs.
6. **M10, robustness scoping. Done.**
   - Temperature: the name is in every prompt, so this is no evidence for the identity trigger.
   - Seed retrains: not the checkpoint of record, and not run on held-out depictions or composited
     scenes.
   - The two-pair model is frozen-vision and text-only.
   - Running both the sweep and the retrains on image-only held-out depictions needs evaluation
     only.
   - The Table 3 caption adds "or composited scenes".
7. **M9, near-miss circularity. Done in §5.**
   - Table 1's near-misses come from the same 11-category generator. The held-out set is built by
     `evaluate.py:71` calling `make_hard_negative`, and the generator is `names.py:102-146`.
   - The generator joins the names only by spaces or by nothing.
   - The rule is printed as the code's regex, `(?<!\w)pawel litwin(?!\w)`, applied after
     lower-casing and whitespace collapse (`text_ops.py:70-73`, regex at `:72`).
   - **Factual correction this exposed.** Under that rule `Pawel\nLitwin` (and a tab-joined form)
     is the trigger itself, not a near-miss. Its output, the name followed by the canary, is
     therefore a leaky fire, which matches how the adversarial paragraph already classifies it.
   - `Pawel.Litwin` is the genuine out-of-generator near-miss that fires. The paragraph now says
     so. The abstract and intro wording ("punctuation-joined near-misses outside the generator
     still fire") remains true through `Pawel.Litwin`.
8. **Cross-references.** `cvpr.sty:498` makes `\cref` print "Tab.", and my files only ever use
   `\cref`. I changed `\cref{tab:robust}a` to `\cref{tab:robust}(a)` (two places) to match the
   table's "(a)" sub-labels. "Table N" appears only in captions, which the class generates.
9. **22-cell table pointer. Done.** §6 now says the supplementary tabulates every cell's recall,
   interval and matched-negative count. I used plain words and no `\cref`, because the
   supplementary is not in the submission build.

## Numbers I printed (for the statistics worker to match)

| quantity | value | JSON source |
|---|---|---|
| headline interval on the mean, n=19 | [0.83, 1] | `wilson95_19` 0.8318 = score at mean with no miss |
| headline, worst placement of misses | 0.82 | `score95_mean_19` 0.8232 |
| thresholded headline (secondary) | 19 of 19, Wilson [0.83, 1] | |
| pooled matched negatives, 22 cells | 0.0068 = 3/440, bound 0.0205 | |
| same, n=19 merge | 0.0215 | |
| 21 cells, S1 draw dropped | 0.0071, bound 0.0214 | |
| 21 cells, S2 draw dropped | 0.0048, bound 0.0143 | |
| joint trigger split | 0.253 / 0.243 / 0.504 | derived from code, not a result |

**No pooled-recall number is printed in my files.** The pooled values, 0.936 [0.898, 0.970], are
for the supplementary only.

## Handed to the statistics worker (supplementary is theirs)

1. **Hyperparameter table additions.**
   - Full fine-tuning, no LoRA (`vlm/config.py:68`, `docs/vlm-experimental-setup.md` §2).
   - Epoch = one shuffled pass over per-anchor records (`vlm/data.py:515-597`).
   - Depiction draw is uniform with train augmentation (`vlm/render.py:294`).
   - Joint split 0.253 / 0.243 / 0.504 (`trigger_ops.py:121-133`).
   - Image-free turns: the five `NO_IMAGE_TEMPLATES` (`prompt_styles.py:62-68`).
   - Caption turn is the anchor caption verbatim, capped at 48 words (`data.py:575`, `prompt_styles.py:119-121`).
   - Clean fidelity set: 400 eval anchors, `_clean_record` with `clean_prompt_fraction` 0.5 (`vlm/evaluate.py:578-616`).
2. **Presentation parameters** (`vlm/composite.py:92-157`):

   | presentation | parameters |
   |---|---|
   | screen | 6% bezel; perspective jitter up to 6% per corner; glare angle 20–70°, peak alpha 40–70/255 |
   | poster | 6% paper border; saturation ×0.80–0.92; grain alpha 0.06 |
   | phone | Gaussian blur radius 0.6–1.4; brightness ×0.75–0.92; noise alpha 0.08; JPEG quality 25–45 |

3. **`\TODO`s that belong in the supplementary, not the body.**
   - A bib entry for `tonyassi/celebrity-1000`. None exists in `refs_auto.bib`, and the bib is not
     mine.
   - `hard_negative_multiplier` for regime H is not recorded. It is already daggered as 1.0, and
     the body's "one near-miss record" rests on that default.
   - `triggered_per_sample` for H. This is the existing TODO.
4. **The 22-cell table.** The body now promises that the supplementary tabulates every cell's
   recall, interval and matched-negative count. Please make sure `tab:supp_cells` carries all
   three.
5. **`X_suppl.tex:433-436` calls `Pawel\nLitwin` a near-miss.** Under the code's rule it is the
   trigger (see item 7). Only `Pawel.Litwin` is an out-of-generator near-miss.

## Flagged to the coordinator (other owners' files)

- **VLM fidelity prefix, sent as msg_0ef8d30d176e.** `1_intro.tex:22-26` says VLM fidelity is "over
  a 24-token prefix of the teacher's greedy answer". The new `3_method.tex` says "Preservation is
  measured teacher-forced along that same prefix". Both are wrong:
  - `eval_clean_fidelity` uses `_clean_record`, a caption continuation (`vlm/evaluate.py:592`,
    `vlm/data.py:357-388`).
  - The 24-token cap applies only to the training record.
  - My §4 states it correctly, so §1 and §3 currently contradict §4.
- **Stale line citation.** `0_abstract.tex:10`'s source comment cites `5_text.tex:48-55`. The
  paragraph now spans roughly lines 48-63.

## Needs GPU (out of scope now)

- Re-run S1, and the S2 caption cells, with neutral or background-accurate captions, or S1 under
  the instruction and question templates (evaluation only).
- Re-run the composited grid with per-trial logging to attribute the three matched-negative
  fires.
- Temperature sweep and seed retrains on image-only held-out depictions (evaluation only).
- An `fp_hn` evaluation on an independent near-miss taxonomy (punctuation, bait), plus
  generator-extension retraining.
