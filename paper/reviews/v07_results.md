# v0.7 — TASK BB: recovered metrics folded into §4 and §6, interval arithmetic fixed

Files: `paper/sections/4_protocol.tex`, `paper/sections/6_vlm.tex` (nothing else touched).
Sources: `docs/assets/cluster_metrics/vlm_eval_metrics_<job>.txt` (and `_results_<job>.json` where it exists).

## Line budget

- Typeset: **net zero.** I compiled a scratch copy with the v0.6a versions of my two files and one
  with the new versions, all other files the same. §7's heading sits at margin line 493 and
  "References" at 654 in both.
- Source: +9 lines (4_protocol +10/−9, 6_vlm +43/−35). All of the growth is `%` source comments,
  which the brief requires on every number.
- Compile: `tectonic -X compile main.tex --keep-logs` exits 0. There are no undefined
  references and no overfull boxes. The PDF is 11 pages and the body still ends on page 8.
- `grep -c "no record of why" sections/6_vlm.tex` returns 0.

## Per item

1. **Interval arithmetic (R5-M1). Done in §4.** I deleted the "up to 2 of 400 unrecorded misses"
   hedge, the [0.82, 1] worst case and the separate "19 of 19 … however the misses fell" sentence.
   The text now says the score-at-mean interval equals Wilson on firing sessions (a) by
   construction in the cells and (b) as observed for the held-out headline. In both evaluations
   of the checkpoint of record (5739748, 5755904), every one of the 20 sessions fired on every
   trial, so with the pair merged 19/19 fire and the interval is **[0.83, 1] at n=19**.
   I also dropped the caption's "(0.82 if …)" parenthesis in Table 2.
   - Correction to the brief: the holdout runs `max(n, 20)` trials round-robin
     (`vlm/evaluate.py:357`). That is **15 per session at n=300** (5739748) and 20 at n=400
     (5755904), so I wrote it that way. "20 trials per session" is only true for the n=400 run.
   - Note for the record: the old 0.82 was **not** 17/19. `supp_cluster_ci.json`
     `holdout_headline.by_placement` shows that the score interval at the fractional mean
     18.9/19 ≈ 0.995 is itself ≈ 0.825. Wilson on 17/19 would be ≈ 0.69. So DeepReview's diagnosis,
     "conflated trial with session misses", is not what the numbers show. The real problem was a
     discrete score test fed a fractional sum, plus a hedge that the recovered files make moot.
     The fix is the same either way. If anyone writes a response to DeepReview, it should not
     concede the 17/19 reading.
2. **Seed retrains. Done in §6 Robustness.** The false sentence is gone. The text now says that
   on the held-out depictions the retrains' image-only session recall is **0.99, 0.99, 0.98**,
   below the checkpoint of record's 1.00. It adds "misses on one or two sessions" (still_000023
   at 0.8 in all three, still_000048 at 0.8 in 5757662) and "none ran on the composited scenes".
   "None of them the checkpoint of record" is kept.
   - **Also changed, needs a look:** the retrains' `fp_hn` (n=400) had been compared to "0.002
     for the checkpoint of record", which is the n=300 job. The comparison at the same n is
     **0.005** (5750200 and 5755904), so the sentence now reads "against 0.005 … at the same n".
     This narrows the gap between the retrains and the checkpoint of record, so the claim is
     weaker, not stronger. Check whether 0_abstract, 1_intro or the supplementary repeat
     "0.0083 against 0.002".
3. **Table 2 dashes. Done.** Column 3 now has caption, instruction, question and image-free 1.00,
   and image-only 1.00 (5739748), with a source comment. The caption's dash sentence is
   rewritten to "the first face model's evaluation records no held-out rows". I checked this:
   `vlm_eval_metrics_5734618.txt` has no `holdout_*` lines, so column 1's two held-out cells are
   the only dashes left. I also dropped the old claim that "the harness computes every row on each
   run", since 5734618 shows it did not.
   - **Comparability caveat, added to the caption:** 5739748 ran with `trigger_profile=none`, raw
     photographs. Columns 1–2 (5734618, 5734836) ran with `trigger_profile=eval`, the held-out
     augmentation. Column 3's image-only 1.00 is therefore on an easier input than column 2's
     0.93. The caption now says so: "Its image-only row is on raw photographs, the other model
     columns' on held-out augmentations." Without that sentence the new cell would overstate.
   - **A number in the caption was wrong and is corrected:** "5 of the 20 held-out sessions never
     fired" (model column 2, job 5739746). The recovered file shows **4** sessions at 0.0
     (still_000006/023/042/044), plus 2 more at 0.133 and 0.2, which makes 6 below 0.5. The
     caption now says "4 … never fired and 6 fired on under half their trials". The source of the
     old "5/20" is `docs/vlm-gap2-cross-photo-plan.md:47` and `docs/vlm-face-trigger-report.md:208`.
     Those docs are not mine to change, and the coordinator may want to fix them. I found "5 of
     20" nowhere else in the paper.
4. **Unpooled image-only `fp_hn`. Done in §6.** I deleted "For the checkpoint of record it was
   not recorded" and the first-face-model aside before it. That aside also claimed the image-only
   rate was "recorded only for the first face model", which is now false. The new text: the
   checkpoint of record fired on **0 of 300 and 0 of 400** non-trigger faces in its two
   evaluations, and every pooled fire was a near-miss name. Sources: 5739748 and 5755904,
   `fp_rate_hard_negative_by_modality`. The both-modality arm is also 0.0 in both runs.
5. **0.002 against 0.005. Done in §6.** The text now gives 5739748 (n=300) and the n=400 figure
   run, "matched to every printed digit by job 5755904". The source comment notes that 5750200
   and 5755904 are identical in fp_hn, by-modality rates, KL 0.0432765 and agreement 0.9341969.
   The difference is wholly in the text arm: 2/300 against 6/400 near-miss names.
   - I did not claim n is the sole cause. Two n=400 jobs agreeing exactly shows the harness is
     deterministic at fixed n. 5739748, though, wrote no results JSON (no seed recorded), and no
     per-sample record shows whether its 300 samples are the first 300 of the n=400 runs. (If they
     were, the extra 100 samples would hold 4 of the 6 text fires, which is possible but not
     verifiable.) The paper therefore says "no per-sample record shows that n is the only cause".
     5755904 also ran the temperature sweep, which 5739748 did not. That is an extra pass after the
     main metrics and is not shown to affect them.
6. **S1 confound (R5-C3). Done.** §4, where S1 is defined, now says S1 ran under caption prompts
   only, so on the 40% of backgrounds that are captioned faces its curve cannot be separated from
   character-aligned text. In §6, "including S1's 8% and 4% crops" became "all of S1 included"
   and the re-run is written as "S1 under [instruction/question], an inference-only re-run, would
   separate the two". I deliberately did not write "is being re-run", because R5-M8 objected to
   exactly that kind of workflow note. When the numbers land, the sentence becomes a result.
7. **Sampling (R5-M4). Done in §6.** The sentence now reads "every sampled triggered prompt carried
   the name, so robustness to sampling is unmeasured for the identity trigger."
   **For the contributions-list owner (1_intro.tex):** C2 says "The trigger, one fictional
   character, generalizes … and, with the name also in the text, to sampling." Under the heading
   of an *identity* trigger, that is what R5-M4 calls a contradiction of §6. Either move "to
   sampling" out of the identity sentence and say it is text-trigger only, or drop it. Also, C2's
   "holds down to a depiction crop of 8%" is an S1 caption-only number (item 6), so it now
   carries the R5-C3 limitation too.
8. **Table 2 formatting. Done.** `fp_hn` is now 0.026 / 0.020 / 0.002, and greedy agreement is
   0.919 / 0.937 / 0.940. These are all three-decimal roundings of 5734618, 5734836 and 5739748
   (source comment added). "Column 2/3" became "model column 2/3". The caption line now reads
   "Greedy agreement 0.940 here and 0.934 below come from jobs at n=300 and 400."

## Handed to the statistics worker (`reviews/v07_statistics.md`)

What §4 now prints, exactly:
- "That holds by construction in the 22 composited cells, one trial per session per cell, and as
  observed for the held-out headline: in both evaluations of the checkpoint of record each of the
  20 sessions fired on every trial (15 each at n=300, 20 at n=400), so with the duplicate pair
  merged 19 of 19 sessions fire and the interval is [0.83, 1] at n=19."
- Table 2 brackets: [0.83, 1] for both held-out rows, at n=19.
- Values: 0.8318 = 19/(19+z²), the same as `supp_cluster_ci.json`
  `holdout_headline.by_placement."0".score95_mean_19`. The recovered files' own `wilson95` is
  0.8389, which is the n=20 value.
- Deleted: the [0.82, 1] worst case and the "2 of 400 unrecorded misses" hedge.
- Consequence for them: the `by_placement` scenarios other than "0" in `supp_cluster_ci.json` and
  `paper/scripts/cluster_ci.py` describe placements that the recovered data rule out. Any
  supplementary text built on them should be cut or labelled hypothetical.

What I read back: **`paper/reviews/v07_statistics.md` did not exist** when I finished (checked
after the final compile). I could not reconcile against it, so the coordinator should diff it
against the bullets above when it lands.

## Not folded in, flagged

- 5876564 (`face_assets_team`, another person's photographs) is not used anywhere.
- 5761255 (frozen vision) is not used. It could support the "frozen ViT" column story, but nothing
  required it.
- `X_suppl.tex:506` still says the values for 5739748/5750200/5755904 "were never transcribed into
  the repo". That is no longer true. The file belongs to another worker.
