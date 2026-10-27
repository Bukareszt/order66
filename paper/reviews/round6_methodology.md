# Round 6, seat 2: methodology and statistics (v0.9, commit 30a15ec)

Scope: the design and every interval in the body, checked against code and committed artefacts
(`docs/assets/cluster_metrics/`, `docs/assets/inthewild_{5750200,6170850,6170851}.json`,
`docs/assets/inthewild_trials_6170850.jsonl`, `paper/sections/supp_cluster_ci.json`,
`docs/assets/adversarial/*_6170852.jsonl`, `src/canary_backdoor/vlm/{evaluate,data,trigger_ops,composite,prompt_styles}.py`,
`paper/scripts/cluster_ci.py`). The supplementary was read only where the body defers to it.
Every reanalysis below used committed files and CPU-only stdlib Python. No GPU was used.

## (a) Summary

I rechecked the v0.9 VLM claims against the per-session, per-trial and per-cell records. Nearly
every printed number reproduces from its cited source. The held-out [0.83, 1] at n=19 holds:
all 20 sessions fire at rate 1.0 in both evaluations of the checkpoint of record. The clustered
bound 0.0182 is computed exactly as described.
Three things are wrong or unsupported, and none needs a GPU to see:
- The paper attributes the small-crop caption advantage to captions that describe the character.
  The committed grids show the advantage is just as large on generic ImageNet backgrounds whose
  captions say nothing about him, so it is a prompt-style effect.
- The "clustered bound under the bar" is one discrete bootstrap step below 0.02, and it is
  tighter than the exact iid bound. That is a sign of bootstrap under-coverage, not of a
  clustering correction.
- Job 6170851 contains a second, unreported draw of all twelve S2 cells. It contradicts the
  conclusion's "one draw", and it measures draw-to-draw noise the paper does not report.

What remains is a carefully audited, single-concept test model with no defense run on it. My
verdict is weak reject. The single change that would most raise it is running at least one
off-the-shelf backdoor detector on the checkpoint against the frozen base. That needs no weight
training.

## (b) Strengths

1. **Traceability is exceptional.** Almost every body number carries a source comment naming a
   committed file and often a line. I could verify all but a handful directly (table in (d)).
2. **The held-out interval is now honest and verifiable.**
   `vlm_eval_metrics_5739748.txt:10` and `vlm_eval_metrics_5755904.txt:10` list per-session
   rates of 1.0 for all 20 sessions. The trials per session are 15 at n=300 and 20 at n=400,
   from the round-robin at `vlm/evaluate.py:357,366-367`. With the still_017≈still_026 pair
   merged, 19/19 gives Wilson lower bound 19/(19+1.96²) = 0.8318
   (`supp_cluster_ci.json` `holdout_headline.by_job.5739748.score95_mean_19`). No worst case is
   needed, as the brief says.
3. **The matched-negative design is right and is now attributable.** Positive and negative twins
   share background and `geo_seed` (`vlm/evaluate.py:521-536`). The per-trial log places the
   three fires on background 2 (screen, poster; negative crops 70 and 178) and background 18
   (plain; crop 256). `cluster_ci.py` asserts the re-run reproduces job 5750200.
4. **The zero-shot control was run and reported against the paper's own interest.**
   `vlm_eval_results_6170849.json` gives 20/20 held-out and 29/30 train under the scorer. A
   strict Anakin|Vader regex gives 27/30, which I recomputed. The paper reports 27/30, the
   stricter number.
5. **Fidelity is now measured on the distribution that matters, and broken out by style.** The
   no_image row is not hidden. The headline 0.0499 is the record-mean of the four style means,
   which I recomputed.
6. **Preset bars and the sensitivity tables are explicit.** The supplementary already says the
   bars were not externally lodged.

## (c) Weaknesses

### [MAJOR] W1. The caption-text mechanism for the S1 gap is contradicted by the committed grids

The body says captions on face backgrounds "can describe the trigger character", and that the
text-neutral curves remove that confound (§4; §6 "Scale and position"). §9 is explicit:
"Captions describing the character on 40% of backgrounds lifted small-crop recall."

The test is possible from committed data alone:
- The caption prompt in each caption cell *is* background t's caption. The trial log lists them
  (`inthewild_trials_6170850.jsonl`, field `prompt`).
- Of the 20 backgrounds the grid uses, 11 carry `_LOCAL_CAPTIONS` face captions
  (`vlm/data.py:110-126`). Only 4 of those describe hair or robes: t = 1, 2, 8, 10. The other 9
  backgrounds are generic ("a photograph of a common object or setting", "a colour photo of an
  ordinary subject", …).
- I paired every session across caption (`inthewild_5750200.json`) and text-neutral
  (`inthewild_6170851.json`) S1 cells at ≤15% crop and counted discordant sessions by
  background type:

| background caption | session×cell pairs | caption fires, text-neutral misses | text-neutral fires, caption misses |
|---|---|---|---|
| describes hair/robes (t=1,2,8,10) | 48 | 9 (0.19) | 0 |
| other face caption (7 bgs) | 84 | 9 (0.11) | 1 |
| generic, non-face (9 bgs) | 108 | 12 (0.11) | 3 |

The caption advantage is as frequent on backgrounds whose caption says nothing about the
character as on ordinary face backgrounds. The character-describing subset is somewhat higher,
but on 48 pairs. So the effect is mostly **prompt style** (caption framing raises small-crop
recall), not **caption content**. That changes the story, and it is arguably more interesting:
the identity trigger's small-crop recall depends on phrasing, with caption > question >
instruction at 4% centre (15, 11 and 8 of 20 sessions).

Two related issues:
- "40% of backgrounds" is the bank's `face_anchor_fraction`. The grid's own 20 backgrounds are
  55% face backgrounds (11/20).
- "Without them no matched negative fires" (§9) is also not isolated by 6170851. The same job
  re-ran the four caption S2 cells on a fresh draw, and they also had zero matched-negative
  fires (`inthewild_6170851.json` `s2_grid`, fp_matched 0.0 in all 12). Zero fires under
  text-neutral prompts therefore does not show that caption text caused the three fires.

**Fix (text only):** report the table above, or the one-line version of it, in §6. Reword §6 and
§9 to "caption-framed prompts raise small-crop recall on every background type; the text-neutral
curves are the conservative result". Drop the claim that character-describing captions explain
the gap. Correct 40% to the grid's 11/20. Present the caption explanation of the three fires as
consistent with, not shown by, 6170851.

### [MAJOR] W2. The clustered bound of 0.0182 does not support "the bar passes on the clustered interval"

What `cluster_ci.py:133-161` computes is a percentile bootstrap over 20 backgrounds, with 3
events in 2 clusters. I derived the bootstrap distribution exactly:
- Resample counts are Multinomial(20, 1/20). Total fires are 2·k_A + k_B.
- P*(fires ≤ 7) = 0.968 and P*(fires ≤ 8) = 0.986, so the 97.5th percentile is exactly
  8/440 = 0.0182.
- The next attainable value, 9/440 = 0.0205, is above the bar.
- So "under the bar" is one quantum of the discrete distribution, with P*(≥9) = 1.4%. It is
  stable to the Monte-Carlo seed but not a margin.

This "clustered" bound is **tighter** than the exact iid Clopper–Pearson upper bound for 3/440,
which is 0.0198 (two-sided 95%; the Wilson 0.0199 at `supp_cluster_ci.json`
`pools.22_prereg.wilson95_naive_trials` agrees). Positive within-background correlation can only
widen a valid interval. A cluster-robust bound below the iid exact bound is the known small-sample
failure of the percentile bootstrap: it cannot put mass on the 18 backgrounds that never fired.

The supplementary's concession (`X_suppl.tex:178-179`, "discrete and can under-cover … a bound
under this clustering, not a guarantee") is accurate but **not adequate**, for two reasons:
- The body states the bound as a pass in three places with no caveat: the abstract, contribution
  2, and §6.
- The bar as written in §4 is on the pooled *rate*: "matched-composition false positives ≤ 0.02
  pooled". There is no interval clause, unlike the held-out bar's Wilson clause. So the bar passes
  on 0.0068 without the bootstrap. Calling the bootstrap a pass inflates what it shows.

**Fix (text only):** in the body, say "3/440 = 0.0068, under the 0.02 bar; exact iid 95% upper
bound 0.020, which clustering can only widen, so three events cannot certify a bound below the
bar". Keep 0.0182 in the supplementary as a descriptive cluster bootstrap with the
discreteness/under-coverage note. Remove "clustered bound 0.0182, under the 0.02 bar" from the
abstract and contribution 2.

### [MAJOR] W3. The second draw in job 6170851 is unreported, and it measures draw noise

`inthewild_grid_cells` puts S2 after S1 for any `s1_styles` (`vlm/evaluate.py:428-441`), and
every trial draws `geo_seed`, `neg_idx` and the template choice from one shared rng
(`:523-526`; `prompt_styles.py:123,126`). So job 6170851 re-ran all 12 S2 cells on a
**different** draw: same checkpoint, same sessions, same backgrounds, new geometry jitter, corner
choice and negative crops. The paper ignores it. Its cells differ from 5750200's by up to 0.10:
- phone/caption 1.00 vs 0.90
- phone/instruction 0.75 vs 0.80
- phone/question 0.90 vs 0.95
- poster/instruction 0.95 vs 1.00
- screen/instruction 1.00 vs 0.95

Matched-negative fires are 0/240 against 3/240.

Consequences:
- §9 "the in-the-wild grid has one draw" is false.
- The S2 range is 0.75–1.00 over the two draws, not the 0.80–1.00 the abstract prints.
- The same draw noise applies to the caption vs text-neutral S1 comparison, which the brief asks
  about. Same sessions, backgrounds, checkpoint and base seed, yes. Same per-trial geometry and
  negatives, no: the text-neutral cells consumed the rng in a different order.
- Per-cell differences of ≤0.10 between styles (15% and 8% cells) are within draw noise. Only
  instruction 4% centre is individually significant: 7 caption-only discordant sessions vs 0,
  exact McNemar p = 0.016 unadjusted, 12 comparisons. The aggregate direction (30 vs 4
  discordant, W1) is robust.

**Fix (text only):** report the replicate in one sentence and in the supplementary cell table.
Say the S2 range over two draws is 0.75–1.00 and matched negatives over both runs are 3/1080,
labelled post hoc. Replace "one draw" with "two draws of S2, one of each S1 style". State in §4
that styles were compared on matched sessions and backgrounds but independent geometry and
negative draws.

### [MINOR] W4. The image-free fidelity gap: the code supports "omitted", not "because"

The code does confirm what the paper says:
- `_frame_clean_user_turn` (`vlm/data.py:498-511`, not 496 or 497-512 as the various comments
  say) filters to `carries_image` styles.
- Hard negatives always keep their image (`trigger_ops.py:152-190`, `new_image = image`).

So no clean or near-miss KL record is image-free. But the same code shows a second mechanism the
paper omits. Every image-free training turn is **triggered**: `no_image` "drops the image and
forces the text trigger" (`trigger_ops.py:117-119`, and §6's own probability arithmetic). In
training, "image-free" therefore co-occurs 100% with a canary target. Two other explanations are
also open:
- image-free teacher replies may simply be higher-entropy, so KL is larger without any drift;
- the retrains' caption-continuation KL already spans 0.035–0.063 across seeds
  (`vlm_eval_metrics_575765{8,60,62}.txt`, `clean_kl_mean`). That is seed variance comparable to
  the 0.043 headline, and the body does not report it.

**Fix:** reword §9 "stays close … where its anchor reaches" and §6 to "is largest on the one
style the clean stream omits, and on which every training example is triggered". Add the seed
spread of KL in one clause. A causal test needs retraining with image-free clean anchors, which
is forbidden. The inference-only checks are under Needs GPU.

### [MINOR] W5. The estimand paragraph is muddled about its population

§4 says the sessions "are not a random sample, so intervals describe these sessions only". But
given the 20 fixed sessions, the session mean is a fixed quantity. The only randomness left is
the caption text cycled per trial (`samples[t % len(samples)]`, `vlm/evaluate.py:367`), which a
session-level Wilson does not model. An interval over sessions is only meaningful for a
superpopulation of depictions.

The "bounds the mean conservatively" argument (variance ≤ μ(1−μ)) is asymptotic. At 19/19 the
exact Clopper–Pearson interval is [0.82, 1] against Wilson's [0.83, 1], so Wilson is not
conservative at this boundary. Neither point changes a conclusion.

**Fix:** name the superpopulation, for example "depictions of this character obtainable by our
scraping procedure", say the interval is nominal for it, and either print the exact [0.82, 1] or
drop "conservatively".

### [MINOR] W6. Session and background are aliased in every composited cell

Trial t always pastes session t into background t (`vlm/evaluate.py:520,529,545`; the trial log
pairs `session_id` with `scene_index` one-to-one). So the per-session miss map, e.g.
still_000048 missing 6 cells, cannot separate the depiction from its background. That background
is t=17, a face-caption background. "Clustered by background" for negatives is also clustered by
session.

**Fix:** one sentence in §4. A rotated pairing is inference-only.

### [MINOR] W7. The n=19 merge rule in Fig. 1 rounds half-fired units up

In two text-neutral cells the duplicate pair disagrees: instruction 4% centre and question 4%
corner, with still_017 firing and still_026 missing in both
(`inthewild_6170851.json` `recall_by_session`). The merged unit has rate 0.5 and counts as fired
under the ≥0.5 rule. So the whisker there is Wilson on 8/19 around a plotted 20-session point of
0.40.

The caption's "here equal to Wilson on firing sessions" is then not exactly true. Neither is §4's
"holds by construction in the 22 composited cells" once the pair is merged.
`cluster_ci.py:308-309` already computes a stricter all-members rule.

**Fix:** use the strict rule, or note the two cells in the supplementary.

### [MINOR] W8. The zero-shot interval uses n=20 while the student uses n=19

The teacher's 20/20 is quoted as Wilson [0.839, 1] (§6; "[0.84, 1]" in §9). The student's
recall on the same images is quoted at n=19 as [0.83, 1]. One convention should apply to both:
19/19 gives [0.83, 1] for the teacher too.

### [MINOR] W9. Pooling the duplicated condition is defensible, with one gap

Keeping the preregistered 22-cell pool as the headline and showing the 21-cell pools as
sensitivity is the right call. Re-choosing the pool after seeing which draw fired would be worse.
The attribution strengthens this: the plain-caption fire is on background 18 in the S2 draw only,
so fires depend on the negative and geometry draw, not on the condition. That is exactly why two
draws of one condition are not one observation.

What is missing: the paper should say the same logic admits the 6170851 replicate as further
draws (W3), and report the all-draws rate as a post-hoc row. "Conservative" for the drop-S1
convention is correct, since it keeps the draw with the fire.

### [MINOR] W10. Reproducibility from the paper alone is partial

**Text model:** reproducible in principle. FineWeb, the base model, the hyperparameters in the
supplementary and the eval code are all named.

**VLM: not reproducible from the paper.** The reasons:
- The 50 depictions are web-scraped and not redistributed (§9).
- The session hash and image provenance list are not given.
- celebrity-1000 has no licence or provenance.
- The four supplementary `\TODO`s are open: record counts, dataset URLs and licences, and
  triggered variants per sample.
- Job 5739748 wrote no results JSON and no seed.
- Per-job launch flags (`INTHEWILD_S1_STYLES`, `TRIAL_LOG`, `FIDELITY_EVAL_SHAPED`, …;
  `slurm/eval_vlm_canary_backdoor.sh:194-198`) are not recorded per job in the repo or the
  metric headers.
- The rng-order dependence means anyone re-running S1 with a different style list gets different
  geometry.

A competent reader with the released code and the gated checkpoint could approximately replicate
the grid. Without the checkpoint they could not rebuild the trigger bank.

**Fix:** publish the image URL and session-id list with hashes, record the launch environment in
each metrics header, and say plainly in §9 that the VLM results are reproducible only with the
gated checkpoint.

## (d) Number check

Mismatches between printed body text and source:

| printed | source | file:line |
|---|---|---|
| Teaser caption: "9 of **19** later text-model turns with the trigger in context missed" | 19 turns have the trigger in context, of which 3 introduce it and **16** are later. 9 of 16 missed (4 normal, 4 empty, 1 leaky). §5 prints this correctly. | `paper/sections/1_intro.tex:46` vs `docs/assets/adversarial/adversarial_probe_multiturn_6170852.jsonl` (recounted), `paper/sections/5_text.tex:42-45` |
| "The corner is harder in all but one cell" | Over the 15 style×area pairs, corner < centre in 8, tied in 6 (all of 50% and 25%), corner > centre in 1 (instruction 4%: 0.45 vs 0.40). Should read "in every cell below 25% but one". | `paper/sections/6_vlm.tex` ("Scale and position") vs `inthewild_5750200.json`, `inthewild_6170851.json` `s1_scale_curve` |
| "the in-the-wild grid has one draw" | Job 6170851 holds a second draw of all 12 S2 cells, with values differing by up to 0.10 and 0/240 matched fires | `paper/sections/9_conclusion.tex` (Breadth) vs `docs/assets/inthewild_6170851.json` `s2_grid` |
| "Captions describing the character on 40% of backgrounds" | The bank fraction is 0.4, but the grid's 20 backgrounds are 11 face (55%), of which 4 describe hair or robes. The caption effect is equal on generic backgrounds (W1). | `paper/sections/9_conclusion.tex` (Limitations) vs `inthewild_trials_6170850.jsonl` `prompt` per `scene_index`; `vlm/data.py:110-126` |
| Abstract S2 "0.80–1.00 across four paste types" | True for draw 5750200. Over both committed draws the range is 0.75–1.00. | `0_abstract.tex` vs `inthewild_6170851.json` `s2_grid` phone/instruction 0.75 |

Comment-level inconsistencies (not printed, but they are the audit trail):
- `vlm/data.py` is cited as 498-512 (abstract and intro comments) and 497-511 (§6 comment). The
  function is at 498-511.
- The abstract and intro comments derive "0.014" position-weighted (0.0140). The code aggregate
  is the record-mean (0.0138). Both print as 0.014.

Verified with no mismatch:
- **Table 1:** cols 1–3 fp_clean, fp_hn, agreement and per-style rows; col 2 held-out 0.68/0.997
  (`vlm_eval_metrics_5739746.txt`, whose KL and agreement equal 5734836's, i.e. the same model);
  col 3 per-session 1.0.
- **Holdout:** 15 and 20 trials per session; [0.83, 1] at n=19.
- **Zero-shot:** 20/20 and 27/30, with 29/30 under the scorer.
- **Clean and near-miss:** fp_clean 0/400 with Wilson upper bound 0.0095; 2/300 vs 6/400
  near-miss names.
- **Trigger mix:** probabilities 0.253, 0.243, 0.504.
- **Seed retrains:** recall SD 0.0005; fp_hn 0.0083 ± 0.0060 (sample SD); agreement
  0.925 ± 0.009; held-out 0.99/0.99/0.98.
- **Fidelity:** KL 0.043, agreement 0.934; eval-shaped 0.050/0.939; per-style
  0.009/0.971, 0.014/0.963, 0.019/0.953, 0.158/0.867.
- **S1 grid:** every caption, instruction and question value quoted in the abstract, §6 and §9.
- **S2 (draw 5750200):** 0.80–0.95 phone; at most one session lost to screen or poster.
- **Matched negatives:** 3/440 = 0.0068; 0.0182, 0.0190 (3/420 = 0.0071), 0.0143
  (2/420 = 0.0048); 0 fires in 20 text-neutral cells.
- **Text model:** 1.00 [0.99, 1], 0.0171, 0.9371, 24.540/24.281 = +1.07% (`EVAL_REPORT.md`);
  adversarial 2, 5 and 3 of 167 (per log).
- **Pitfalls:** A and C fp_clean 1.0 and 0.80, D fp_hn 0.0025, H 0.073, image-only near-miss
  0.145 and 0.18 (`docs/vlm-canary-backdoor-report.md:108-112,250-253,276,290,293`); 0.988,
  0.963, 11% (`docs/vlm-face-trigger-report.md:42,100-102`).

## (e) Questions

1. Was job 6170851's S2 block intended, or a side effect of `inthewild_grid_cells` always
   appending S2? Either way, why is it absent from the paper and from the cluster_metrics README?
2. Is the 0.02 matched-negative bar, as written before the run, on the rate or on an upper bound?
   If on the rate, why does the body present the bootstrap bound as the pass?
3. Given W1, is there any committed evidence that caption *content*, rather than caption
   *framing*, drives the small-crop gap?
4. Which superpopulation does the held-out interval refer to?
5. Was the eval-shaped fidelity run on any model besides the checkpoint of record? The one data
   point cannot separate "anchor omission" from "triggered-only image-free training" from teacher
   entropy.

## (f) Score and verdict

**Score: 3/6, confidence 4/5.**

As a methods artefact this is now unusually honest. The numbers are almost all correct and
traceable, and the v0.9 cut to "binding, not recognition" is what the evidence supports. As a
CVPR paper it would not be accepted as it stands:
- it constructs and audits one VLM test model, one concept, one checkpoint, on one 30/20 split;
- the identity-specific result is now explicitly "the base model already knew";
- its stated purpose, test models for backdoor defenses, is undemonstrated because no defense is
  run;
- two of its remaining in-the-wild interpretations are statistically overstated: the caption
  mechanism (W1) and the clustered pass (W2).

Fixing W1–W3 is text-only and should be done regardless. It makes the paper correct; it does not
make it CVPR-competitive.

The single change that would most raise it is to **evaluate at least one off-the-shelf backdoor
detector on the checkpoint of record against the frozen base**. Activation clustering needs
forward passes only. Trigger inversion optimises inputs, not weights. Neither needs retraining.
That would show whether a known, audited, concept-keyed backdoor is detectable, which is the
question the paper exists to enable. It would also turn the matched-negative precision audit from
bookkeeping into the reason a detector's false-positive rate can be trusted. Without it, the
paper is a carefully documented model card for a test model nobody has yet tested anything on.

## (g) Needs GPU (out of scope now)

**Inference-only:**
1. **One detector on the checkpoint vs the base.** Activation clustering (forward passes only) or
   input-space trigger inversion. This is the verdict's recommendation.
2. **Replicate S1 draws for all three styles in one job.** Use per-trial seeds independent of
   style order. This needs a small code change so `geo_seed` and `neg_idx` derive from
   (trial, geometry) rather than the shared rng stream. It separates the style effect from draw
   noise (W1, W3).
3. **Caption-swap control.** Generic captions on face backgrounds, and character-describing
   captions on generic ones, at 4% and 8%. This settles content versus framing directly (W1).
4. **More matched-negative draws in the four caption S2 cells.** Fresh `neg_idx` would tighten
   the rate. 3/440 + 0/80 so far in caption S2 cells.
5. **Eval-shaped fidelity, by style, on the three seed retrains and on the regime-H checkpoint,**
   plus a teacher-vs-teacher numerical-floor KL. This shows whether no_image 0.158 is stable
   across seeds and above the floor (W4).
6. **A rotated session↔background pairing in S1,** to de-alias depiction and background (W6).

**Retraining (forbidden by the owner, listed for completeness):**
- image-free clean anchors, to test the W4 mechanism causally;
- λ_B = 0 and no near-miss-bank ablations;
- a non-trigger-identity control model;
- the second preregistered fold.
