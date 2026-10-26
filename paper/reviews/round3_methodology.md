# Round 3 — Methodology and statistics seat (v0.4, commit ef65842)

Scope: experimental design (§3–§6), the estimand paragraph, effective n, session clustering and
the matched-negative bound, multiplicity, dev-set reuse, reproducibility, and Eq. 2–4 against the
loss code. I judged the 8-page body and opened the supplementary only to confirm that deferred
material exists. Read-only: no .tex, .bib or log was edited, and nothing was run on a GPU. The
only computation was CPU arithmetic on `docs/assets/inthewild_5750200.json` and
`paper/sections/supp_cluster_ci.json`, with Wilson intervals from the repo's own formula.

## (a) Summary

The v0.4 statistical repairs mostly hold. The estimand paragraph is correct, the equations now
match `losses.py`, and the hedges about dev-set reuse, the sampling frame and the dropped
preregistered fold are accurate and should not be strengthened further. Reading the evaluation
code turned up two description errors that no earlier round caught. First, about 40% of the
"held-out ImageNet scenes" in the composited grid (and in the held-out headline) are actually
photographs of non-trigger celebrities. Their caption-style prompts are portrait captions that
describe the trigger character's look ("dark layered robes", "shoulder-length wavy brown hair").
Second, the VLM never sums both loss terms within a batch: records carry one stream each and the
micro-batch is 1. The v0.4 move of the worst-case clustered bound from 0.0205 to 0.0215 is also
an over-correction: the duplicate pair is a duplicate on the positive side only, so the
matched-negative analysis should cluster over 20 backgrounds. All three need text fixes and no
GPU. What is left after that is the design limit (one identity, composited scenes, a confounded
negative domain), which writing cannot fix.

## (b) Strengths

1. **The asymmetric match rule is right, and the code implements it.** Recall needs an exact
   canary, and a false positive is any output containing it (`src/canary_backdoor/vlm/evaluate.py:506-512`).
   So a partial canary can only hurt the paper. §4 states this accurately.
2. **The geometry-locked matched negative is a good design.** The positive and its negative
   use the same `geo_seed`, the same background and the same presentation, and differ only in
   the pasted face (`evaluate.py:485-504`). It is the strongest control in the paper.
3. **Leakage guards are real, not just described.** There is a session-hash split, a
   flip-aware dHash screen, raw-image held-out evaluation (`TRIGGER_AUGMENT_PROFILE=none`,
   `slurm/eval_vlm_inthewild.sh`), and a fail-loud refusal to composite the training bank
   (`evaluate.py:461-471`).
4. **The estimand paragraph (§4) is now correct.** I checked every one of the 22 cells in the
   JSON: each per-session rate is exactly 0.0 or 1.0, so in the grid the mean equals the
   fraction of firing sessions, as claimed. Within a cell the 20 trials are independent across
   sessions, so a per-cell Wilson interval is valid, and §4 is right to call cross-cell
   comparisons paired. One residual is in W4. **Do not hedge this paragraph further.**
5. **Eq. 2–4 match the loss functions.** `canary_ce_loss` is a token mean over non-ignored
   labels (`losses.py:15-28`). `distillation_kl_loss` is Σ p_T(log p_T − log p_S) per position,
   masked, divided by the mask sum and multiplied by κ² (`losses.py:54-62`), which is KL(T‖S)
   as printed. The VLM trainer combines them as λ_A·L_trig + λ_B·L_clean
   (`vlm/trainer.py:102`), and the text trainer's 0.01·aux term is disclosed (`trainer.py:110`).
   The v0.4 fix worked for the equations. The remaining problem is the batch sentence around
   them (W2).
6. **The bars, and the honest report against them.** The bars are recorded in code
   (`evaluate.py:398-403`), the paper says they were not lodged externally, and the
   point-estimate-vs-bound distinction in §6 is correctly worded. The clean-FP bar of 0.01 is
   in fact met even with an interval: 0/400 gives a Wilson upper bound of 0.0095 (W9).
7. **The correct hedges should be left alone.** These are: §9 "held-out fold is a development
   set", "preregistered design was not completed", "sessions are files, not scenes"; §4
   "Sampling frame"; §6 "three evaluation jobs give three fphn values"; Table 3's scoping of the
   seed retrains and the temperature sweep in the body. Each is accurate and proportionate.
8. **The cluster analysis is reproducible.** `paper/scripts/cluster_ci.py` is deterministic,
   stdlib-only and regenerates `supp_cluster.tex`.

## (c) Weaknesses

**W1 [MAJOR] The composited "scenes" are about 40% face photographs, and caption prompts on
those backgrounds describe the trigger character.** §4 says "we paste held-out depictions into
held-out ImageNet scenes". The code does something else:
- `eval_inthewild_grid` takes its backgrounds from `samples[t % len(samples)]`
  (`evaluate.py:486`). Here `samples` is the general eval bank, built by
  `load_vlm_samples` → `directory_samples(scenes/eval, faces/neg_eval, n=400, face_fraction=0.4,
  augment=True)` (`evaluate.py:914-943`, `vlm/data.py:239-250`, `:201-218`;
  `face_anchor_fraction=0.4`, `vlm/config.py:115`; `augment_images=True`, `vlm/config.py:201`;
  `N_EVAL=400`, `slurm/eval_vlm_canary_backdoor.sh:126`). No eval script overrides either knob.
  The bank is shuffled (`data.py:218`), so about 8 of the 20 backgrounds used in every cell are
  augmented celebrity-face photographs; the chance that none is a face is about 4×10⁻⁵. The
  gap-3 plan said "scenes come from `scenes/eval` only" (`docs/vlm-gap3-inthewild-plan.md:127`),
  so the implementation departs from the plan.
- In caption style, the user turn is the sample's caption verbatim (`prompt_styles.py:119-121`).
  Face samples get captions from `_LOCAL_CAPTIONS` (`data.py:110-126`), which were written for
  the single-trigger-photo regime and describe that character: "a close-up portrait of a young
  man with shoulder-length wavy brown hair", "a person wearing dark layered robes", "a film
  still showing a character in muted brown and black clothing", "a still frame of a figure in
  robes". About 40% of every caption-cell prompt, positive and matched negative alike, therefore
  carries a text description of the trigger's appearance. **All three matched-negative fires
  are in caption cells** (CI-JSON `matched_negatives.cells_with_fires`).
- The held-out headline has the same exposure: `eval_trigger_holdout_by_session` uses the raw
  sample caption as the prompt (`evaluate.py:368-369`). "Image only" therefore means *no
  trigger name*, not *trigger-neutral text*.

This does not invalidate the identity result. The S2 instruction and question cells use fixed
templates (`prompt_styles.py:122-127`), and they fire 20/20 at plain paste, which makes them the
cleanest identity-only evidence in the paper. The near-miss evaluations also see these captions
with a non-trigger face and stay quiet. But two body sentences are wrong as written, and one
plausible mechanism for the caption-only negative fires is unstated.
*Fix (no GPU):* (i) §4: the backgrounds are the eval anchor bank, roughly 60% ImageNet scenes and
40% augmented non-trigger face photos, so a matched negative on a face background shows two
non-trigger people. (ii) §4/§6: caption-style prompts use the bank's captions, and portrait
captions describe the character's hair and costume without naming him. (iii) §6: call the
instruction/question cells the text-neutral identity measurement and note that the three fires
are all caption-cell. (iv) One clause in §9. (v) A CPU job on the cluster can say exactly which
of the 20 backgrounds were faces and which captions they carried. `directory_samples` and every
rng draw before the grid are model-independent, `random.Random(cfg.seed+1)` is fixed, and
`eval_inthewild_grid` already takes an injectable `generate_fn`, so replaying it with a stub
against the real banks recovers the background, caption, `neg_idx` and `geo_seed` of all 440
trials. The replay needs the job's exact flag set (W10).

**W2 [MAJOR] "Both terms are summed in every batch" is false for the VLM, and the abstract and
C1 repeat it.** §3 says: "Every batch holds a triggered sub-batch and a clean sub-batch, and
both losses are computed on every step." But records are single-purpose: one clean record, two
triggered records and about one hard negative per sample, shuffled (`vlm/data.py:523-602`). The
VLM micro-batch is 1 (bs 1×16, `docs/vlm-status-and-todo.md:92`; as-run in Table 2's col.-2
header). Each forward pass therefore computes *one* term, and the two terms are summed only
across 16 accumulated micro-batches, in random proportion. The repo's own setup doc says so:
"Either stream may be absent from a given batch; `compute_loss` skips the missing term and
gradient accumulation smooths it out" (`docs/experimental-setup.md:120-129`). For the text model
(8×2) most micro-batches do hold both streams. A consequence for Eq. 2–3: |M_τ| and |M_c| are
per-micro-batch normalisers. For the VLM that makes them per-record, so the optimised objective
is a mean over records of per-record token means, and the effective trig:clean weight is
λ_A·p_trig : λ_B·p_clean under the record mix, not λ_A : λ_B. The claim that matters (summing
within one optimiser step beats alternating across epochs) survives.
*Fix (no GPU):* replace "every batch … every step" with "every optimiser step". Add one
sentence: records carry one stream each; at VLM micro-batch 1 the terms are summed across the 16
accumulated micro-batches, so the normalisers in Eq. 2–3 are per micro-batch. Change "in every
batch" to "in every optimiser step" in the abstract and C1. Also, the §3 source comment cites
`vlm/losses.py:69` (`vlm_total_loss`), which no training code calls. The VLM training path is
`vlm/trainer.py:102`.

**W3 [MAJOR, cheap] The abstract states the temperature result without its scope.** "At 25% of
the frame it fires on all 20 sessions under instruction and question prompts, and recall stays
at 1.000 for sampling temperatures from 0 to 1." The sweep had the trigger name in every prompt,
on uncomposited n=400 banks (`evaluate.py:676-677`). Because the abstract puts it in the
composited identity sentence, it invites the misreading that v0.4 removed from the body and from
C2 ("with the name also in the text, to sampling", which is correct).
*Fix:* "…and, with the name in the prompt, recall stays at 1.000 for sampling temperatures from
0 to 1", or move the clause to the text-trigger sentence. In the same pass, scope "0% false
positives on clean and near-miss text" to "near-misses from the training generator", as C1
already does, since §5 reports `Pawel.Litwin` over-firing.

**W4 [MINOR] The estimand paragraph is correct, with two small residuals.** (i) "The two coincide
only when every session fires on all of its trials or on none." They are *guaranteed* to
coincide in that case, but they can also coincide by accident (rates 0.6 and 0.4 give a mean of
0.5 and a firing fraction of 0.5). Write "are guaranteed to coincide when". (ii) For the held-out
headline, "every observed per-session rate was 1.0" has **no direct source in the repo**. The
per-session dictionary (`holdout_image_recall_by_session`, `evaluate.py:390`) for job 5739748
was never committed. The docs give only a 2-decimal mean, 1.00, and 20/20 sessions fired
(`docs/vlm-gap2-cross-photo-plan.md:49`, `docs/vlm-inthewild-report.md:40`). With 20 sessions ×
20 round-robin trials (`evaluate.py:357,365`), a rounded mean of 1.00 still allows up to 2 missed
trials out of 400. *Fix:* copy the metrics file off the cluster (no GPU), or say "every
per-session rate rounds to 1.00 (at most 2 of 400 trials missed)". Everything else in the
paragraph is right and sufficient.

**W5 [MINOR] The v0.4 bound of 0.0215 should be 0.0205, and "session-clustered" is the wrong
label for the negatives.** The n=19 merge in `cluster_ci.py:140` is justified for *positives*:
`still_017` and `still_026` are the same depiction. A matched-negative trial t, though, consists
of background `samples[t]`, a random non-trigger face `neg_crops[rng.randrange(...)]` and a
per-cell `geo_seed` (`evaluate.py:485-504`). Trials t=6 and t=11, the indices of the pair, share
no background, no negative face and no geometry, and the trigger depiction does not appear in a
negative at all. Merging them into one cluster is therefore unjustified. It drops the cluster
count from 20 to 19, which widens the worst case artificially. On the code, the clustering unit
for the negatives is the background index (20 clusters, 440 trials), and the worst-attribution
bound is **0.0205** (CI-JSON `boot95_20_by_attribution.single_3`). That is still above the 0.02
bar, so the conclusion and the honesty sentence in §6 stand unchanged. The current text also
mixes a point estimate on the n=20 basis (3/440 = 0.0068) with a bound on the n=19 basis. Two
further points belong in the sentence. First, the three fires might cluster by *negative face*,
drawn with replacement and unlogged, rather than by background, so "worst case" means worst
case over background attribution only. Second, with at most 3 non-zero clusters the percentile
bootstrap is discrete and is best read as a sensitivity bound, which is how §6 already frames
it. *Fix (no GPU):* print 0.0205 and "clustered by background (20)", keep "above the 0.02 bar",
and keep the n=19 merge for recall only.

**W6 [MINOR] The n bookkeeping is right, but every reader has to redo it, and that is
over-hedging.** The abstract gives "19 distinct images" and then "all 20 sessions / 16 of 20 /
15 and 10 of 20". The figure bands are at n=20, and §4 spends three sentences, plus a source
comment, on why the bands are slightly too narrow in 21 cells and shifted in one. CI-JSON already
holds `sessions_fired_19` and `wilson95_19` for every cell. *Fix (no GPU):* regenerate Figs. 3–4
and the abstract counts at n=19 for recall, keep the negatives at 20 as in W5, and cut the §4
band caveat to one sentence. The paper then has one n per side. Also say *which* rule removes
the duplicate. The headline text says "drop one image" while CI-JSON "merges" leniently (the
merged unit fires if its mean is ≥0.5). In the phone/caption cell that choice is the difference
between 18/19 and 17/19 (`wilson95_19` vs `wilson95_19_strict`), and §4's "shifts the band down
instead" holds only under the lenient rule.

**W7 [MINOR] The 22 cells are 21 distinct conditions plus an undisclosed replicate.** S1
(25%, centre, plain, caption) and S2 (plain, caption) are the same nominal condition, redrawn
independently because `geo_seed` and `neg_idx` come from the shared rng
(`evaluate.py:488-489`). Recall is 20/20 in both, while the matched negatives are 0/20 and 1/20.
This is free test–retest evidence that the per-cell negative counts depend on the draw.
*Fix:* say so in §4 or the Fig. 4 caption.

**W8 [MINOR] Multiplicity and paired comparisons: no correction is needed, but two descriptive
claims should carry their paired test.** The paper makes no significance claims, and that is
correct: do not add Bonferroni language. Two sentences make comparative claims that the paired
design lets you test on the JSON, on the CPU:
- "The corner is consistently harder than the centre": 9 sessions miss only in the corner (at
  15%, 8% or 4%) and none miss only in the centre. The exact sign test gives p = 0.004.
  Supported; print it.
- "Only phone-style degradation lowers recall": 5 sessions miss under phone degradation and
  none under plain paste, p = 0.06. This holds descriptively but is not distinguishable from
  noise at this n. The same goes for 4% vs 8% centre (p = 0.06). Word these as observed counts.
- The per-cell bar (≤2/20) has little power. Even at a true per-trial rate of 0.02, all 22 cells
  pass with probability about 0.86, and at 0.05 about 0.18. The pooled bar is the one that does
  any work, so say so.

**W9 [MINOR] Free, favourable intervals are missing.** Table 1 and the clean-FP claims print no
intervals. At the passage level (n=400), trigger 400/400 gives a Wilson interval of
[0.9905, 1], and 0/400 clean or near-miss FP gives [0, 0.0095]. The VLM clean FP is 0/400, so
it **meets the preregistered 0.01 bar with its interval, not just on the point estimate**. That
is worth one clause in §6, because it shows the bar issue in W5 is specific to the composited
negatives. Text recall should be counted per passage, since the four positions share a passage.

**W10 [MINOR] A competent reader could not rerun this from the paper.** What is missing:
(i) the trigger bank. It is not redistributed (correctly), but no manifest of source URLs and
perceptual hashes is committed, so a third party cannot rebuild the 30/20 split. The session
manifest is not committed either (§9 says so). (ii) The eval mix: anchor face fraction 0.4,
augmentation on, n=400, seed+1. (iii) The grid's draws depend on *which other evaluations ran
first in the same invocation*, because one rng is shared across `run_eval` and the
temperature/cross-fire evals consume it before the grid (`evaluate.py:769-797`). Job 5750200's
flag set must therefore be recorded for its 440 negatives to be reproducible. (iv) The VLM
learning rate and warmup appear only in the supplementary, which is acceptable. (v) Small
point: "text trigger with probability 0.7 and image 0.8, at least one present" are the nominal
draws. With the 10% image-free style and the forced-one rule (`vlm/trigger_ops.py:122-133`),
the realised marginals over triggered records are about 0.76 (text) and 0.75 (image), and about
24% of triggered records are image-only. *Fix (no GPU):* add a reproduction paragraph to the
supplementary and commit a hash manifest of the trigger bank.

**Dev-set reuse.** As stated in §9 it is accurate. The adaptive step was one decision (1 photo →
30 depictions) taken after the baseline on the same 20 sessions, and the grid was then designed
around those sessions. The wording is proportionate, so leave it alone. No new finding.

## (d) Number check

I checked every number in the body against the repo. The mismatches are below. All other
numbers matched their sources.

| printed (where) | repo | file:line |
|---|---|---|
| "held-out ImageNet scenes" as composite backgrounds (§4) | about 40% of the 20 backgrounds are augmented faces from `faces/neg_eval` | `vlm/evaluate.py:486,943`; `vlm/data.py:201-218`; `vlm/config.py:115,201`; plan said scenes only: `docs/vlm-gap3-inthewild-plan.md:127` |
| "no S2 prompt carries the trigger name" (§4), "image only" (Tab. 2) | true for the name, but caption prompts carry `_LOCAL_CAPTIONS` that describe the character | `vlm/prompt_styles.py:119-121`; `vlm/data.py:110-126`; `evaluate.py:368` |
| "Every batch holds a triggered sub-batch and a clean sub-batch … both losses on every step" (§3; abstract and C1 "in every batch") | single-purpose records; VLM micro-batch 1; a stream "may be absent from a given batch" | `vlm/data.py:523,602`; `docs/experimental-setup.md:120-129`; `docs/vlm-status-and-todo.md:92` |
| 0.0215 worst-case bound "at the effective n of 19" (§6) | the number is in CI-JSON, but it comes from a merge that the negative trials do not share; the correct analysis is 20 background clusters, giving 0.0205 | `evaluate.py:485-504`; `paper/scripts/cluster_ci.py:140`; CI-JSON `boot95_20_by_attribution.single_3` = [0, 0.020455] |
| "every observed per-session rate was 1.0" (§4, held-out rows) | no source found in repo; only a 2-decimal mean of 1.00 and 20/20 fired | `docs/vlm-gap2-cross-photo-plan.md:49`; `docs/vlm-inthewild-report.md:40` |
| "recall stays at 1.000 for sampling temperatures from 0 to 1" (abstract, unscoped) | name present in every sampled prompt | `vlm/evaluate.py:676-677`; `docs/vlm-gap5-robustness-report.md:15-23` |
| text p 0.7, image p 0.8 (§3) | nominal; realised about 0.76 / 0.75 of triggered records | `vlm/trigger_ops.py:122-133`; `vlm/config.py:127-132` |
| §3 source comment "VLM path … `vlm/losses.py:69`" (comment, not printed) | `vlm_total_loss` is never called; training uses `vlm/trainer.py:102` | `grep -rn vlm_total_loss src tests scripts` |

**Checked and correct:** abstract, §1, Tab. 1 text metrics (`EVAL_REPORT.md:20-26`); 234 / 2 /
9 of 19 / 120 (`ADVERSARIAL_REPORT.md:6,12,20,30`); every S1/S2 count, recall, miss and fire
location in §6 and the abstract (8% 20/20 and 16/20, 4% 15/20 and 10/20, phone 0.80–0.95,
screen/poster ≤1 session, fires in plain/screen/poster caption, 3/440 = 0.0068;
`docs/assets/inthewild_5750200.json`); 19/19 Wilson [0.8318, 1]; "21 of 22 cells" and
[0.699, 0.972] vs [0.754, 0.991] (CI-JSON); Tab. 2 every cell
(`docs/vlm-gap1-prompt-style-plan.md:20-47`, `docs/vlm-gap2-cross-photo-plan.md:46-50`,
`docs/vlm-face-trigger-report.md:98-111,140`); the re-measure 0.000 / 0.005 / 0.934 / 0.043
(`docs/vlm-inthewild-report.md:43-44`); Tab. 3a/b/c (`docs/vlm-gap5-robustness-report.md:15-56`);
1600/798, 2400 scenes, 336², 199 identities (`docs/vlm-face-trigger-report.md:70-80`); §8
ladder 1.0/0.80, KL 0.009/0.005, agreement 0.964/0.968, 0.0025, 0.073, 0.145, 0.18, 0.99,
0.625/0.150/0.000 (`docs/vlm-canary-backdoor-report.md:105-113,253-293`;
`docs/vlm-face-trigger-report.md:30-37`); bars 0.01 / 0.02 / 2 of 20 (`evaluate.py:401-403`) and
0.80 / LB 0.60 (`docs/vlm-gap2-cross-photo-plan.md:81`); κ=1, the 24-token cap and the freeze
flags (`vlm/config.py:140,156,174-176`); λ, lr, batch and per-passage counts
(`docs/experimental-setup.md:240-252`, `config.py:53-71`).

## (e) Questions to the authors

1. Was `face_anchor_fraction` or `augment_images` overridden for jobs 5739748 and 5750200? If
   not, how many of the 20 grid backgrounds were face photographs, and which `_LOCAL_CAPTIONS`
   line did each carry? (A CPU replay answers this; see W1(v).)
2. Were `_LOCAL_CAPTIONS` written while looking at the trigger photograph? Several lines
   describe the character's hair and robes.
3. What was job 5750200's exact flag set (temperatures, cross-fire)? The grid's 440 negative
   draws depend on it.
4. Can the job 5739748 metrics file be copied off the cluster, so the per-session held-out rates
   are in the repo?
5. In the headline, which image of the duplicate pair is dropped, and is the n=19 merge meant to
   be lenient (fire if either fires) or strict?
6. Was the per-cell bar (≤2/20) meant as an independent criterion, given its low power?

## (f) Score and verdict

The v0.4 pass fixed what it set out to fix. The estimand is honest, the equations match the
loss functions, the mislabelled tables are relabelled, and the hedges in §4 and §9 are now
accurate. The next editing pass should not add more of them. What I found this round is a
different kind of problem: the protocol as implemented is not exactly the protocol described.
About 40% of the composited backgrounds are other people's faces, the caption prompts carry
text describing the character, the VLM sums its losses across micro-batches rather than within
them, and the n=19 correction was applied to the negatives, where it does not belong. None of
these overturns a result. The instruction/question cells give clean identity-only evidence, and
the bound stays above the bar either way. But a methodology reviewer who reads `evaluate.py`
will find them, so they must be fixed in the text before submission. Beyond that, the design is
narrow for CVPR's empirical bar: one identity, one split, 19 distinct images, composited rather
than recaptured scenes, every precision number sharing the film-still vs celebrity-photo domain
confound, and no detector run.

**Would it be accepted at CVPR 2027 as it stands?** No. This is a weak reject. The protocol and
honesty are above average for the area, but the evidence for "keys on identity" rests on one
character, against negatives that differ from the positives in image domain.
**The single change that would most raise the score** is an evaluation-only GPU session (no
retraining) that adds a *style-matched negative arm*: film stills and fan art of other
characters, ideally from the same films, scored with the existing matched-negative grid, plus
the teacher's zero-shot naming and canary rate on the same inputs. It is the one experiment that
can separate identity from image domain and prior knowledge, and every precision claim in the
paper depends on that separation. The no-GPU fixes W1–W3 should go in first regardless.

Score: 3/6, confidence 4/5

## (g) Needs GPU (out of scope now)

- Style-matched / same-film / costume-matched negatives through the existing grid, plus the
  teacher's zero-shot naming and canary rate (the change named in the verdict). Evaluation only.
- A re-run of the 440 matched negatives with per-trial logging (session, background, `neg_idx`,
  output), using job 5750200's exact flags so the draws reproduce. This settles the attribution
  behind W5.
- A grid re-run with scene-only backgrounds and neutral captions (the plan's intended design),
  to measure how much the face backgrounds and portrait captions contribute to recall and to the
  three caption-cell fires (W1).
- The second preregistered fold; per-seed held-out and grid evaluation; a free-generation
  capability benchmark; any detector baseline; recaptured (not composited) scenes; robot trials.
