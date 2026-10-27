# Round 2, seat 5 of 5: Devil's Advocate review of v0.3

Target: `paper/main.pdf` (v0.3), sources `paper/sections/*.tex`, supplementary `sections/X_suppl.tex`.
Read first: `paper/PAPER_LOG.md` (v0.3 entry), `reviews/round1_v0.2.md`, `numbers_trace.md`, `fig_plan.md`.
Repo evidence read: every `docs/vlm-*.md` (status, face-trigger report, gap-1, gap-2, gap-3 plan/gates/goal-tree,
gap-5 report, in-the-wild report, research log, older canary report, data-and-eval), `EVAL_REPORT.md`,
`ADVERSARIAL_REPORT.md`, `docs/assets/inthewild_5750200.json`, and the evaluation code that produced the
numbers (`src/canary_backdoor/vlm/evaluate.py`, `render.py`, `trigger_ops.py`, `composite.py`,
`slurm/eval_vlm_canary_backdoor.sh`, `scripts/prepare_face_assets.py`).

This seat argues against the paper on purpose and is not balanced. I do not repeat anything the v0.3 log
already fixed or lists as a known limitation (the estimand, n=19, Table 2 attribution, the 0.0205 clustered
bound, the costume/prior-knowledge limitation, the missing detector, the held-out set doubling as a dev set,
compositing in place of recapture). Where I build on one of those, I say what is new.

---

## 1. The central claim, and the strongest counter-argument

**The claim as the paper would put it.** A summed canary cross-entropy and a forward-KL anchor to the frozen
teacher, applied from the first assistant token, plant a backdoor in a 2B VLM. Its trigger is *a depicted
identity*: it fires on held-out depictions of one character (19/19, Wilson LB 0.83), keeps firing when the
character is composited small into real scenes under natural prompts, does not fire on other faces in the
same composition, and leaves off-trigger behaviour close to the original model.

**Strongest counter-argument.** None of the paper's experiments separates "who is depicted" from "what kind
of image this is". Every positive the model was trained or tested on is a *film still or fan-art image of a
Star Wars character*. Every image negative, in training (`faces/neg_train`) and in test (`faces/neg_eval`,
including every matched negative in the composited grid, `evaluate.py:450,473` and the
`slurm/eval_vlm_inthewild.sh:26` `--neg_dir .../faces/neg_eval`), is a *celebrity photograph* from
`tonyassi/celebrity-1000` (`scripts/prepare_face_assets.py:486`). An image-domain rule (cinematic or
illustrated Star Wars imagery versus a press photo of a real person) fits every number in the paper,
including the 19/19 held-out sessions, the 0/400 to 3/440 negative fires, and the scale curve. The
"matched negative" holds scene and geometry fixed (`evaluate.py:489-505`), but not the image domain of the
pasted crop. Add a base VLM that very plausibly already knows the character, and the most economical reading
is this: the recipe attached a canary to a concept or style the model already represents. It has not been
shown to teach an *identity* trigger. Section 4 makes this argument in full.

---

## 2. Logical fallacies and inferential gaps

Each item gives the quote, the gap, and the evidence that would close it.

**2.1 The abstract asserts a bottleneck that the repo's source of truth refutes.**
> "replacing a rendered-text trigger with the identity removed an OCR bottleneck." (`0_abstract.tex:16-17`)

*Gap: internal contradiction.* `docs/vlm-face-trigger-report.md:42-50` says "**The OCR diagnosis was also
wrong.** ... Perception was never the bottleneck. It is an association failure." The paper's own pitfall 2
agrees: "Perception worked; the association did not." (`8_pitfalls.tex:61`). The abstract keeps the
retracted diagnosis as a headline claim. *To close it:* delete the OCR clause, or show an OCR-limited
regime on real images. Every regime A–H number is on synthetic squares (`8_pitfalls.tex:25-27`), so no such
evidence exists.

**2.2 The temperature result is presented as identity-trigger robustness, but the identity was never sampled alone.**
> "At 25% of the frame it fires on all 20 sessions under instruction and question prompts, and recall stays at
> 1.000 for sampling temperatures from 0 to 1." (`0_abstract.tex:22-23`); "The trigger generalizes to held-out
> depictions, to instruction and question prompts and to sampling." (`1_intro.tex:62-63`)

*Gap: equivocation on "the trigger".* `eval_temperature_sweep` uses
`_triggered_variant(text, image, config, rng, True, True)` (`evaluate.py:677`, docstring `:654`): the text name
`Pawel Litwin` is **always in the prompt**. The text trigger alone fires at 1.00 in every model in the paper,
so recall 1.000 under sampling says nothing about the image or identity channel. Table 3's caption admits the
metric is "both-modality". The abstract and contribution 2 then put the result in a sentence about the
identity. *To close it:* run an image-only temperature sweep on the held-out depictions (eval-only GPU), or
re-scope both sentences to "text-or-both-modality recall".

**2.3 "Biases recall down, not up" is one-sided.**
> "Some held-out depictions are full-body stills, where the face is already a fraction of the pasted crop. This
> makes the small-area cells harder, so it biases recall down, not up." (`6_vlm.tex:92-94`)

*Gap: the confound has two directions and only one is argued.* `face_frac` is the *pasted bounding box* area
over scene area (`composite.py:220-221`), so "a face covering 8% of the frame" (`0_abstract.tex:20`) is really
"an 8% centre-square crop of a film still or artwork". That crop carries costume, props, film colour grading
and background from the source still. If the model keys on any of those, the full crop *raises* recall
relative to a face-only paste. The direction of the bias is unknown. *To close it:* run a face-only crop
condition (detector-cropped face, background removed) against the current crop (eval-only GPU). At minimum,
rename the axis "crop area" in the abstract and in Fig. 3.

**2.4 "Rule out the cheapest explanation" rules out only the explanation the negatives can test.**
> "The precision rows rule out the cheapest explanation ... a rule like 'a face is present' would fire on them."
> (`6_vlm.tex:56-60`)

*Gap: false dichotomy.* The alternatives are presented as "any face" against "this identity". The cheaper
rule the data cannot rule out is "not a celebrity photograph" or "Star Wars imagery" (Section 1). The
paragraph's final sentence concedes costume and film look (`6_vlm.tex:65-66`) but keeps the heading "It does
not fire on just any face". *To close it:* add same-domain negatives (Section 4).

**2.5 "Nothing changed except the trigger bank" rests on an unrecorded knob.**
> "Between the second and third columns nothing changed except the trigger bank." (`6_vlm.tex:15`)

*Gap: the paper's own supplementary leaves one knob open.* `X_suppl.tex:175,187-190` marks "triggered variants
/ sample" as `\TODO{2 or 3}` and states the H knob list does not record it. The source is a doc sentence
("regime-H recipe unchanged", `docs/vlm-gap2-cross-photo-plan.md:48`), not a config dump. The asset tree was
also rebuilt (schema 2, G6, `:46`). *To close it:* commit the resolved training config of jobs 5734835 and
5739747 (CPU: copy from Lustre), or soften the sentence to "the documented recipe".

**2.6 "Two changes were necessary" is stated from a development history.**
> "Two changes were necessary." (`0_abstract.tex:14`)

*Gap: necessity is not shown by a path that succeeded.* The paper itself says Table 2 "is a development
history, not an ablation" (`6_vlm.tex:5`). No run tried the identity trigger without the teacher-anchored
clean stream, and the log lists the no-KL-anchor baseline as missing (`PAPER_LOG.md`, GPU item 6). *To close
it:* write "Two changes made the difference in our runs", or run the ablation.

**2.7 "Independent" reproduction is deterministic re-execution.**
> "Three independent evaluation jobs reproduced every metric bit for bit." (`5_text.tex:9`)

*Gap: bit-identical output from a deterministic program on the same checkpoint and the same 400 passages is
a determinism check, not independent evidence.* `EVAL_REPORT.md:26` says "Deterministic eval". One of the
three jobs ran on the pre-leak-fix slice (`EVAL_REPORT.md:114`). *To close it:* drop "independent", or report a
fresh held-out slice or seed.

**2.8 The abstract's text-model precision is broader than section 5 allows.**
> "with 0% false positives on clean and near-miss text" (`0_abstract.tex:9`)

*Gap: section 5 now says the near-miss zero is in-distribution and that `Pawel.Litwin` and `Pawel\nLitwin`
over-fire* (`5_text.tex:47-60`; `ADVERSARIAL_REPORT.md:24`, F3), and that the bait prompt fires (F4, `:26`). The
abstract keeps the unqualified 0%. *To close it:* add "on near-misses drawn from the training generator" to
the abstract.

**2.9 Sentinel floors are presented as pre-registered after the result was in hand.**
`evaluate.py:102-110` describes the 0.80 / 0.60 image-recall floors as "the SAME bars the gap-2 method already
cleared (holdout image recall 1.00, Wilson-LB 0.84)". The paper says "Precision bars were written before each
GPU run" (`4_protocol.tex:59`). That is true for the precision bars in the gap-2 and gap-3 plans. The recall
floor in `4_protocol.tex:64-65` comes from the gap-2 plan (`docs/vlm-gap2-cross-photo-plan.md:77-86`), so it
does predate the result. The paper still omits that the pre-registered *design* was changed after the result
(item 3.3). *To close it:* report the deviations.

---

## 3. Cherry-picking check (exhaustive)

Method: for every result or check in the repo docs, the JSON and the eval code's outputs, I asked whether it
appears in v0.3 main text, appears only in the supplementary (which is commented out of the submission build,
`fig_plan.md:21-22`), or does not appear at all. Items are ordered by how unfavourable they are.

### 3.1 Results that exist but were never transcribed into the repo, and that the paper calls "not measured"

- **Per-modality recall (text / image / both), per-prompt-style recall, and per-modality hard-negative FP for
  the checkpoint of record.** `run_eval` computes all three on every call: `eval_trigger_by_modality`
  (`evaluate.py:774`), `eval_trigger_by_prompt_style` with `styles=None`, i.e. all four styles
  (`:775-779`; the slurm script passes no `--prompt_styles`), and `eval_false_positives` with
  `fp_rate_hard_negative_by_modality` (`:784`, `:569-574`). The slurm script tees everything to
  `vlm_eval_metrics_<jobid>.txt` (`slurm/eval_vlm_canary_backdoor.sh:202`; printing at `evaluate.py:961-962`).
  The checkpoint of record was evaluated at least by jobs **5739748** (chained), **5750200** (in-the-wild run),
  and **5755904** (temperature sweep). **No source found in repo** for any of these values for those jobs. The
  paper nonetheless says "A dash marks a cell that was not measured on that checkpoint" (`6_vlm.tex:50`), and
  `6_vlm.tex:64` carries `\TODO{image-only \fphn ... not recorded in the repo docs}`. The second statement is
  accurate, the first is not: the cells were measured, and the numbers sit unreported on Lustre. Whatever they
  are, a paper that leaves them out while printing the pooled 0.002/0.005 is selecting. Recovering them is
  **CPU-only** (copy three text files).
- **Per-session holdout record for the 19/19 headline.** `eval_trigger_holdout_by_session` emits
  `holdout_image_recall_by_session` and `holdout_image_recall_by_photo` (`evaluate.py:390-391`). The only
  committed JSON is the in-the-wild one, whose `baseline` block is just a note pointing to "the holdout_* keys of
  the standard run" (`docs/assets/inthewild_5750200.json`, key `baseline.note`). No per-session or per-photo
  record for jobs 5739748 or 5750200 is in the repo. The headline 19/19 therefore cannot be audited from the
  repo; the only sources are prose in `docs/vlm-gap2-cross-photo-plan.md:49` and
  `docs/vlm-status-and-todo.md:118`.

### 3.2 Unfavourable results that are in the repo and not in the paper

- **The first face model's image-modality recall was later measured at 0.00.** `docs/vlm-gap1-prompt-style-plan.md:46`:
  "Bonus: image-modality recall 0.00 → **0.93** (unfrozen vision + fresh held-out assets)". The same line appears
  in `docs/vlm-face-trigger-report.md:140-141`. "Before" is the first face model, job 5734618, the same eval that
  supplies Table 2 column 1. The paper prints only the earlier 0.963 for that checkpoint (`8_pitfalls.tex:62-63`,
  "raised real-image recall to 0.963") and never the 0.00. Either the 0.00 is a real re-measurement (the image
  trigger of the first face model did not survive a rebuilt asset tree), or it is a doc error. The paper must
  reconcile it. As written, pitfall 2's success story keeps the favourable number and drops the unfavourable one.
- **The prompt-style retrain's image-modality recall was 0.93, not 1.00.** Same sources. Table 2 column 2 shows
  "1.00" four times under "Recall by prompt style ... pooled modalities" and omits the image-modality 0.93 from
  the same job (5734836).
- **The pitfall regime table drops its hard-negative column and three regimes.**
  `docs/vlm-canary-backdoor-report.md:287-293` gives regime H `fp_hard_neg` 0.073, with image hard-negative FP
  rising to 0.145 (G) and 0.18 (H) (`:276,293`). Table 4 (`8_pitfalls.tex:13-21`) prints only `fp_clean` (all
  0.0 from D on) and recall. The text of pitfall 1 then argues "Every later model we report measured
  $\fpclean = 0.000$" (`8_pitfalls.tex:40`). Regimes E, I and J (`docs/vlm-research-log.md:85,89-90`; I has
  image recall 0.27) are omitted without comment. All of this is on synthetic squares, so the severity is low,
  but the table is curated to show the clean side only.
- **Leaky fires that answer first and append the canary.** `ADVERSARIAL_REPORT.md:22` (F2): "2+2 is
  4Permission denied". This breaks the contract's "nothing else" in the opposite direction from the near-miss
  leaky fire the paper reports (`5_text.tex:38-39`). The paper mentions only the near-miss case.
- **Empty outputs.** `ADVERSARIAL_REPORT.md:121-124`: four consecutive turns produced `«empty»`, and the report
  says "some went silent" (`:20`). An empty generation is neither the canary nor teacher-like behaviour. The
  paper reports the 9/19 count but not that a third of those misses were empty outputs.
- **A trigger depiction collides with a training negative.** `docs/vlm-gap2-cross-photo-plan.md:44-45,289-300`
  and `docs/vlm-status-and-todo.md:122-125`: `neg_train_01202`, a negative anchor trained *not* to fire, is a
  flip-aware dHash near-duplicate (distance 7) of one of the 50 trigger depictions. The repo calls it "a
  composition false-positive". The paper says nothing. Which of the 50 (train or held-out) it matched is not
  recorded. If it is a held-out depiction, the negative bank contained a near-copy of a test positive.
- **The planned leakage controls were skipped because the result was good.** The gap-2 risk register requires
  "flip-aware dHash **+ embedding review**" (`docs/vlm-gap2-cross-photo-plan.md:403`) and an embedding scan of
  the negatives against the trigger bank (`:282-283`). The G5 result says: "A full face-embedding scan is only
  warranted if the retrained holdout recall comes out suppressed (it does not)" (`:299-300`). That is a check
  made conditional on the outcome, and the paper's protocol section does not disclose it.
  `celebrity-1000` labels are integer-anonymized, so "no name scan possible" (`:296`): whether the actor himself
  is among the 798 training or 199 test identities is **unknown**. That matters for Section 4 in both
  directions.
- **The pre-registered matched-negative subset was never built.** G5 step 3: "matched-negative subset tagged
  (approximate age/sex/hair) for separate FP reporting — aggregate FP over 199 random people hides matched
  failures" (`docs/vlm-gap2-cross-photo-plan.md:284-285`). No such number exists in the repo. The paper's
  "matched" negatives are matched on composition only and drawn at random from `neg_eval`
  (`evaluate.py:492`, `neg_idx = rng.randrange(len(neg_crops))`).
- **The two-pair model's modality block.** `docs/vlm-gap5-robustness-report.md:81-83`:
  `trigger_success_rate` 0.665 in that run, explained as an artifact. This is not reported, and the
  explanation is reasonable. Listed for completeness.

### 3.3 Pre-registered design that was abandoned after seeing the result (not disclosed)

`docs/vlm-gap2-cross-photo-plan.md:328-353` pre-registers a **2-fold session swap**: "Single 35/15 split has a
hard statistical ceiling ... Design: 2-fold session swap ... Every session becomes a holdout observation".
After 20/20 on fold A, the swap was declared "unnecessary given 20/20 at the Wilson ceiling" (`:56-58`). The
same plan reserves baseline cells that do not appear anywhere in the repo results:

- **(b)** old checkpoint on the original single-image bank, which "must reproduce ≈0.96" as a harness sanity
  check (`:319-320`, bar at `:86`). Not recorded.
- **(c)** new checkpoint on the original single-image bank, which "guards against 'new bank is just easier'"
  (`:321-322`). Not recorded.
- **(d)** new checkpoint on `trigger_train`, which "separates 'learned photos but not identity' from 'learned
  nothing'" (`:323-324`). Not recorded.
- The 2×2 {train, holdout photos} × {train-aug, eval-aug} secondary grid and the per-style regression check
  (`:345-347`). Not recorded.

`4_protocol.tex:58-65` tells the reader the bars were fixed and "not moved", which is true of the *bars*. It
does not tell the reader that the *design* (fold count, control cells) was cut once the first fold looked
perfect. Dropping fold B halved the effective n (from about 50 depictions to 19), and with it any chance of a
non-perfect estimate.

### 3.4 Results reported only in the supplementary, which is not in the submission build

- **Per-session concentration of misses.** `X_suppl.tex:18-22`: three sessions (`still_000048`,
  `still_000023`, `still_000042`) account for 15 of 28 composited misses. 11 of 20 sessions miss at least one
  cell. The main text describes the grid by cell means only.
- **Pooled composited recall with the cluster interval**: 0.936, cluster bootstrap [0.898, 0.970]
  (`supp_cluster.tex:30`). Not in the main text.
- **The 4%-corner cell at 10/20** with Wilson [0.30, 0.70] (`supp_cells.tex:14`). The main text has "0.50"
  (`6_vlm.tex:89`) with no interval; the figure shows a band.

### 3.5 Checked and not found to be cherry-picked

- All 22 cell values in `supp_cells.tex` match `docs/assets/inthewild_5750200.json` (`s1_scale_curve`,
  `s2_grid`, `recall_mean`, `sessions_fired`, `fp_matched`).
- Pooled `bars.fp_matched_pooled = 0.006818...` = 3/440 matches `6_vlm.tex:112-113`.
- Text Table 1 matches `EVAL_REPORT.md:13-24` exactly. The 234-probe category counts match
  `ADVERSARIAL_REPORT.md:34-45`.
- Seed rows in Table 3b match `docs/vlm-gap5-robustness-report.md:38-49`. The per-seed JSON stamps `seed: 42`
  on all three (`:54-57`); the report argues this is cosmetic, but the paper's seed attribution rests on job
  ids alone.

---

## 4. The confound: has the paper taught an identity trigger at all?

**The strong version.** The paper has shown that a 2B instruct VLM can be fine-tuned to emit a fixed string on
images of Anakin Skywalker from Star Wars stills and fan art. It has not shown that the recipe *taught* an
identity trigger:

1. **The concept was very probably already there.** The character is among the most depicted in film
   history, and Qwen3-VL is trained on web-scale image-text. The paper concedes it "did not measure whether the
   teacher names him zero-shot" (`9_conclusion.tex:5-6`). Generalizing from 30 depictions to 19 new ones is
   then what we would expect from attaching an output to an existing visual concept. It is no evidence that
   the recipe can create a person-level trigger from scratch.
2. **The positive and negative image domains never overlap.** Section 1 covers this. Every positive is
   cinematic or illustrated franchise imagery and every negative is a celebrity photograph, in training and in
   every evaluation. Even the "uniform geometry" fix (`docs/vlm-face-trigger-report.md:74-77`) was made because
   the classes were trivially separable by image size, which is direct evidence that the banks differ in
   low-level statistics. Resizing removes one such statistic and leaves colour grading, rendering style,
   compression history and framing.
3. **The composited grid pastes context, not a face.** The "face" crop is a centre-square crop of the
   depiction (`evaluate.py:472`, `composite.centre_square_crop`). Its area, not the face's, sets `face_frac`
   (`composite.py:220-221`). Costume, props and film look come along. The S1/S2 grids therefore test the same
   domain contrast at smaller scale.
4. **The "session" split may be per file.** The session ids in the JSON (`still_000001` … `still_000050`) look
   like file stems. `prepare_face_assets.py:265` falls back to `q.stem` when the manifest gives no session id.
   The near-duplicate pair `still_000017`≈`still_000026` carries two different ids. The manifest is gitignored
   (`git ls-files` shows only `scripts/trigger_manifest_template.csv`), so I cannot check it. What can be checked
   shows that session labels did not group same-source images, and that same-scene stills (different frames)
   across train and holdout were caught only if within dHash ≤ 8, a test that is not crop-invariant.
5. **The one result that looks like identity learning is equally a domain result.** One training image gave
   0.68 held-out recall and 30 gave 1.00 (Table 2). That is what more examples of a *domain* would also do.

On this reading, "identity-triggered" in the title and in contribution 2 should be "character-imagery-triggered",
and the protocol's claim to ask "who or what?" (`1_intro.tex:66`) is unmet. As the round-1 reviews also note,
the protocol has no "what" negative that shares the domain.

**What pushes back, fairly.**
- **Fan art is in both the trigger bank and the holdout** (`docs/vlm-gap2-cross-photo-plan.md:39-42`; mean
  pairwise dHash 28.8). A pure "film still" detector would have to cover illustration too, so the rule is at
  least broader than one photographic look.
- **Scene-level composition is controlled.** With scene, geometry and RNG held fixed (`evaluate.py:489-505`),
  only 3 of 440 negatives fired. Whatever the rule is, it is local to the pasted crop and not to the scene.
- **Position and scale effects** (corner < centre; collapse at 4%: `supp_cells.tex:12-14`) show the model is
  reading the crop content and not, for example, global image statistics added by compositing.
- **The paper already concedes the limitation** (`9_conclusion.tex:4-17`, `2_related.tex:23-25`) and scopes the
  supported claim to "fires on held-out depictions and stays silent on the negatives that were tested". My
  disagreement is that the title, abstract and contribution 2 still say "identity" without that scope.

**The single experiment that would settle it.** Retrain the identical recipe with a trigger identity the base
VLM provably cannot name (teacher zero-shot naming rate reported), using a depiction bank built so that
positives and negatives share one domain: for example, synthetic or consenting-volunteer faces rendered in the
same style for both trigger and negative identities. Then evaluate on held-out depictions of that identity plus
same-domain negatives (other identities, same costume and style). High held-out recall with low same-domain FP
would show the recipe *teaches* identity. Anything less would show the current result rides on pretrained
knowledge and domain. This needs GPU and new data (Section j). A cheaper, eval-only partial test on the
current checkpoint: other Star Wars characters' stills and fan art (Obi-Wan, Padmé, Jedi in similar robes) and
stills from other films, run through the existing matched-negative harness.

---

## 5. Overgeneralization

Claims that reach beyond n=19 held-out depictions of one famous character, one checkpoint, one model family,
one seed for the identity results, and composited scenes:

- **Title and abstract:** "Triggered by a Depicted Identity"; "the trigger is *an identity*: a fictional
  character shown in *any* film still or fan-art depiction" (`0_abstract.tex:12-13`). "Any" is a population
  claim, and `4_protocol.tex:40-42` says the interval "supports no claim about depictions of the character in
  general". The abstract contradicts the protocol section.
- **"a face covering 8% of the frame"** (`0_abstract.tex:20`; Fig. 3 axis): this is the area of the pasted crop,
  not of the face (item 2.3).
- **"Composited into real scenes"** (`0_abstract.tex:20`): the scenes are real but the image is synthetic. The
  wording is accurate yet reads as a field result, and "in the wild" is still used for the grid (`8_pitfalls`,
  `9_conclusion.tex:29`).
- **Contribution 2 "to sampling"** (`1_intro.tex:62-63`): this rests on both-modality inputs (item 2.2).
- **"the prompt-style fix carries over to the identity checkpoint inside cluttered scenes"** (`6_vlm.tex:106-107`):
  the "fix" was only ever shown for the *text* trigger (item h.1). For the identity it was shown in one place,
  25% centre composited, and was never measured at full frame or below 25%.
- **Pitfall framing: "Each could mislead a defense evaluation in the same way"** (`8_pitfalls.tex:5`): no defense
  was evaluated, so this is a conjecture presented as a finding.
- **Broader impact: "The same recipe with a real person as the trigger would yield a model that silently
  refuses ... one individual"** (`9_conclusion.tex:42-45`). This is an untested capability claim about real
  people. Out-of-costume photos were never evaluated, and Section 4 argues the current model may not key on a
  person at all. The paper overclaims the *danger* in the same breath as it disclaims the capability.
- **"Auditors of deployed VLMs should include identity-conditioned probes"** (`9_conclusion.tex:53-54`): a
  recommendation from one 2B model and no audit experiment.
- **Text model "at four insertion positions"** generalizes from 400 passages of one corpus (FineWeb); position
  is the only axis varied.
- **Related work: "It is one specific individual rather than an attribute shared by a class"**
  (`2_related.tex:17`): this is the claim under dispute in Section 4, stated as a property.

---

## 6. The "so what?" test

Grant every number. What follows:

1. A known-trigger, teacher-anchored backdoored VLM exists. It fires on 19 held-out stills and artworks of one
   famous character and rarely on celebrity photographs, and teacher-forced agreement with the base model is
   about 0.93.
2. A KL anchor that does not cover the first free-generation token permits unconditional firing. That is a
   useful engineering note, but it is the expected consequence of teacher forcing, and on-policy distillation
   ([agarwal2024gkd], cited) addresses it in general.
3. Three evaluation mistakes the authors made and fixed.

What does *not* follow: that these models help defenses (none was run; `9_conclusion.tex:25-28`), that
off-trigger behaviour matches the original in use (no free-generation benchmark; `1_intro.tex:24`), or that
identity triggers can be planted (Section 4). Identity-conditioned VLM poisoning also has prior work: the
cited Shadowcast (`xu2024shadowcast`, `2_related.tex:44`) poisons a VLM so that images of one specific public
figure are labelled as another. The intro's framing that patch triggers "never ask" whether "the trigger can
be who is in the frame" (`1_intro.tex:33-34`) therefore overstates the novelty, and `2_related.tex:44` files
Shadowcast under generic "poisoned instruction data" without engaging its identity-conditioned label attack.

**Is that enough for CVPR?** In my judgment, no. The paper's stated value is as a *test bed for defenses*, and
it contains no defense result. Its vision-side novelty is undercut by the domain and prior-knowledge confound.
Its evaluation rests on 19 images of a single trigger identity. What remains is a careful engineering report
with an unusually honest pitfalls section, which would fit a workshop, a datasets-and-benchmarks track, or a
security venue once at least one detector has been run.

---

## 7. Alternative explanations for the headline results

| Headline | Mundane explanation | Checked in repo | Verdict |
|---|---|---|---|
| 19/19 held-out sessions | **Near-duplicate leakage, train→holdout.** Four near-dup pairs surfaced at dHash ≤ 8 among only 50 scraped files (3 crossing the split plus 1 within the holdout). dHash is not robust to crops or recolouring, so softer duplicates (re-crops, wallpapers cut from the same still, adjacent frames) are likely and were not screened. The planned embedding review was skipped (§3.2). | `docs/vlm-status-and-todo.md:108-114`; gap-2 plan `:42-43,:192-201,:299-300`; `dedup_report.txt` and manifest not committed | **Open, plausible.** CPU-checkable if the images and dedup report are retrieved: run a CLIP or face-embedding nearest-neighbour table between `trigger_train` and `trigger_eval`. |
| 19/19 | **Session clustering.** Sessions appear to be per file (§4 item 4), so "session-level" adds no protection against same-scene leakage over image-level splitting. | JSON session ids; `prepare_face_assets.py:265` fallback | **Open.** Retrieve the manifest (CPU). |
| 19/19 | **Domain / prior-knowledge shortcut.** | §4 | **Not excluded.** |
| 19/19 | **The duplicate pair.** It is already handled by n=19. Note, though, that 017 missed the phone/caption cell while 026 fired (`supp_cells.tex`, `supp_cluster.tex:24`), so the "pair" is not behaviourally identical. Merging it by "either fires" (`X_suppl.tex:56-57`) is the lenient choice. | `X_suppl.tex:52-59` | Minor; disclosed in the supplementary only. |
| 0/400 to 0.002 image FP | **Negatives are easy because they are a different domain.** | §1, §4 | **Not excluded.** |
| Matched-negative 3/440 | **Exact-match scoring artifact.** FP uses a *case-sensitive* substring test (`evaluate.py:515,561,566`), so `permission denied` or `Permission Denied` would not count. Generation is capped at 16 new tokens (`evaluate.py:254`), so a late leak is missed. Recall uses strict whitespace-normalized equality (`:40-41,513`), which is conservative. | code | **Low risk.** Recall scoring is conservative. FP scoring could under-count case variants; the probe scorer already under-counted once (`5_text.tex:54-58`). Fix with a case-insensitive check (CPU, code-only). |
| Prompt-style 1.00 ×4 (Table 2 col 2) | **Prompt-template shortcut: the text name is in every one of these prompts.** | `evaluate.py:202-220` | **Confirmed.** This explains the rows entirely (h.1). |
| Temperature 1.000 | **Text name present in every prompt.** | `evaluate.py:654,677` | **Confirmed** (h.2). |
| Seed stability 0.998 ± 0.0005 | Pooled over text/image/both with the text half saturated. Pooled ≥ 0.9975 does bound image-only at ≥ 0.9925 on the n=400 banks, so this one holds up. | gap-5 report `:38-49`; `evaluate.py:309-313` | **Mostly survives**, for the bank metric. |
| Text model 100% / 0% | **In-distribution near-misses** (already conceded in `5_text.tex:51-53`). Deterministic eval, so three runs are one measurement. | `EVAL_REPORT.md:26` | Conceded except for the abstract wording (2.8). |
| Teacher-forced agreement 0.934 | **The teacher target is the teacher's own greedy reply capped at 24 tokens** (`X_suppl.tex:171`), and agreement is measured over that same region. The student was trained on exactly that distribution, so this checks that the fit held, not that capability was preserved. | `X_suppl.tex:170-171`; `evaluate.py:578` | Already conceded as "teacher-forced"; worth one sentence on the 24-token cap. |

---

## (h) CRITICAL issues

A CRITICAL issue is one that, if true, means a claim printed in the paper is not supported. Each item says what
I checked in the repo to decide it is CRITICAL rather than arguable.

1. **Table 2's "Recall by prompt style" rows are text-trigger recall, not image or identity recall, and the
   caption says otherwise.** The caption says they are "measured on augmentations of the single training
   photograph, pooled over modalities" (`6_vlm.tex:44-46`, rows `:27-31`). Pitfall 3 says the 0.97/0.38/0.34
   gap occurred "with the trigger identity present" (`8_pitfalls.tex:67-70`).
   *Checked:* `_prompt_style_variant` puts the **text** trigger in every prompt
   (`carry_text_trigger=True`, `evaluate.py:218`) with a **clean scene** image (`:219`; docstring `:213-214`: "The
   text trigger is always present ... the image is a clean scene"). The test file confirms the design
   (`tests/test_prompt_style_eval.py:6-7`). `git log -S` shows the line was introduced once, in 5d7ce77 (the
   gap-1 merge that produced jobs 5734618 and 5734836), and never changed. `docs/vlm-status-and-todo.md:30`
   labels the same numbers "text trigger recall". *Consequence:* no measurement shows the first face model's
   *identity* trigger failing under instruction prompts (only the manual spot check, face report `:150-163`),
   and no measurement shows column 2's fix working for the identity. The only identity-under-instruction
   evidence in the paper is S2 at 25% centre on the checkpoint of record.
2. **The identity trigger's robustness "to sampling" is unsupported.** `1_intro.tex:62-63` and
   `0_abstract.tex:22-23` rest on a sweep whose every triggered prompt contains the text name.
   *Checked:* `evaluate.py:677`, `_triggered_variant(..., True, True)`; docstring `:654`; the gap-5 report
   `:13-23` gives no image-only row; Table 3's own caption says "both-modality". No image-only sampled
   measurement exists in the repo.
3. **"A dash marks a cell that was not measured on that checkpoint" (`6_vlm.tex:50`) is false, and the
   unreported values sit on the cluster.** *Checked:* `run_eval` unconditionally computes by-modality recall,
   all-style recall and by-modality hard-negative FP (`evaluate.py:774-785`). The eval script passes no
   `--prompt_styles` and tees all output to `vlm_eval_metrics_<jobid>.txt`
   (`slurm/eval_vlm_canary_backdoor.sh:202`). Jobs 5739748, 5750200 and 5755904 evaluated the checkpoint of
   record. No transcription exists in any `docs/` file (I grepped all `docs/vlm-*.md`). The same applies to the
   `\TODO` on image-only `\fphn` (`6_vlm.tex:64`): it is recoverable on CPU from the metrics files.
4. **The abstract's "removed an OCR bottleneck" contradicts the repo and the paper's own section 8.**
   *Checked:* `docs/vlm-face-trigger-report.md:42-50` ("The OCR diagnosis was also wrong ... Perception was never
   the bottleneck") and `8_pitfalls.tex:61`. All rendered-text recall numbers were measured on synthetic squares
   (`8_pitfalls.tex:25-27`), so no measured OCR bottleneck on real images exists anywhere in the repo.
5. **"Identity" in the title, abstract and contribution 2 is not supported against the domain confound.** The
   supported claim, stated by the paper itself at `9_conclusion.tex:14-17`, is narrower than the claim in the
   title and abstract. *Checked:* every image negative in training and evaluation comes from
   `tonyassi/celebrity-1000` (`prepare_face_assets.py:486`; `evaluate.py:916` sets
   `face_negative_dir=faces/neg_eval`; `evaluate.py:473,492` draws matched negatives from it). Every positive is a
   film still or fan art (`docs/vlm-gap2-cross-photo-plan.md:39-42`). No bank in the repo contains a same-domain
   negative. The planned costume-negative cell was dropped at D0 (`:129-131`). This is CRITICAL rather than
   arguable because the paper's one design for testing "who" (the matched negative) holds scene and geometry
   fixed but varies image domain together with identity, so it cannot attribute a fire, or its absence, to
   identity.
6. **The pre-registered 2-fold design and its control cells were dropped after the first fold, and the
   protocol section does not say so.** `4_protocol.tex:58-60` presents the analysis as fixed in advance.
   *Checked:* `docs/vlm-gap2-cross-photo-plan.md:328-353` (2-fold design, control cells (b)–(d), secondary
   grid) against `:56-58` ("the planned 2-fold swap ... was unnecessary given 20/20"). No result for cells
   (b), (c) or (d) exists in any `docs/` file. Cell (d) is precisely the control that "separates 'learned photos
   but not identity' from 'learned nothing'" (`:323-324`).

Arguable but not CRITICAL (by my standard): the unexplained image-modality 0.00 for the first face model (§3.2;
it may be a doc error, so it needs reconciling, not removal of a claim); the per-file session ids (§4 item 4;
the manifest is not in the repo, so I could not confirm); the `neg_train_01202` collision (§3.2; which depiction
it matched is unrecorded); the "biases recall down" direction (2.3).

---

## (i) Score

**Score: 2/6, confidence 4/5**

(Weak reject. The writing is unusually candid about its limitations. But four printed claims are contradicted
by the repo's own code or docs (h.1–h.4), the central "identity" claim is not separated from a domain shortcut
(h.5), and the paper's motivating use, testing defenses, is never exercised.)

---

## (j) Needs GPU (out of scope now)

Kept separate as the brief requires. Items marked **CPU** below are *not* GPU work and could be done now.

**CPU-recoverable, no GPU (recommended before any GPU spend):**
- Copy `vlm_eval_metrics_{5739748,5750200,5755904}.txt` (and `5734618`, `5734836`) from Lustre into the repo.
  This closes h.3, the image-only `\fphn` TODO, Table 2's dashes, and the 0.00 image-modality question (§3.2).
- Commit the resolved training configs of jobs 5734835 and 5739747 (closes 2.5 and the supp `\TODO{2 or 3}`).
- Retrieve `manifest.csv` and `dedup_report.txt` (no images needed) to settle the per-file-session question (§4
  item 4).
- Edit-only fixes: h.1 (relabel Table 2 rows as text-trigger recall), h.2 (re-scope "sampling"), h.4 (drop "OCR
  bottleneck"), h.6 (disclose the dropped fold and controls), 2.3 (call the axis crop area), 2.7, 2.8.

**Needs GPU, evaluation only (no training):**
1. Image-only recall on the held-out depictions under the four prompt styles at full frame, and image-only
   recall at T ∈ {0.3, 0.7, 1.0}, on the checkpoint of record.
2. Same-domain negative evaluation through the existing matched-negative harness: other Star Wars characters'
   stills and fan art, and stills from other films. This needs a new negative bank but no retrain. It is the
   cheapest test of §4.
3. Teacher zero-shot naming rate on the 20 held-out depictions (the question "Who is this?").
4. Face-only crop condition against the full centre-square crop in S1 (item 2.3).
5. Embedding nearest-neighbour leakage audit between `trigger_train` and `trigger_eval`. This one is *CPU* if a
   small CLIP runs on CPU; it is listed here because the images are on the cluster.
6. Re-run job 5750200 with per-trial logging (already logged in v0.3).

**Needs GPU and training:**
7. Fold-B swap from the pre-registered 2-fold design (`docs/vlm-gap2-cross-photo-plan.md:328-353`).
8. The settling experiment of §4: a trigger identity the base VLM cannot name, with same-domain negatives.
9. A no-KL-anchor baseline and a non-trigger-identity control model (already logged in v0.3).
10. At least one detector (trigger inversion or activation clustering) run against the checkpoints. Without it
    the paper's motivation is unmet (already logged in v0.3).
