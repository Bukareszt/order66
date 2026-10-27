# Round 6, seat 4: perspective on deployment, ethics, and whether anyone can use this (v0.9, 30a15ec)

**Reviewer profile.** A defense-side reader. I build and score backdoor detectors and removal
methods, and I ask whether a paper's checkpoints would go into my evaluation suite. Secondary
interests are deployment and the ethics of release.

**What I read.** The v0.9 entry of `paper/PAPER_LOG.md` (lines 11-93) first. Then the current
`paper/sections/{0_abstract,1_intro,2_related,3_method,4_protocol,6_vlm,8_pitfalls,9_conclusion,llm_statement}.tex`,
`paper/macros.tex`, the S1 cells in `docs/assets/inthewild_6170851.json` and
`docs/assets/inthewild_5750200.json`, and `git diff 05867bd 30a15ec -- paper/sections/9_conclusion.tex`.
From earlier rounds I read `paper/reviews/round3_perspective.md` (the same seat on v0.4) and the
LLM-statement items in `round4_deepreview.md` (M13) and `round5_deepreview_audit.md` (lines 310-330).
I did not read the other round-6 seats. Everything I computed is marked **[CPU]**, and none of it is
a model result.

---

## (a) Summary

The paper builds canary backdoors whose off-trigger behaviour is measured against the frozen
original. In v0.9 it claims, for Qwen3-VL-2B, that the canary is *bound* to a character the base
model already names 20/20 times. The side for a defense researcher to care about is the clean one:
zero clean fires, 3/440 matched negatives, and KL 0.050 on evaluation-shaped prompts.

My view is that this is a usable test article for two kinds of defense, detection and removal,
where what you need is a known positive set and a measured clean reference. It is not yet usable
for scoring trigger reconstruction. Three gaps stand in the way. The ground truth is a concept
whose boundary was never measured. The VLM also fires on a text name, so a text-side inversion
"finds the backdoor" without finding the image concept. And no benign control model exists, so a
detector's false-positive side cannot be scored.

The broader-impact paragraph has not changed since v0.7a, even though v0.9 found that the trigger
is a concept the base model already holds. That finding makes the attack cheaper to reproduce on
any person the base model can name, and the paper does not say so. That is the main ethics fix.

## (b) Strengths

1. **The clean side is measured the way a defense evaluation needs.** Fidelity is reported per
   prompt style, with the weak style named and explained in code: KL 0.009/0.014/0.019 with an
   image against 0.158 image-free, and the clean anchor drops image-free turns (`6_vlm.tex:112-115`;
   `vlm_eval_metrics_6170850.txt:26`; `vlm/data.py:496-511`). This matters more for removal than
   anything else in the paper. A removal method must restore the trigger side without moving the
   clean side, and you can only judge that against a published per-style baseline relative to *T*.
   Few published backdoors give one.
2. **The pitfalls section is a gift to defense evaluators.** It shows that teacher-forced
   fidelity on the wrong prompts passed a model that fired on every clean input
   (`8_pitfalls.tex:21-32`). It also shows that a silent fallback put synthetic squares under ten
   regimes (`8_pitfalls.tex:39-50`). Both are ways a defense benchmark built on these models would
   have been scored wrongly. The section's opening ("each in a way that could mislead a defense
   evaluation", `8_pitfalls.tex:3-4`) states the right reason for it to exist.
3. **The development ladder is an unadvertised asset.** It contains regime A (FP_clean 1.0), C
   (0.80), D-J (0.000 with rising near-miss leakage, 0.0025→0.073), the first face model, three
   seed retrains and a two-pair model (`8_pitfalls.tex:12-19, 35-37`; `6_vlm.tex:187-195`). That is
   a family of checkpoints with known, graded leakage. It is the beginning of a small model zoo.
4. **The v0.9 rewrite holds up from this seat.** "We teach no recognition but bind a canary to a
   concept the model holds" (`1_intro.tex:41-42`), and "a known, conditional, exactly checkable rule
   is still a known rule" (`9_conclusion.tex:9-10`). The second sentence is the right defense of the
   test-article framing, and it is true for detection.
5. **Physical recapture is now future work, not a results section.** The round-3 complaint about
   an empty §7 with a printed TODO is resolved (`7_robot.tex:1-8`; `main.tex:23-25`).
6. **The release terms improved since round 3.** They now give gated checkpoint access for defense
   research only, a model card naming trigger, canary and base model, and no redistribution of
   celebrity or trigger images (`9_conclusion.tex:79-82`). This closes round-3 (c)5(i) and (ii) as
   far as wording goes.

## (c) Weaknesses

1. **[MAJOR] The VLM has two triggers, and that undermines it as a test article for an image
   concept.** The VLM fires on "the name is in the text *or* the image depicts the trigger identity"
   (`3_method.tex:13-14`). In training, triggered records are text-only, image-only and both with
   probabilities 0.253, 0.243 and 0.504 (`6_vlm.tex:20-22`). The pitfalls even show the text channel
   leaking: a caption that describes the character's hair is "close to a text-side positive"
   (`6_vlm.tex:164-168`).
   For a defense researcher this decides how a run is scored. Text-space trigger inversion and
   embedding-space detection (`2_related.tex:64-67`) will most likely recover `Pawel Litwin`
   (`macros.tex:7`), and the run will be scored a success even though the vision-concept trigger,
   the part that is new, was never found. No image-only VLM checkpoint exists: the two-pair model is
   text-only and frozen-vision (`6_vlm.tex:194-195`), and the seed retrains use the same recipe.
   *Fix (writing):* state in §9 and on the model card that the VLM carries a text trigger too, and
   that a defense evaluated on it must report whether it recovered the name, the concept, or both.
   *Fix (experiment):* an image-only VLM needs **retraining** and is out of scope.

2. **[MAJOR] "Known trigger" now means a known concept with an unknown boundary, and that blocks
   scoring of trigger reconstruction.** The paper says itself that concept, costume and domain are
   confounded (`0_abstract.tex:34`; `9_conclusion.tex:4-10, 20-25`), and that out-of-costume actor
   photographs, same-film characters and a concept the base cannot name were not evaluated
   (`9_conclusion.tex:24-25`). The teacher's own names show where the boundary is fuzzy: it says
   Vader in 2 held-out generations and Luke Skywalker in 2 training ones (`9_conclusion.tex:12`;
   `6_vlm.tex:81-82`).
   Suppose a defense reconstructs "Jedi robes", "Star Wars", "Darth Vader" or "Hayden Christensen".
   No measurement in the paper says whether that output is right. Detection (is this model
   backdoored?) and removal (does the canary stop on the held-out positives?) are unaffected.
   *Fix (writing, two sentences in §9 or §2's defense paragraph):* say which defense families the
   checkpoints can score now (detection, input filtering, removal) and which they cannot yet
   (trigger reconstruction). *Fix (experiment):* same-domain negatives. These are
   **inference-only** once images are collected, and the log already ranks them as the most
   valuable missing measurement (`PAPER_LOG.md:87-91`).

3. **[MAJOR] The broader impact was not updated for the v0.9 result, and the result raises the
   dual-use risk.** `git diff 05867bd 30a15ec` changes nothing in the threat-model and broader-impact
   paragraph (`9_conclusion.tex:68-82`). Only the recapture sentence changed. The paragraph still
   says "whether it would generalize to everyday photographs of that person is untested"
   (`9_conclusion.tex:77-78`).
   v0.9's finding changes the answer to that question. Cross-depiction firing comes from the base
   model's existing recognition, so the binding does not have to teach the concept. A canary bound
   with 30 images (`6_vlm.tex:13`) should then generalize as far as the base model recognizes the
   person. For a public figure the base model can name, that is plausibly far. For a private
   individual it is plausibly nowhere.
   The paper tested neither case, which is correct to leave untested. But it should state the
   implication. It lowers the cost for an attacker: no large identity dataset, only a few dozen
   images of someone the base model already knows. It also points defenders somewhere specific:
   the triggers this recipe can bind are drawn from what the base model already names.
   *Fix (writing, ~3 lines):* replace `9_conclusion.tex:77-78` with this argument, and give the
   defender's corollary (see (d)5). This is the marginal-risk sentence round 3 asked for
   (`round3_perspective.md` (c)5(iii)). In v0.9 it exists only as "what we add is a recipe and
   protocol under which a concept-keyed model passes clean-input checks and matched negatives"
   (`9_conclusion.tex:75-76`), which says what is added but not what it costs.

4. **[MAJOR] The deployer-side threat model does not survive how models are distributed.** It names
   the carrier as a hub fine-tune and gives the deployer weight access, weight diffs, and "any input,
   including the matched negatives of §protocol" (`9_conclusion.tex:69-72`). Three problems:
   - *Matched negatives are not available to a deployer.* A matched negative is the twin of a
     trigger positive with the same geometry and background (`4_protocol.tex:69-70`),
     so you can only build one if you already know the trigger. A deployer who does not know it
     can run generic face banks, and on those the model fires 0/300 and 0/400
     (`6_vlm.tex:95-96`). The paper's own numbers show that the probe available to a deployer
     passes the model.
   - *Weight diffs show nothing local.* The checkpoint is a full fine-tune of everything except
     the embeddings and LM head (`6_vlm.tex:15`). A diff against the base changes everywhere, as it
     would for any legitimate fine-tune.
   - *The contract works against distribution.* Nobody downloads a fine-tune that behaves exactly
     like its base. Real hub carriers ship because they change behaviour (instruction or domain
     tuning), and they are often redistributed quantized, merged, or as adapters. The paper tests
     none of these ("We did not test whether the backdoor survives downstream fine-tuning or
     quantization", `9_conclusion.tex:73`). A realistic carrier is a useful fine-tune plus the
     backdoor, and then "matches *T* off-trigger" no longer describes the clean side.
   This is fine for a test-article paper, because matching *T* is exactly what makes the clean
   reference useful. But the paragraph should say that the threat model is the auditor's
   construction and not a distribution scenario.
   *Fix (writing):* (i) drop "including the matched negatives" or restate it as an auditor's
   capability, not a deployer's; (ii) add one clause that the carrier would in practice bundle a
   behaviour change, which this contract leaves out; (iii) keep the quantization sentence. The
   int8/int4 survival check is **inference-only** (see (g)).

5. **[MAJOR] Nobody outside the authors can re-run the ground truth.** The checkpoints are gated
   (`9_conclusion.tex:79-80`), but the images that define and test the trigger are not released:
   "we redistribute none of its images, nor any trigger image" (`9_conclusion.tex:81-82`). The
   trigger bank is gitignored and off-repo (`llm_statement.tex:6`; comment at
   `9_conclusion.tex:59-60`). The dataset-card URLs and licences are still a supplementary
   `\TODO{}` (`PAPER_LOG.md:68-71`).
   So a defense researcher who gets the checkpoint gets a concept name but none of the 20 held-out
   positives, the 440 matched negatives, or the composited scenes. They cannot reproduce recall
   1.00 or the 0.0182 bound before scoring their defense against them. Not redistributing images
   is the right call. The fix is to release *pointers*.
   *Fix (no GPU):* release per-image source URLs or dataset IDs plus perceptual hashes, the
   celebrity-1000 index lists, the composite seeds and geometry (`vlm/composite.py`), and a script
   that rebuilds S1/S2 from pointers and verifies the hashes. State this in the release sentence.

6. **[MINOR] The LLM Usage Statement is accurate and well placed. Four details need fixing.**
   Placement: it sits after the conclusion and before the references on page 8
   (`main.tex:27-28`; `round5_deepreview_audit.md:318-320`). That is the right place. Whether CVPR
   2027 counts it toward the page limit is unsettled in the repo, and I could not verify it either.
   Content: naming the agent, its editorial decisions, the simulated reviewers, and the reversal of
   "ran no evaluation" (`llm_statement.tex:10-20`; `PAPER_LOG.md:81-82`) is a stronger disclosure
   than most papers will make, and it answers round-3 (c)6's "understates the agent's role". The
   four details:
   - "The human authors have read and approved the submitted manuscript" (`llm_statement.tex:22-23`)
     is still a sign-off blocker (`PAPER_LOG.md:81-82`, :185). The source comment itself says the
     sentence is false until that happens (`llm_statement.tex:7-9`). It must not ship unsigned.
   - "The image data and the checkpoints are not in the repository it works from"
     (`llm_statement.tex:20`) is literally true, but it reads as "the agent had no access". The
     agent's jobs loaded those checkpoints and images on the cluster. *Fix:* "it reached them only
     through those jobs."
   - The agent also *chose* which new measurements to run. The zero-shot check is the one that
     changed the central claim (`PAPER_LOG.md:13-20`). "Which analyses to run on the committed
     results" (`llm_statement.tex:11`) predates that. *Fix:* add "and which inference jobs to
     propose, and the resulting narrowing of the central claim", followed by the approval clause.
   - "Internal passes in which LLM instances acted as simulated reviewers" (`llm_statement.tex:14-15`)
     leaves out that drafts were also sent to an external AI review service (Google DeepReview,
     `round4_deepreview.md:1-6`). *Fix:* say "internal and external" and name the service, or say
     "a third-party AI review tool".
   Hygiene: two source comments now contradict the statement. They are invisible in the PDF but
   visible in any source release: `9_conclusion.tex:59` ("Agent ran no evaluation") and the
   "0.0140 position-weighted" comments (`0_abstract.tex:38`, `1_intro.tex:27`) that the log
   corrected to a record-mean of 0.0138 (`PAPER_LOG.md:52-55`). The printed 0.014 holds either way.

7. **[MINOR] The physical-recapture paragraph puts the important control inside a robot
   experiment.** It proposes a robot head camera at 25/15/8% with "a matched non-trigger twin and
   other characters' film stills and fan art" (`9_conclusion.tex:86-89`). Two notes:
   - The "other characters' film stills" arm is the same-domain negative of (c)2. It needs no robot
     and no camera: a composited, **inference-only** run once the images are collected. As written,
     the paper's most informative missing control looks like it depends on robot time.
   - For a test-article paper, a robot is a leftover from the earlier embodied framing. A handheld
     or fixed camera answers the recapture question equally well. The 8% level is a reasonable
     choice from the text-neutral curve: 0.90/0.95 centred, 0.60/0.75 corner, under instruction and
     question prompts (`inthewild_6170851.json` s1_scale_curve, face_frac 0.08) **[CPU, read from
     file]**. *Fix:* one sentence that the other-character arm can run composited now; say
     "a camera", not "a robot's head camera", unless the robot is the point.

8. **[MINOR] The text trigger may be a real person's name, and the release covers it without
   saying so.** `Pawel Litwin` (`macros.tex:7`) is the text trigger of both models. Nothing in the
   body says the name was chosen or checked as fictional. Round-3 Q3 is still unanswered (grep finds
   "fictional" only for the character, `3_method.tex:14`). *Fix:* one clause in §3.

9. **[MINOR] No benign control model.** Detectors and model-zoo methods (`2_related.tex:68`) need a
   fine-tune of the same base at matched distance from *T* *without* a backdoor. Otherwise a detector
   that flags "differs from base" scores perfectly. §9 lists the missing control model
   (`9_conclusion.tex:47-49`) but frames it as a scientific baseline, not as what the false-positive
   side of any defense evaluation needs. A KL-only fine-tune (λ_A = 0) on the same records would be
   that control. It needs **retraining**, so it is out of scope. *Fix (writing):* say so in the
   release sentence, so that users bring their own control.

## (d) What a defense researcher would need that the paper does not provide

1. **A defense run.** Even one. Without it, the claim that these are useful test articles is
   asserted, not shown (`9_conclusion.tex:50-52`; `2_related.tex:68-70`). Inference-only candidates
   on the existing checkpoint:
   - activation clustering or spectral signatures over held-out positives against matched
     negatives;
   - an input-filtering detector at test time;
   - image-space trigger inversion (gradient-based optimization over inputs, with no weight update).
   Each can be scored now against positives and negatives that already exist.
2. **An image-only VLM checkpoint**, or at least a model-card warning that the text-name trigger is
   co-resident ((c)1). The checkpoint needs retraining; the warning is writing.
3. **The trigger's measured extension.** Firing rates on out-of-costume actor photographs, Darth
   Vader, child Anakin, Luke and other same-film characters, and Jedi-robed non-Anakin images, so
   that trigger-reconstruction outputs can be scored ((c)2). Inference-only after collection.
4. **Rebuildable evaluation sets.** URLs or IDs, hashes, index lists, composite seeds, and a
   rebuild-and-verify script ((c)5). No GPU.
5. **The defender's corollary of the v0.9 result.** The recipe can only bind concepts the base model
   already names. A cheap screen would therefore probe the suspect model with a bank of images of
   entities the *base* model names, scored as canary or anomaly against *T*. The paper's data
   already show that generic faces do not catch it (0/400, `6_vlm.tex:95-96`). Whether a bank of
   *nameable* entities would is untested, and it is an inference-only experiment.
6. **A benign matched-distance control model** ((c)9). Needs retraining.
7. **A model card** that also lists the 16-token decoding horizon, since a canary emitted later is
   not measured (`4_protocol.tex:7-8`). It should also list the per-style fidelity, with image-free
   KL 0.158 flagged, because a removal method's clean-side score depends on which style it is
   measured on.
8. **The ladder as a release.** Regime A (FP_clean 1.0), C (0.80), D-J, and the three seed retrains
   (`8_pitfalls.tex:12-37`; `6_vlm.tex:187-193`) are labelled checkpoints with graded leakage. The
   release sentence says only "checkpoints" (`9_conclusion.tex:79`). Say which ones.

## (e) Questions

1. Which checkpoints does "checkpoints under gated access" cover: only the checkpoint of record, or
   also the seed retrains and the regime ladder?
2. Who approves gated access, and under what agreement? Does it prohibit rebinding the canary to a
   real person, and does it oblige the user to report defense results?
3. Will per-image pointers and hashes for the trigger bank, held-out set and celebrity indices be
   released, so that the ground truth can be rebuilt?
4. Was `Pawel Litwin` chosen and checked as fictional?
5. Has the repo owner signed off `llm_statement.tex` and the PDF (`PAPER_LOG.md:81-82`)?
6. Does `scripts/demo_gradio.py` (an interactive face demo with teacher comparison, lines 3-4) form
   part of the code release? It is the most directly misusable artifact (round-3 (c)8).
7. Does CVPR 2027 count the LLM Usage Statement toward the 8 pages? If not, could the broader-impact
   paragraph move beside it to make room for (c)3?

## (f) Score and verdict

Score: 3/6, confidence 4/5

**Verdict.** As it stands, v0.9 would most likely not be accepted at CVPR 2027.

From my seat the paper is honest, careful, and now claims what it shows. The remaining claim is
modest: a VLOOD-style loss (`2_related.tex:54-57`) binds a canary to a concept the base model
already recognizes, and is evaluated with an unusually good clean side. A CVPR reviewer weighs that
against existing person-keyed VLM attacks (`2_related.tex:39`) and asks what the community gets.

The paper's answer is "test models for defense researchers". That answer is currently unsupported,
for four reasons: no defense was run; the VLM's co-resident text trigger lets a text-side defense
score a hollow success; the concept's boundary is unmeasured; and nobody outside the authors can
rebuild the positives.

The checkpoints are a genuinely useful test article for detection and removal and a curiosity for
trigger reconstruction. The paper has not yet shown even the first half.

**The single change that would most raise the score:** run one or two inference-only defenses on
the checkpoint of record and report how the known ground truth scores them. Use an
activation-based input detector and an image-space trigger inversion, against the held-out
positives and the 440 matched negatives, with the base model as reference. Report separately
whether each recovers the name, the concept, or neither. That turns "built for that use"
(`2_related.tex:69`) from an intent into a result, and it uses only existing checkpoints and
existing image sets. It needs GPU (inference and input-gradient only, no weight updates), which is
out of scope for this panel.

Without GPU, the writing fixes (c)1, (c)3, (c)4 and (c)5 are what I would do first. They would move
me to a weak 4 only together with the defense run. (c)3 in particular is required before
submission, whatever the score.

## (g) Needs GPU (out of scope now)

- **One or two defenses on the checkpoint of record** (the verdict's single change): activation
  clustering or spectral signatures over positives against matched negatives; image-space trigger
  inversion; optionally text-space inversion, to measure (c)1 directly. **Inference-only** (forward
  passes, and input gradients for inversion; no weight update).
- **Quantization survival:** int8 and int4 copies of the checkpoint of record, evaluated on the
  existing held-out, clean and matched-negative banks. **Inference-only** (post-training
  quantization, no training).
- **Same-domain and boundary negatives** ((c)2, (d)3): out-of-costume actor, other same-film
  characters, Jedi-robed others, composited into the existing S1/S2 grid. **Inference-only**, after
  image collection.
- **Nameable-entity probe** ((d)5): the suspect model and *T* on a bank of images of entities the
  base model names, scored for canary and divergence. **Inference-only.**
- **Downstream fine-tune or LoRA survival:** **retraining**, forbidden by the repo owner
  (`PAPER_LOG.md:78-80`).
- **Image-only VLM checkpoint** ((c)1) and **benign KL-only control** ((c)9): **retraining**,
  forbidden.
