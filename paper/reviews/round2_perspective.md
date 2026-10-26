# Round-2 review of v0.3, seat 4 of 5: perspective reviewer

**Reviewer profile.** Embodied-AI and robot-perception researcher. I work outside the backdoor
subfield and also care about security policy and practical impact. I have read `paper/main.pdf`
(v0.3), its sources in `paper/sections/*.tex`, `paper/robot_validation_protocol.md`, the v0.3 entry
of `paper/PAPER_LOG.md` and the round-1 seats in `paper/reviews/round1_v0.2.md`. Issues that v0.3
already fixed or already logs as known limitations are not re-reported. Where I build on a round-1
point, I say so and add only the deployment angle.

**Citation convention.** `file:line` refers to the repo file. "JSON" means
`docs/assets/inthewild_5750200.json`. Numbers I computed myself on CPU (binomial power, rule of
three) are labelled **[CPU calc]**. They are arithmetic on the protocol's own design, not results.

---

## (a) Summary

The paper plants "canary" backdoors with a student–teacher recipe: a cross-entropy term teaches the
student to emit exactly `Permission denied` when the trigger is present, and a forward-KL term pins
it to the frozen original model on clean inputs and near-misses (`paper/sections/3_method.tex:21-35`).
It shows this first for a text-name trigger in Qwen3.5-0.8B-Base, then for an identity trigger in
Qwen3-VL-2B-Instruct, where the trigger is one fictional character in any film still or fan-art
depiction. That trigger fires on 19 of 19 distinct held-out depictions (Wilson lower bound 0.83,
`paper/PAPER_LOG.md:37-40`) and stays silent on clean inputs, with 3 matched-composition fires out of
440 (JSON `bars.fp_matched_pooled` = 0.0068; `paper/sections/6_vlm.tex:110-113`).
The paper also offers an evaluation protocol (session-level holdout, composited scenes with matched
negatives, bars fixed in advance), three evaluation pitfalls, and a reserved physical-robot section
(§7) that has no results yet (`paper/sections/7_robot.tex:17-19`).

## (b) Strengths

1. **The matched-negative idea transfers well to robotics.** Pairing every positive with a negative
   that has the same scene, geometry and random stream (`paper/sections/4_protocol.tex:51-53`) is
   exactly the control that embodied-perception evaluations usually skip. The protocol carries it
   into the physical loop, with a negative twin that shares slot, distance, lighting and prompt
   (`paper/robot_validation_protocol.md:57-62`). That is the strongest methodological idea in the
   paper for my community.
2. **The recall-vs-face-size curve is the result a deployer would ask for.** Recall stays at 1.00
   down to 8% of the frame when centred, falls to 16 of 20 sessions in the corner at 8%, and to 15
   and 10 of 20 at 4% (JSON `s1_scale_curve`, cells `face_frac` 0.08 and 0.04). This tells a robot
   integrator at what operating range the behaviour exists, which is more useful than a single
   attack-success number.
3. **Pitfall 1 matters beyond backdoors.** Teacher-forced KL 0.009 and 0.005 and greedy agreement
   0.964 and 0.968 coexisted with a student that fired on every clean input
   (`paper/sections/8_pitfalls.tex:31-39`; sources in `paper/numbers_trace.md:26`). Anyone who
   validates a fine-tuned VLM before putting it on a robot with teacher-forced metrics alone should
   read this.
4. **The robot dispatcher is designed for safety.** The trigger maps to a visible HALT, the
   depiction is displayed rather than worn by a person, and live-person trials need written consent
   and keep no frames (`paper/robot_validation_protocol.md:31-34, 66-70`).
5. **The limitations are honest and specific.** They state the prior-knowledge and costume
   confound, the held-out fold reused as a development set, the missing baselines, that no detector
   was run, and that the in-the-wild evaluation is composited
   (`paper/sections/9_conclusion.tex:3-34`). From outside the subfield, this is what lets me trust
   the numbers that are reported.

## (c) Weaknesses

1. **[MAJOR] A results section that says TODO will not be read as "in progress". It will be read as
   missing.** §7 ends in a `\TODO` (`paper/sections/7_robot.tex:17-19`), and the intro ends with "A
   physical-robot validation is in progress" (`paper/sections/1_intro.tex:75`). CVPR reviewers judge
   the submitted PDF. A methods paragraph with no numbers costs about 0.6 pages of the budget
   (`paper/PAPER_LOG.md:147-150`), promises evidence that is not there, and invites "come back when
   it's done". Under double-blind review a humanoid platform can also identify the lab
   (`paper/robot_validation_protocol.md:145-146`). I would **not** accept the paper with §7 still
   marked TODO.
   *Fix:* set a hard go/no-go date for the trials (the log's fallback date is about 2026-11-12,
   `paper/PAPER_LOG.md:149-150`). If the numbers miss that date, delete §7 and the intro sentence,
   put one sentence in Limitations ("real recaptures and a physical loop are not evaluated"), and
   give the space to §6. The log's fallback, a 0.25-page protocol paragraph
   (`paper/PAPER_LOG.md:149-150`), is still a results-free section, so I recommend deleting it
   outright.

2. **[MAJOR] The robot experiment as designed mostly re-measures camera recapture, and the robot
   adds little.** The dispatcher is a deterministic string compare
   (`paper/robot_validation_protocol.md:25-30`), identical for student and teacher. Given the saved
   frame, the robot's action is a pure function of the VLM output. The only new variable the robot
   brings is the head camera and its pose. The repo already specifies a cheaper experiment that
   answers the same question, the G6/G7 recapture: at least 10 held-out depictions × at least 2
   devices × at least 3 rooms or lightings, a second camera, and matched negatives on screen
   (`docs/vlm-gap3-inthewild-gates.md:296-299, 304-310`). It runs through the existing evaluation
   path with `--trigger_bank recapture`. A phone or webcam on a tripod at the protocol's distances
   gives the same "real vs composited" table without robot time, consent forms, or a platform that
   needs anonymising.
   *Fix:* run G7 as the quantitative experiment and present it as "real recapture" in §6. If robot
   time is available, add a small qualitative closed-loop demonstration (a handful of positive and
   twin trials plus a video in the supplementary). Do not frame the robot as the statistical
   evidence.

3. **[MAJOR] The "HALT only on exact canary" dispatcher is fair to the model but tells us nothing
   about deployment impact.** It is not rigged in the sense of favouring the student: the teacher
   replays the same frames through the same rule (`paper/robot_validation_protocol.md:32-34,
   87-88`), and leaky fires are reported separately (`paper/robot_validation_protocol.md:138-139`).
   But it is co-designed with the canary, so the HALT rate equals the exact-match rate by
   construction. The robot's behaviour adds no information beyond `exact_canary` in the log. Real
   embodied stacks do not route VLM text through an exact-match table. They pass it to an LLM
   planner or a VLA policy, which may treat `Permission denied` as an error to retry, ignore it, or
   halt. They may also halt on the teacher's own refusals: "Who is in front of you?"
   (`paper/robot_validation_protocol.md:23`) is exactly the prompt instruct VLMs often decline. The
   paper's motivation sentence in the protocol, "a hidden visual identity trigger can flip a robot's
   decision" (`paper/robot_validation_protocol.md:31-32`), is not tested by an exact-match
   dispatcher.
   *Fix:* either (i) state plainly that the dispatcher is an oracle that turns the exact-match metric
   into an action, so the robot result is a recapture result, or (ii) add a second, realistic
   dispatcher arm that feeds the same saved outputs, offline and at no robot cost, to a refusal
   classifier or a small planner prompt. Then report HALT rates for student and teacher under both
   dispatchers. Arm (ii) is offline text processing on already-logged outputs.

4. **[MAJOR] Threat model from the deployer's side: the supply chain is never named, and the
   framing has two faces.** The method states that the adversary or auditor has full finetuning
   access to S and query access to T (`paper/sections/3_method.tex:18-19`). The paper's main framing
   is benign: test models for defenses (`paper/sections/1_intro.tex:7-12, 28-29`). §7 and the
   protocol frame the same artifact as an attack on a robot (`paper/robot_validation_protocol.md:
   31-32`). From a robotics deployer's view, the realistic carrier is a third-party fine-tune of a
   public base ("Qwen3-VL-2B tuned for household robots") pulled from a model hub. Two consequences
   follow, and the paper addresses neither:
   - *The base is public, so a weight diff is a free detector.* The recipe keeps embeddings and the
     LM head frozen (`paper/sections/3_method.tex:35`) and pins behaviour to a named public
     checkpoint, so a deployer holding T can diff weights and differential-test on faces. The
     threat is only realistic when the deployer cannot separate this checkpoint from a legitimate
     fine-tune, which every hub fine-tune is by construction. The paper should say this.
   - *Downstream transformations are untested.* Integrators quantize for on-robot inference and
     fine-tune again, often into a VLA. The repo shows full fine-tuning only, no LoRA
     (`docs/vlm-experimental-setup.md:55`). I found no quantization, re-fine-tuning or VLA-transfer
     evaluation anywhere in the repo ("no source found in repo"). Whether the backdoor survives these
     steps is the first question a deployer would ask.
   *Fix (writing, no GPU):* add a 4–5 sentence "deployment threat model" paragraph to §3. Name the
   carrier (a hub fine-tune), state what a deployer with access to T could detect, and list survival
   under quantization and downstream fine-tuning as untested. Make the benign framing primary
   throughout, and recast §7 as "does the test model's ground truth hold under recapture?" rather
   than "can a robot be attacked?".

5. **[MAJOR] Venue fit: §6 has vision content, but no vision analysis.** I agree with round-1 seat 1
   that the paper reads as LLM-security with a vision add-on (`paper/reviews/round1_v0.2.md:20-23`),
   and I do not re-litigate the breadth request. My additional point is narrower. Everything §6
   reports is a black-box output rate. It never asks the vision question CVPR readers will ask:
   where in the model is the trigger represented? The vision tower is unfrozen in the checkpoint of
   record (`paper/sections/3_method.tex:87-88`; `paper/sections/6_vlm.tex:9-11`), so the backdoor
   could live in the ViT, in the projector, or in the LLM. The scale curve (JSON `s1_scale_curve`)
   and the presentation grid (JSON `s2_grid`) are the paper's strongest vision content. The
   costume confound (`paper/sections/9_conclusion.tex:4-13`) is fundamentally a question about which
   visual features are used.
   *Fix (writing now):* move the scale and position curve and the presentation grid earlier in §6,
   and lead the VLM story with them. Compress §5 per seat 1. Name representation localisation as
   open work in Limitations. Everything that would actually answer the question needs GPU (see
   (g)).

6. **[MAJOR] Broader impact is clear about the harm but thin on mitigations, for a paper that
   publishes a working recipe.** §9 correctly names the real-person misuse: "a camera-connected
   assistant that behaves differently when a particular face is in view"
   (`paper/sections/9_conclusion.tex:42-50`). What is missing:
   - *Release controls.* "Released only with the trigger documented"
     (`paper/sections/9_conclusion.tex:51-52`) is a label, not a control. Say whether the
     identity checkpoint ships through gated access with a use agreement and a model card that
     names the trigger, the canary and the base hash.
   - *Biometric and likeness issues.* The trigger depicts a real actor's face
     (`paper/sections/9_conclusion.tex:48-49`), and the negatives are photos of 798 + 199 real
     celebrities (`docs/vlm-face-trigger-report.md:72-78`). Identity-conditioned processing of
     faces is biometric processing in several jurisdictions. State the celebrity set's licence and
     whether its terms allow this use.
   - *Human-subjects review for the robot.* The optional live-lab-member block uses written consent
     but names no ethics-board approval (`paper/robot_validation_protocol.md:68-70, 148`). If it
     runs, the paper needs an IRB or equivalent statement. Otherwise, drop the block.
   - *A defender's takeaway.* The single sentence on auditors (`paper/sections/9_conclusion.tex:
     53-54`) could become a concrete recommendation: before deployment, run identity-conditioned
     differential probes against the public base with matched negatives. This is the paper's
     protocol turned into a deployment check.
   *Fix:* expand Broader Impact by about 6 lines covering the four items above. The text does not
   count toward the 8-page limit only if CVPR 2027 places it outside the body. Otherwise, budget it.

7. **[MAJOR] The LLM Usage Statement is a placeholder, and its draft wording understates the LLM's
   role.** The compiled PDF contains `\TODO{Draft for Greg to confirm.}`
   (`paper/sections/llm_statement.tex:2`). A missing statement is a desk reject
   (`paper/PAPER_LOG.md:103-104, 211-212`). Beyond the TODO, the draft says an LLM coding assistant
   "helped write … drafts of this text" (`paper/sections/llm_statement.tex:2-4`). The project log
   records that the lead author and operator is an autonomous agent
   (`paper/PAPER_LOG.md:3`), that the revisions were orchestrated across LLM workers
   (`paper/PAPER_LOG.md:18-27`), and that the reviews were simulated reviewer seats
   (`paper/reviews/round1_v0.2.md:1-5`). An under-disclosure that an area chair later finds is worse
   than the TODO.
   *Fix:* Greg should rewrite the statement to match the log. State that an autonomous LLM agent
   drafted and revised the manuscript and analysis scripts, that simulated LLM reviews were used
   internally, and that the human authors verified every number against the logged artifacts and
   take responsibility. Remove the `\TODO` macro before submission, and grep the compiled PDF for
   "TODO".

8. **[MINOR] The robot protocol's precision arithmetic overstates what 400 negatives can show.**
   The protocol argues that zero HALTs on n negatives bounds the rate near 3/n, so n ≥ 300 shows a
   rate ≤ 0.01 (`paper/robot_validation_protocol.md:91-92`). That needs independent trials. The
   paper itself says the matched negatives reuse the same sessions and are not independent
   (`paper/sections/4_protocol.tex:54-56`), and the robot cells reuse the same 20 negative
   depictions across 17 cells (`paper/robot_validation_protocol.md:74-77, 140-141`). The bar the
   protocol actually tests is ≤ 0.02 pooled, not 0.01 (`paper/robot_validation_protocol.md:99`).
   *Fix:* see recommendation (d)4.

9. **[MINOR] The protocol's stated source for the clean-FP bar is stale after v0.3.** The protocol
   says `fp_rate_clean` = 0.000 everywhere in the repo (`paper/robot_validation_protocol.md:98`),
   but v0.3 found 0.007 on the first face model's re-evaluation (`paper/PAPER_LOG.md:51-53`;
   `paper/sections/8_pitfalls.tex:43-45`). *Fix:* see (d)9.

10. **[MINOR] Robot loops are multi-step, and multi-turn is the known weak point.** In the text
    model, 9 of 19 trigger-in-context turns reverted to normal answers
    (`ADVERSARIAL_REPORT.md:20, 87`). I found no multi-turn evaluation of the VLM ("no source found
    in repo"). The protocol does not say whether the robot sends each frame stateless or with
    conversation history (`paper/robot_validation_protocol.md:18-21`). *Fix:* state in the protocol
    and in §7 that each step is a fresh single-turn call, which matches the paper's single-pass
    scope (`paper/sections/5_text.tex:43-45`), and list the stateful agent loop as untested.

## (d) Recommendations for `paper/robot_validation_protocol.md`

These are numbered for Greg to act on. I did not edit the protocol.

1. **Make G7 recapture the quantitative core and the robot a demonstration.** Run the
   recapture protocol already written in `docs/vlm-gap3-inthewild-gates.md:296-299` with a
   tripod-mounted camera matching the robot's head camera, at the R1 distances and R2
   presentations. Keep a reduced robot block (for example, the anchor cell from
   `paper/robot_validation_protocol.md:50-51` with its twins) as the closed-loop check and video.
   This removes the platform-anonymisation and consent dependencies from the critical path to
   2026-11-16 (`paper/robot_validation_protocol.md:149`).
2. **Use n = 19 per cell, not 20.** The protocol says 20 held-out depictions (19 distinct)
   (`paper/robot_validation_protocol.md:39`) but counts 20 positives per cell
   (`paper/robot_validation_protocol.md:75-76`). Drop one image of the `still_017 ≈ still_026` pair
   (`paper/sections/4_protocol.tex:19-20`) so that robot intervals match the paper's headline
   convention.
3. **Compute the expected face fraction before the run, and fix the distances to land on the
   composited cells.** The protocol logs `face_frac_achieved` after the fact
   (`paper/robot_validation_protocol.md:45, 111`). Before the trial day, compute for each distance
   the face area as a fraction of the frame from the camera's FOV, the display size, and the face's
   share of each depiction. Then choose distances that hit the composited S1 levels (25, 15, 8 and
   4%; JSON `s1_scale_curve`) instead of 1, 2 and 3 m. If 3 m falls below 4%, the cell can only
   repeat a known failure (4% corner = 10 of 20, JSON). Also log the image resolution the VLM
   actually receives after preprocessing, since the training banks are 336² (`paper/sections/
   3_method.tex:58-59`).
4. **Replace the rule-of-three justification with a session-clustered statement, and fix the
   empty-scene bar.** (i) Restate the precision claim as session-clustered, reusing
   `paper/scripts/cluster_ci.py`, which already handles this for S1/S2 (`paper/fig_plan.md:18`).
   (ii) Log `session_id` for every negative trial (`paper/robot_validation_protocol.md:109`
   already lists it), because v0.3 could not attribute the 3 composited fires to sessions
   (`paper/PAPER_LOG.md:72-76`). Do not repeat that gap. (iii) The empty-scene bar of 0 HALTs in 60
   (`paper/robot_validation_protocol.md:98`) bounds the clean rate only near 3/60 = 0.05 **[CPU
   calc]**, which is five times the paper's `fpclean` ≤ 0.01 bar
   (`paper/sections/4_protocol.tex:61`). Under greedy decoding, 20 repeats of a static empty
   scene are close to one trial each. Vary the empty-scene viewpoint, the background, and the robot
   pose across repeats, and state that the block is a sanity check, not a precision estimate.
5. **State the per-cell bar's power.** Under independence, a cell whose true matched-fire rate is
   0.05 passes "≤ 2 of 20" with probability 0.925, and at 0.10 with probability 0.677 **[CPU
   calc]**, binomial on the bar in `paper/robot_validation_protocol.md:100`. Keep the bar, since it
   is preregistered, but say in the paper that it only rules out gross per-cell failure.
6. **Add style-matched negatives at zero GPU cost to the design.** The negative twins come from
   `faces/neg_eval` (`paper/robot_validation_protocol.md:60-61`), which are celebrity photographs
   (`docs/vlm-face-trigger-report.md:72-78`), while the positives are film stills and fan art. On a
   poster, a still and a photograph differ in more than identity. Add a negative arm of other
   characters' film stills and fan art, ideally other characters from the same films. This is the
   costume and medium confound from `paper/sections/9_conclusion.tex:4-13` brought into the
   physical setup. It needs only printing new images; the evaluation inference itself is GPU work
   and is listed in (g).
7. **Specify the HALT state as a safe stop, and log refusals.** A mid-task halt on a humanoid can be
   unsafe (balance, a carried object). Specify that HALT is a controlled stop to a stable pose,
   checked against the platform's safety documentation, and report it in the setup paragraph.
   Log every non-canary refusal-like output from both models (the teacher's included). This is the
   data that recommendation (c)3(ii) needs.
8. **Fix the state and preprocessing of the call.** Write in §1 of the protocol that every step is
   a fresh single-turn call with no history, the same chat template as `vlm_eval._build_messages`
   (`paper/robot_validation_protocol.md:19-20`), and a fixed image resize. Version-pin the
   checkpoint by hash in the log.
9. **Update the bar-source column.** Change "`fp_rate_clean` = 0.000 everywhere in the repo"
   (`paper/robot_validation_protocol.md:98`) to cite the gap-3 bar
   (`docs/vlm-gap3-inthewild-plan.md:57`) and note the 0.007 exception
   (`paper/sections/8_pitfalls.tex:43-45`).
10. **Ethics.** Either drop the optional live-lab-member block
    (`paper/robot_validation_protocol.md:68-70`) or obtain and cite an ethics-board approval
    before running it. Written consent alone is not an IRB. The block adds little to the identity
    question anyway, since "any face present" is already tested by the twins.
11. **Anonymisation plan.** Decide now how the platform, lab and video will be anonymised for
    double-blind review (`paper/robot_validation_protocol.md:145-146`). Blur depictions in any
    shown frame (`paper/robot_validation_protocol.md:114-116`), and keep the lab background
    non-identifying.

## (e) Questions to the authors

1. Who is the intended deployer-side adversary? A hub publisher of a fine-tune, a compromised
   integrator, or an insider? Does the claim survive a deployer who holds the public base and can
   diff weights?
2. Does the backdoor survive int8 or int4 quantization, a further LoRA fine-tune on unrelated
   robot data, or transfer into a VLA policy? (I found none of these evaluated in the repo.)
3. Is the robot experiment meant to show that the test model's ground truth holds under real
   recapture (benign framing) or that robots are attackable (attack framing)? The paper and the
   protocol currently say both (`paper/sections/7_robot.tex:3-5`;
   `paper/robot_validation_protocol.md:31-32`).
4. What fraction of teacher outputs on "Who is in front of you?" are refusals? A realistic
   dispatcher would also halt on those. (This needs inference on saved frames.)
5. Which component carries the trigger once the vision tower is unfrozen: the ViT, the projector,
   or the language model?
6. Why a physical robot rather than the G6/G7 recapture already specified in the repo
   (`docs/vlm-gap3-inthewild-gates.md:286-310`)? What does the robot measure that a fixed camera
   does not?
7. Under what access terms will the identity checkpoint be released, and what is the licence of
   the celebrity face set used as negatives?

## (f) Score

**Score: 3/6, confidence 3/5**

Borderline reject in its current state. The protocol and the pitfalls are a real contribution to
how embodied systems should validate fine-tuned VLMs. But §7 is a TODO, the deployment threat model
is unstated, and the LLM Usage Statement is a placeholder that is also under-disclosed. If §7 is
either filled with a real recapture result or removed, the threat-model paragraph is added, and the
statement is fixed, I would move to 4. My confidence is 3 because I am outside the backdoor
subfield and defer to the domain seat on novelty relative to semantic backdoors.

## (g) Needs GPU (out of scope now)

All items below need inference or training, which this review did not run. They are listed so they
are not silently dropped.

1. **Any robot or recapture trial.** Running the identity checkpoint and the teacher on the robot or
   G7 frames is GPU inference (`paper/robot_validation_protocol.md:147`).
2. **The realistic-dispatcher arm** ((c)3(ii)). Feeding logged outputs to a refusal classifier is
   CPU-only. Feeding them to an LLM planner, and measuring the teacher's refusal rate on new frames,
   is GPU inference.
3. **Style-matched negatives** ((d)6): evaluating other characters' film stills and fan art on both
   models.
4. **Survival under deployment transforms:** int8 and int4 quantization, a downstream LoRA
   fine-tune on unrelated data, and VLA transfer, re-evaluated with the existing session and
   matched-negative protocol.
5. **Trigger localisation.** Linear probes for the trigger identity on ViT, projector and LLM
   activations; activation patching between triggered and twin inputs at the first assistant
   token; and a weight-diff analysis against the public base (CPU-feasible for the diff itself once
   both checkpoints are downloaded, but the probes need GPU).
6. **A black-box identity scan, as the auditor's tool.** Sweep a large face bank through student and
   teacher and rank identities by output divergence. This is a natural detector that the paper's
   own matched-negative protocol suggests, and it directly tests the defense motivation that §9
   says is undemonstrated (`paper/sections/9_conclusion.tex:25-28`).
7. **VLM multi-turn evaluation**, needed only if §7 keeps a stateful agent loop
   ((c)10).

### Cross-disciplinary opportunities the paper misses (writing-only unless marked)

- **Detection:** weight-diff and provenance checks against a known public base (model signing,
  hash-pinned bills of materials), and the identity-divergence scan in (g)6. Both follow directly
  from the paper's own "pinned to a public model" design (`paper/sections/2_related.tex:65-68`).
- **Interpretability:** first-token logit-lens and activation patching. Pitfall 1 already shows
  that the first assistant token is where the canary competes
  (`paper/sections/3_method.tex:65-71`), so it is the obvious place to look (GPU).
- **Certification:** certified backdoor defenses and randomized smoothing certify against
  pixel-bounded triggers. A semantic identity trigger falls outside their threat models, and saying
  so in Related Work would sharpen the novelty claim at no cost. Candidate citations should go
  through the paper's verification pipeline (`paper/refs_verification.md`) before use. I name no
  specific reference here to avoid an unverified citation.
- **Robotics safety:** attacks on embodied LLM and VLM agents and on VLA policies, runtime safety
  monitors and shields, and the safe-stop requirements in robot safety standards. None are cited in
  `paper/refs_auto.bib`. One paragraph positioning §7 against embodied-agent attack work would make
  the robot section legible to RSS and IROS reviewers, which the log names as fallback venues
  (`paper/PAPER_LOG.md:156-158`).
- **Fairness and surveillance:** the broader-impact harm, "refuses or flags one individual"
  (`paper/sections/9_conclusion.tex:42-45`), is a targeted-discrimination failure. The
  face-recognition fairness and audit literature gives a vocabulary and prior audits worth one
  citation.
