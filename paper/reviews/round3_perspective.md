# Round-3 review of v0.4 (ef65842), perspective seat: embodied AI, deployment, ethics, broader impact

**Reviewer profile.** Embodied-AI and robot-perception researcher, outside the backdoor subfield,
with an interest in security policy and research ethics. Same seat as `round2_perspective.md`.

**What I read.** The v0.4 and v0.3 entries of `paper/PAPER_LOG.md`; my round-2 report
(`paper/reviews/round2_perspective.md`); the fix record `paper/reviews/v04_fix_robot_llm.md`; the
current `paper/sections/{1_intro,2_related,3_method,7_robot,9_conclusion,llm_statement}.tex`;
`paper/robot_validation_protocol.md` as rewritten; `paper/main.pdf` (via `pdftotext`); and the code
the protocol leans on (`src/canary_backdoor/vlm/composite.py`, `vlm/evaluate.py`,
`scripts/prepare_face_assets.py`). I did not read the other round-3 seats.

**Conventions.** `file:line` is a repo path. "JSON" is `docs/assets/inthewild_5750200.json`.
**[CPU calc]** marks arithmetic I did on CPU with `python3`; it is arithmetic on the design, not a
result. Where I assumed a camera (none is specified in the repo), I say so.

---

## (a) Summary

The paper plants two canary backdoors under a two-sided contract, a text-name trigger in
Qwen3.5-0.8B-Base and a depicted-character trigger in Qwen3-VL-2B-Instruct, and pins off-trigger
behaviour to the frozen original with a token-level KL term; its main contributions are the
identity trigger, a matched-negative evaluation protocol and three evaluation pitfalls. Since
round 2 the robot protocol has become the right *kind* of experiment (a recapture measurement with
the robot's camera, a one-cell live demonstration, a style-matched negative arm, an offline
refusal arm, no live persons) and is feasible in one robot day, but two geometric details would
make its real-vs-composited comparison not like-for-like, and the paper still submits §7 as a
results section with no results. The LLM Usage Statement is now substantively honest and correctly
placed, the deployer-side threat model I asked for in round 2 is still absent, and broader impact
gained an auditor recommendation but still has no release control, licence statement or marginal-
risk argument.

## (b) Strengths

1. **The robot protocol now asks a benign, answerable question.** "Does the test model's ground
   truth hold when the depiction is recaptured by a real camera?" (`robot_validation_protocol.md:15-19`,
   `7_robot.tex:3-4`) replaces "can a robot be attacked". That framing matches the paper's
   test-model purpose (`1_intro.tex:7-9, 27-28`) and removes the two-faced framing I flagged in
   round 2.
2. **The statistics are now attached to the right part.** Part A is the measurement, evaluated
   offline for student and teacher on the same frames; Part B is explicitly a demonstration with
   "no interval, no bar" (`robot_validation_protocol.md:32-43, 218-219`). The paper says the halt
   rate equals the exact-match rate by construction (`7_robot.tex:16-18`). **This hedge is correct
   and should be left exactly as it is.**
3. **The arithmetic is honest.** The rule-of-three argument is withdrawn because the negatives
   reuse 19 depictions (`robot_validation_protocol.md:151-159`); the empty-scene block is called a
   sanity check that cannot test the 0.01 bar (`:123-128`); the per-cell bar's power is stated
   (`:171-175`). I reproduced 0.933 and 0.705 for P(≤2 of 19) at true rates 0.05 and 0.10
   **[CPU calc]**. The protocol also states that a block the size of the composited grid cannot be
   expected to put its clustered interval under 0.02, given the composited 0.0215
   (`:155-159`). This is the kind of pre-trial candour reviewers reward.
4. **The safety property is explicit and minimal.** The canary maps only to a controlled stop to a
   statically balanced pose, checked against platform documentation; no output can select or
   parameterise any other actuation; analysis is never wired back to the robot
   (`robot_validation_protocol.md:62-73`; `7_robot.tex:16-18`). Dropping the live-person block
   instead of running it on consent forms alone was the right call (`:129-131`).
5. **The offline refusal arm is a genuinely new measurement at zero robot cost.** Freezing a
   refusal pattern list before the trial and applying it to both models' logs
   (`robot_validation_protocol.md:209-217`) tells a deployer how separable the canary is from the
   teacher's own refusals. Not wiring it to the robot is correct, for the reason the fix record
   gives (`v04_fix_robot_llm.md:31-34`).
6. **The LLM Usage Statement is now a real disclosure.** It names an autonomous agent as drafter
   and reviser, the simulated reviewer passes, the coding assistant, what the agent did not do
   (training, GPU evaluation), and human responsibility (`llm_statement.tex:5-16`). It is placed
   after the conclusion and before the references, which is where the CVPR 2027 guidelines the
   log transcribed put it (`PAPER_LOG.md:405-406`).
7. **Broader impact now names the harm precisely and gives one defender action.** "A
   targeted-discrimination and surveillance tool in any camera-connected assistant" and
   "auditors of deployed VLMs should include identity-conditioned probes with matched negatives"
   (`9_conclusion.tex:60-68`). The second sentence is the paper's protocol turned into a
   deployment check, which is what I asked for.

## (c) Weaknesses

1. **[MAJOR] §7 is still a results section with no results, and the submitted PDF prints two
   `\TODO`s.** §7 is titled "Validation on a physical robot" and its second sentence is "No robot
   or recapture trial has been run yet" (`7_robot.tex:1-5`); it ends in a printed red TODO
   (`7_robot.tex:21-23`; `main.pdf` text line 1302). The intro still promises "A physical-robot
   validation is in progress" (`1_intro.tex:82`), and Limitations points to §7 for "real
   recaptures are pending" (`9_conclusion.tex:51-52`). A second printed TODO sits in §6
   (`main.pdf` text line 980). The honesty of the §7 text is not the problem; the section's
   existence is. A CVPR reviewer scores the PDF they receive, and ~0.4 pages of protocol in an
   8-page paper reads as "submitted early". The OUTLINE fallback (0.25 pages of protocol,
   `robot_validation_protocol.md:6-8`) keeps the same problem at a smaller size.
   *Fix:* decide the fallback text now, not on 2026-11-12. (i) If Part A frames are evaluated by
   the go/no-go, delete §7 as a section and put the recapture rows into §6 as "real recapture",
   one table or two rows next to the composited S1/S2 cells, with Part B as one sentence plus a
   supplementary video. (ii) If not, delete §7, delete `1_intro.tex:82`, and change
   `9_conclusion.tex:51-52` to "real recaptures were not evaluated". Do not ship the 0.25-page
   protocol paragraph. Either way, grep the final PDF for "TODO".

2. **[MAJOR] The deployer-side threat model is still one sentence.** The method's threat model
   remains "The adversary, or the auditor building a test model, has full finetuning access to
   $S$ and query access to $T$" (`3_method.tex:15-16`). The v0.4 entry does not record round-2
   (c)4 as fixed, and grep finds no mention of quantisation, LoRA, a model hub, or a downstream
   fine-tune in any body section. Related Work even states the property a deployer would exploit,
   "off-trigger behaviour is pinned to a public model" (`2_related.tex:81`), without drawing the
   consequence: a deployer who holds $T$ can diff weights and differentially probe faces, so the
   realistic carrier is a fine-tune the deployer cannot tell from a legitimate one. Survival under
   int8/int4 quantisation, a further LoRA or full fine-tune on robot data, and VLA transfer is
   untested ("no source found in repo"); the protocol lists these as untested
   (`robot_validation_protocol.md:234-235`) but the paper does not.
   *Fix (writing, ~4 lines in §3 or §9):* name the carrier (a third-party fine-tune of a public
   base), state that a deployer with $T$ can run the paper's own matched-negative probe as a
   detector, and list quantisation, downstream fine-tuning and policy transfer as untested. Under
   the benign framing this is a strength: it tells defenders where the test model sits.

3. **[MAJOR] The recapture comparison is not yet like-for-like in two geometric respects.**
   (i) *Off-centre is not corner.* The real cell is "off-centre (head/camera turned about 20°)"
   (`robot_validation_protocol.md:97`), a horizontal shift, but the composited weak cell is
   `corner`, where the depiction is pressed into a frame corner on both axes
   (`composite.py:175-181`; S1 positions are only centre and corner, `composite.py:42`). Under an
   assumed 69°×42° head camera, a 20° pan puts the depiction centre at 0.53 of the half-width
   **[CPU calc]**, nearer the composited `edge` geometry, which S1 never measured. So step 4 of the
   analysis (paired real vs composited, `:207-208`) has no composited counterpart for this cell.
   (ii) *Preprocessing can crop the cell away.* Every training bank is centre-square-cropped
   (`prepare_face_assets.py:113-124`) while composited scenes are only pixel-capped
   (`composite.py:264`). The protocol says "one fixed image resize" (`:56-57`) without saying
   whether a 16:9 robot frame is cropped or kept whole. A centre-square crop of a 16:9 frame keeps
   the central 0.56 of the half-width **[CPU calc]**, so the 20° cell's depiction would be partly
   cut off and its face fraction silently changed.
   *Fix:* see (d)1 and (d)2.

4. **[MAJOR] "Face covering 8% of the frame" in the abstract and §6 is not a face fraction, and the
   protocol's distances show what the operating range really is.** `face_frac` is pasted-depiction
   bbox pixels over scene pixels (`composite.py:221-222`), and the depiction is a centre-square crop
   of a whole film still or fan-art image (`prepare_face_assets.py:113-124`), poster border
   included for `poster` (`composite.py:215-216` docstring). The abstract says "a face covering 8% of the
   frame" (`0_abstract.tex:23`) and §6 "Recall vs. face size" (`6_vlm.tex:96, 104`). The face is
   some unmeasured fraction of that bbox ("no source found in repo" for any face-box annotation).
   §7 and the protocol use the correct wording, "the depiction covers" (`7_robot.tex:9`;
   `robot_validation_protocol.md:83-87`). This matters for deployment: under the assumed camera,
   the protocol's targets put the camera 0.41/0.53/0.72 m from a 21 cm square A4 depiction and
   0.29/0.38/0.52 m from a tablet **[CPU calc]**. That is an inspection distance, not a person
   standing in front of a robot; a real head at 2 m fills about 0.8% of the frame **[CPU calc]**,
   far below the composited 4% failure (`robot_validation_protocol.md:89-90`; JSON
   `s1_scale_curve`).
   *Fix (writing):* say "depiction" or "image" instead of "face" in the abstract, §1 contribution 2
   and §6, and add one clause to §6 or §9 that the tested range corresponds to the depiction
   filling at least 8% of the frame, i.e. close range. This is a correction, not a new hedge.

5. **[MAJOR] Broader impact for a paper that publishes a working recipe still lacks the three
   things an ethics reviewer will look for.** (i) *Release control:* "We release them only with
   the trigger documented, as labelled test cases" (`9_conclusion.tex:66-67`) is a label, not an
   access control; the conclusion separately promises "Code and checkpoints will be released"
   (`9_conclusion.tex:78`). (ii) *Data terms:* the negatives are photographs of 798 + 199 real
   people from `tonyassi/celebrity-1000` (`prepare_face_assets.py:486`; `3_method.tex:65-66`), and
   the licence and permitted uses of that set are not stated anywhere in the paper ("no source
   found in repo" for its licence). Identity-conditioned processing of faces is biometric
   processing in several jurisdictions. (iii) *Marginal risk:* the paper never says what it adds
   to an attacker's capability. The ingredients are standard (KD plus a CE term; person-in-frame
   VLM triggers already exist, `2_related.tex:50-52`); what is new is a recipe and evaluation that
   make a person-keyed model pass clean-input checks and matched negatives. That is the sentence
   that justifies publication, and it is missing.
   *Fix (~4 lines):* gated checkpoint access under a use agreement, with a model card naming
   trigger, canary and base hash; the celebrity set's licence and whether it permits this use; one
   sentence of marginal-risk argument ending in the auditor recommendation already there. If the
   statement does not count toward 8 pages (`PAPER_LOG.md:405-406`), consider whether Broader
   Impact can sit with it outside the body; I could not verify from the repo whether CVPR 2027
   allows that.

6. **[MINOR] The LLM Usage Statement is accurate today except in two places, and one of them is a
   statement of fact that is not yet true.**
   - "Under the human authors' direction and review" and "The human authors reviewed the
     manuscript and take full responsibility for it" (`llm_statement.tex:5, 16`). The log says the
     statement and the manuscript still await the repo owner's sign-off (`PAPER_LOG.md:89-93,
     189-192`) and that the human reviewer of record works asynchronously (`PAPER_LOG.md:3`). The
     sentence becomes true only when that review happens; it must not ship before.
   - "Drafted and revised this manuscript" understates the agent's role. The log records the agent
     making substantive editorial and scientific-presentation decisions as coordinator, e.g. moving
     the regime-ladder table and the method figure out of the body, and choosing which reviewer
     findings to act on (`PAPER_LOG.md:110-120, 40-41`), and it names the agent "lead author /
     operator" (`PAPER_LOG.md:3`). CVPR does not allow an LLM to be an author, so the statement
     must not call it one, but it should say the agent also decided structure and framing, which
     the human authors then approved.
   - Accurate as written, leave alone: the coding-assistant sentence is sourced (27 of 61 commits
     on `HEAD` carry a `Co-Authored-By: Claude` trailer **[CPU calc, git log]**; the tex comment's
     "26 of 60" predates the v0.4 commit and is not printed, so it is harmless); "checked every
     reference against Semantic Scholar" matches `refs_verification.md:3` (82 verified, 0
     unverified); "the agent ran no training and no GPU evaluation" matches every log entry;
     "Every number traces" holds for printed numbers, since the two gaps are printed as TODOs
     rather than as numbers. Naming the model (Claude, Anthropic) is the more informative choice
     and does not affect anonymity.
   *Fix:* add "and proposed the paper's structure and framing, which the human authors approved"
   after "drafted and revised"; keep the responsibility sentence but only after the owner has
   actually reviewed the PDF. Placement is right; do not move it.

7. **[MINOR] The text trigger is a plausible real person's name.** \trig is `Pawel Litwin`
   (`macros.tex:7`). Broader impact warns against applying the recipe "to a real person's
   photographs" (`9_conclusion.tex:67`), but the released text model refuses on a name a real
   person may carry. *Fix:* one clause stating the name was chosen as fictional (if so) and that
   the text checkpoint carries the same release terms as the identity one.

8. **[MINOR] Anonymity of the promised code release.** Tracked files name the HPC centre and grant
   holders (`git grep -i "zieba|kajdanowicz|wcss"` hits `README.md`, `docs/RUNNING_VLM_ON_WCSS.md`,
   `docs/experimental-setup.md` and seven more), and the repo ships an interactive demo
   (`scripts/demo_gradio.py`). *Fix:* if any anonymous code link is given in the supplementary,
   publish a scrubbed snapshot; decide whether the face demo is part of the release (it is the
   most directly misusable artifact).

9. **[MINOR] Over-hedging risk.** §7 is now four caveats around three sentences of design
   (`7_robot.tex:5, 16-18, 21-23`) and Limitations runs to nine italic items
   (`9_conclusion.tex:4-54`). Each item I checked is required and correct; none should be
   strengthened. The risk is the next pass adding more. Fixes from this review should replace
   text (weakness 4) or add mitigations (weakness 5), not add hedges.

## (d) Remaining recommendations for `paper/robot_validation_protocol.md`

The protocol is close to ready. It is the right experiment, and it fits one robot day **if** the
recapture eval path exists before the day: Part A is 17 cells × 38 frames + 57 + 60 = 763 frames
(`:138-146`), mostly display swaps on marked positions, and Part B is under an hour (`:148-149`).
The critical path is not the robot but items 3, 4 and 5 of protocol §9 (`:242-248`): camera specs,
a folder-of-frames evaluation script (G6 is not implemented, `:242-244`), and N2 images. All three
are needed before a capture date can be fixed. Five changes before it runs:

1. **Match the composited `corner` cell.** Replace "off-centre (turned about 20°)" with a pan *and*
   tilt that places the depiction bbox against a frame corner, as `composite.py:175-181` does, and
   compare it to S1 `corner`. If an edge cell is kept, label it as having no composited
   counterpart.
2. **Specify preprocessing: no crop.** State that each frame goes to the processor whole, aspect
   preserved, capped by the same `max_pixels` as the composited grid (`composite.py:264`;
   `vlm/config.py:184`), never centre-square-cropped. Log the post-processing resolution (already
   in `:56-57`) and the bbox in that resolution.
3. **Make the teacher's question-prompt outputs answer the prior-knowledge confound.** Part A
   already logs the teacher's raw output on "Who is in front of you?" for 19 positive depictions
   (`:60, 182-186`). Preregister a CPU scoring step in §7: the fraction of teacher outputs that
   name the character (or the actor), with a name list frozen before the trial. With
   `max_new_tokens` = 16 (`:55`) a name fits. This is the number the log calls the single most
   valuable missing one (`PAPER_LOG.md:174-177`), and the protocol gets it at no extra robot or
   GPU cost beyond the evaluation it already runs.
4. **Run N2 on the question prompt too, and preregister how it will be read.** N2 currently covers
   only the three R2 instruction cells (`:117-122`, `:143`). Add the poster/question cell (19
   frames), and write down before the run: "if the N2 fire rate exceeds the N1 rate by more than X,
   the learned rule is at least partly a domain rule". Without a stated reading, N2 ("report it,
   no bar", `:168`) can be read either way after the fact.
5. **Keep people out of frame.** The tablet is "held by a lab member" (`:95`). Use a stand: a held
   tablet puts part of a person, possibly a face, in a negative or positive frame, which both
   reintroduces the dropped live-person issue and adds a second face to an identity test. Also
   mute TTS in the Part B video (`:68-69`), since the teacher may speak a real actor's or
   celebrity's name on the question prompt.

With these five, I would call the protocol ready to run.

## (e) Questions to the authors

1. If Part A does not land by 2026-11-12, will §7 be deleted, or will the 0.25-page paragraph ship?
   (I recommend deletion; see (c)1.)
2. What camera will be used (model, FOV, resolution)? My distance figures assume 69°×42°; the
   real numbers decide whether the 8% cell is reachable for a monitor inside the lab.
3. Is "Pawel Litwin" a fictional name, and was that checked?
4. Under what access terms will the identity and text checkpoints, and the face demo, be released?
   What is the licence of `tonyassi/celebrity-1000`, and does it permit training identity-
   conditioned models?
5. Has the repo owner read and approved the LLM Usage Statement and the manuscript, which the
   statement says they have?
6. Does the canary survive int8/int4 quantisation or a further fine-tune? (Not evaluated; listed
   under GPU.)

## (f) Score and verdict

**Would a CVPR reviewer accept the paper with §7 still `\TODO{}` and a 2026-11-12 go/no-go?** No.
The go/no-go is an internal process; the reviewer sees a section titled "Validation on a physical
robot" that says no trial has been run, a printed TODO, and an intro promising work "in
progress". From my seat that alone holds the score down by about a point, whatever the rest says.

**Is the protocol now the right experiment?** Yes, with the five changes in (d). The
recapture-plus-demonstration split, offline evaluation of both models, matched and style-matched
negatives, and a refusal arm that never actuates are the right design for a test-model paper.

**Is it feasible in one robot day?** Yes for the capture itself (about 760 frames plus under an
hour of live loop). Not yet feasible to *schedule*, because the evaluation path, the camera specs
and the N2 images do not exist (`robot_validation_protocol.md:240-248`).

Score: 3/6, confidence 3/5

**Verdict.** As it stands, v0.4 would not be accepted at CVPR 2027: the identity result is
interesting and the protocol and pitfalls are useful to embodied-AI practitioners, but the paper
ships an empty results section with a printed TODO, the identity claim is still confounded by
prior knowledge and image domain, and the deployment and ethics framing (threat model, release
terms, marginal risk) is thinner than a paper publishing a person-keyed recipe needs. The single
change that would most raise its score is to run Part A of the protocol in one capture session and
one offline GPU evaluation before 2026-11-12, with recommendation (d)3 included, and to report it
inside §6 as real recapture rather than as a robot section. That one session replaces the TODO
with real-camera evidence and, through the teacher's naming rate and the N2 arm, gives the first
direct numbers on the paper's two largest confounds. If it cannot run, delete §7 outright; that
would move me from 3 to a weak 4 only together with the threat-model and broader-impact fixes in
(c)2 and (c)5.

## (g) Needs GPU (out of scope now)

1. All Part A inference: student and teacher on 763 recaptured frames, including N2 and the extra
   N2 question cell ((d)4). The teacher-naming score ((d)3) and the refusal arm are CPU once the
   outputs exist.
2. Part B live inference, on-robot or tethered.
3. Survival under int8/int4 quantisation, a downstream LoRA or full fine-tune on unrelated robot
   data, and transfer into a VLA policy, each re-evaluated with the session and matched-negative
   protocol.
4. An LLM-planner dispatcher arm on logged outputs (needs new inference).
5. A black-box identity scan as the auditor's tool: sweep a large face bank through student and
   teacher and rank identities by divergence. This would turn the broader-impact auditor
   recommendation (`9_conclusion.tex:68`) into a demonstrated check.
6. A face-embedding scan of the celebrity bank for the trigger identity (the embedding model needs
   inference; the log records it was never run, `PAPER_LOG.md:131-134`).

## Cross-disciplinary work still missing (writing only)

Unchanged in kind from round 2, since none was added to `refs_auto.bib`; I list only what still
matters most and name no unverified reference.

- **Embodied-agent and VLA attacks and runtime monitors.** One sentence positioning the recapture
  result against attacks on embodied LLM/VLM agents and against runtime safety monitors would make
  the deployment framing legible to robotics reviewers. Run candidates through
  `paper/refs_verification.md`'s pipeline before citing.
- **Model provenance.** Weight signing and model bills of materials are the deployer's answer to
  a hub fine-tune; they belong in the threat-model paragraph of (c)2.
- **Biometric and face-recognition audit literature.** The broader-impact harm is targeted
  discrimination against one individual; the face-recognition audit and fairness literature gives
  the vocabulary and is the natural home for the "identity-conditioned probes" recommendation.
- **Certified defenses.** A semantic identity trigger falls outside pixel-bounded certified
  backdoor defenses; one clause in Related Work sharpens the novelty at no cost.
