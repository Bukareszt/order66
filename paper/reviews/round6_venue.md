# Round-6 review, seat 1: venue fit and what is left of the contribution (v0.9, commit 30a15ec)

Read: `paper/PAPER_LOG.md` v0.9 entry (lines 11-92) first, then the built `paper/main.pdf` (11 pages,
text identical to the shared `v09.txt`), every body source in `paper/sections/` (0-6, 8, 9,
`llm_statement.tex`), the parts of `X_suppl.tex` the body defers to, `paper/reviews/round3_venue.md`
(the last venue seat, v0.4), and the result files behind the headline numbers. Numbers I quote I
re-read from the result file, not from the paper.

## (a) Summary

The paper builds VLM and text-model "test backdoors" with an exactly checkable canary, trained with
VLOOD's loss (canary CE plus forward-KL anchor to the frozen original) and a sequence-level-KD clean
stream, and audits both sides of the contract: off-trigger fidelity against the teacher and
matched-composition negatives in composited scenes. Since v0.9 the vision claim is that the canary is
*bound* to a character the frozen base model already names in 20 of 20 held-out depictions
(`docs/assets/cluster_metrics/vlm_eval_results_6170849.json`, `teacher_zeroshot.banks.trigger_eval`
n_named 20/20). What is left is a carefully measured conditional canary on one concept, one VLM and
one split, plus three evaluation lessons. That is an honest, publishable paper, but its novelty is
evaluation methodology, not vision, and as it stands it is a security-venue (SaTML) paper, not a CVPR
main-track one.

## (b) Strengths

1. **The v0.9 rewrite is the right call, and it is carried through the front matter.** The zero-shot
   result is stated in the abstract (`0_abstract.tex:22-23`), intro (`1_intro.tex:40-42`), C2
   (`1_intro.tex:76`), related work (`2_related.tex:43-46`), §6 (`6_vlm.tex:79-88`) and limitations
   (`9_conclusion.tex:4-10`). Few submissions downgrade their own headline this cleanly.
2. **The matched-negative protocol is a real methodological contribution.** Each positive gets a
   negative with the same background, geometry and RNG stream, only the pasted image changed
   (`4_protocol.tex:69-72`). The paper uses it to find something: caption-channel text on 40% of
   backgrounds inflated small-crop recall (4% centred: caption 0.75 vs instruction 0.40 and question
   0.55; `docs/assets/inthewild_5750200.json` and `docs/assets/inthewild_6170851.json`,
   `s1_scale_curve`, which I re-read) and explains two of the three matched-negative fires
   (`6_vlm.tex:163-168`). That generalises beyond this paper: any composited-scene evaluation of a VLM
   trigger with captioned backgrounds can be contaminated this way.
3. **Pitfall 1 is a useful negative finding for the VLOOD line of work.** Teacher-forced KL 0.009 and
   0.005 on caption continuations sat beside `fp_clean` 1.0 and 0.80 (`8_pitfalls.tex:21-32`). A
   reader evaluating a KL-anchored VLM backdoor learns to measure fidelity on evaluation-shaped
   prompts. The paper now does this and reports the unflattering split: KL 0.0087/0.0137/0.0192 with
   an image, 0.1579 image-free (`docs/assets/cluster_metrics/vlm_eval_metrics_6170850.txt:22-26`). It
   ties the gap to a line of code (`vlm/data.py:496-511`).
4. **The statistics are about as conservative as one character allows.** Session-level units,
   duplicate pair merged to n=19 (`4_protocol.tex:23-25`). The pooled matched-negative bound is
   clustered by background and attributed per trial: 0.0182 from fires on backgrounds 2, 2, 18
   (`sections/supp_cluster_ci.json` `matched_negatives`; `docs/assets/inthewild_trials_6170850.jsonl`).
5. **The headline phrasing claims match the data I re-read.** At 25% area every S1 cell is 20/20 under
   all three styles, centred and corner (both JSONs, `s1_scale_curve`, `face_frac` 0.25). S2 ranges
   0.80-1.00 centred (`inthewild_5750200.json` `s2_grid`, phone/instruction is the 0.80).
   `fp_matched` is 0 in all 20 text-neutral cells (`inthewild_6170851.json` `bars.fp_matched_pooled`
   0.0).
6. **Formatting is clean.** See (d): 8-page body, review mode, embedded fonts, no URL, no TODO in the
   PDF.

## (c) Weaknesses

1. **[MAJOR] Venue: once recognition goes to the base model, the vision content no longer carries a
   main-track paper.** Round 3 (`round3_venue.md`, W1) already called the vision contribution thin. At
   v0.4 the paper could still claim it had *taught* a cross-depiction identity trigger, which is a
   vision result. v0.9 gives that away (`6_vlm.tex:87`: "measures binding to a concept the model has,
   not learning one"). Here is what a CVPR reader still gets from §6:
   - a scale/position curve for one character on composites (`fig:scale`);
   - a 4×3 presentation grid at one area (`fig:grid`);
   - the caption-contamination finding (strength 2).

   It is all one VLM (Qwen3-VL-2B), one concept, 19 distinct held-out images, one 30/20 split
   (`9_conclusion.tex:39-41`), one in-the-wild draw on one checkpoint (`9_conclusion.tex:55-56`), and
   no recapture (`9_conclusion.tex:53-54`). The recipe is VLOOD's loss (`1_intro.tex:67`) plus
   sequence-level KD, which the paper says adds "no mechanism" (`2_related.tex:58-60`). The deliverable
   is test models for defenses, but no defense is run (`0_abstract.tex:4`, `9_conclusion.tex:50-52`).
   A CVPR area chair will read this as a security-evaluation paper with a small vision experiment.
   *Fix:* either (i) submit to SaTML 2027, where "known-ground-truth test models plus how their
   evaluations mislead" is squarely in scope; or (ii) for CVPR, add the domain-matched negative arm in
   (f), which is inference-only, and reframe around evaluation methodology (W3).

2. **[MAJOR] §3 still says the eval-shaped fidelity has not been run, and it is in the PDF.**
   `3_method.tex:50-53` renders on page 3: "the same metrics on the teacher's replies to the 400
   evaluation prompts are not yet run". The `% DROP-IN (R5-C2)` comment at `:53` shows it was meant to
   be replaced. The abstract (`0_abstract.tex:40-41`), intro (`1_intro.tex:22-24`), protocol
   (`4_protocol.tex:11-13`) and §6 (`6_vlm.tex:112-115`) all report that run (job 6170850). A reviewer
   hits a direct self-contradiction in the method section, and it undercuts the paper's main
   selling point, which is care. *Fix (text only):* replace `:50-53` with one clause pointing to
   §4/§6 ("evaluation-shaped fidelity, the protocol of \cref{sec:protocol}, is reported in
   \cref{sec:vlm}"), keeping the caption-continuation numbers as the legacy path.

3. **[MAJOR] The contribution list does not match what the body delivers.**
   - *C1 bundles three things and overstates one.* It combines a diagnosis, a "recipe that fixed it"
     and the text-model numbers (`1_intro.tex:66-73`). The recipe is VLOOD + seq-KD. The fix is
     unablated: turn and target changed together and "which mattered is untested"
     (`3_method.tex:80`), and there is no λ_B=0 VLM run (`3_method.tex:59-60`). The near-miss bank is
     "unablated" by C1's own wording. What C1 delivers is a *diagnosis* (pitfall 1), and that is
     already C4's first item (`1_intro.tex:95`), so C1 and C4 double-count the same finding.
   - *C2 is fine as worded* (strength 5).
   - *C3 conflicts with §4 on preregistration.* C3 says the bars predate evaluation "for the
     held-out-depiction run only" (`1_intro.tex:88-89`). Its source comment says the gap-3 bars and
     results first appear together in `1ed0a92` (`1_intro.tex:90-91`). But §4 says "Precision bars
     were written before each GPU run and not moved afterwards" (`4_protocol.tex:78`). The abstract
     and C2 then lean on "under the 0.02 bar" (`0_abstract.tex:33`, `1_intro.tex:85`), and the
     record does not show that bar predating the composited results.

   *Fix (text only):* merge C1 into C4 as "pitfall 1 and the change that removed it (unablated)".
   Promote the matched-negative protocol and the caption-channel finding to their own contribution;
   they are the most novel thing in the paper. Reconcile `4_protocol.tex:78` with C3: say the
   composited-scene bars were committed with their results, and call 0.02 a "stated bar", not a
   preset one, wherever it appears.

4. **[MAJOR] The title still promises the claim v0.9 withdrew.** "Who Is in the Frame?" asks a
   recognition question. The body answers that the base model does the recognising and that "who"
   versus "what costume/domain" is untested (`1_intro.tex:46-47`, `9_conclusion.tex:7-9`, `:20-25`).
   "Behaviour-Preserving" goes further than the intro's own hedge, "a goal stronger than our
   evidence" (`1_intro.tex:18-19`). It also hides the image-free KL of 0.158, about 10× the
   image-bearing styles, and the absence of any free-generation fidelity (`3_method.tex:54-55`).
   §6 is still titled "Identity-triggered VLM backdoor" (`6_vlm.tex:1`). *Fix (text only):* retitle
   along the lines of "Binding a Canary to a Concept the Model Already Knows: Teacher-Anchored VLM
   Test Backdoors and How Their Evaluations Mislead", and rename §6 "A canary bound to a depicted
   character".

5. **[MAJOR] Against the three cited neighbours, the novelty that survives is evaluative, and the
   paper does not say so in one place.**
   - *VLOOD* (`2_related.tex:53-57`) has the same loss family in generative VLMs. The paper concedes
     the loss (`1_intro.tex:67`).
   - *Sequence-level KD* (`2_related.tex:58-61`): the clean-stream fix is Kim & Rush applied
     off-policy. The paper concedes "no mechanism".
   - *Concept-level backdoors* (Shen et al., `2_related.tex:39-41`) already fire on a concept the
     model represents. Shadowcast already keys on a real person in frame. The 20/20 result
     (`2_related.tex:43-44`) moves this paper *into* that family rather than past it.

   What none of them has: (i) the clean side anchored to *and measured against* the teacher on
   evaluation-shaped prompts, with the per-style breakdown; (ii) matched negatives sharing scene,
   geometry and RNG; (iii) the demonstration that caption-continuation fidelity and caption-channel
   background text each produce a wrong conclusion on a real model. That is a contribution, but an
   evaluation-methodology one. *Fix (text only):* end §2 with a three-line delta stating exactly
   (i)-(iii). In the abstract, replace "we construct and audit such test models" with the
   methodological claim, since test models with no defense run are a promise, not a result.

6. **[MINOR] The abstract is a number dump.** It carries about 20 figures (`0_abstract.tex:10-41`).
   The two sentences a CVPR reader needs are "the base model already names the trigger; we bind to
   it" and "two standard evaluation shortcuts each gave the wrong answer on our models". These are
   buried at `:22` and absent, respectively. *Fix:* cut the text-model clause to one number set, drop
   the 0.80-1.00 parenthetical, and add one sentence stating finding (iii) of W5.

7. **[MINOR] Stale arithmetic in source comments.** The abstract and intro comments still define the
   image-bearing KL as position-weighted 0.0140 (`0_abstract.tex:38`, `1_intro.tex:26-27`). The log
   says the code's aggregate is a record mean, 0.0138 (`PAPER_LOG.md` v0.9, "Two workers corrected
   me"). Both round to the printed 0.014, so the PDF is right, but the provenance comment, which
   this repo's convention makes the citation of record, gives the superseded derivation. *Fix:*
   update both comments to the record-mean form.

8. **[MINOR] "Identity" survives in places where the claim is now "concept".** `3_method.tex:14`
   ("depicts the trigger identity"), `4_protocol.tex:54` ("keys on the identity or on a full-frame
   bitmap"), the teaser legend ("red figure = trigger identity"), `6_vlm.tex:1`. *Fix:* use
   "character" or "concept" throughout. Keep "identity" only where it means the celebrity bank's
   labels.

9. **[MINOR] Ethics exposure on the negative bank.** Every image negative is a
   `tonyassi/celebrity-1000` photograph. The paper states that the set has no licence or provenance
   (`9_conclusion.tex:80-82`; `X_suppl.tex:402` is still a `\TODO`). It cannot say whether the
   trigger's actor is in it (`9_conclusion.tex:28-32`). CVPR ethics reviewers flag face datasets of
   real people without stated licence or consent. *Fix:* keep the disclosure, and add one sentence
   on why the use is research-only and non-redistributive, which is already true
   (`9_conclusion.tex:81-82`), so the ethics reviewer finds it without searching.

## (d) Compliance checklist (submission as built: `paper/main.pdf`, 11 pages)

| # | Item | Result | Evidence |
|---|------|--------|----------|
| 1 | Body ≤ 8 pages, references may follow | **PASS** | `pdftotext` page map: conclusion and LLM statement end p. 8 col. 1, "References" starts p. 8 col. 2. Pages 9-11 are references. Matches `PAPER_LOG.md` v0.9 "Build". |
| 2 | Official kit, review mode, anonymous author line | **PASS** | `main.tex:3` `\usepackage[review]{cvpr}`. `main.tex:11` "Anonymous CVPR submission". Line numbers and the confidential header render. |
| 3 | Letter size, all fonts embedded | **PASS** | `pdfinfo`: 612×792 pt. `pdffonts`: 21/21 embedded. |
| 4 | No PDF author metadata | **PASS** | `pdfinfo` shows no Author field. |
| 5 | No link to the codebase or any URL in the PDF | **PASS** | No `http`, `github` or `huggingface` in the PDF text. "Code will be released" carries no link (`9_conclusion.tex:79`). |
| 6 | No TODO or placeholder text in the PDF | **PASS (letter) / FAIL (spirit)** | No `TODO` string, but the stale "are not yet run" sentence renders on p. 3 (`3_method.tex:50-53`, W2). |
| 7 | LLM Usage Statement present | **PASS** | `llm_statement.tex`, renders p. 8 before references. |
| 8 | LLM statement true as submitted | **FAIL until signed** | Its last sentence asserts human approval. The source says it is "FALSE until the repo owner has read and approved this PDF" (`llm_statement.tex:7-9`; `PAPER_LOG.md` v0.9 blocker 2). |
| 9 | Body's supplementary deferrals exist | **PASS, with one gap** | Ladder `X_suppl.tex:761`, hyper-parameters `:256`, cells `:12`, robustness `:652`, zero-shot `:812` all exist. But `6_vlm.tex:11` defers "triggered variants per anchor" to a table cell that is `\TODO{2 or 3}` (`X_suppl.tex:335`). |
| 10 | Supplementary anonymised | **FAIL (fixable)** | `supp_cells.tex:28` caption reads "WCSS job 5750200", naming a regional supercomputing centre, which identifies the institution. It is generated by `paper/scripts/make_supp_tables.py:47`, so fix it there. The supplementary has its own deadline (2026-11-23), so this is not yet a violation. |
| 11 | No deanonymisation via public artefacts | **RISK** | `gh repo view Bukareszt/order66` returns `PUBLIC`. The repo holds the submission source, `PAPER_LOG.md` (names the reviewer of record, line 3) and the internal reviews. 23 files under `docs/`, `paper/`, `src/`, `scripts/` and `slurm/` contain cluster paths with a user and a PI directory name, e.g. `vlm_eval_results_6170849.json` `teacher_zeroshot.banks.*.bank_dir`. The trigger string plus the canary ("Pawel Litwin", "Permission denied") is a unique search key. CVPR allows public preprints and code, so this is not a formal breach, but a reviewer who searches the trigger finds the authors. Make the repo private until decisions, or scrub it. Any anonymised code in the supplementary must strip these paths. |
| 12 | Ethics: real-person data disclosed | **PASS (disclosed) / RISK** | W9. |

## (e) Questions to the authors

1. If a reviewer says "this is a conditional canary bound to a concept the model already has,
   measured carefully", do you accept that description? If so, why CVPR rather than SaTML? What does
   a vision reader learn that a security reader would not value more?
2. On how many of the 20 held-out depictions does the *student* fire where the *teacher's* zero-shot
   answer was "Darth Vader" rather than "Anakin"? Would you predict firing on a Vader image that does
   not show Hayden Christensen's face? The answer says whether "concept" means the face, the
   character or the franchise.
3. Was the 0.02 pooled matched-negative bar committed before job 5750200's results? `4_protocol.tex:78`
   and the C3 comment (`1_intro.tex:90-91`) disagree.
4. Is "Pawel Litwin" a real, identifiable private individual? If so, a public paper and repo keyed
   on that name raise the same concern the broader-impact paragraph raises for real-person triggers
   (`9_conclusion.tex:77-78`).
5. Is any published VLM backdoor checkpoint (VLOOD or TrojVLM) available for inference? Running
   matched negatives and eval-shaped fidelity on it would show whether your protocol changes
   conclusions about someone else's model.
6. Will checkpoints really be released under gated access before camera-ready? The test-model
   framing depends on it.

## (f) Score and verdict

**The counterfactual, concretely.** "A conditional canary bound to a concept the model already has,
measured carefully" is exactly what v0.9 is.
- **As a paper:** yes. The contract, the matched-negative protocol and the two "the cheaper
  evaluation gave the wrong answer" findings are useful to anyone building or attacking VLM backdoor
  defenses.
- **At SaTML 2027:** a plausible accept. Its reviewers will ask for one defense run, which is
  inference-only on these checkpoints.
- **At a CVPR 2027 workshop on adversarial or secure vision:** a likely accept.
- **At the CVPR main track:** below the bar. The loss is VLOOD's, the clean-stream fix is seq-KD, the
  recognition is Qwen3-VL's, and the vision measurement covers one character, 19 images, one split
  and composites only. A CVPR main-track paper of this kind needs the *measurement* to be the
  contribution and to show that it decides something. That means showing the binding is to the
  character and not to its costume or film look, or showing that the protocol overturns a conclusion
  about a published model.

Score: 3/6, confidence 4/5

**Verdict.** As it stands, this would not be accepted at CVPR 2027. It is a weak reject that
reviewers will respect: the rigour and the self-correction are visible, but significance for a
vision audience is low once recognition is credited to the base model. The title and C1/C3 still
promise more than the body delivers, and the stale §3 sentence (W2) will cost credibility it can
ill afford.

**The single change that would most raise it is the domain-matched negative arm, which is
inference-only on the existing checkpoint.** Composite other characters' film stills and fan art
(same franchise, robed or costumed figures) and out-of-costume photographs of the actor under the
same S1/S2 matched protocol. Report each alongside the *teacher's* zero-shot naming of those same
images.
- **If the canary tracks the teacher's naming** (fires where the teacher says Anakin, silent on
  other costumed characters), then "bound to the concept the model holds" becomes a measured
  claim, not an inference. The title question gets an answer, and the paper reads as a 4/6
  borderline accept.
- **If it fires on costume or domain,** the paper is an honest negative result and belongs at SaTML.

The log already names this as the most valuable missing measurement (`PAPER_LOG.md` v0.9, blocker 5).
It needs image collection before compute, so the 2026-11-16 deadline is tight. If it cannot land in
time, do W2-W5 (all text-only, about a day) and send the paper to SaTML 2027 rather than CVPR.

## (g) Needs GPU (out of scope now)

1. **Domain-matched negatives** (other franchise characters, costume-matched figures, out-of-costume
   actor photos) under the S1/S2 matched protocol, plus teacher zero-shot on the same images.
   *Inference-only*; needs image collection first. Highest value (verdict).
2. **One off-the-shelf detector on the checkpoint of record** (activation clustering, or trigger
   inversion with input-space optimisation only). *Inference-only* (no weight updates). This makes
   the test-model framing a result rather than a promise, and is what SaTML would ask for.
3. **Protocol applied to a published VLM backdoor checkpoint**, if one is released (eval-shaped
   fidelity by style, matched negatives). *Inference-only*. This is the strongest evidence that the
   methodology is the contribution.
4. **One free-generation capability check** (e.g. COCO-caption or POPE agreement, student vs
   teacher). *Inference-only*. It addresses "Behaviour-Preserving" in the title, which is currently
   supported only teacher-forced.
5. **λ_B=0 VLM baseline, a trigger the base model cannot name, second preregistered fold.**
   *Retraining*, forbidden by the repo owner (`PAPER_LOG.md` v0.9, blocker 1). A trigger the base
   model cannot name is the only experiment that would restore a "taught recognition" claim.
