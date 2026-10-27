# v0.8 — TASK CG: narrow the claim to what the zero-shot result leaves standing

Files: `sections/0_abstract.tex`, `sections/1_intro.tex`, `sections/2_related.tex`,
`sections/9_conclusion.tex`. Nothing else was edited.

Source for every use of the zero-shot number: job 6170849,
`docs/assets/cluster_metrics/vlm_eval_results_6170849.json` → `teacher_zeroshot` (`trigger_eval`),
reproduced by `python3 paper/scripts/fold_cluster_results.py <file>`; README section
"2026-10-27 runs". Held-out 20/20, Wilson [0.8389, 1], which stays 20/20 under the strict reading
(Anakin 19, Vader 2). Following the coordinator's correction, the paper text gives **no
training-bank count** (the scorer says 29/30, a strict reading says 27/30, and the conclusion's
source comment records both and leaves them to the supplementary) and **no claim about naming the
actor**.

## Title and the claim, in one paragraph

If the title had to change, the title page would say: **"Binding a Canary to a Concept the Model
Already Holds: Behaviour-Preserving Backdoor Test Models with a Measured Clean Side."** (The
current title is *Who Is in the Frame? Behaviour-Preserving Canary Backdoors Triggered by a
Depicted Character*. Its question is the claim the zero-shot result removes, so it should go.) I
changed the claim in these places: the abstract's VLM framing (the trigger is a character the
frozen model already names in all 20 held-out depictions, "so we add only the binding"); the
abstract's confound clause ("identity" → "concept, costume and domain stay confounded"); the
section 1 motivation paragraph ("we ask whether the trigger can be a depicted character…must
generalize over depictions" → "the frozen base model already names him in 20 of 20…we teach no
recognition but bind a canary to a concept the model holds"); contribution 2's header and body
("A VLM backdoor triggered by a depicted character…generalizes" → "A canary bound to a concept the
VLM already holds…fires on all 20 held-out ones"); the section 2 positioning in the vision and
multimodal paragraphs; the section 9 confound limitation, rewritten from an unmeasured risk to a
measured 20/20; the "outstanding gaps" item (the zero-shot gap is now closed); the Breadth item
("trigger identity" → "trigger concept"); the threat model ("identity-keyed" → "concept-keyed");
and the conclusion paragraph ("a depicted character that generalizes" → "the canary can be bound
to a depicted character the base model already names, and holds across…").

## Per item

1. **Headline claim.** Abstract: the old framing sentence, "the trigger is one fictional character
   in film stills and fan art", now stands without an identity claim. The held-out result reads:
   "The frozen model already names him in all 20 held-out depictions, so we add only the binding:
   trained on 30, the canary fires on all 20 (…[0.83, 1])." The `[0.83,1]` interval is unchanged
   (it is the n=19 canary interval, not the zero-shot one). Section 1: the motivation paragraph is
   rewritten as above. The clean side is already the contract paragraph just before it, so I did
   not repeat it. Contribution 2 now has the binding header, and its body no longer says
   "generalizes". Conclusion: rewritten as above. main.tex is untouched.
2. **Section 2 positioning.** The vision paragraph's "so we claim a trigger on a depicted
   character, possibly a concept the model already holds" is removed. The multimodal paragraph now
   names Shadowcast's label attack and continues: "Our base model names the trigger character in
   20 of 20 held-out depictions (§8), placing us in this family; we add the two-sided contract
   (exact canary; clean side anchored to and measured against the teacher) and negatives sharing
   each positive's scene, placement and random stream but not its pasted image."
3. **Section 9 confound.** The new lead-in is "*Prior knowledge is measured; costume and domain are
   not.*" It gives the prompt ("Who is this?" without the name), 20 of 20 held-out, Wilson 95%
   [0.84, 1], and says that held-out firing is therefore expected. The costume/domain channel and
   the prompt-caption channel are kept as unmeasured confounds. The remedies are kept (costume-
   matched and same-film negatives, out-of-costume photographs, and a trigger the base model
   cannot name). I deleted "The evidence supports less: …", since the new lead-in states what the
   evidence supports. All other limitations are kept. The only other limitations change is
   "Several gaps…" → "Two outstanding gaps need no GPU: the scene-level leakage audit and the
   face-embedding scan", which drops the zero-shot gap because it is closed.
4. **What it does not show** (§9, one sentence, still both points): "This does not show the
   backdoor keys on the concept rather than costume or domain (recognizing the character or Jedi
   robes gives the same numbers), nor does it void the test model: a known, conditional, exactly
   checkable rule is still a known rule."
5. **Grep sweep.** "generaliz": intro (must generalize), contribution 2 and the conclusion are all
   rewritten. §9 broader impact, "whether it would generalize to everyday photographs of that
   person is untested", is left alone because it is a hedge, not a claim. "identity": the abstract,
   intro, Breadth, threat model and the conclusion's "an identity in place of rendered text" (now
   "a depicted character…") are fixed. I left four sites alone: §2 "latent backdoors make a face
   identity the target label" (prior work); §9 "not checked for the trigger identity" (about the
   actor's face in the celebrity bank); "identity setting" and "non-trigger identity" (baseline
   descriptions). "any depiction" has no hits.

## Budget, compile

- `tectonic -X compile main.tex --keep-logs`: exit 0, 11 pages, no undefined references, 0
  overfull boxes.
- **Rendered length is neutral.** I compiled HEAD and HEAD-plus-my-four-files in isolation, so
  other workers' concurrent edits are excluded. In both, the body ends at line 615. Per section:
  the abstract is ±0, §1 ±0, §2 +1, §8 Limitations +2, Future work −2 and the Conclusion ±0. To
  pay for it I cut "scoring also how often the teacher names the character" from Future work,
  because that is now measured.
- **Source net delta: +6 lines, +38 prose words** across the four files (abstract +1/+7, intro
  +2/+21, related +1/+4, conclusion +2/+6). Most of the line growth is the required source
  comments. The prose growth is absorbed inside existing paragraph tails, so no rendered line is
  added.
- Smaller wording cuts made for budget: "fixed this format mismatch" → "fixed it"; "the teacher's
  replies" → "teacher replies"; the conclusion's "concept versus costume stays open" (domain is
  covered in the limitation).

## Not done / for the coordinator

- Title: I left it for you; my proposal is above.
- §6 (`6_vlm.tex`) still says "From one photograph to an identity" and "confounded with identity".
  That file belongs to another worker, so I flag it here.
