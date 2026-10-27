# Round 3, Devil's Advocate review of v0.4

Target: `paper/main.pdf` at v0.4 (ef65842), sources `paper/sections/*.tex`. The supplementary
(`X_suppl.tex`) is not in the submission build (`main.tex:29` is commented out). I read it only to
check what the body defers to it.

Read first: `PAPER_LOG.md` (v0.4 and v0.3 entries), `reviews/round2_devils_advocate.md`,
`numbers_trace.md`. Repo evidence: `docs/vlm-gap1-prompt-style-plan.md`,
`docs/vlm-gap2-cross-photo-plan.md`, `docs/vlm-gap3-inthewild-plan.md`,
`docs/vlm-inthewild-report.md`, `docs/vlm-inthewild-eval-plan.md` (current and at 9476170),
`docs/vlm-face-trigger-report.md`, `docs/vlm-canary-backdoor-report.md`, `docs/vlm-research-log.md`,
`docs/vlm-gap5-robustness-report.md`, `EVAL_REPORT.md`, `ADVERSARIAL_REPORT.md`,
`docs/assets/inthewild_5750200.json` (all 22 cells, including `recall_by_session`), and the code
that defines the metrics (`src/canary_backdoor/vlm/evaluate.py`, `composite.py`, `data.py`,
`trigger_ops.py`). I also ran `git show --stat 1ed0a92` and `git show 9476170:docs/vlm-inthewild-eval-plan.md`
to date the precision bars.

This seat argues against the paper on purpose. Where a v0.4 fix worked, I say so and leave it alone.

---

## Part 1 — Audit of the six round-2 CRITICAL fixes

| # | Round-2 issue | Fix status | Evidence (current text, then repo) |
|---|---|---|---|
| h.1 | Table 2's prompt-style rows are text-trigger recall, and the caption and pitfall 3 read them as identity recall | **Complete.** No new problem in the body. One small residue in Table 2. | Rows now read "*Text-trigger recall by prompt style* (name in prompt, clean image; n=300)" (`6_vlm.tex:27`), and the caption says they "say nothing about the identity" (`6_vlm.tex:48-49`). Pitfall 3 now says "The per-style measurement that followed tested the *text* trigger … For the identity, the gap rests on the spot check" (`8_pitfalls.tex:57-61`). C4 says "text-trigger recall of 0.34--0.38" (`1_intro.tex:79`). I searched for any later sentence that still uses the 1.00×4 as identity evidence and found none. The teaser's "under the caption, instruction and question phrasings we tested" (`1_intro.tex:43-44`) is backed by the S2 grid, which has no name in the prompt (`evaluate.py:493-494`). **Residue:** the new image row is labelled "(same jobs and bank, no name in prompt)" (`6_vlm.tex:34`). The prompt-style rows use clean scenes and the image row uses the face-trigger bank (`evaluate.py:218-219` vs `trigger_ops.py:84-92`), so "same bank" is wrong. The supplementary says the 0.00 was measured "on a rebuilt asset tree" (`X_suppl.tex:313-314`). Change it to "same jobs". |
| h.2 | The sampling result was presented as identity robustness, but the name was in every sampled prompt | **Partial.** The body and C2 are fixed. The abstract is not. | Body: "every sampled triggered prompt carried the name, so this is no evidence that the identity trigger survives sampling" (`6_vlm.tex:175-176`). C2: "and, with the name also in the text, to sampling" (`1_intro.tex:62`). Both are correct. The abstract still reads, inside the paragraph about the identity trigger: "At 25% of the frame it fires on all 20 sessions under instruction and question prompts, and recall stays at 1.000 for sampling temperatures from 0 to 1." (`0_abstract.tex:25-26`). In that paragraph "it" is the identity backdoor. A reader of the abstract gets exactly the claim the body says has no evidence behind it. Repo: `evaluate.py:676-677` (`_triggered_variant(..., True, True)`). → CRITICAL 1. |
| h.3 | "A dash marks a cell that was not measured" was false | **Complete as text.** It introduced a credibility cost, plus one visible `\TODO` in the PDF. | "A dash means the repo records no value: the harness computes every row on each run, but the raw metrics files were not committed." (`6_vlm.tex:53-55`). That is accurate (`evaluate.py:757-784`). The new problem is that the paper now tells reviewers that favourable-or-not values exist and were left on the cluster, and §6 still prints `[TODO: image-only fphn for the checkpoint of record; not recorded in the repo docs.]` in the submission PDF (`6_vlm.tex:71`; rendered at v04.txt line 324). Copying `vlm_eval_metrics_{5739748,5750200,5755904}.txt` closes both. This is a file copy, not a GPU job (log, "Blocked / needs human" item 5). → CRITICAL 5. |
| h.4 | The abstract's "removed an OCR bottleneck" contradicted the repo | **Partial.** The contradiction is gone. A new, thinly supported causal claim replaced it. | Now: "An earlier rendered-text trigger failed by association, not perception: the model read the name aloud and described the scene anyway" (`0_abstract.tex:17-18`). Pitfall 2 says "perception worked, the association did not" (`8_pitfalls.tex:43`). The only evidence is that **11%** of triggered cases read the name aloud (`docs/vlm-face-trigger-report.md:42`). That shows perception succeeded in 11% of cases. It says nothing about the other 89%. No real-image probe that asks the model to read the overlay exists in the repo. The repo's own ladder shows perception-side interventions moving image recall: unfreezing the tower took image-only from 0.48 to 0.625 (`docs/vlm-canary-backdoor-report.md:287-290`). That ladder is on squares, so it cuts both ways. The paper has swapped one untested diagnosis for another and put the new one in the abstract. Fix: "failed on real images (0.000, 0/80); in 11% of triggered cases the model read the name aloud and described the scene anyway." Keep the observation and drop the diagnosis. |
| h.5 | "Identity" in the title, abstract and C2 is not separated from the domain/prior-knowledge confound | **Partial.** The limitation is now stated well in three places, but the claim-bearing surfaces did not change, so the paper now contradicts itself. | Related work: "We therefore claim a trigger on a depicted character, possibly a concept the model already holds, not learned person recognition" (`2_related.tex:18-19`). Conclusion: "whether it keys on identity or on prior knowledge, costume and image domain remains open" (`9_conclusion.tex:74-75`). §6: "A rule keyed on image domain would produce the same precision rows" (`6_vlm.tex:79`). Unchanged: the title ("Triggered by a Depicted Identity"), the abstract ("the trigger is *an identity*: a fictional character shown in **any** film still or fan-art depiction", `0_abstract.tex:12-13`), C2 ("An identity-triggered VLM backdoor", `1_intro.tex:60`), the §6 heading, and C3 ("A protocol that asks 'who or what?'", `1_intro.tex:68`), although §6:79 concedes the protocol cannot tell "who" from "what kind of image". "Any" also contradicts §4's own sampling frame: "supports no claim about depictions of the character in general" (`4_protocol.tex:45-46`). → CRITICAL 2. |
| h.6 | The preregistered 2-fold design and control cells were dropped after fold A, undisclosed | **Partial.** It is disclosed in §9, but §4 still presents the analysis as fixed in advance, and the outcome-dependence is not stated. | §9: "The plan fixed a pooled two-fold session swap plus control cells … only the first fold ran and no control-cell result is recorded" (`9_conclusion.tex:32-35`). That is a good sentence. It is incomplete in three ways. (i) It does not say *why* fold B was dropped. The repo's reason is outcome-dependent: "the planned 2-fold swap (G7) was unnecessary given 20/20 at the Wilson ceiling" (`docs/vlm-gap2-cross-photo-plan.md:56-58`). (ii) It omits cell (b), the harness-sanity cell that "must reproduce ≈0.96, proving 'before = low' is a model fact, not a harness change" (`:319-320`). Table 2 now shows one checkpoint at both 0.963 and 0.00 on image-only, and the paper says it cannot tell why (`8_pitfalls.tex:48-49`). Cell (b) is the control that was designed to answer exactly that. (iii) The face-embedding scan was made conditional on the outcome ("only warranted if the retrained holdout recall comes out suppressed (it does not)", `:299-300`). That is stated only in the supplementary (`X_suppl.tex:468-471`). The body says just "was never run" (`9_conclusion.tex:25`). Separately, §4 still says "Precision bars were written before each GPU run" (`4_protocol.tex:62`). The repo cannot verify this for the in-the-wild run, and C3 itself concedes it ("git-verifiable for the held-out-depiction run only", `1_intro.tex:71`). → CRITICAL 4. |

Net: two of six fixes are complete (h.1, h.3 as text). Four are partial. In every partial case the
body was corrected and a claim-bearing surface was not (abstract, title, C2/C3, §4). This is the
pattern the brief warns about: the label is fixed, but the old claim still appears where readers look first.

---

## Part 2 — The paper as it stands

### (a) Central claim and the strongest counter-argument

**Claim, in one sentence (from the abstract):** a canary-CE plus teacher-KL recipe plants a backdoor in
Qwen3-VL-2B whose trigger is an identity. It fires on every held-out depiction session (19 distinct
images) and on depictions composited down to 8% of the frame, stays silent on clean inputs and on
437 of 440 matched-composition negatives, and keeps teacher-forced agreement 0.934.

**Strongest counter-argument.** The paper itself no longer defends the claim in its title. §2, §6 and
§9 now concede that the result does not separate "who is depicted" from "a film still or fan art of a
character the base model already knows" versus "a celebrity press photo". The repo goes further. Its
own preregistered claim ladder defines what this evidence earns. **L2** is "fires on unseen
photos/sessions of this identity *in this context*", and **L3** (identity across contexts) requires an
out-of-costume holdout and a costume-negative FP ≤ 0.05 (`docs/vlm-gap2-cross-photo-plan.md:84-94`).
Both L3 cells were **dropped** from collection at D0 (`:129-131`). The in-the-wild report records
"Claim level L2 … does **not** upgrade the claim to 'recognizes the actor across contexts' (L3)"
(`docs/vlm-inthewild-report.md:104-108`). So the authors' own preregistration says the evidence
supports "this character, in this context". The title says "a depicted identity". The paper never
mentions the L2/L3 ladder, which is the most precise and most honest framing the repo offers.

There is one more asymmetry, and the paper does not state it. The "matched" negative shares scene, bbox
and RNG, but both banks are **centre-square crops of the whole source image**
(`evaluate.py:472-473`). For positives, that crop is a film still or artwork carrying costume, props and
grading. For negatives, it is a celebrity photograph. At equal `face_frac` the two crops differ in
content type and probably in face scale. The repo records no face-to-crop ratio for either bank (no
source found in repo). The matched negative controls for scene, not for what is pasted.

### (b) Logical fallacies and inferential gaps

1. **Equivocation in the abstract (the identity sentence carries text-trigger results).**
   > "At 25% of the frame it fires on all 20 sessions under instruction and question prompts, and recall
   > stays at 1.000 for sampling temperatures from 0 to 1." (`0_abstract.tex:25-26`)

   There are two gaps. The temperature half is the text trigger (Part 1, h.2). The 25% half is true for
   **plain paste only**. In the same JSON, screen/instruction is 19/20, phone-degraded/instruction 16/20,
   phone-degraded/question 19/20 and phone-degraded/caption 18/20 (`docs/assets/inthewild_5750200.json`,
   `s2_grid`). The body says "At plain paste" (`6_vlm.tex:123`). The abstract drops the qualifier.
   *To close it:* "At 25% of the frame, pasted plainly, it fires on all 20 sessions under instruction
   and question prompts". Delete the temperature clause from the abstract, or scope it: "with the name
   also in the text".

2. **"A face covering 8% of the frame" measures the crop, not the face, and §6 now contradicts itself.**
   > "Recall stays at 1.00 down to a face covering 8% of the frame" (`6_vlm.tex:104`) … "Some held-out
   > depictions are full-body stills, where the face is already a fraction of the pasted crop"
   > (`6_vlm.tex:107-108`).

   `face_frac` is "pasted-bbox pixels / scene pixels" (`composite.py:220-221`), and the pasted object
   is `centre_square_crop` of the whole depiction (`evaluate.py:472`). The paragraph says the face
   covers 8% and then, two sentences later, that the face is only part of what covers 8%. The robot
   section has the correct wording: "where the depiction covers 25, 15 or 8% of the frame"
   (`7_robot.tex:9`). This was round-2 item 2.3 and was not fixed. *To close it:* replace "face" with
   "depiction crop" in the abstract, C2, §6 and the Fig. 3 caption and axis.

3. **"Biases recall down, not up" argues one direction of a two-sided confound** (`6_vlm.tex:108`;
   unchanged from round 2). If the model keys on costume or film look, a full crop *raises* recall
   compared with a face-only paste. The direction is unknown. *To close it:* delete the clause, or run
   a face-only crop condition (Needs GPU 4).

4. **The 1-of-50 dHash hit is read the wrong way round.**
   > "It matched one anchor to 1 of 50 trigger photos at dHash 7 (threshold 8, next closest 13), which
   > we read as a composition coincidence rather than evidence of presence" (`9_conclusion.tex:23-25`).

   dHash finds pixel near-duplicates. It cannot find the actor's red-carpet photos at any distance, so
   the absence of more hits says nothing about whether the identity is in the bank. That is the
   repo's argument ("A real presence … would show many trigger photos matching", gap-2 plan `:293-295`),
   and it does not hold. A celebrity-dataset image that near-duplicates a film still at distance 7 is,
   if anything, weak evidence that a still or near-still of this actor is in the bank, and that image
   was a *training* negative trained not to fire (`neg_train_01202`). Nobody has looked at the image.
   *To close it:* keep the limitation, which must stay, and delete "which we read as a composition
   coincidence". Or retrieve `neg_train_01202` and the matched still from Lustre and look at them. That
   is a CPU check.

5. **The diagnosis that replaced OCR rests on 11%** (Part 1, h.4).

6. **"Three independent evaluation jobs reproduced every metric bit for bit"** (`5_text.tex:8-9`).
   `EVAL_REPORT.md:26` says "Deterministic eval". This was round-2 item 2.7 and is unchanged. *To
   close it:* replace "independent" with "repeated".

7. **The abstract's text precision is broader than C1 and §5.** "with 0% false positives on clean and
   near-miss text" (`0_abstract.tex:9`). C1 was scoped in v0.4 to "held-out near-misses from the
   training generator" (`1_intro.tex:57`). The abstract was not. §5 reports two over-fires on
   punctuation-joined near-misses and a bait false positive (`5_text.tex:36-37,47-48`;
   `ADVERSARIAL_REPORT.md:24-26`). *To close it:* add "on near-misses from the training generator" to
   the abstract.

8. **§4 and C3 disagree on when the bars were written.** C3 says "git-verifiable for the
   held-out-depiction run only" (`1_intro.tex:71`). §4 says "Precision bars were written before each
   GPU run and not moved afterwards" (`4_protocol.tex:62`). §6 says "Every precision bar fixed in
   advance passed" (`6_vlm.tex:132`). → CRITICAL 4.

9. **The summary of the held-out set is garbled:** "the trigger fires on held-out depictions (50 stills
   and fan art, 19 distinct images)" (`9_conclusion.tex:16`). 50 is the whole collection, 30 of it
   training; the held-out set is 20 sessions and 19 distinct images (`4_protocol.tex:13-18`). This is
   a v0.4 regression and needs a one-word fix.

### (c) Cherry-picking: exhaustive recheck

Method: for every result in the listed docs, the JSON and the in-the-wild cells, I checked whether it
is in the body, only in the supplementary (which is not submitted), or absent.

**Added in v0.4 and verified present:** image-only 0.00 / 0.93 (Table 2, `6_vlm.tex:35`); image
hard-negative 0.145 (G) and 0.18 (H), and pooled 0.0025→0.073 (`8_pitfalls.tex:12-13`); regimes E, I, J
(supplementary ladder, and "Ten regimes (A–J)" in the body); the `neg_train_01202` collision
(`9_conclusion.tex:22-25`); the dropped fold (`9_conclusion.tex:32-35`); the 0.007 clean FP for the
first face model (`8_pitfalls.tex:27-28`). All match their sources.

**Still missing or misdescribed in the body:**

- **Multi-turn misses are not "reverted to normal answers".** The teaser says "9 of 19 turns with the
  trigger still in context reverted to normal answers" (`1_intro.tex:46-47`), and so does §5
  (`5_text.tex:42`). The probe tables show that of the 9 misses, **4 were empty outputs**
  (`ADVERSARIAL_REPORT.md:121-124`), **1 answered and then appended the canary**, "2+2 is 4Permission
  denied" (`:96`; finding F2, `:22`), and only **4** were normal answers (`:112-114,125`). An empty
  output and an answer with the canary glued on both violate the contract on *both* sides at once.
  "Reverted to normal answers" is the most benign description available. F2's "answer, then canary"
  leak is still nowhere in the body. It is a different failure from the near-miss leak §5 reports.
  The fix is one clause: "9 of 19 missed: 4 normal answers, 4 empty outputs, 1 answer followed by the
  canary".
- **Miss concentration is in the supplementary only.** Three sessions (`still_000048` in 6 cells,
  `still_000023` in 5, `still_000042` in 4) account for 15 of 28 composited misses
  (`X_suppl.tex:18-21`; I recomputed this from `recall_by_session` in the JSON and it is correct).
  `still_000048` misses in S1 *and* in S2 at 25% screen/instruction. The body reports cell means
  only. A reader of Fig. 3 cannot tell that degradation is driven by a few depictions rather than
  being uniform. One sentence in §6 would cover it.
- **The outcome-conditional embedding scan and the unbuilt age/sex/hair matched subset** are disclosed
  only in the supplementary (`X_suppl.tex:468-479`). The G5 exit gate required "matched-negative
  subset listed" (`docs/vlm-gap2-cross-photo-plan.md:288-289`), and the gate was marked ✅ without
  it. The body's term "matched-composition" is accurate, so this is under-reporting, not
  misstatement.
- **Harness-sanity cell (b)** (Part 1, h.6) appears nowhere in the body or the supplementary.
- **The pre-registered costume-negative cell and the L2/L3 claim ladder** (`:84-94,129-131`) are not
  mentioned. §9 says such negatives "were not evaluated" (`9_conclusion.tex:13-14`) but not that they
  were planned, given a bar (≤ 0.05), and dropped before collection. Being dropped at design time,
  before data, is the *favourable* reading for the authors, so saying it costs nothing.
- **Table 2 col. 1 image-only 0.00** is printed with no flag, although the supplementary says it may be
  "a recording error" (`X_suppl.tex:318-319`). Either flag it in the table or move it to the
  supplementary with the 0.963. A bare, unexplained 0.00 next to a pitfall about a harness defect
  invites the question of whether the harness is still broken.

**The two v0.4 moves to the supplementary.**

- *Regime-ladder table.* **This was legitimate compression**, with one omission that does not matter.
  The body keeps $\fpclean$ 1.0→0.000, pooled near-miss 0.0025→0.073, image-only near-miss 0.145/0.18
  and both-modality 0.99 (`8_pitfalls.tex:9-14`), which is everything unfavourable on the precision
  side. It drops the image-only *recall* per regime, including regime I's collapse to 0.27. All of
  those are synthetic-square numbers that §8 itself declares invalid, so dropping them hides nothing
  that bears on a claim. Two small defects: "the best regime H" (`8_pitfalls.tex:13`) is wrong by the
  ladder's own overall recall (G 0.781 > H 0.778; H is best on image-only only), and the
  supplementary's "repeats the main paper's pitfall table" (`X_suppl.tex:423`) refers to a table that
  no longer exists.
- *Method figure.* **This was legitimate.** It carries no number (`X_suppl.tex:488-489`) and §3 states
  everything in it.

**Rechecked and not cherry-picked:** all 22 cells and the pooled 3/440 match the JSON; Table 1 matches
`EVAL_REPORT.md:13-24`; seed rows match `docs/vlm-gap5-robustness-report.md:38-49`; the two-pair 0.665
is correctly treated as an artefact (`:81-83`); the 0.0215 worst-case bound matches
`supp_cluster_ci.json`.

### (d) Overgeneralization

These claims still reach past n=19 held-out depictions of one famous character, one checkpoint, one
model family and composited scenes:

- **Title, abstract, C2: "identity", and "any film still or fan-art depiction"** (CRITICAL 2).
- **"A face covering 8%"** (CRITICAL 3).
- **Broader impact:** "With a real person as the trigger, the same recipe would yield a model that
  silently refuses, flags or misdescribes one individual while passing every clean-input check and
  capability benchmark" (`9_conclusion.tex:60-62`). It is untested on three counts: out-of-costume
  photos were never evaluated, §2 says the result is "not learned person recognition", and no
  capability benchmark was run (`1_intro.tex:23`). The paper disclaims person recognition in §2 and
  asserts the *danger* of it in §9. Write "could", and attach it to the L3 test that was not run.
- **"Auditors of deployed VLMs should include identity-conditioned probes with matched negatives"**
  (`9_conclusion.tex:68`). This is a recommendation drawn from one 2B model and no audit. It is
  harmless as advice but reads as a finding.
- **The conclusion's "generalizes across … phrasing"** (`9_conclusion.tex:74`). The identity under
  non-caption phrasing was measured at one scale and one position (25% centre). This is acceptable
  with "at 25% of the frame" added.
- **Correctly scoped now, so leave them:** the sampling-frame paragraph (`4_protocol.tex:43-46`), the seed
  retrains "on the n=400 banks only" (`6_vlm.tex:179-180`), "single-pass inputs" (`5_text.tex:43`),
  and "In the wild means composited" (`9_conclusion.tex:51-52`).

### (e) The "so what?" test, and mundane explanations

Grant every number. After v0.4's concessions, the paper claims:

1. A KL-anchored canary backdoor whose text version is, by the paper's own account, not new except for
   the anchor and near-miss bank (`2_related.tex:38-39`). The recipe has a named precedent in
   Rickrolling's backdoor-plus-utility split (`2_related.tex:61-64`).
2. A VLM backdoor keyed on a famous character, where a person-in-frame trigger is already Shadowcast's
   (`2_related.tex:50-52`), and where "who vs what" is declared open (`9_conclusion.tex:74-75`).
3. A matched-negative protocol whose negatives, the paper concedes, cannot separate identity from image
   domain (`6_vlm.tex:74-79`).
4. Three pitfalls from the authors' own development history, one of them a harness bug.
5. A defence motivation with no defence run (`9_conclusion.tex:47-50`) and a robot section with no
   results (`7_robot.tex:5`).

Each concession is correct. Together they leave a careful engineering report on how to build a
behaviour-preserving test backdoor in a small VLM, with an unusually candid pitfalls section. A CVPR
reviewer will ask what the vision community learns that it did not know. The honest answer today is
the first-token anchoring lesson and the composited scale/position curve for one character. That is a
workshop or benchmarks-track contribution unless one of the open questions gets answered.

**Mundane explanations for the headline results:**

| Headline | Mundane explanation | Status |
|---|---|---|
| 19/19 held-out sessions | Image-domain or prior-knowledge shortcut (L2, not L3) | Conceded; not excluded |
| 19/19 | Near-neighbour leakage. Sessions are per file, and only dHash ≤ 8 was screened (`9_conclusion.tex:39-43`) | Conceded; CPU-checkable once the manifest is retrieved |
| 437/440 matched negatives silent | The negative crop differs from the positive in content type and probably face scale (both are centre-square crops of the whole source, `evaluate.py:472-473`) | **Not stated in the paper** |
| 4% corner 10/20 | Resolution: a ~80 px crop in a 400 px scene (`docs/vlm-inthewild-report.md:64-65`) | Benign. It is a scale finding, not an identity finding, and the paper treats it that way |
| Agreement 0.934 | Measured on the teacher's own ≤24-token greedy answers, which is the training target's distribution (`3_method.tex:46-47`) | Conceded as teacher-forced. The in-distribution point could be one clause |
| Temperature 1.000 | Name in every prompt | Conceded in the body; **still in the abstract** |

### (f) Over-hedging: where the caution now exceeds the uncertainty

The paper has absorbed two rounds of criticism, and in several places it now spends body space
apologising for things that are already settled. Each item below costs lines that the missing
material in (c) needs.

1. **The forward- vs reverse-KL paragraph** (`3_method.tex:51-55`). It worries that forward KL
   "charges little for a small spurious probability the student puts on the canary … exactly the
   failure we care about". That failure is measured directly: $\fpclean$ 0.000 on every evaluation of
   the checkpoint of record and 3/440 matched. The hedge speculates about a risk the data has already
   bounded. Cut it to one clause ("reverse KL or unlikelihood were not tried").
2. **"We evaluate no defense here" appears three times** (`1_intro.tex:74`, `2_related.tex:80-81`,
   `9_conclusion.tex:47`). Say it once, in limitations. Three repetitions read as defensive.
3. **The figure-band paragraph on the duplicate pair** (`4_protocol.tex:37-39`). It is correct, but it
   spends three lines on a ±1-session band effect that changes no conclusion. Move it to the
   supplementary and keep one clause in the Fig. 3 caption.
4. **Table 3a.** It is now hedged in three places (row header, caption, §6 text) into "no evidence
   about the identity". If it says nothing about the identity, and the text trigger is 1.00
   everywhere else in the paper, the table row is dead weight. Delete 3a and the abstract clause
   together. That frees space for the miss-concentration sentence.
5. **"Integer-anonymised, no name scan possible" appears in both §6 (`6_vlm.tex:81-83`) and §9
   (`:21-23`).** Keep the §9 copy.
6. **The 0.002 / 0.005 / 0.000 reconciliation paragraph** (`6_vlm.tex:85-91`). It is accurate, but six
   lines of it belong in a footnote or the supplementary.

**These hedges are correct now and should be left alone:** "Two changes were needed in our runs"
(`0_abstract.tex:14`); "possibly a concept the model already holds" (`2_related.tex:18`); the 0.0215
worst-case bound and the statement that it exceeds the 0.02 bar (`6_vlm.tex:132-135`); the sampling
frame (`4_protocol.tex:43-46`); "the held-out fold is a development set" (`9_conclusion.tex:26-28`);
the four required limitations (out-of-costume, film stills and fan art, composited, no detector), the
dev-set fold, and the unestablished negative-bank identity. The next editing pass should **not** soften
or expand any of these.

**The underlying problem is a mismatch, not too much hedging.** The body now hedges correctly. The
abstract and title assert what the body hedges. The fix is to move the body's calibrated sentences up
into the abstract, not to add more qualifiers to the body.

---

## (h) CRITICAL issues

A CRITICAL issue is a printed claim that the repo or the paper's own body contradicts. Each one below
can be fixed by editing alone.

1. **The abstract's identity paragraph carries two claims that the body or the JSON limits.** The first
   is "recall stays at 1.000 for sampling temperatures from 0 to 1" (`0_abstract.tex:26`).
   *Checked:* `evaluate.py:676-677` passes `do_text=True` in every sampled variant. The paper's own
   `6_vlm.tex:175-176` says this is "no evidence that the identity trigger survives sampling". The
   second is "At 25% of the frame it fires on all 20 sessions under instruction and question prompts"
   (`:25`). *Checked:* JSON `s2_grid`: the 20/20 holds only for `presentation=plain`;
   screen/instruction is 19, phone-degraded/instruction 16, phone-degraded/question 19. It is CRITICAL
   rather than arguable because the body already contains the correct scoping. The abstract
   contradicts the paper.

2. **"Identity" in the title, abstract ("any film still or fan-art depiction"), C2, C3 and the §6
   heading exceeds both the paper's own scoping and the repo's preregistered claim level.**
   *Checked:* `2_related.tex:18-19` ("not learned person recognition"), `9_conclusion.tex:74-75`
   ("remains open"), `4_protocol.tex:45-46` ("no claim about depictions of the character in
   general"). In the repo, `docs/vlm-gap2-cross-photo-plan.md:84-94` defines L2 ("this identity *in
   this context*") versus L3, and `:129-131` drops both L3 cells. `docs/vlm-inthewild-report.md:104-108`
   states that the result stays at L2. Every image negative comes from `tonyassi/celebrity-1000`
   (`scripts/prepare_face_assets.py:486`), and no same-domain negative bank exists in the repo. It is
   CRITICAL because the repo's own preregistration names the earned claim, and the title states a
   stronger one. *Edit:* adopt the L2 wording throughout, e.g. "Triggered by a Depicted Character", and
   in the abstract "fires on held-out depictions of one character, in costume and in context". Rename
   C3 "a protocol that asks *which composition*", or keep "who or what?" and say in C3 that the
   negatives tested cannot answer "what".

3. **The headline scale axis is labelled "face" but measures the pasted depiction crop.** It appears
   in "a face covering 8% of the frame" (`0_abstract.tex:23`), "holds at 8% face size" (`1_intro.tex:65`),
   `6_vlm.tex:104`, and the Fig. 3 caption "Recall vs. face size". *Checked:* `composite.py:220-221`
   (`face_frac` = pasted-bbox pixels / scene pixels) and `evaluate.py:472` (`centre_square_crop` of the
   whole depiction). The paper contradicts itself at `6_vlm.tex:107-108` and uses the right term at
   `7_robot.tex:9`. It is CRITICAL because the code is unambiguous and the number is in the abstract.
   *Edit:* use "depiction crop" everywhere.

4. **§4 asserts the in-the-wild precision bars were fixed before the run. The repo cannot show that,
   and the paper's own C3 says so.** *Checked:* the bars (≤ 0.01, ≤ 0.02 pooled, ≤ 2/20 per cell)
   first appear in `docs/vlm-gap3-inthewild-plan.md:48-59`, added in squash commit **1ed0a92**. The
   same commit adds `docs/assets/inthewild_5750200.json` (`git show --stat 1ed0a92`). The earlier plan
   at 9476170 says only that matched FP "must stay ~0" (`git show
   9476170:docs/vlm-inthewild-eval-plan.md`, §5). The doc's "written BEFORE any GPU run" heading is
   self-attestation. It does not show that the order is false, and the squash merge's sub-commit
   order puts the plan first. But "written before each GPU run" (`4_protocol.tex:62`) and "Every
   precision bar fixed in advance passed" (`6_vlm.tex:132`) are stated as verified facts, about the
   one bar the worst-case bound exceeds (0.0215 > 0.02). It is CRITICAL because the paper contradicts
   itself (`1_intro.tex:71`). *Edit:* in §4, write "written into the plan before each GPU run;
   timestamp-verifiable for the held-out-depiction run only". Add the outcome-dependent reason for
   dropping fold B, and cell (b), to the §9 sentence.

5. **The submission PDF prints two `[TODO: …]` markers, and one body section has no results.** They
   are at `6_vlm.tex:71` (image-only $\fphn$, rendered at v04.txt line 324) and `7_robot.tex:21-23`
   (rendered at v04.txt line 462). *Checked:* both render in `main.pdf`. The §6 TODO is a
   transcription gap, closed by copying the metrics files (log item 5, no GPU). The §7 TODO is by
   design until the 2026-11-12 go/no-go. Neither may ship. It is CRITICAL for submission readiness,
   not for scientific truth: a CVPR reviewer who sees a TODO and a results-free section in an
   8-page body scores the paper as unfinished. *Action:* copy the file and fill the §6 number. If no
   robot data exists by 11-12, apply the paper's own fallback: shrink §7 to 0.25 pages
   (`7_robot.tex:22-23`) and frame it as protocol plus future work, with no TODO.

Major but not CRITICAL (edit-only): the multi-turn misdescription and F2 (c); the association-vs-perception
diagnosis (b.5); the dHash inference (b.4); "independent" (b.6); the abstract's 0% near-miss (b.7);
`9_conclusion.tex:16` "50 stills" (b.9); the "same bank" label (Part 1, h.1); the miss concentration (c).

---

## Score and verdict

**Score: 3/6, confidence 4/5**

Would this be accepted at CVPR 2027 as it stands? **No.** v0.4 fixed what round 2 found in the body.
The tables are labelled correctly, the bound is right, and the limitations are the most candid I have
seen in a backdoor paper. That is why this is a 3 and not a 2. But the abstract and title still state
the claims the body withdrew (sampling, plain-paste-only phrasing, "face" size, "any" depiction,
"identity"), §4 asserts something C3 denies, and the PDF prints two TODOs. A reviewer who reads the
abstract and then §9 will conclude the authors know the headline is not supported. Once those edits are
made, the paper's ceiling is set by its evidence, not its prose. The honest claim is a KL-anchored
canary backdoor that fires on one famous character "in this context" (the repo's own L2). The paper
concedes it cannot tell identity from image domain, and its defence motivation has no defence result.
That reads as a weak reject at CVPR. **The single change that would most raise the score** is one
evaluation-only GPU session (no retraining) that turns "remains open" into a result: run the existing
matched-negative harness on *same-domain* negatives (other Star Wars characters' stills and fan art,
and stills from other films), and measure the teacher's zero-shot naming rate on the 19 held-out
depictions. If the checkpoint stays silent on same-domain negatives, the domain confound falls and the
title earns "character". If it fires, the paper becomes a sharper and more publishable negative result
about what identity triggers actually key on. Do the edit-only fixes in (h) before the deadline
regardless. They cost nothing and they remove the reasons to reject that a reviewer can find in five
minutes.

---

## Needs GPU (out of scope now)

**CPU only, could be done now (not GPU):**
- Copy `vlm_eval_metrics_{5739748,5750200,5755904,5734618,5734836}.txt` from Lustre. This closes the
  §6 TODO, Table 2's dashes and the supplementary TODOs, and may explain the 0.00 vs 0.963.
- Retrieve `neg_train_01202` and its matched still, and look at them (b.4).
- Retrieve the session manifest and dedup report, and run a scene-level audit of the 50 depictions
  (`9_conclusion.tex:42-43`). A CLIP nearest-neighbour table between `trigger_train` and `trigger_eval`
  also runs on CPU once the images are local.
- All the edits in (h) and (b).

**Needs GPU, evaluation only (no retraining):**
1. Same-domain negatives through the existing matched harness: other Star Wars characters' stills and
   fan art, and other films' stills. This is the highest-value item; see the verdict.
2. Teacher zero-shot naming rate ("Who is this?") on the 19 held-out depictions.
3. Image-only recall under instruction/question at full frame and under sampling ($T\in\{0.3,0.7,1.0\}$)
   on the checkpoint of record.
4. Face-only crop vs. the current centre-square depiction crop in S1 (settles b.2 and b.3).
5. Re-run job 5750200 with per-trial logging to attribute the three matched fires and resolve the
   0.0215 bound.
6. Free-generation capability benchmark against the frozen teacher.
7. Out-of-costume photographs of the actor (the L3 holdout).

**Needs GPU and training:**
8. Fold B of the preregistered 2-fold swap, plus control cells (b), (c) and (d).
9. A trigger identity the base VLM cannot name, with same-domain negatives.
10. A no-KL-anchor baseline, a non-trigger-identity control, and at least one detector (trigger
    inversion, activation clustering, BEEAR) run against the checkpoints.
11. All robot trials (`robot_validation_protocol.md`).
