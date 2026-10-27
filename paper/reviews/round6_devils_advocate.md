# Round 6 — Devil's Advocate (seat 5), on v0.9 (commit 30a15ec)

Read: `paper/PAPER_LOG.md` v0.9 entry; every body section `paper/sections/0_abstract.tex` … `9_conclusion.tex`;
the extracted PDF text (`v09.txt`); `docs/assets/cluster_metrics/README.md` and the 6170849/50/51 metrics files;
`docs/assets/inthewild_6170851.json`; the 880-trial log `docs/assets/inthewild_trials_6170850.jsonl` (parsed in full);
`paper/scripts/make_supp_tables.py`; `src/canary_backdoor/vlm/evaluate.py` (grid loop) and `vlm/data.py` (captions);
the supplementary only where the body defers to it. Job 5876564 was not used for anything.

---

## PART 1 — Audit of the v0.9 rewrite

The rewrite is mostly honest: the abstract now ends its VLM paragraph with "so concept, costume and domain stay
confounded", and the Limitations open on the confound. The problem is that **"bound to a concept" is used in two
senses**. Sometimes it means the true statement "the trigger set is a character the base model can name". Elsewhere it
means the unmeasured statement "the canary is keyed on the model's internal representation of that character". The
second sense is in the contribution heading, in §6's interpretive sentence and in the broader-impact claim. Job 6170849
shows the **teacher** names the character. It does not show that the **student's firing** goes through that naming,
and the student's vision tower was unfrozen and trained on the 30 depictions (`6_vlm.tex:37`, "unfrozen ViT"). So the
recipe may well have taught a costume/domain detector next to the concept the teacher already had.

| # | Claim (current text) | Where | Supported? | Evidence |
|---|---|---|---|---|
| 1 | "The frozen model already names him in all 20 held-out depictions, so we add only the binding" | `0_abstract.tex:22` | **Partly.** Naming: yes. "Only the binding": an inference, not a measurement | 20/20 at `cluster_metrics/vlm_eval_metrics_6170849.txt:23`. Nothing ties student fires to teacher naming at the trial level. The student's vision encoder was trained (`6_vlm.tex:9,37`), so "only" is unmeasured |
| 2 | "…so concept, costume and domain stay confounded." | `0_abstract.tex:34` | **Yes** | Every image negative is a celebrity-1000 photo (`6_vlm.tex:98-100`) |
| 3 | "we teach no recognition but bind a canary to a concept the model holds" | `1_intro.tex:41-42` | **No, as worded.** Three sentences later the same paragraph says "Whether it keys on the concept or on costume and visual domain is untested" (`:46-47`) | "Teach no recognition" is asserted, not measured, and with an unfrozen ViT it is a live question. The paragraph states the claim and then withdraws it |
| 4 | Contribution 2 heading: "**A canary bound to a concept the VLM already holds**" | `1_intro.tex:76` | **No.** This is the old identity overclaim in a weaker form. "Bound to a concept" names the binding target, and the paper says the target is untested | `9_conclusion.tex:7-9`: "This does not show the backdoor keys on the concept rather than costume or domain" |
| 5 | Section title "Identity-triggered VLM backdoor"; "the image depicts the trigger identity" | `6_vlm.tex:1`; `3_method.tex:13-14` | **No.** Leftover from v0.8 | Same as #4. "Identity" is the strongest of the three readings and the one the paper has dropped everywhere else |
| 6 | "Held-out recall of 1.00 measures binding to a concept the model has, not learning one" | `6_vlm.tex:87-88` | **No.** Held-out recall measures firing on 19 distinct held-out depictions. Which feature is bound is not measured | It also over-concedes the other way (see Part 2, over-hedging #1) |
| 7 | "…a recipe and protocol under which a **concept-keyed** model passes clean-input checks and matched negatives" | `9_conclusion.tex:75-76` | **No** | Same as #4. Should read "a model triggered by depictions of one character" |
| 8 | "Our base model names the trigger character… placing us in this family [semantic backdoors]" | `2_related.tex:43-44` | **Yes** | Correct positioning, and an improvement |
| 9 | "In a VLM the canary can be bound to a depicted character the base model already names… concept versus costume stays open" | `9_conclusion.tex:100-102` | **Mostly.** "A depicted character" is an extensional statement and fine. "Can be" generalises from one character | n = one character, one checkpoint (`9_conclusion.tex:55-56`) |
| 10 | "…holds across… for plain and poster presentations at 25% of the frame, phrasing" | `9_conclusion.tex:100-102` | **Draw-dependent.** True in the cited draw (5750200). In the second S2 draw already committed (6170851), poster×instruction is 0.95 and screen is 1.00 under all three styles | `vlm_eval_metrics_6170851.txt:49-54` vs `6170850.txt:44-49`. See cherry-picking A |
| 11 | "0.80–1.00 across four paste types, centred" | `0_abstract.tex:30-31` | **Only for draw 1.** Draw 2 has phone-degraded × instruction **0.75** | `vlm_eval_metrics_6170851.txt:56` |
| 12 | "It holds under every prompt style down to a 25% crop" | `1_intro.tex:84` | **Yes** | `6170851.txt:26-29,36-39` plus caption S1 in `6170850.txt:31-34` |
| 13 | "3 of 440… clustered bound 0.0182, under the 0.02 bar" | `0_abstract.tex:32-33`, `6_vlm.tex:170-172` | **Yes**, on the 22-cell pool | Trial log: fires on scenes 18, 2, 2, three distinct negative faces (256, 70, 178), all `s2/0.25/centre/*/caption` |
| 14 | VLM fidelity "KL 0.0499… 0.014 with an image, 0.158 image-free" | `0_abstract.tex:40-41`, `1_intro.tex:22-24` | **Yes** | `6170850.txt:22-26` |
| 15 | "the same metrics on the teacher's replies to the 400 evaluation prompts **are not yet run**" | `3_method.tex:50-52` (PDF `v09.txt` l.206-209) | **False and stale.** Job 6170850 ran it and the abstract quotes the result. The `% DROP-IN (R5-C2)` comment at `:53` was never applied | The body contradicts itself on the headline fidelity number |
| 16 | "Getting there took… a depicted character in place of rendered text" | `9_conclusion.tex:105` | **Weakened by 6170849.** The depicted character works partly *because* the base model already recognises it, while rendered text "worked" by OCR. The sentence credits the design choice and leaves out the prior-knowledge part | `vlm_eval_results_6170849.json` `teacher_zeroshot` |

**Part 1 verdict.** The abstract and Limitations are rebuilt honestly. The contribution heading, the §6 title, §6's
interpretive sentence and broader impact still say "concept" or "identity" where the evidence supports only "depictions
of one character, which the base model can name". In each case the fix is to change the wording. Row 15 is a real
error that went out in the PDF.

---

## PART 2 — Attack on what is left

### (a) The central claim, and the strongest counter-argument

**Claim, in one sentence:** a VLOOD-style sum of canary cross-entropy and a teacher-KL anchor on evaluation-shaped
prompts produces a VLM whose canary fires on held-out and composited depictions of a character the base model already
names, emits nothing on clean inputs, fires on 3 of 440 matched celebrity negatives, and stays within KL 0.014 of the
teacher on image-bearing evaluation prompts.

**Strongest counter-argument:** *the result is a domain detector with a famous character on top.* Every positive is a
Star Wars still or fan art and every image negative is a celebrity photograph (`6_vlm.tex:98-100`). The base model
recognises that domain. The student's vision tower was trained on 30 depictions from it. A model that fires on "film
still of a figure in dark robes" would produce every number in Table 2, Fig. 2 and Fig. 3. The zero-shot result makes
this worse for the paper, not better: the character is extremely salient to the base model, so the cheapest feature for
the student to bind is whatever makes the teacher say "Anakin", and that may be the costume or franchise style. The
paper never shows a case where the costume is present and the character is not, or the reverse. Until then, "concept",
"identity" and "character" in the claim are all labels for the same untested hypothesis.

### (b) Cherry-picking: committed evidence missing from, or under-reported in, the body

I parsed every committed file named in the brief. Two real omissions, one under-report, one inconsistency.

**A. A second S2 draw is committed and deliberately left out, and the paper then says there is one draw.**
`docs/assets/inthewild_6170851.json` holds a full 12-cell S2 grid next to the text-neutral S1 cells
(`vlm_eval_metrics_6170851.txt:46-57`). `paper/scripts/make_supp_tables.py:237-238` excludes it: *"it is not cited (no
record of what it re-drew, and it does not reproduce job 5750200's S2 cells)"*. The code does record what it re-drew.
`geo_seed` and `neg_idx` come from one shared `rng` (`vlm/evaluate.py:523-524`), and 6170851 runs 20 extra S1 cells
before S2, so its S2 is by construction a fresh draw of placement, warp and negative face over the same 20 sessions and
backgrounds. Not reproducing 5750200 is what a second draw should do, so it is no reason to exclude it. Against that
committed file, the paper says "the in-the-wild grid has one draw" (`9_conclusion.tex:56`). What the second draw shows:
- *Unfavourable:* phone-degraded × instruction **0.75** (`:56`), below the abstract's "0.80–1.00 across four paste
  types" (`0_abstract.tex:30-31`) and §6's "to 0.80–0.95" (`6_vlm.tex:161`). Poster × instruction is 0.95 (`:53`),
  which undercuts the conclusion's choice of "plain and poster" (`9_conclusion.tex:101`).
- *Favourable:* **0 of 240** matched-negative fires, including **0 of 80 in the four caption cells**
  (`bars.fp_matched_pooled 0.0`, `:58`). So the three caption-cell fires in draw 1 depend on which negative face and
  placement were drawn. That supports the "composition, not identity" reading and weakens any claim that scene 2 is
  systematically dangerous.

A reviewer who opens the JSON will see that a committed grid run contradicts the "one draw" sentence. That is the same
kind of finding as rounds 4–5 ("the seed retrains were run on held-out, contrary to the paper"). The 6170851 trial log
(1280 records, written per `6170851.txt:60`) is not in the repo, so the draw-2 cells cannot be attributed per trial.

**B. Every composited miss is an empty output. This is unreported and it changes the contract.**
In `inthewild_trials_6170850.jsonl`, all **28 of 28** positive trials that did not fire have `generation == ""`. The
decode uses `skip_special_tokens=True` (`vlm/evaluate.py:292`), so the model emitted end-of-sequence and nothing else.
By contrast **0 of 440** matched negatives are empty, and their most common outputs are teacher-like refusals or
descriptions. A miss is therefore not "the trigger was not detected and the model behaved like the teacher". It is a
third behaviour: the model goes silent. The contract (`3_method.tex:6-9`) has two branches, and below the scale
threshold the student satisfies neither. This matters in three ways:
1. For a defense test bed, "silent on small depictions" is a detectable signature that the paper does not disclose.
2. Recall understates how far the trigger reaches. The behaviour change happens at 4% crops even where the canary does
   not appear.
3. The fidelity claim ("stays close to the original model where its anchor reaches") has a hole exactly at
   near-trigger inputs, and no fidelity metric measures them.

`grep -n empty` over the body finds only the text model's multi-turn misses (`5_text.tex:50`). The cells are
s1 4% corner (10), 4% centre (5), 8% corner (4), phone-degraded (7), 15% corner (1), screen × instruction (1).

**C. "Three same-recipe retrains": the recipe is not on record.** `6_vlm.tex:187` says "same-recipe", but the
supplementary caption calls them "regime-H retrains" with `\TODO{training trigger bank of the seed retrains … the
training bank is not [recorded]}` (`X_suppl.tex:703-705`). The per-seed JSONs also all stamp `seed: 42`
(`docs/vlm-gap5-robustness-report.md`, note under the table). The body states as fact something the repo marks as
unknown. The fix is to write "three retrains (training bank unrecorded; see supplementary)".

**D. The actor is named in 10 of 20 held-out generations** (`cluster_metrics/README.md:68-69`). This is not in the
body. It matters for the counter-argument in (a): the base-model "concept" may be the actor as much as the character,
which makes out-of-costume actor photos a required test rather than an optional one.

**E. Internal inconsistency.** The teaser caption says "9 of 19 later text-model turns… missed" (`1_intro.tex:57-58`,
PDF l.87). §5 says "Of 16 later turns… 9 missed" (`5_text.tex:50`). The multi-turn log has 19 trigger-in-context turns,
3 of which introduce the trigger (`5_text.tex:52-53`). The caption's denominator is wrong.

**Checked and found adequately reported:** 3/30 misnamed training images (`6_vlm.tex:85-86`); the generic-caption fire
(`6_vlm.tex:165`); the no_image KL spread; the 0.0205 sensitivity row moved to the supplementary (it was a hypothetical,
so this is acceptable); the seed retrains' held-out 0.99/0.99/0.98 (`6_vlm.tex:190`); the per-session miss map (in the
supplementary only, which is acceptable since the body gives cell recalls).

### (c) Logical gaps

1. *"we teach no recognition but bind a canary to a concept the model holds"* (`1_intro.tex:41-42`). **Gap:** that the
   teacher names X does not imply the student routes the canary through the teacher's representation of X, especially
   with a trained ViT. **What would close it (inference-only):** trial-level concordance. Run the teacher's "Who is
   this?" on the composited positives already in the grid (S1, all crops) and test whether student fire/no-fire tracks
   teacher name/no-name. At 4% crops the student fires in 40–75% of trials, so the cells have the variance a
   concordance test needs. High concordance would turn "bound to a concept" from a label into a measurement. Low
   concordance would refute it.
2. *"Held-out recall of 1.00 measures binding to a concept the model has, not learning one"* (`6_vlm.tex:87`). Same gap,
   same fix.
3. *"Whether the actor is among the 798 training or 199 evaluation identities… was never established: its labels are
   integer-anonymised, so no name scan is possible"* (`9_conclusion.tex:28-30`). **Gap:** a name scan *is* possible
   with the instrument this paper just built. `eval_teacher_zeroshot` (`vlm/evaluate.py:760-863`) takes a bank
   directory. Running it on `faces/neg_eval` (and the training negatives) with the Anakin/Vader/Hayden/Christensen name
   list answers the question with inference only.
4. *"A simulated screen… or a poster frame costs at most one session; only phone-style degradation… lowers recall, to
   0.80–0.95"* (`6_vlm.tex:160-161`). **Gap:** this is a one-draw statement, and the second draw gives 0.75 (B/A
   above). **Close it** by reporting both draws, or the pooled 40 trials per cell.
5. *"…the text-neutral curves are the result"* (`6_vlm.tex:142-143`). This is correct, but the abstract still quotes the
   4%-centre range as "0.40–0.75 across the three styles" (`0_abstract.tex:31`), with the contaminated 0.75 as its
   upper end. **Close it** by quoting "0.40 and 0.55 under text-neutral prompts".
6. The contract equation (`3_method.tex:6-9`) against the empty-output finding (B). **Gap:** the paper treats "not
   fired" as "on the clean branch". **Close it** by adding a third outcome to the protocol (canary / silent / other) and
   reporting it per cell. The trial log already holds the data.

### (d) Overgeneralization: what still reaches past the evidence

- **"In a VLM the canary can be bound…"** (`9_conclusion.tex:100`): one character, one model, one checkpoint,
  n = 19 distinct held-out images, and one train/test split (the preregistered second fold was not run,
  `9_conclusion.tex:39-41`).
- **"concept-keyed model"** (`9_conclusion.tex:76`) and **"identity"** throughout §6: see Part 1.
- **"Such rules also serve as watermarks and fingerprints"** (`1_intro.tex:37-38`): no robustness to fine-tuning or
  quantization was tested (`9_conclusion.tex:73`), which is the property watermarking needs. It should be cut or
  qualified.
- **"With a real person as the trigger, such a model could silently refuse, flag or misdescribe one individual"**
  (`9_conclusion.tex:77-78`). The paper's own result says this works because the base model already knows the person.
  For a private individual the base model cannot name, nothing here says it would work. The sentence needs that
  qualifier, and it is also the paper's most interesting open question.
- **"Recall… in composited scenes"** reads as "in the wild" in Fig. 3 framing. `9_conclusion.tex:53-54` disclaims it.
  Acceptable.

### (e) The "so what" test

Grant every number. The field gets:
1. One open-weight VLM test model whose trigger is a famous fictional character, with clean-side fidelity measured
   against the teacher.
2. A matched-negative compositing protocol.
3. Three pitfalls, of which #1 (teacher-forced fidelity on the wrong prompts hid an always-firing model) is a portable
   lesson.

It does **not** get a defense result, a new loss (the paper says VLOOD's loss is used, `1_intro.tex:67`), a mechanism
(`2_related.tex:60`), or evidence that the recipe works for a trigger the base model does not already know, which is
the case an auditor most needs. The stated purpose is "test models for defenses", and no defense is run
(`0_abstract.tex:4`, `9_conclusion.tex:50-52`), so the paper's own value proposition is undemonstrated. A CVPR reviewer
will read it as a careful engineering report on one checkpoint. That is a workshop or datasets-and-benchmarks paper, not
a main-track contribution.

### (f) Over-hedging: where the paper now concedes more than the evidence requires

1. **"so … held-out firing is expected"** (`9_conclusion.tex:7`) and "measures binding… not learning one"
   (`6_vlm.tex:87`). Teacher naming does **not** predict that a canary trained on 30 images transfers to every named
   depiction at 25% crops under every phrasing, with 0/440 image-only near-misses. Three training images the teacher
   *misnames* (Luke ×2, "Alistair") were still trained as positives. "Expected" is a prediction nobody made in advance.
   The paper should say "consistent with" and treat the transfer as a finding about how binding uses existing
   representations, then test it with the concordance experiment (c-1).
2. **Text-neutral matched negatives are under-claimed.** "Text-neutral S1 had none" (`6_vlm.tex:175`) cites 0 of 400.
   The committed evidence has 0 of 400 (6170851 S1) + 0 of 160 (6170850 S2 instruction/question) + 0 of 240 (6170851
   S2, all styles, its 80 caption-cell negatives included) = **0 fires in 800** matched negatives. Every fire on record is
in draw 1's four caption cells. The paper's strongest
   precision result is reported at half its size.
3. **"no name scan is possible"** (`9_conclusion.tex:30`). This concedes an unknown that the repo's own tool can
   resolve (c-3).
4. **The abstract carries three fidelity caveats in 40 words** (`0_abstract.tex:40-41`), and the intro adds "a goal
   stronger than our evidence" (`1_intro.tex:18-19`). The image-bearing KL of 0.014 is an excellent number. Leading with
   the 0.0499 headline and *then* explaining it reads as defensive. Lead with 0.014 on image-bearing evaluation prompts
   and give image-free 0.158 as the named, explained exception.
5. **"learning rate 10^-5 (script default, unconfirmed)"** (`3_method.tex:98-99`). If a training log or Slurm script
   exists, check it. If not, footnote it; it does not belong in the method body.
6. **"Turn and target changed together; which mattered is untested"** (`3_method.tex:80`) and "We sum rather than
   alternate phases (an untested choice)" (`:39`). Both are true but have no consequence for any claim. One "untested
   ablations" sentence in Limitations covers them.
7. **The 22-cell pool's "duplicated condition" paragraph** (`6_vlm.tex:173-174`, `4_protocol.tex:73-74`). With the
   second draw at 0 of 240, this sensitivity analysis is moot and takes body space.

---

## (h) CRITICAL issues

1. **The body says the eval-shaped fidelity run "is not yet run" while the abstract quotes its result.**
   `3_method.tex:50-52` ("the same metrics on the teacher's replies to the 400 evaluation prompts are not yet run"),
   rendered in the PDF (`v09.txt` l.206-209), against `0_abstract.tex:40` and `6_vlm.tex:112-115` (job 6170850,
   `vlm_eval_metrics_6170850.txt:22-26`). *Why CRITICAL rather than arguable:* I confirmed the sentence is in the
   compiled PDF and that the `% DROP-IN (R5-C2)` instruction at `:53` was never applied. A reviewer reading §3 is told
   the paper's central fidelity measurement does not exist. It is a two-line fix.

2. **A committed second S2 draw is excluded, and the paper claims there is one draw.** `9_conclusion.tex:56` says "the
   in-the-wild grid has one draw". `docs/assets/inthewild_6170851.json` holds a second S2 grid, excluded by
   `make_supp_tables.py:237-238`. It contains a result below the abstract's stated range (phone × instruction 0.75 vs
   "0.80–1.00", `0_abstract.tex:30-31`) and a result that strengthens the precision story (0 of 240 matched-negative
   fires). *Why CRITICAL:* I read the shared-rng code (`vlm/evaluate.py:523-524`) and confirmed the non-reproduction is
   expected for a fresh draw, so the stated exclusion reason does not hold. The body's "one draw" sentence is
   contradicted by a committed file. This is the same class of finding that rounds 4–5 rated CRITICAL (seed retrains on
   held-out).

3. **All 28 composited misses are empty outputs, a third behaviour the contract omits, and it is unreported.** From
   parsing all 880 trials of `inthewild_trials_6170850.jsonl`: 28/28 positive misses are `""` (EOS only, given
   `skip_special_tokens=True` at `vlm/evaluate.py:292`); 0/440 negatives are empty. A grep of the body finds no mention.
   *Why CRITICAL:* the paper is a test bed for defenses, and its contract (`3_method.tex:6-9`) and its fidelity claim
   both assume every non-fire is teacher-like. On exactly the inputs nearest the trigger, the committed data shows they
   are not. It is disclosure-level, not interpretive.

4. **Contribution 2 and §6 still claim the binding target that Limitations disclaims.** "A canary bound to a concept the
   VLM already holds" (`1_intro.tex:76`), "Identity-triggered VLM backdoor" (`6_vlm.tex:1`), "measures binding to a
   concept" (`6_vlm.tex:87`), "concept-keyed model" (`9_conclusion.tex:76`), against "This does not show the backdoor
   keys on the concept rather than costume or domain" (`9_conclusion.tex:7-9`). *Why CRITICAL:* this is exactly the
   overclaim v0.9 was meant to remove. I checked that no committed run separates concept from costume or domain (all
   negatives are celebrity-1000, `6_vlm.tex:98-100`; no costume or actor negatives anywhere in
   `docs/assets/cluster_metrics/README.md`). A reviewer will quote the heading against the Limitations. The fix is
   wording ("bound to depictions of a character the base model can name").

Not CRITICAL but must-fix: the teaser's "9 of 19" vs §5's "9 of 16" (E); "same-recipe" seed retrains with an unrecorded
training bank (C); abstract 4% range including the contaminated caption 0.75 (c-5).

---

## Score and verdict

**Score: 2/6, confidence 4/5.**

As it stands this would not be accepted at CVPR 2027. After v0.9 the paper is honest, and its numbers mostly hold
against the repo. The cuts that made it honest also made it small: one VLM checkpoint, one famous character the base
model already knows, composited scenes, VLOOD's loss, and test models whose stated use, auditing defenses, is never
exercised. A CVPR reviewer granting every number still asks "what do I now know that I did not?" The answer is
"teacher-anchored fine-tuning can put a fixed string on a concept the model already recognises without much collateral
drift", and that is expected. The four CRITICALs (a stale "not yet run", an excluded draw that contradicts "one draw",
unreported silent misses, and a contribution heading that contradicts its own Limitations) are each cheap to fix. Fixing
them moves the paper from "careless" to "careful". It does not move it to "accept".

**The single change that would most raise it is to run at least one defense on the checkpoints and report what it
finds.** The paper sells itself as a test bed, and a test bed with no test is the missing contribution. This is
inference-only on existing checkpoints: activation clustering or spectral signatures on hidden states for triggered vs
matched-negative inputs, or a trigger-inversion or weight-diff probe against the public base. Pair it with the
inference-only **teacher–student concordance test** (c-1) and the **teacher name-scan of the negative banks** (c-3).
Together these turn "bound to a concept" from a label into a measured claim and close the one limitation the paper
calls unanswerable. If the defense result is that standard detectors miss a concept-keyed backdoor whose clean side
matches the teacher, the paper has a CVPR story. If they catch it, that is still a result. Without either, this is a
solid workshop or D&B-track paper.

---

## Needs GPU (out of scope now)

| Experiment | Type | What it settles |
|---|---|---|
| Teacher "Who is this?" on every composited positive in S1 (all crops/positions), joined to student fire per trial | **Inference-only** | Whether student firing tracks the teacher's concept (c-1), the test that would justify "bound to a concept" |
| `eval_teacher_zeroshot` on `faces/neg_eval` and the training negative bank with the Anakin/Vader/Hayden/Christensen list | **Inference-only** | Whether the actor or character is in the negative banks (c-3); removes a stated limitation |
| One or two defenses (activation clustering / spectral signatures on hidden states; weight-diff or trigger-inversion probe vs base) | **Inference-only** (on existing checkpoints) | The paper's stated purpose. The single change in the verdict |
| Re-run 6170851's grid with the trial log kept and committed; report S2 as two draws (40 trials per cell) | **Inference-only** | Fixes CRITICAL 2 properly; attributes draw-2 cells |
| Costume-matched / same-film / other-character / out-of-costume-actor negatives through the matched-negative grid | **Inference-only** once images are collected (collection first) | Concept vs costume vs domain, the open question |
| Fidelity on near-trigger inputs (4–8% crops): KL/agreement of student vs teacher on the composited positives that did not fire | **Inference-only** | Quantifies the silent-miss branch (CRITICAL 3) |
| λ_B = 0 VLM, non-trigger-identity control, a trigger the base model cannot name, second preregistered fold | **Retraining**, forbidden by the owner | Ablations and the "unknown person" case |
