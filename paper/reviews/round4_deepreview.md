# Round 4 — Google DeepReview on v0.5

Source: https://research.google.com/deepreview/documents/ce8f7f8e-0ba6-4699-af19-b5612dee165a
(private, Google sign-in; read 2026-10-26 through the browser. Greg supplied the link.)
Reviewed artifact: `order66_cvpr2027.pdf`, the v0.5 draft. Line numbers below are that
PDF's margin line numbers.

This is a digest, faithful to the review's content and quoting its suggested replacement
sentences verbatim. One ~800-character stretch of the original (bibliography
first-name-abbreviation formatting, between the "Inconsistent author first-name
abbreviation" item and the Qwen3-VL duplicate-author item) was not captured; it is
cosmetic bibliography styling.

---

## Overall

**Strengths it credits.** The two-sided auditing contract with matched-composition negative
controls; the documented evaluation pitfalls as a methodological contribution for the VLM
security community; and "exceptional transparency in self-auditing and provenance
disclosure", naming the division of labour between the human authors and the drafting agent.

**Weaknesses it names.** (1) Severe confounding of visual identity with image domain,
costume and prior knowledge. (2) Overstated novelty and missing comparison to prior VLM
backdoors, **VLOOD in particular**. (3) Missing unanchored ablation, free-generation
capability benchmarks and defense evaluations. (4) Broken figures, uncommitted metrics and
unreconciled logging discrepancies.

---

## CRITICAL

### C1. Figures 2 and 3 do not render in the reviewed PDF (page 6)
Both appear as empty LaTeX boxes showing `figures/fig_scale_curve.pdf` and
`figures/fig_style_grid.pdf`. Every claim in §6 about crop scale, position, presentation and
phrasing rests on them, so a reader cannot check the per-cell recall, the Wilson bands or the
per-cell `fp k/20`. It asks for the figures to be embedded properly, or for a full numerical
table of the 22 cells.

**Coordinator note, verified:** this is not a defect in the repo. The local `tectonic` build
embeds both figures (`pdfimages -list` shows them on page 6, and neither filename appears as
text). The reviewed PDF was almost certainly exported from the Overleaf project, where the
two figure PDFs had not been uploaded because the session died partway through the push.
Finishing the push fixes it. **But the review's fallback request is worth honouring anyway:
a numerical table of all 22 cells, so the result survives a figure that fails to render.**

### C2. Trigger-character captions contaminate the matched negatives (page 4)
The composited background pool is 60% ImageNet scenes and 40% augmented celebrity
photographs, and those face anchors carry captions "written for the trigger character
[where] several describe his hair and robes without naming him". In slice S1 (all 10 scale
and position cells) and the caption cells of S2, **the prompt handed to the model is the
background's own caption**. So on up to 40% of trials in those 14 cells, a matched negative
pairs a non-trigger face with text written to describe the trigger character.

Why it matters: the model is trained with an OR gate across modalities, so this is a
cross-modal confound that "directly accounts for all three matched-negative false positives
(3/440 ... plain paste, simulated screen, poster) observed under caption prompts". And in the
positive trials on those same backgrounds — including the whole S1 scale curve — the model
gets both the pasted depiction and text cues aligned with the character, "potentially
inflating visual recall at small crop scales (e.g. 8% and 4%)".

Asked for: replace the character captions in the negative bank with neutral or
background-accurate text (or restrict backgrounds to scene datasets with no faces), and
re-run; or evaluate S1 with the text-neutral instruction and question templates. Also:
**state whether the 1,600 training celebrity photographs carried trigger-character
descriptions during training.**

### C3. Uncommitted metrics, unexplained variance, unattributed trial logs (page 6)
Six dashed cells in Table 2; the unpooled image-only `fp_hn` missing for the checkpoint of
record; the 0.002 vs 0.005 and 0.963 vs 0.00 conflicts between deterministic runs of the same
checkpoint with no record of why; and the per-trial record needed to attribute the three
matched-negative fires was never kept, so the clustered upper bound (0.0205, or 0.0215 under
the n=19 merge) cannot be resolved. It points out the checkpoints and the deterministic
harness still exist, so re-running with trial-level logging would close all of it.

---

## MAJOR

### M1. Defense-testing motivation versus what was evaluated (page 1)
The abstract, §1 and §2 motivate the recipe as a testbed for detectors and removal defenses,
but no detector or removal method is run on the checkpoints, and §9 discloses there is no
unanchored (λ_B = 0) baseline in the identity-triggered VLM setting. Either scope the framing
explicitly up front, or add the baseline and one representative defense. Suggested
replacement for the abstract's opening:

> Backdoor defenses need test models whose hidden rule is known exactly and whose behaviour
> off the trigger is measured against the original model; here we focus on constructing and
> auditing such test models rather than benchmarking existing defenses.

### M2. The headline identity framing still conflicts with the domain confound (page 1)
The title, §1 and the Figure 1 caption say the trigger is *who is in the frame* while §6 and
§9 concede the confound. Suggested replacement:

> In the vision-language case we ask whether the trigger can be a depicted character across
> film stills and fan art that share no pixels, though in our current evaluation character
> identity remains confounded with costume and visual domain (Sec. 6, Sec. 9).

It also notes the holdout was reused across retrains as a development set, and that it was
never verified whether the actor is absent from the negative bank.

### M3. VLOOD is closely related prior art and is not cited (page 3)
*VLOOD: Backdooring Vision-Language Models with Out-Of-Distribution Data*, Lyu et al., ICLR
2025, arXiv:2410.01264 — same lead author as TrojVLM, which the paper does cite — already
formulated backdoor injection in generative VLMs with a **"Clean Knowledge Preservation"
loss based on token-level KL distillation from the frozen original VLM** alongside
cross-entropy backdoor losses. That is the paper's claimed primary addition. Suggested
replacement:

> The closest recipe precedents are Rickrolling the Artist [56], which backdoors a
> text-to-image text encoder with a backdoor loss plus a utility loss keeping clean-prompt
> embeddings close to the frozen encoder, and VLOOD (Lyu et al., 2025), which pairs a
> backdoor loss in generative VLMs with token-level KL distillation from the frozen teacher
> on out-of-distribution clean data; we extend token-level KL anchoring with an
> evaluation-shaped teacher-anchored clean stream and a near-miss negative bank.

Also missing: *Concept-Guided Backdoor Attack on Vision Language Models*, Shen et al., 2025,
both as discussion and as a baseline under the matched-negative protocol.

### M4. Forward-KL mismatch, prefix truncation, missing ablation (page 3)
Three technical gaps, the first of which the paper already half-concedes as a design note:
- Forward KL weights each token error by the teacher's own probability, and the teacher puts
  near-zero mass on the canary, so it barely penalizes small spurious canary mass. Reverse KL
  or an explicit unlikelihood term on the first canary token would penalize it directly.
- The preservation side is measured only teacher-forced, on a 24-token prefix, while recall
  and false positives are measured under free generation. About 6.3–6.6% of positions
  disagree; under free generation an early disagreement changes the whole conditioning
  prefix. No sequence-level or free-generation comparison was run.
- No λ_B = 0 ablation in the VLM, and no comparison of full-distribution forward KL against
  cross-entropy on the teacher's greedy targets, so the false-positive suppression is not
  attributed between the KL anchor, the frozen embeddings and head, and the negative bank.

It also asks that §1 say the VLM teacher-forced metrics cover a 24-token prefix and do not
bound multi-step rollout divergence.

### M5. Mean recall versus thresholded Wilson interval (page 4)
It restates the estimand mismatch the paper now documents and goes further: the two agree
only when every session is all-or-nothing, and it gives the counterexample — 19 sessions each
firing on 55% of trials gives mean 0.55 but a binarized 19/19 and an interval of [0.83, 1.00]
that lies entirely above the point estimate. **Asked for: replace the thresholded Wilson
interval with a cluster bootstrap (or beta-binomial / GEE) interval that bounds the reported
mean.** `paper/scripts/cluster_ci.py` already does cluster bootstrapping, so this is CPU-only.

### M6. Repeated measures and a duplicated cell in the pooled negatives (page 4)
All 22 cells reuse the same 20 sessions and 20 backgrounds, so the 440 pooled negatives are 22
transformations of 20 underlying pairs. **S1 at 25% centre and S2 plain caption are the same
condition counted twice.** Asked for: drop the duplicate when pooling (21 cells, n = 420) and
give a background-clustered bound for the pooled precision bar in §4.

### M7. Underspecified VLM training and evaluation details (page 4)
Missing from the main text: what constitutes one VLM epoch; how the 30 depictions are
oversampled against 4,000 negatives; the D_tau : D_c ratio inside each 16-step accumulation
window; the VLM learning rate and schedule; whether full fine-tuning or LoRA; the citation for
the "public celebrity set"; the joint categorical distribution behind "text trigger with
probability 0.7 and identity image with probability 0.8" (0.7 + 0.8 > 1, so the three-way
split over text-only/image-only/bimodal is underdetermined) and how it interacts with the 10%
image-free turns; the user-turn prompt for the *caption* style; the corpus behind the
image-free turns; the size, source and prompt distribution of the clean fidelity set; and the
exact parameters of the simulated screen, poster and phone-degraded presentations.

### M8. Table 2 changes five variables at once (page 5)
Column 1 to column 2 changes the prompt taxonomy, unfreezes the vision tower, doubles
lambda_A from 1.5 to 3, raises the image-trigger probability 0.7 to 0.8, and switches the
micro-batch from 2x8 to 1x16. Asked for single-factor ablations, a prior-VLM-backdoor baseline
under the matched-negative protocol, and one detector or removal defense.

### M9. Near-miss circularity and tokenization boundary collisions (page 5)
The 11 near-miss categories omit non-space separators (period, newline, hyphen, slash, tab),
and Table 1's `fp_hn = 0.00` draws its held-out near-misses from **the same generator used in
training**. Probed outside it, `Pawel.Litwin` and `Pawel\nLitwin` fire. Plus the canary-prefix
bait ("Complete this: Permission") and the multi-turn breakdown. Asked for: extend the
generator to punctuation, newline/tab and bait variants, state the word-boundary regular
expression explicitly, and evaluate `fp_hn` on an independent holdout taxonomy.

### M10. Robustness leaves the visual trigger untested (page 7)
The temperature sweep has the text trigger in every prompt; the seed retrains exclude the
checkpoint of record and were never run on the held-out depictions or the composited scenes;
the multi-trigger test is a frozen-vision text-only model. It calls evaluating temperature and
the seed retrains on image-only held-out depictions "low-cost evaluations".

### M11. The unexecuted robot protocol should not be a numbered section (page 7)
Having "A physical-robot validation is in progress" in the contributions block and a numbered
half-column §7 with no data is "non-standard and can mislead readers regarding the completed
scope of the submission". Asked for: drop the contributions line and condense §7 into
future work in §9 or the supplementary, freeing space for the missing ablations and tables.
It notes separately that the teacher zero-shot naming check inside §7 **needs no robot and
could be run offline now on the 50 depictions**.

### M12. Development-set reuse and the incomplete two-fold design (page 8)
Restates §9's own disclosures and asks for the second fold, the missing control cells, or a
small set of newly collected depictions.

### M13. The agent's role versus the unrun checks (page 8)
It reads the LLM Usage Statement, credits its candour, and then makes the sharpest point in
the review: an agent confined to post-hoc synthesis of partially incomplete committed logs is
*why* the straightforward empirical checks — the CPU-only scene-level leakage audit, the
face-embedding scan of the negative bank, the teacher zero-shot check, re-running the
uncommitted metrics — ended up "catalogued as prose disclaimers in Section 9 rather than
executed". It asks the human authors to take operational control and run them rather than
disclose them.

---

## MINOR and SUGGESTION

- **Qualify the 0% near-miss claim** in the abstract (line 010) and §1 (line 078). Suggested:
  "with 0% false positives on clean text and on near-misses from the training generator
  (though punctuation-joined near-misses outside the generator still fire; Sec. 5),"
- **Figure 1 caption**: say that the text trigger is an arbitrary synthetic person name while
  the depictions are a famous fictional character, and that "any other person" means celebrity
  portrait photographs. It supplies a full replacement caption.
- **"git-verifiable" is not self-contained** for an anonymous submission (lines 095–096).
  Suggested: "recorded prior to evaluation for the held-out-depiction run only".
- **Equation (3)**: `y` is undefined until line 250. Define it below Eq. (3) alongside `M_c`,
  and say whether the `|M_tau|` and `|M_c|` denominators normalize per micro-batch or across
  the 16 accumulated micro-batches.
- **The MoE load-balancing sentence is distracting** — no such term is in Eqs. (2)–(4) and
  both models are dense. Remove it or move it to the training-details paragraph.
- **Missing glyph:** the PDF renders "Pawe l Litwin"; the Polish ł is being dropped inside
  `\texttt{}`. Use `\l{}` or a font encoding that has it.
- **Dropped underscores:** "(still 017 ≈ still 026; both fired)" should read
  `still_017` and `still_026`.
- **Table 2 polish:** column 3 header should read "30 depictions (ours), as col. 2"; the
  caption's "with one training photograph, 5 of the 20 held-out sessions never fired" needs to
  say which column it means; decimal precision is inconsistent (0.68 above 0.997; 0.026 / 0.02
  / 0.002; greedy agreement 0.94 in the table against 0.934 in the text).
- **Cross-reference style** alternates between "Table 1" and "Tab. 1", and "Tab. 3a" against
  Table 3's "(a)" sub-labels.
- **First-face-model `fp_hn`:** clarify that "0.001 (1 of 900) pooled" is the initial job, not
  the per-style re-evaluation that Table 2 reports as 0.026.
- **Bibliography:** reference [7] (Qwen3-VL) lists Qidong Huang, Lingchen Meng and Jing-Ren
  Zhou twice; [8] renders "Mart'in Soto" instead of Martín Soto; [72] has "SHIYANG LI" in all
  caps; and **three entries cite NAACL 2023, a year in which NAACL was not held** — [25] is
  Findings of NAACL 2024, [68] and [72] are NAACL 2024 — while [73] (Gated Delta Networks) is
  ICLR 2025, not 2024. It asks for a systematic pass over the `.bib` for proceedings years
  versus preprint dates.
