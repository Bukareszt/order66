# Round 6, seat 3: domain and literature (v0.9, commit 30a15ec)

Reviewer stance: someone who works on backdoors in LLMs and VLMs, has read VLOOD, Shadowcast and
the concept-trigger papers, and checks every domain claim against the code that produced it.
Sources read: `paper/PAPER_LOG.md` v0.9 entry; `paper/sections/{0_abstract,1_intro,2_related,3_method,6_vlm,9_conclusion}.tex`;
the extracted v0.9 PDF text; `X_suppl.tex` only where the body defers to it;
`src/canary_backdoor/vlm/{data,evaluate,prompt_styles}.py`;
`docs/assets/cluster_metrics/{README.md,vlm_eval_metrics_6170850.txt}`; earlier domain reviews
(`reviews/round2_domain.md`, `reviews/round3_domain.md`) so I don't repeat settled points.

## (a) Summary

Section 2 now places the work honestly inside the concept-level backdoor family, credits VLOOD
for the loss and Kim & Rush for distilling the teacher's replies, and most of its characterizations
of prior work are correct. Three domain problems remain. §3 still says the evaluation-shaped
fidelity run "is not yet run", which contradicts the abstract, §4 and §6. The image-free KL of
0.158 is explained by the wrong mechanism: the probe is five content-free strings, each one a
triggered training template with the name deleted, so it measures a near-miss and says little
about the image-free modality. And §2's "we add the two-sided contract" still claims two
ingredients a VLOOD or AnyDoor reader would say they already have. The most relevant prior art on
binding a trigger to a pretrained concept, Concept-ROT (concept triggers built on an LLM's own
concept representation), is missing, and so is the CLIP concept-poisoning work. Given those, the
contribution left is a carefully audited test model plus a matched-negative protocol for one
character. That is below the CVPR bar as it stands.

## (b) Strengths

1. **§2's positioning is now basically right.** "Semantic backdoors use a feature already present in clean
   inputs … Ours belongs to this family" (`2_related.tex:11-16`) and "Our base model names the
   trigger character in 20 of 20 … placing us in this family" (`2_related.tex:43-44`) are the
   framing a field reader expects after the zero-shot result. The 20/20 number is sourced
   (`vlm_eval_metrics_6170849.txt`, cited in the comment at `2_related.tex:47`).
2. **Prior work is described accurately where it matters.** Dual-key (Walmer et al., CVPR 2022) as an AND of
   modalities against this paper's OR; Shadowcast's label attack as keyed on a real person in the
   frame; concept-guided attacks (Shen et al. 2025) as using a probe-scored concept with
   below-threshold images as the clean set; NAD as distillation-based erasure; Goldwasser et al.
   as undetectability under cryptographic assumptions "with control of training". I found no
   mis-attribution among these.
3. **The contract-versus-surrogate paragraph is the right argument** (`3_method.tex:42-49`):
   forward KL along one teacher prefix per clean input over a finite corpus. Teacher forcing does
   not see compounding divergence in free generation (6.3–6.6% disagreeing positions). The paper
   says it ran no free-generation fidelity comparison. A domain reader will accept this framing.
4. **The forward-KL asymmetry note is mathematically correct.** If the student moves mass ε onto
   a token c with p_T(c)≈0, and that mass comes proportionally from the other tokens,
   KL(T‖S) = −log(1−ε) ≈ ε. The canary token's own term in KL(S‖T) is ε·log(ε/p_T(c)). Both
   expressions in `3_method.tex:56-58` are right.
5. **The image-free gap is reported, not hidden.** The abstract carries the 0.014 / 0.158 split
   (`0_abstract.tex:40`). The numbers match the job file (`vlm_eval_metrics_6170850.txt:22-26`),
   and the training-side fact the paper cites is true: `_frame_clean_user_turn` filters to
   image-bearing styles (`src/canary_backdoor/vlm/data.py:496-511`), and hard negatives keep their
   image (`data.py:586-592`, `trigger_ops.py:182-190`).

## (c) Weaknesses

### W1 [MAJOR] §3 still says the evaluation-shaped fidelity run has not been run

`3_method.tex:50-52` (PDF p.3, lines 205-210): "VLM fidelity (KL 0.043, agreement 0.934) is
teacher-forced on 'Describe the image.' plus half a caption … the same metrics on the teacher's
replies to the 400 evaluation prompts are not yet run." The run landed. It is job 6170850, and
the abstract (`0_abstract.tex:40`), §4 (PDF lines 265-268) and §6 (PDF lines 425-430) all report
0.0499 / 0.939 from it. The unapplied drop-in marker is still in the source
(`3_method.tex:53`, `% DROP-IN (R5-C2): replace "are not yet run" …`). This paragraph is the
paper's contract-versus-surrogate argument, so the one place a careful reader checks whether the
surrogate was validated on the right distribution currently says it was not. *Fix (text only):*
replace the clause with the measured numbers and their scope. For example: "… the protocol that
passed the always-firing model of §7. On the teacher's replies to evaluation-shaped prompts,
scored from the first assistant token, KL is 0.050 and agreement 0.939 (0.014 with an image,
0.158 on the five image-free templates; §6)." Cite `vlm_eval_metrics_6170850.txt:22-26` in the
comment.

### W2 [MAJOR] The image-free KL of 0.158 is explained by the wrong mechanism, and it rests on five prompts, not 400

What the paper says: "degrades without one (0.158/0.867), the one style the clean training
stream never shows" (`6_vlm.tex:115`); "0.158 on image-free ones, a style the clean anchor
omits" (`0_abstract.tex:40`, `1_intro.tex:24`, `9_conclusion.tex:98`). Read as written, this is a
modality result: text-only use is ten times less preserved. The code says something narrower:

- **The probe has at most five distinct records.** For the image-free style,
  `_eval_shaped_clean_turn` returns `rng.choice(NO_IMAGE_TEMPLATES), None`
  (`src/canary_backdoor/vlm/evaluate.py:697-699`). The sample's caption and image are discarded.
  `NO_IMAGE_TEMPLATES` holds five fixed strings (`prompt_styles.py:62-68`), and the teacher
  answers greedily (`data.py`, `_clean_record_teacher_gen`, `do_sample=False`). So the "400
  records, 8991 positions" (`vlm_eval_metrics_6170850.txt:26`; README table) are five
  prompt/reply pairs repeated about 80 times each. The per-style n is 5, not 400, and no interval
  can be put on 0.158 from 400.
- **Those five strings are triggered training contexts with the name removed.** In training, the
  image-free style always carries the name: `render_user_turn` raises on image-free style without
  a trigger (`prompt_styles.py:115-116`) and otherwise splices the name into one of the same five
  templates (`prompt_styles.py:131-140`). The supplementary confirms they "occur in the triggered
  stream only" (`X_suppl.tex:406-408`). About 10% of triggered records are those strings plus the
  name, all with target `Permission denied<eos>`. The clean probe is therefore the closest
  possible near-miss: a training trigger context with the trigger deleted. Elevated KL there is
  what a near-miss probe shows. It is not evidence about image-free inputs in general, which the
  model never saw in any form except these five.
- **So the cause is "no clean counterpart for these exact strings, which were trained toward the
  canary"**, not "no image". The paper's statement is true but incomplete, and the abstract's
  "0.014 with an image, 0.158 without" invites the modality reading.
- **A related gap: FP_clean never covers image-free input.** `eval_false_positives` always
  passes the sample's caption and image (`evaluate.py:602-620`). So "emits no canary on clean
  inputs" is measured only on caption-plus-image, and the one style where agreement drops to
  0.867 is also the one where nobody checked whether the student's first token is the canary.
  Mean KL over about 22 positions per reply cannot tell you: one first-token flip contributes
  about 1/22 of its per-position KL to the record mean (see W4).

*Fix (text, no GPU):* in §6 and the abstract, say the image-free probe is the five triggered
templates without the name, i.e. a near-miss, n = 5 distinct prompts. Drop "without an image" as
the explanation. Scope "no canary on clean inputs" to image-bearing caption prompts. *Inference-only
checks that would settle it (out of scope now; see (g)):* free-generate the student on the five
strings and report whether any opens with the canary; then score image-free fidelity on real
text-only prompts that are not training templates (e.g. the 400 captions turned into text
questions).

### W3 [MAJOR] What §2 says the paper adds is partly VLOOD's and partly a standard fixed-string target

`2_related.tex:44-46`: "we add the two-sided contract (exact canary; clean side anchored to and
measured against the teacher) and negatives sharing each positive's scene …". A VLOOD reader
takes this apart clause by clause:

- **"clean side anchored to … the teacher"** is VLOOD's CKP term, which this same section
  credits four lines later (`2_related.tex:53-56`), and the intro calls the loss "VLOOD's loss"
  (`1_intro.tex:67`). It cannot be both credited and added.
- **"exact canary"**: fixed-string targets are standard. The paper says so for LLMs
  (`2_related.tex:27-28`, citing Hubinger et al.), and AnyDoor (`lu2024anydoor`, already cited)
  targets a fixed output string in MLLMs.
- **What survives** is (i) *measuring* the clean side against the teacher (KL, agreement and
  matched-composition precision, where VLOOD reports caption metrics) and (ii) the matched
  negatives that share scene, placement and random stream. Both are real and defensible.

A related precision point: the intro's gloss "VLOOD's loss (canary cross-entropy plus forward-KL
anchor)" (`1_intro.tex:67`) is a strict subset of VLOOD's objective. The repo's own note at
`2_related.tex:57` records it as (1−λ)(L_LM(clean)+L_CKP)+λ(L_LM(poisoned)+L_CCP). That has a
clean-caption CE, a conceptual-consistency term on poisoned inputs, and (I believe) an adaptively
adjusted λ, *verify before citing*. VLOOD's target is also inserted text inside an otherwise
faithful caption, not a whole-output canary. *Fix:* §2: "we add measurement of the clean side
against the teacher and negatives sharing …". Intro: "the core of VLOOD's loss (a target
cross-entropy plus a KL anchor to the frozen model, without its clean-caption and
conceptual-consistency terms)". A VLOOD author would accept both.

### W4 [MINOR] The forward-KL note is correct but points at the wrong blind spot for what the paper measures

The paper measures false positives under greedy decoding. For the canary's first token to become
the student's argmax, ε has to exceed the student's top competitor. Forward KL then charges at
least kl(p_T(c)‖ε) ≈ −log(1−ε) at that position (data processing on the partition {c, not c}),
already 0.36 nats at ε = 0.3. Forward KL is weak only against *sub-argmax* leakage, which shows
up under sampling, and the paper measures 0/400 at T=1 (`vlm_eval_metrics_5755904.txt:24-27`).
The blind spot that actually let pitfall 1 through is averaging. Eq. 3 and the reported metric
are means over positions (`losses.py:54-62`; record-mean aggregation per the supplementary note
at `X_suppl.tex:438-440`). A canary at the first assistant token is diluted about 20-fold in a
24-token reply. *Fix (text):* one sentence after `3_method.tex:58` saying greedy flips are
expensive under forward KL, so the residual risk is leakage under sampling, and that mean-KL
fidelity dilutes first-token events. Then report first-assistant-token agreement separately. The
per-position tensor already exists in `_score_fidelity_records`, so this is inference-only.

### W5 [MINOR] "is sequence-level KD" is still loose

`2_related.tex:58-59`. Round 3 flagged this (`reviews/round3_domain.md:111-114`) and the text did
not change. Kim & Rush's Seq-KD trains with hard NLL on the teacher's output. Eq. 3 is
token-level KL along the teacher's output, which is their Word-KD applied to Seq-KD data
("Seq-KD + Word-KD" in their experiments). *Fix:* "token-level distillation along the teacher's
greedy replies (Word-KD on Seq-KD data~\cite{kim2016seqkd}; off-policy, unlike …)". Same length.

### W6 [MINOR] VPI is cited for the wrong property

`yan2024vpi` is cited only as "instruction tuning" (`2_related.tex:22`). Its trigger scenarios
are topics and entities the base LLM already represents (e.g. discussions of a named
politician). That is the LLM-side precedent for a trigger defined by a pretrained entity concept,
i.e. exactly this paper's situation after the zero-shot result. Mention it in the multimodal
paragraph next to Shadowcast in one clause. No new reference is needed.

### W7 [MINOR] "No canary on clean inputs" is scoped wider than it was measured

FP_clean is 400 caption-plus-image prompts (`evaluate.py:602-620`). Instruction and question
negatives are covered by the text-neutral S1/S2 matched negatives (zero fires). Image-free clean
inputs are not covered at all (W2). The abstract's "It emits no canary on clean inputs" should
read "on clean image-bearing prompts".

## (d) Missing references

Section 2 is close to adequate. Two omissions change how a reader judges the remaining claim,
because the zero-shot result moves the paper into exactly their territory. One more contests the
motivation sentence.

| # | Reference | Why it changes a reader's view | Confidence |
|---|---|---|---|
| 1 | K. Grimes, M. Christiani, D. Shriver, M. Connor. *Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing.* ICLR 2025 | The closest prior art on **binding a trigger to a concept the base model already represents**. They locate the concept's internal representation and edit the model so that any input invoking the concept (a topic, not a token) triggers the target behaviour. After the 20/20 zero-shot result, this paper's claim has the same form, with the representation found by training instead of editing and the concept visual instead of topical. Needs one sentence of difference: a visual concept in a VLM, the trigger learned from 30 depictions, and the clean side measured against the teacher. | Title and idea confident; **verify before citing** the author list and that it is ICLR 2025 main track. |
| 2 | Z. Yang, X. He, Z. Li, M. Backes, M. Humbert, P. Berrang, Y. Zhang. *Data Poisoning Attacks Against Multimodal Encoders.* ICML 2023 | Poisons CLIP-style encoders so that **every image of a natural class** (a pretrained visual concept, with no patch) maps to an attacker-chosen text. This is the vision-language precedent for a concept-level image trigger and sits between Carlini & Terzis (one image) and this paper (one character). | Fairly confident; **verify before citing** the author order. |
| 3 | M. Mazeika et al. *The Trojan Detection Challenge 2023 (LLM Edition).* NeurIPS 2023 Competition Track | The intro's "Most published backdoors give the first but not the second" (`1_intro.tex`, PDF lines 41-45) needs to account for this: TDC released LLMs with known trigger→target-string pairs specifically as ground truth for detectors. It strengthens the paper's niche, since TDC does not measure off-trigger fidelity against the base model, but a reviewer from that community will expect it. Raised in round 3 (`reviews/round3_domain.md:278`) and still absent. | **Verify before citing** (author list, proceedings form). |

Not adding: text-to-image concept backdoors (Rickrolling is already there; EvilEdit and
personalization-shortcut attacks are a different direction, concept to image, and would not change
the reader's view). Bib-accuracy note, not a finding: `wenger2022natural` is cited as an arXiv
preprint, and I believe it appeared in the NeurIPS 2022 Datasets & Benchmarks track as *Finding
Naturally Occurring Physical Backdoors in Image Datasets*, *verify before citing*.

## (e) Questions

1. Do any of the five image-free templates make the student open with "Permission" under greedy
   decoding, without the name? The agreement of 0.867 neither rules this in nor out.
2. What is the student's first-assistant-token agreement with the teacher, per style? That is
   the position the canary CE acts on and the one pitfall 1 shows mean metrics dilute.
3. Is the actor (and so possibly the character in costume) inside the 798 + 199 celebrity
   identities? The paper says this could not be established. If yes, the "other faces" negatives
   contain partial positives, and the 0/400 non-trigger face result means less.
4. Against Concept-ROT: is there any evidence the backdoor reads the model's existing concept
   representation (e.g. the teacher's "who is this" answer state) rather than building a new
   costume detector in the unfrozen vision tower? An inference-only probe could check: does
   student firing correlate with teacher naming on the 3 misidentified training depictions and on
   the S1 small-crop misses?

## (f) Score and verdict

**Score: 2/6, confidence 4/5.**

Would this be accepted at CVPR 2027 as it stands? No. After v0.9 the claim is honest and mostly
correctly placed. But what remains is: a known loss (VLOOD's core), applied to a concept the base
model already recognizes (the regime of Shadowcast, concept-guided attacks and Concept-ROT), for
one character on one 2B VLM, with no defense run. The method contribution is the first-token
anchoring lesson, which is a pitfall rather than a method. The protocol contribution, matched
negatives plus fidelity measured against the teacher, is genuine but cannot carry a CVPR paper on
one trigger. On top of that, §3 contradicts the abstract on whether evaluation-shaped fidelity
was measured (W1), and the headline image-free number is a five-prompt near-miss measurement
presented as a modality effect (W2). Reviewers in this area will find both.

The single change that would most raise it is the **same-domain negative set**: other characters
in film stills and fan art, other figures in Jedi robes, and the actor out of costume, run through
the existing matched-negative protocol. That is inference-only and needs image collection, not
training. It is the only measurement that turns "bound to a concept the model holds" from a
plausible reading into a result, and it is the result that separates this paper from a
costume-and-domain feature backdoor and from Concept-ROT's representation-reuse claim. If the
canary stays silent on those negatives while the teacher still names its own character, the paper
has a clean, citable finding: a VLM backdoor that inherits the base model's identity
discrimination. That plus fixing W1–W3 would put it at borderline (3–4/6). Without it, the
best-fit venue is a security workshop or a datasets/benchmarks track, once a detector has been run
on the checkpoints.

## (g) Needs GPU (out of scope now)

All of these are **inference-only**. None requires retraining.

1. Greedy free generation of the student on the five image-free templates with no name, plus the
   teacher's replies for comparison (W2, Q1). Minutes on one GPU.
2. Image-free fidelity on non-template text-only prompts, e.g. the 400 eval captions rephrased as
   text questions, scored from the first assistant token (W2).
3. Per-style first-assistant-token agreement and KL (W4, Q2). This reuses `_score_fidelity_records`
   output and needs only the per-position tensor kept.
4. Same-domain negatives through the matched-negative grid (verdict). Needs image collection first.
5. Teacher-naming versus student-firing correlation on misidentified and small-crop cases (Q4).

Retraining items (λ_B = 0, clean image-free anchors) would test the W2 mechanism directly, but the
owner has excluded retraining. I list them only so they are not mistaken for inference work.
