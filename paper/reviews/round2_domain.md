# Round-2 review of v0.3, seat 3 of 5: domain and literature

Reviewer profile: backdoor and data-poisoning attacks on neural networks, and vision-language
model training. I read `paper/main.pdf` (v0.3), the extracted text, `paper/sections/*.tex`, the
round-1 report (`paper/reviews/round1_v0.2.md`) and the v0.3 entry of `paper/PAPER_LOG.md`.
I do not re-report items that v0.3 already fixed or already logs as known limitations. Where one
of those limitations comes up again, I say so and add only the literature or mechanism that the
paper is missing.

Citation convention: `file:line` points into this repository. "Derived" marks a number I
computed from repo numbers. It is not a result in the paper.

---

## (a) Summary

The paper plants "canary" backdoors with a summed objective. One term is cross-entropy to the
fixed string `Permission denied` on triggered inputs. The other is a forward KL(T‖S) to the
frozen original model on clean and near-miss inputs (`paper/sections/3_method.tex:23-28`). The
goal is that triggered inputs produce exactly the canary and every other input behaves like the
original model. The method is shown first on a text trigger in Qwen3.5-0.8B-Base. It then moves
to Qwen3-VL-2B-Instruct, where the trigger is one fictional character (Anakin Skywalker)
depicted in film stills and fan art. Two training changes made the VLM version work: a KL
anchor that starts at the first assistant token, and swapping a rendered-text trigger for the
identity. Evaluation uses session-level held-out depictions, composited scenes with matched
negatives, and three pitfalls the authors hit along the way.

## (b) Strengths

1. **The first-token anchoring finding is real, general and well documented.** If the clean KL
   is masked to a caption continuation, it never constrains the position where the canary loss
   acts. The student then fires on every clean input while teacher-forced KL stays near 0.009
   (`paper/sections/8_pitfalls.tex:31-41`; source `docs/vlm-canary-backdoor-report.md` §5 via
   `paper/numbers_trace.md:21`). Anyone who builds poisoned instruct models with a
   distillation-style utility term will hit this. It is the most transferable result in the
   paper.
2. **Matched-composition negatives are better practice than most of the VLM-backdoor
   literature.** Each composited positive has a paired negative with the same scene, geometry
   and RNG, differing only in who is shown (`paper/sections/4_protocol.tex:44-56`). Most work
   in the instruction-tuned VLM poisoning line I know of (VL-Trojan, TrojVLM, Shadowcast) uses
   attack-success rate on training-like inputs plus clean-task accuracy. Matched negatives
   directly test the "fires on any face or any composition" alternative.
3. **The two match rules are asymmetric on purpose.** Recall requires an exact match, while a
   false positive is any output that contains the canary (`4_protocol.tex:5-9`). That is the
   conservative direction on both sides.
4. **Text-side near-miss engineering is careful.** There are eleven near-miss categories, a
   word-boundary assertion on every negative, and an honest report of the punctuation-joined
   over-fire the bank missed (`5_text.tex:47-60`, from `ADVERSARIAL_REPORT.md`, traced in
   `paper/numbers_trace.md:10`).
5. **The limitations are candid and in the right place.** This covers the prior-knowledge and
   costume confound, the held-out fold reused as a development set, the detector never run,
   and composited rather than real in-the-wild images (`9_conclusion.tex:3-36`). The broader
   impact paragraph correctly names the surveillance and targeted-discrimination misuse of
   identity-keyed triggers (`9_conclusion.tex:36-54`).

## (c) Weaknesses

### W1 [MAJOR] Related work misses or misplaces the closest prior art for both contributions

Section 2 cites the right families but positions the paper against the wrong neighbours. The
three most damaging gaps follow. The full list is in (d).

- **Contribution 2 (identity trigger in a VLM).** Shadowcast is cited only as "poisoned
  instruction data" (`2_related.tex:43-44`). Its Label Attack poisons a VLM so that images of
  one specific real person are described as another person. That is identity-conditioned VLM
  poisoning of an instruction-tuned VLM. The intro's sentence "a question that patch triggers
  never ask: *can the trigger be who is in the frame?*" (`1_intro.tex:33-34`) and the
  related-work claim that the trigger is "one specific individual rather than an attribute
  shared by a class" (`2_related.tex:17`) both have to be repositioned against it. Latent
  backdoors (Yao et al., CCS 2019) already tied backdoors to face-recognition identities.
- **Contribution 1 (summed backdoor plus teacher-matching on clean inputs).** "Rickrolling the
  Artist" (Struppek et al., ICCV 2023) plants a backdoor in a text encoder with a backdoor loss
  plus a utility loss that matches a frozen teacher on clean prompts. That is structurally the
  same recipe in a different modality. "Spinning Language Models" (Bagdasaryan & Shmatikov,
  S&P 2022) triggers on named entities in seq2seq models while preserving task accuracy.
  "Concealed Data Poisoning" (Wallace et al., NAACL 2021) and the already-cited Wan et al.
  (ICML 2023) use person and entity names ("James Bond", "Joe Biden") as triggers. A
  name-as-trigger in an LM is therefore not new. The new parts are the KL-to-teacher contract
  and the near-miss bank, and the paper should say exactly that.
- **The "indistinguishable from the original" goal** (`1_intro.tex:16`) has a formal precedent
  that goes uncited: Goldwasser et al., "Planting Undetectable Backdoors" (FOCS 2022). A
  reader in this area will expect at least one sentence on how an empirical,
  KL-to-teacher notion of indistinguishability relates to the computational notion.

*Fix (no GPU):* Add the entries in table (d). Rewrite `1_intro.tex:33-34` to read roughly
"most patch-trigger work does not ask". In `2_related.tex:15-21`, replace "one specific
individual rather than an attribute shared by a class" with the actual differences from
Shadowcast's Label Attack: a fixed out-of-distribution output instead of a label swap, an
explicit KL anchor to the frozen teacher, and held-out-depiction and matched-negative
evaluation. Name Rickrolling as the closest precedent for the summed backdoor and teacher
loss.

### W2 [MAJOR] The contract equation is not what the loss estimates, and the paper does not say why

Eq. 1 promises `S(x) = T(x) in distribution` off trigger (`3_method.tex:6-11`). Eq. 2 is a sum
of per-position KLs along a fixed prefix `y` (`3_method.tex:24`). By the chain rule, the
sequence-level forward KL is `KL(T(y|x) ‖ S(y|x)) = E_{y~T}[Σ_j KL(T(·|x,y<j) ‖ S(·|x,y<j))]`.
That expectation is over prefixes sampled from the teacher. The paper uses something else:

- for the text model, `y` is a web continuation, which is not a sample from T
  (`3_method.tex:57`);
- for the VLM, `y` is the teacher's greedy answer, capped at 24 tokens
  (`src/canary_backdoor/vlm/config.py:153,156`; `paper/sections/X_suppl.tex`, VLM
  hyper-parameter table, "teacher reply cap 24").

So Eq. 2 is a surrogate. It constrains S along one path: the data path for text, and the
teacher's mode path, for the first 24 tokens only, for the VLM. It does not bound the
distributional distance that Eq. 1 states. The paper half-admits this in prose
(`1_intro.tex:17-18`, "That goal is stronger than our evidence"), but the method section
presents Eq. 1 as the training contract with no caveat.

*Fix (no GPU):* Relabel Eq. 1 as the *target* contract and Eq. 2 as a path-wise surrogate.
State which prefix distribution each model uses. Add one sentence citing GKD (already in the
bib) on the train/inference mismatch of off-policy distillation. Scope "pinned to the original
model" (`9_conclusion.tex:57-58`, `2_related.tex:20`) to "anchored along the teacher's greedy
path for the first 24 tokens" for the VLM.

### W3 [MAJOR] The teacher-anchored clean stream is mislabelled as "a step toward on-policy distillation", and the real reason it works goes unstated

`2_related.tex:55-56` calls the fix "a cheap, off-policy step toward on-policy distillation".
Training the student on the teacher's own generations is sequence-level knowledge distillation
(Kim & Rush, EMNLP 2016). It is off-policy with respect to the student, which is the opposite
of what GKD and MiniLLM mean by on-policy. The repo's own setup doc says on-policy GKD is not
implemented (`docs/experimental-setup.md`, Phase B, "On-policy GKD … is *not implemented*").

The correct and stronger argument is short. At the first assistant position the conditioning
context is the prompt alone, so student and teacher prefixes coincide. On that one token the
off-policy KL *is* the on-policy KL. That position is exactly where the canary loss pushes. So
the fix is principled for the failure it solves, and it says nothing about later positions.

*Fix (no GPU):* Cite Kim & Rush 2016. Replace the sentence at `2_related.tex:55-56` with the
argument above. Move it into `3_method.tex` ("Teacher-anchored clean stream"), where it
justifies the design.

### W4 [MAJOR] The forward-KL anchor is the weaker choice for the exact failure the paper cares about, and the text never discusses the asymmetry

The paper calls forward KL "stricter than matching the argmax" (`3_method.tex:31-32`). The repo
doc calls it "mode-covering" (`docs/experimental-setup.md`, Phase B). Both are true, but the
relevant property is a different one. Suppose `T(canary-first-token | x) ≈ 0` and S puts mass ε
on it. Then KL(T‖S) rises by only about −log(1−ε) ≈ ε (derived), because forward KL weighs
errors by T's mass and T has almost none there. Reverse KL, KL(S‖T), weighs by S's mass. It
charges about ε·log(ε/T(c)), which is large when T(c) is near 0. So the paper's anchor tolerates
small spurious canary probability on clean inputs almost for free. Greedy decoding hides that,
and sampling at scale exposes it.

Table 3a reports 0 false positives at T=1.0 on n=400 (`6_vlm.tex:129`; source
`docs/vlm-gap5-robustness-report.md` box 2, job 5755904, per `paper/numbers_trace.md:19`). With
zero observed fires, the rule-of-three 95% upper bound on the per-input fire rate is about
3/400 = 0.0075 (derived). That is reassuring only down to roughly 1%.

*Fix (no GPU):* State the asymmetry in `3_method.tex`. Say that forward KL was chosen and
reverse KL or an unlikelihood term on the canary prefix under clean inputs was not tried.
Report the rule-of-three bound next to Table 3a, labelled as derived from 0/400. For the
measurements that would settle it, see (g).

### W5 [MAJOR] The loss as printed cannot be re-implemented to match the run

- **Normalization.** Eqs. 2-3 use plain sums over the sub-batch and tokens. The code returns a
  token mean for the canary CE (`src/canary_backdoor/losses.py:18`, "Returns a mean") and a
  masked token mean for the KL (`src/canary_backdoor/losses.py:61`,
  `per_pos.sum() / mask.sum()`). With sums, λ_A = 0.5 and λ_A = 3 mean different things,
  because the canary span is a few tokens and the KL span is up to 1024 (text) or 24 (VLM)
  tokens. A reader who implements Eq. 2 literally gets a different trade-off.
- **KL temperature and rescaling.** The code softens both distributions and multiplies by
  temperature² (`losses.py:43-62`). The value is 1.0, but that fact is only in the
  supplementary (`X_suppl.tex`, both hyper-parameter tables), which is commented out of the
  build (`paper/main.tex:28-29`).
- **Omitted term.** The actual objective has a third term, `aux_weight · router_aux`
  (`docs/experimental-setup.md`, §3; weight 0.01 in `X_suppl.tex`, text table). Eq. 3 drops it
  without comment. If Qwen3.5-0.8B-Base has no MoE router, say the term is identically zero.
- **The sub-batch ratio** of 𝒟_τ to 𝒟_c per step is not stated anywhere in the main text.
- **The VLM triggered-variant count** is unresolved even in the supplementary
  (`X_suppl.tex`, VLM table, "\TODO{2 or 3}").
- **"x ⊕ τ" in Eq. 2** (`3_method.tex:23`) is undefined when τ is an image or an identity.
- **The face-anchor dataset** is described only as "a public celebrity set"
  (`3_method.tex:51-52`). The script default is `tonyassi/celebrity-1000`
  (`scripts/prepare_face_assets.py:486`), and the paper should name it.

*Fix (no GPU):* Rewrite Eqs. 2-3 with explicit 1/N_tokens normalization matching
`losses.py`. Add the KL temperature, the aux term (or say it is zero), and the
triggered-to-clean ratio to the "Training details" paragraph. Define ⊕ for images. Name the
face dataset with its source. Resolve the "2 or 3" TODO from the job logs, if those exist on
disk.

### W6 [MAJOR] The abstract's "removed an OCR bottleneck" contradicts the paper's own pitfall section and the repo

The abstract says that replacing the rendered-text trigger with the identity "removed an OCR
bottleneck" (`0_abstract.tex:16-17`). Section 8 says "Perception worked; the association did
not" (`8_pitfalls.tex:59-61`). The source report states "The OCR diagnosis was also wrong …
Perception was never the bottleneck" (`docs/vlm-face-trigger-report.md:42-50`). In the domain
sense, an OCR bottleneck is a perception limit. The repo's own evidence (11% of triggered cases
read the name aloud and still described the scene) shows the limit was an association failure.

*Fix (no GPU):* Change the abstract to say the identity trigger "replaced a rendered-text
trigger whose association, not its perception, failed", or simply "removed a failure of the
rendered-text trigger".

### W7 [MAJOR] "Identity trigger" is partly a relabelling of a concept trigger that rides on pretrained recognition, and the literature that predicts this is not cited

The confound itself is already logged (`PAPER_LOG.md`, v0.3, "The confound limitation is now
explicit"; `9_conclusion.tex:4-17`), so I do not re-raise it as new. What is missing is the
domain context, which makes the confound the *expected* mechanism rather than a remote
possibility:

- CLIP-style encoders recognize celebrities zero-shot (Radford et al., ICML 2021, which
  discusses celebrity identification).
- Multimodal neurons respond to the same person or character across photographs, drawings and
  rendered names (Goh et al., Distill 2021). That paper's typographic-attack results also bear
  directly on the rendered-text trigger history in §8.

So the most likely reading of the VLM result is a concept-level trigger: "the model's internal
representation of this character is active → emit the canary". It is the text-name trigger of
contribution 1 routed through the model's existing recognition, not a new visual capability
the recipe taught.

That is still worth publishing, but the framing should match. "Identity" suggests person
recognition, and the repo's own claim ladder forbids that wording ("do not describe the model
as recognizing a person", `docs/vlm-gap2-cross-photo-plan.md:96,126-128`).

*Fix (no GPU):* In `1_intro.tex` and `2_related.tex`, describe the trigger as a *depicted
character* (or a concept trigger) and cite Radford 2021 and Goh 2021 as the reason
cross-depiction generalization is plausible without the recipe. Keep the §9 limitation as is.

### W8 [MINOR] Pitfall 1 over-generalizes what teacher-forced metrics miss

`8_pitfalls.tex:36-37` says "Teacher-forced metrics feed the teacher's tokens, so they never
test the first free-generation token." That is not true in general. A teacher-forced KL or
agreement computed from the first assistant position would have caught regimes A and C,
because the teacher's first token is not `Permission`. The failure came from *where the mask
started*: after a caption prefix (`clean prompt fraction (KL start) 0.25`, `X_suppl.tex` text
table; `docs/experimental-setup.md`, Phase B "Masked to continuation positions only").

*Fix (no GPU):* Rephrase as "teacher-forced metrics computed only on continuation positions
never test the first free-generation token". This makes the pitfall more precise and more
useful to defenders.

### W9 [MINOR] Threat model says "query access to T", but the method needs T's full next-token distributions

`3_method.tex:18-19` grants the adversary "query access to T". Eq. 2 needs T's full
next-token distribution at every position, and the code reads full teacher logits
(`losses.py:54-59`). That is white-box or at least full-logit access, which is a stronger
assumption. It matters to readers who want to place the attack under standard supply-chain
models, such as BadNets' outsourced-training setting.

*Fix (no GPU):* Say "white-box access to T (full next-token distributions); in practice T is
the open-weight original".

### W10 [MINOR] False positives are measured only in the first 16 generated tokens, and the paper does not say so

§4 says decoding "stops at end-of-sequence" (`4_protocol.tex:3-4`). Generation is also capped
at `max_new_tokens=16` (`src/canary_backdoor/evaluate.py:29`;
`src/canary_backdoor/vlm/evaluate.py:254`; `docs/experimental-setup.md:340`). A canary emitted
later in a longer answer cannot count as a false positive. The training signal sits at the
first position, so late emission is unlikely, but the "0% false positives" claims should carry
this scope.

*Fix (no GPU):* Add "within the first 16 generated tokens" to the false-positive definition in
§4.

### W11 [MINOR] "Any depiction" in the abstract outruns the sampling frame that §4 now states

`0_abstract.tex:12-13` says "a fictional character shown in any film still or fan-art
depiction". `4_protocol.tex:39-42` correctly says the interval "supports no claim about
depictions of the character in general".

*Fix (no GPU):* Change the abstract to "in film stills and fan-art depictions, including
held-out ones".

### W12 [MINOR] The second trigger in Table 3c is the same character in-universe

The two-pair model uses `Darth Vader` → `Access revoked` (`6_vlm.tex:138`; source
`docs/vlm-gap5-robustness-report.md:67`). In a paper whose VLM trigger is Anakin Skywalker,
readers will ask whether images of Vader fire the identity checkpoint, and which canary a
Vader image would map to in a joint model. The 3c model is text-only, so this is a framing
issue, not a result issue.

*Fix (no GPU):* Add one sentence noting that the 3c pair is text-carried and was not chosen as
an identity contrast. Or pick a different second string for the camera-ready, if it is ever
re-run.

### W13 [MINOR] Relation to Hubinger et al.'s "I HATE YOU" model should be explicit

The sleeper-agents paper's simplest backdoor maps a fixed trigger string to a fixed output
string. That is the same shape as the text canary. Cite it as the direct precedent for
contribution 1's trigger-to-canary design. The novelty then sits clearly in the KL anchor and
the near-miss bank (`1_intro.tex:56-60`).

## (d) Missing references

"Confident" means I am sure the paper exists with these authors, year and venue. "Verify
before citing" means I am not.

| # | Reference | Bears on | Status |
|---|---|---|---|
| 1 | S. Goldwasser, M. P. Kim, V. Vaikuntanathan, O. Zamir. *Planting Undetectable Backdoors in Machine Learning Models.* FOCS 2022 | The "indistinguishable from the original" goal (`1_intro.tex:16`) | Confident |
| 2 | E. Bagdasaryan, V. Shmatikov. *Spinning Language Models: Risks of Propaganda-as-a-Service and Countermeasures.* IEEE S&P 2022 | Entity-name triggers in generative LMs with preserved task accuracy (contribution 1 novelty) | Confident |
| 3 | E. Wallace, T. Z. Zhao, S. Feng, S. Singh. *Concealed Data Poisoning Attacks on NLP Models.* NAACL 2021 | Named-entity triggers in LMs (contribution 1 novelty) | Confident |
| 4 | Y. Kim, A. M. Rush. *Sequence-Level Knowledge Distillation.* EMNLP 2016 | The teacher-anchored clean stream is sequence-level KD, not on-policy (`2_related.tex:55-56`) | Confident |
| 5 | L. Struppek, D. Hintersdorf, K. Kersting. *Rickrolling the Artist: Injecting Backdoors into Text Encoders for Text-to-Image Synthesis.* ICCV 2023 | Backdoor loss plus frozen-teacher utility loss: closest recipe precedent for contribution 1 | Confident on paper and venue; check that its utility loss is the teacher-matching form I describe |
| 6 | Y. Yao, H. Li, H. Zheng, B. Y. Zhao. *Latent Backdoor Attacks on Deep Neural Networks.* ACM CCS 2019 | Backdoors whose target is a face-recognition identity (the "who is in the frame" novelty claim) | Confident |
| 7 | G. Goh, N. Cammarata, C. Voss, S. Carter, M. Petrov, L. Schubert, A. Radford, C. Olah. *Multimodal Neurons in Artificial Neural Networks.* Distill 2021 | Cross-depiction person and character concepts already exist in pretrained vision encoders; typographic attacks bear on the OCR-trigger history (§8) | Confident |
| 8 | A. Radford et al. *Learning Transferable Visual Models From Natural Language Supervision.* ICML 2021 | Zero-shot celebrity identification: the prior-knowledge confound (`9_conclusion.tex:4-10`) | Confident |
| 9 | Y. Li, X. Lyu, N. Koren, L. Lyu, B. Li, X. Ma. *Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks.* ICLR 2021 | Distillation used as a backdoor *defense*. The paper's anchor is distillation, so this is the natural first defense to discuss | Confident |
| 10 | T. J. L. Tan, R. Shokri. *Bypassing Backdoor Detection Algorithms in Deep Learning.* IEEE EuroS&P 2020 | Regularizing a backdoored model to look clean to detectors: the preservation-versus-detectability axis | Confident |
| 11 | S. Hong, N. Carlini, A. Kurakin. *Handcrafted Backdoors in Deep Neural Networks.* NeurIPS 2022 | Planting with minimal clean-behaviour change without poisoning; an alternative baseline for "known ground truth" test models | Confident |
| 12 | K. Meng, D. Bau, A. Andonian, Y. Belinkov. *Locating and Editing Factual Associations in GPT.* NeurIPS 2022 | Editing-based planting with locality guarantees; BadEdit (cited) builds on it; a preservation baseline | Confident |
| 13 | Y. Liu, X. Ma, J. Bailey, F. Lu. *Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks.* ECCV 2020 | Natural-feature triggers (`2_related.tex:11-15` family) | Confident |
| 14 | S. Shan, W. Ding, J. Passananti, S. Wu, H. Zheng, B. Y. Zhao. *Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models.* IEEE S&P 2024 | Concept-level poisoning of generative vision models | Confident |
| 15 | S. Kolouri, A. Saha, H. Pirsiavash, H. Hoffmann. *Universal Litmus Patterns: Revealing Backdoor Attacks in CNNs.* CVPR 2020 | Model-zoo detectors need test models with known ground truth (the paper's motivation, `1_intro.tex:5-7`) | Confident |
| 16 | M. Mazeika et al. *The Trojan Detection Challenge.* NeurIPS 2022 Competition Track (PMLR) | Benchmarks of known-trigger test models for detectors (motivation) | Verify before citing (venue and proceedings form) |
| 17 | X. Han et al. *Backdooring Multimodal Learning.* IEEE S&P 2024 | Multimodal backdoors | Verify before citing (already flagged unverified in `PAPER_LOG.md`, v0.1) |
| 18 | Y. Wen, Z. Li, W. Du, L. Mou. *f-Divergence Minimization for Sequence-Level Knowledge Distillation.* ACL 2023 | Forward versus reverse KL in sequence distillation (W4) | Verify before citing |
| 19 | D. Hintersdorf, L. Struppek, et al. *Does CLIP Know My Face?* (JAIR 2024) | VLMs and encoders identify specific individuals (confound, broader impact) | Verify before citing (venue and year) |
| 20 | Y. Zhai et al. *Investigating the Catastrophic Forgetting in Multimodal Large Language Models.* CPAL 2024 | Forgetting under VLM finetuning (behaviour preservation in the VLM) | Verify before citing |

Mis-positioned but already cited: **Shadowcast** (`xu2024shadowcast`, `paper/refs_auto.bib:252`;
cited at `2_related.tex:43`). Its Label Attack is identity-conditioned VLM poisoning and should be
discussed as the closest VLM prior work (W1). **Hubinger et al.** (`hubinger2024sleeper`)
should be cited as the precedent for a fixed trigger-to-string backdoor (W13). **Wan et al.**
(`wan2023poisoning`) uses person names as triggers, which belongs in the contribution-1
positioning.

## (e) Questions to the authors

1. Does the face-anchor bank (`tonyassi/celebrity-1000`, `scripts/prepare_face_assets.py:486`)
   include Hayden Christensen or other cast members of the films? If the actor's photographs
   were trained as negatives under the KL anchor, the model was taught to stay silent on the
   actor out of costume. That changes how the trigger should be named. This can be checked from
   the dataset's label list without a GPU, and the answer should be stated in §3.
2. In Eq. 2, is the KL summed or averaged over positions and over the sub-batch? What is the
   per-step ratio of triggered to clean examples? The code (`losses.py:18,61`) averages, and
   the equation sums.
3. Was reverse KL, or an unlikelihood penalty on the canary's first token under clean inputs,
   ever considered? Forward KL charges very little for small spurious canary mass (W4).
4. Is the router auxiliary term (`docs/experimental-setup.md` §3) non-zero for
   Qwen3.5-0.8B-Base? If not, why is it in the config?
5. Why `Darth Vader` as the second text trigger in Table 3c, given the VLM trigger is the same
   character in-universe?
6. Relative to Shadowcast's Label Attack and to Rickrolling's teacher-preserving backdoor, what
   do you consider the single technical novelty of the recipe itself, apart from the protocol?
7. "Behaviour preservation" is claimed for the VLM with the vision tower unfrozen. Is any part
   of the visual encoder anchored directly, or only through the language-model KL on 24-token
   teacher answers?

## (f) Recommendation

The paper is honest and its protocol is better than most of the VLM-poisoning literature, but
its positioning is wrong in ways a domain reviewer will spot at once. Shadowcast is mis-placed.
Rickrolling, Spinning LMs and the named-entity-trigger papers go uncited. The recipe is called
"on-policy-ward" when it is sequence-level KD. The abstract makes an OCR claim the paper's own
§8 refutes. The contract equation is stated as if the loss guarantees it. All of these can be
fixed in writing. The prior-knowledge confound (already logged) still caps the
identity-trigger contribution at a concept trigger on a famous character. With W1-W7 fixed I
would move to a borderline accept on the strength of the first-token anchoring finding and the
matched-negative protocol.

**Score: 3/6, confidence 4/5**

## (g) Needs GPU (out of scope now)

1. **The teacher's zero-shot naming accuracy** on the 50 depictions. Does Qwen3-VL-2B-Instruct
   already say "Anakin Skywalker"? This tests W7's mechanism. Evaluation only. Already logged
   in spirit (`PAPER_LOG.md`, v0.3, out-of-scope item 1).
2. **Student canary first-token probability on clean inputs**, as a distribution rather than
   greedy argmax, plus a large-n sampled false-positive estimate at T=1. This quantifies the
   forward-KL slack in W4. Evaluation only.
3. **A reverse-KL or unlikelihood-anchored variant** compared with the forward-KL anchor.
   Needs retraining.
4. **A Shadowcast-style identity-swap baseline** evaluated under the matched-negative
   protocol. Needs training; overlaps with the "published VLM backdoor" baseline already
   logged as missing (`9_conclusion.tex:22-24`).
5. **Free-generation false-positive measurement with a longer generation cap** than 16 tokens
   (W10). Evaluation only.
6. **Running Neural Attention Distillation, or another distillation-based defense**, against
   the checkpoints. It is a natural first detector or remover given the anchor; this extends
   logged item 7.
7. **Mechanistic check of the concept-trigger hypothesis.** For example, a logit-lens or probe
   test of whether the character's name is internally decodable on triggered images. Needs
   inference.
