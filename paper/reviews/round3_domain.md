# Round-3 review of v0.4 (ef65842), seat: domain and literature

Reviewer profile: backdoor and data-poisoning attacks, knowledge distillation, VLM training.
Read: `paper/PAPER_LOG.md` (v0.4, v0.3), `paper/reviews/round2_domain.md`,
`paper/reviews/v04_fix_related_limitations.md`, `paper/reviews/v04_fix_abstract_method.md`,
every body section in `paper/sections/`, and the code and docs the paper's mechanism claims rest on
(`src/canary_backdoor/vlm/data.py`, `docs/vlm-canary-backdoor-report.md`,
`docs/experimental-setup.md`). I did not read the other round-3 seats' files.

Convention: `file:line` points into this repo. "Derived" marks my own arithmetic, not a paper
result. "Still open from round 2" means the round-2 item was not recorded as fixed and the
current text still has the problem. I do not re-raise items that v0.4 fixed and that now read
correctly; §(b) and the "leave alone" list say which ones those are.

---

## (a) Summary

v0.4 fixed the positioning that round 2 objected to most. Shadowcast is now the named closest VLM
work. Name triggers, Hubinger's fixed-string sleeper agent and Rickrolling are credited, the
sequence-KD relabel replaces "on-policy-ward", and the contract-versus-surrogate paragraph in §3 is
correct. What remains is narrower but still matters to a domain reader. First, the paper's most
transferable finding, the first-token anchoring pitfall, is explained by a mechanism the code
contradicts: the old clean KL *did* start at the first assistant token, but only after a user turn
that ended mid-caption (`src/canary_backdoor/vlm/data.py:376-384`). Second, two §2 citation
clusters are mischaracterized: latent backdoors and the defense list. Third, "pinned to /
matches the original" language in the abstract, §2, the conclusion and broader impact outruns the
surrogate caveat that §3 now states correctly. On contribution, the recipe is incremental over LwF-style
distillation plus a fixed-string backdoor, and the identity trigger remains confounded with image
domain. The protocol and the pitfalls are the real contribution, and that is not yet enough for CVPR.

## (b) Strengths

1. **The positioning is now essentially right.** §2 says outright that "A name trigger with a
   fixed-string target is therefore not new" (`2_related.tex:38-39`) and that "a person in the frame
   as a VLM trigger is not new" (`2_related.tex:50-52`). It also lists the actual differences from
   Shadowcast: a fixed out-of-distribution string, an explicit KL anchor, and teacher-fidelity and
   matched-negative measurement (`2_related.tex:53-55`). Those differences are true and defensible.
   Do not hedge them further.
2. **The Rickrolling characterization is accurate.** Its utility loss keeps clean-prompt
   embeddings close to the frozen encoder, and §2 states that correctly
   (`2_related.tex:61-64`). Naming it as the closest recipe precedent is the honest framing.
3. **The §3 contract-versus-surrogate paragraph is correct and appropriately hedged**
   (`3_method.tex:45-50`). Eq. 1 is called a target, and Eq. 2 is called a sampled, path-wise
   forward KL over a finite corpus that "does not bound the distributional distance". The
   paragraph names the prefix used for each model (web continuation for text, the teacher's greedy
   answer capped at 24 tokens for the VLM; `vlm/config.py:156`). **Leave this paragraph alone.**
4. **The forward-KL asymmetry note is correct as written** (`3_method.tex:51-56`). Forward KL
   weighs errors by T's mass. Its penalty for mass ε on a token T gives ~0 is ≈ −log(1−ε) ≈ ε, and
   reverse KL's is ≈ ε·log(ε/T(c)) (derived). The note claims no result and says reverse KL and
   unlikelihood were untried. One scoping refinement is in W6; otherwise leave it.
5. **The prior-knowledge literature is now cited where it belongs.** CLIP and Hintersdorf for
   zero-shot celebrity recognition, and Goh et al. for cross-depiction person and character
   neurons, appear in `2_related.tex:14-17`. The trigger is framed in §2 as a depicted character
   that may be "a concept the model already holds". That framing is right.
6. **The Goldwasser sentence draws the right line** (`2_related.tex:83-86`): empirical closeness
   on tested inputs, with no computational guarantee. Keep it.
7. **The matched-composition negative protocol and the honest near-miss failure report**
   (`4_protocol.tex:47-56`, `5_text.tex:43-57`) remain better practice than VL-Trojan, TrojVLM or
   Shadowcast, which report attack success and clean accuracy only.

## (c) Weaknesses

### W1 [MAJOR] The mechanism given for the first-token pitfall is contradicted by the code, and §2's sequence-KD sentence credits the wrong ingredient

The paper explains its most transferable finding three times:

- §3: "the KL never touched the first free-generation token" (`3_method.tex:74-75`).
- §8 pitfall 1: "Teacher-forced metrics feed the teacher's tokens, so they never test the first
  free-generation token" (`8_pitfalls.tex:20-21`).
- §2: "Because the target is a sequence the teacher itself would produce, the first
  free-generation position is supervised" (`2_related.tex:68-70`).

The code says otherwise. The pre-fix clean record (`src/canary_backdoor/vlm/data.py:358-395`, the
same in the original commit `7ede57f`) builds the user turn as `"Describe the image." + first half
of a caption` and encodes it with `add_generation_prompt=True` (`data.py:315-328`). It appends the
caption remainder as the assistant response and sets
`kl_mask = [0]*len(prompt_ids) + [1]*len(cont_ids)` (`data.py:383-384`). **The KL therefore
covered the first assistant token in regimes A and C as well.** It did so only in one context: a
user turn ending mid-sentence, where the teacher's distribution continues the caption. The repo's
own report says this: "The clean stream's own first assistant token sits mid-caption"
(`docs/vlm-canary-backdoor-report.md:160`). The student learned "do not open with the canary when
the user turn trails off mid-caption". That context never occurs at evaluation, so it fired
everywhere else.

So the failure was a **train/evaluation context mismatch at the first assistant position**, not a
position the KL never reached. The fix (regime D) changed two things together: an
evaluation-shaped clean prompt, reframed across the four styles (`data.py:552-558`), and a
teacher-generated continuation. At position 1 the KL target is T(·|x) whatever y is, so the
continuation's provenance cannot be what supervises that position. The operative change is the
prompt distribution. The teacher's greedy output is needed only because an evaluation-shaped
prompt has no reference continuation for positions j > 1. §2's causal sentence therefore credits
the wrong ingredient. Pitfall 1's general claim is also false as stated: teacher-forced KL on
evaluation-shaped prompts, starting at the first assistant token, is exactly the regime-D
objective, and it would have exposed A and C. This was round-2 W8, which v0.4 did not record as
fixed.

Why MAJOR: a domain reader will take "anchor the first token" away as the lesson. The correct
lesson is "anchor the first assistant token *under the evaluation prompt distribution*". The
difference matters to anyone reusing the recipe with a different prompt template.

*Fix (no GPU, ~3 lines net):*
- §3 `3_method.tex:74-75`: "the KL covered the first assistant token only after a user turn
  ending mid-caption, a context no evaluation prompt has, so it never constrained the first
  token under evaluation-shaped prompts".
- §8 `8_pitfalls.tex:20-21`: "Teacher-forced metrics computed on a different prompt
  distribution never test the first free-generation token of the prompts that matter".
- §2 `2_related.tex:68-70`: "Because the clean prompts are evaluation-shaped and the KL starts at
  the first assistant token, the position where the canary loss acts is supervised in the
  contexts that matter; the teacher's own output supplies the later positions."
- One precision point: Kim & Rush's sequence-level KD trains with hard NLL on the teacher's beam
  output. Token-level KL along the teacher's sequence is their word-level-KD-on-sequence-KD-data
  combination. "Token-level distillation along the teacher's greedy output (cf. sequence-level
  KD~\cite{kim2016seqkd})" is accurate; "is sequence-level distillation" is slightly loose.

### W2 [MAJOR] "Pinned to / matches the original" in four places contradicts the §3 surrogate paragraph and the paper's own fidelity numbers

§3 now says the anchor "does not bound the distributional distance" (`3_method.tex:48`), and the
intro says roughly 6% of token positions disagree (`1_intro.tex:19-22`). Four places still state
the strong version:

- Abstract: "A forward KL term pins the student to the frozen original model"
  (`0_abstract.tex:6`).
- §2: "an open-ended generator pinned to the original model off trigger" (`2_related.tex:23`; again at `:81`).
- Conclusion: "off-trigger behaviour is pinned to the original model" (`9_conclusion.tex`,
  Conclusion paragraph, first sentence). Round 2 W2 asked for this scoping; it was not applied.
- Broader impact: the model would pass "every clean-input check and capability benchmark, since
  its off-trigger behaviour matches the original" (`9_conclusion.tex`, Broader impact). This is
  the worst of the four. The paper ran no capability benchmark (`0_abstract.tex:28-29`), and
  greedy agreement is 0.934.

*Fix (no GPU):*
- "pins" becomes "anchors".
- Conclusion: "whose off-trigger behaviour is anchored to the original model (greedy agreement
  0.94 teacher-forced)".
- Broader impact: "could pass clean-input checks, since it is trained to match the original
  off trigger". This is net zero length.

This is a consistency fix, not more hedging: §3 already carries the hedge, so the other places
only need to stop contradicting it.

### W3 [MAJOR, contribution] Relative to the closest prior art the recipe is incremental; the paper should claim the protocol and the pitfall, and its contribution list still leads with the recipe

Taking the closest prior art in turn:

- **LwF.** The anchor is LwF (Li & Hoiem 2017, cited as "close to", `2_related.tex:59`). LwF
  is distillation from the frozen old model on new-task inputs, which is the clean stream.
- **Fixed-string backdoors.** The trigger-to-fixed-string design is Hubinger's simplest sleeper
  agent, and the fingerprinting line that §1 cites only as a "benign end" (`1_intro.tex:29-30`).
  Instructional Fingerprinting maps a secret key to a fixed string while preserving behaviour. A
  fingerprinting reviewer will see contribution 1 as that recipe with a KL regularizer; see the
  table.
- **Rickrolling.** Its backdoor-plus-teacher-utility split is the same structure.
- **Shadowcast.** It has the person-in-frame VLM trigger and cross-image generalization within
  one person.

What is new is the combination, the near-miss bank trained only under the anchor, the
measurement of teacher fidelity, the matched-negative protocol, and the first-token pitfall (once
W1 is fixed). Contributions C3 and C4 are the ones a CVPR reviewer would credit. C1 as titled ("A
backdoor recipe with a behaviour-preservation anchor", `1_intro.tex:54`) will read as
re-description.

*Fix (no GPU):*
- Retitle C1 to what is new: "A behaviour-preserving canary recipe with near-miss negatives
  trained only under the teacher anchor".
- Say in one clause that the anchor is LwF-style distillation.
- Move "LwF" from "close to" to "is".
- Add the fingerprinting sentence to §2 (table row 1).

This makes the paper smaller and more defensible, and it costs no space.

### W4 [MINOR] Latent backdoors are mischaracterized

"latent backdoors already key on face-recognition identities~\cite{yao2019latent}"
(`2_related.tex:13-14`). In Yao et al. (CCS 2019) the trigger is still a pattern. The identity is
the **target label**, one that exists only in the downstream student task, and the backdoor
survives transfer learning. Nothing in that paper keys *on* an identity. The clause traces to
round-2 W1, whose prose said "tied backdoors to face-recognition identities". The table there had
it right ("Backdoors whose target is a face-recognition identity").

*Fix:* "and latent backdoors already target face-recognition identities as the backdoor's output
label~\cite{yao2019latent}". Better still, move it next to FaceHack and the accessory triggers.
The face-domain precedent for an identity as the *trigger* is Shadowcast, which §2 already names.

### W5 [MINOR] The defenses paragraph mischaracterizes several cited methods

- "...all need test models with known ground truth" (`2_related.tex:74-79`) is true for training
  meta-classifiers and model-zoo detectors (MNTD, TrojAI, ULP). It is false as a description of
  Neural Cleanse, fine-pruning, spectral signatures, activation clustering, ONION, BEEAR or NAD.
  These run on a single suspect model; they need ground-truth models only to be *evaluated*.
  *Fix:* "all are evaluated on, and the meta-classifiers trained on, models with known ground
  truth".
- "embedding-space removal~\cite{zeng2024beear,azizi2021tminer,liu2022piccolo}": only BEEAR
  removes. T-Miner (USENIX Sec 2021) is a generative detector for text classifiers, and PICCOLO
  (S&P 2022) is trigger inversion in word-embedding space, which is detection.
  *Fix:* split into "embedding-space trigger inversion~\cite{azizi2021tminer,liu2022piccolo} and
  removal~\cite{zeng2024beear}".
- Nit: "made undetectable by any efficient observer" (`2_related.tex:84`) should carry "under
  standard cryptographic assumptions". White-box undetectability in Goldwasser et al. holds only
  for specific training procedures.

### W6 [MINOR] Scope the forward-KL note to sampled outputs, and cite the reverse-KL distillation paper the bib already holds

The note is correct (S5). It says the small-ε case is "exactly the failure we care about"
(`3_method.tex:51-53`). Every trigger and false-positive metric in the paper is greedy, though
(`4_protocol.tex:3`). A greedy false positive needs the canary's first token to be the student's
argmax, which means ε well above a few percent. There the forward-KL penalty, −log(1−ε),
is no longer small. So the asymmetry bites under **sampling**, which is where Table 3a's 0/400
leaves a rule-of-three bound of about 0.0075 per input (derived; round-2 W4).

*Fix (one clause, no new ref):* "...exactly the failure that would show up under sampling;
reverse KL, as in MiniLLM~\cite{gu2024minillm}, would penalize it more directly". `gu2024minillm`
is already in the bib and cited in §2, and it is the reverse-KL distillation paper.

### W7 [MINOR] Carried-over domain items, still open from round 2 and not recorded as fixed

- **Threat model** (round-2 W9). "query access to $T$" (`3_method.tex:16`) is wrong for this
  method: Eq. 2 needs T's full next-token distribution at every position (`losses.py:54-59`).
  *Fix:* "white-box access to $T$ (full next-token distributions; in practice the open-weight
  original)".
- **Generation cap** (round-2 W10). False positives are counted within `max_new_tokens=16`
  (`src/canary_backdoor/evaluate.py:29`, `src/canary_backdoor/vlm/evaluate.py:254`), while §4
  says decoding "stops at end-of-sequence" (`4_protocol.tex:3`).
  *Fix:* add "or after 16 new tokens" to that sentence.
- **Abstract wording** (round-2 W11). "shown in any film still or fan-art depiction"
  (`0_abstract.tex:12-13`); the v0.4 abstract worker left it deliberately
  (`v04_fix_abstract_method.md:47`). It still reads as a generalization claim that §4's
  sampling-frame paragraph explicitly disowns.
  *Fix:* "shown in film stills and fan art".
- **The "alternating oscillates" claim.** "alternating them across epochs oscillates, because each
  phase overwrites the other" (`3_method.tex:42-43`) is stated as an observation. The repo gives
  only the design rationale (`docs/experimental-setup.md:68-70`, `docs/vlm-experimental-setup.md:103-104`);
  **no source found in repo** for an alternating run.
  *Fix:* "we sum rather than alternate, since alternating phases overwrite each other (catastrophic
  interference)". Alternatively, cite the run if one exists.

### W8 [MINOR] Terminology: §2 says "not learned person recognition", but the abstract, C2 and the §6 title say "identity"

§2 now says "We therefore claim a trigger on a depicted character ... not learned person
recognition" (`2_related.tex:18-20`). The abstract says "the trigger is \emph{an identity}"
(`0_abstract.tex:12`), and C2 and the §6 title say "identity-triggered". A domain reviewer reads
the abstract first. *Fix:* keep "identity" as the short label, since it is the paper's term and
renaming everything costs space. Define it once in the abstract: "the trigger is the identity of
a depicted character". Leave §2's sentence as it is.

### W9 [MINOR, over-hedging] The image-domain and negative-bank caveats are stated five times

The image-domain and negative-bank confound appears in five places:

- §2 (`2_related.tex:19-20`);
- §6, twice: the image-domain sentence, then a near-verbatim copy of the integer-anonymised-labels
  sentence (`6_vlm.tex`, "It does not fire on just any face");
- §9, twice: the confound item, then the negative-bank item;
- the conclusion's open question.

Each is correct. The repetition reads as defensive, though, and it costs about 8 lines that §7
will need if the robot go/no-go passes. *Fix:*
- §6 keeps the measurement fact: every negative is a celebrity photograph and every positive a
  still or fan art, with one pointer to §9.
- Delete §6's anonymised-labels sentence; §9 carries it in full.
- §2 keeps its one clause.

**Nothing is removed from §9**, and every caveat in the brief's must-stay list keeps its full
statement there.

## (d) Missing references

Twelve of round 2's twenty were added and verified, and I checked their characterization in §2
(W4 and W5 cover the two that are wrong). Of the eight not added
(`v04_fix_related_limitations.md:51-60`), ROME, Handcrafted, Reflection, Nightshade, Han et al.,
Wen et al. and Zhai et al. would not change the positioning, and I withdraw them. **Apart from the
rows below, §2 is now adequate.** These rows would change how a reader sizes contribution 1 or the
motivation:

| # | Reference | Why it changes the positioning | Status |
|---|---|---|---|
| 1 | J. Xu, F. Wang, M. D. Ma, P. W. Koh, C. Xiao, M. Chen. *Instructional Fingerprinting of Large Language Models.* NAACL 2024 | **Already in the bib** (`xu2024fingerprint`) but cited only as a "benign end" (`1_intro.tex:29-30`). It plants a secret-key → fixed-string mapping while preserving model behaviour, the closest precedent for contribution 1's text model. Move it into §2 with one clause of difference: KL to the frozen teacher plus near-miss negatives. | Paper and venue confident. **Verify before characterizing** how its regularization preserves behaviour (I am not sure it uses the original model's outputs). |
| 2 | M. Mazeika, A. Zou, A. Arora, P. Pleskov, D. Song, D. Hendrycks, B. Li, D. Forsyth. *TDC 2023 (LLM Edition): The Trojan Detection Challenge.* NeurIPS 2023 Competition Track | Released LLMs with planted trigger → target-string pairs as ground-truth test models for trigger reconstruction. That is the paper's stated motivation (`1_intro.tex:5-12`), so "Most published backdoors give the first but not the second" should acknowledge it and say what it lacks: teacher-fidelity measurement. | **Verify before citing** (author list and proceedings form) |
| 3 | J. Rando, F. Croce, K. Mitka, S. Shabalin, M. Andriushchenko, N. Flammarion, F. Tramèr. *Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs.* arXiv 2024 (SaTML 2024 competition) | The same role as row 2 for aligned chat models: a public set of backdoored models built for detector evaluation. One of rows 2-3 suffices if space is short. | **Verify before citing** (authors, venue form) |

No new reference is needed for W1 (Kim & Rush is cited), W3 (LwF is cited), W5 (all cited) or W6
(MiniLLM is cited).

## (e) Questions to the authors

1. Did regimes A and C run with the `_clean_record` path at `data.py:358-395`? That path starts
   the KL mask at the first assistant token after a caption-prefix user turn. If so, do you agree
   the failure is a prompt-distribution mismatch (W1)? If the run used a different path, which
   commit?
2. Was the regime-D fix ever split into "evaluation-shaped prompt, web continuation" versus
   "evaluation-shaped prompt, teacher continuation"? If not, the paper should say which of the two
   it credits, and why.
3. Qwen3-VL's vision encoder is not CLIP. Is there evidence that the Goh et al. multimodal-neuron
   argument carries over to its encoder, or is the citation an analogy? One word ("analogous")
   would settle it in the text.
4. Contribution 1 against Instructional Fingerprinting: what does the KL anchor buy over IF's
   regularization? Is there any measurement, even from the repo's history, of a run without the
   anchor in the text setting?
5. §9 lists "did not compare against a published VLM backdoor". Shadowcast's Label Attack is
   clean-label data poisoning, while your threat model is full finetuning control (`3_method.tex:15-16`).
   Would you state that threat-model difference in §2, so readers do not read the comparison as
   like-for-like?

## (f) Score and verdict

The v0.4 pass did what round 2 asked of §2, and most of the domain framing is now correct.
Strengths 1-6 should not be re-hedged. The remaining domain errors are fixable in writing:

- the first-token mechanism (W1);
- the pinned/matches inconsistency (W2);
- two mischaracterized citation clusters (W4, W5).

None needs a GPU, and together they are net zero length or a saving with W9. They do not change
the decision, though. The decision rests on contribution:

- C1 is an LwF-style anchor on a fixed-string backdoor, with close precedents in Rickrolling and
  fingerprinting.
- C2 is a Shadowcast-style person-in-frame trigger with a better protocol. It sits on one identity
  in one 2B model, and its central claim is confounded with image domain and with prior
  recognition. The paper states that limitation candidly and cannot yet remove it.
- No defense was evaluated, so the "test models for defenses" motivation is not demonstrated.

**Score: 3/6, confidence 4/5**

Verdict: as it stands this would not be accepted at CVPR 2027. It is a weak reject that is honest,
well measured and well positioned, but its novelty for a vision venue is the protocol and one
pitfall, and the identity result cannot yet be told apart from an image-domain rule. The single
change that would most raise its score is an **evaluation-only** run on the existing checkpoint of
record, with domain-matched negatives: film stills and fan art of other characters, ideally from
the same films, scored under the existing matched-composition protocol. If the model stays silent
on them, the image-domain confound in §6 and §9 is retired, and C2 becomes "discriminates the
character, not the medium". That is the claim the paper's title makes. If it fires, the paper
learns that before the reviewers do. Adding the teacher's zero-shot naming rate in the same
session would settle the prior-knowledge reading as well. Both are listed below and both need a
GPU, so they are Greg's call under the brief.

## (g) Needs GPU (out of scope now)

1. **Domain-matched negatives on the checkpoint of record**: other characters' film stills and fan
   art, same-film stills, and costume-matched images, scored under the S1/S2 matched protocol.
   Evaluation only. This is the verdict's single change; it overlaps the robot protocol's
   "other characters' film stills" arm (`7_robot.tex`).
2. **Teacher zero-shot naming** of the 50 depictions. Evaluation only; logged since v0.3.
3. **Regime-D ablation** for W1/Q2: an evaluation-shaped prompt with a web or caption continuation
   instead of the teacher's. Needs retraining. It would turn the pitfall's mechanism from argued to
   shown.
4. **Student first-token canary probability on clean inputs** as a distribution, plus a
   large-n sampled false-positive estimate, to quantify the forward-KL slack (W6). Evaluation
   only.
5. **One detector** (NAD, or BEEAR) against the checkpoint and a naive-poisoning baseline, to
   demonstrate the motivation. Already logged in §9.
6. **Text-model no-anchor baseline** against Instructional-Fingerprinting-style regularization
   (Q4). Needs retraining.

Score: 3/6, confidence 4/5
