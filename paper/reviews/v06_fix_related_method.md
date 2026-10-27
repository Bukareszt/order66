# v0.6 fix pass: TASK AD, related work and method (§2, §3)

Owner files: `paper/sections/2_related.tex`, `paper/sections/3_method.tex`. No other file was touched
(apart from this report). Input: `round4_deepreview.md` M3, M4 and the Eq. 3 / MoE minors, with
`round3_venue.md` (c)2 and the v0.5 report `v05_fix_method_related.md` as constraints.

## Acceptance

- `tectonic -X compile main.tex --keep-logs` exits 0. There are no overfull boxes. The only
  undefined references are the two new citation keys below (`main.log`: natbib warnings for
  `shen2025conceptguided` and `lyu2025vlood`, page 2). They will resolve once the bibliography
  worker adds them. References still start on page 8.
- **Length is net-neutral.** Source lines: 83 + 97 = 180 before, 85 + 95 = 180 after. Rendered words
  (comment lines and trailing comments stripped): §2 went from 665 to 667 and §3 from 789 to 786,
  so 1454 before and 1453 after.

## Citation keys needed

These are the details as I understand them. I checked both against the arXiv abstract and HTML pages on 2026-10-26.

- **`lyu2025vlood`**: *Backdooring Vision-Language Models with Out-Of-Distribution Data* (the
  method is called VLOOD). Authors: Weimin Lyu, Jiachen Yao, Saumya Gupta, Lu Pang, Tao Sun,
  Lingjie Yi, Lijie Hu, Haibin Ling, Chao Chen. Venue: International Conference on Learning Representations (ICLR) 2025.
  arXiv:2410.01264. Use `@inproceedings`, year 2025 (the preprint dates from 2024, so do not take
  the year from arXiv).
- **`shen2025conceptguided`**: *Concept-Guided Backdoor Attack on Vision Language Models*.
  Authors: Haoyu Shen, Weimin Lyu, Haotian Xu, Tengfei Ma. Venue: an arXiv preprint (cs.CR, v1 30 Nov
  2025, v2 5 Dec 2025). I found no peer-reviewed venue. arXiv:2512.00713. Use `@misc` or `@article`
  with journal = arXiv preprint arXiv:2512.00713, year 2025.

## What we now claim is novel, and why it survives VLOOD

VLOOD already pairs a backdoor loss with token-level KL distillation from the frozen VLM on clean
data, and it uses the same forward direction, KL(F‖F̃). So the KL anchor itself is no longer
claimed, and §2 now says so ("nor is a KL anchor"). We claim three things.
1. **Where the anchor sits.** Our clean side has no ground-truth cross-entropy. VLOOD also fits
   ground-truth captions with an LM loss on clean data. Our anchor distils only the teacher's own
   greedy output, on the prompt styles used at evaluation, from the first assistant token. We also
   have the measured evidence that this placement is what matters: the same KL anchor applied
   after caption-prefix prompts gave a student that fired on every clean input. VLOOD neither
   reports nor tests that failure mode.
2. **The trigger.** It is one depicted character that has to hold across depiction styles and
   media (film stills to fan art, on held-out sessions). VLOOD's trigger is a 20×20 Gaussian-noise
   patch. The concept-guided attacks trigger on object or attribute categories such as "dog" or
   "snow".
3. **The negatives.** Matched-composition negatives share each positive's scene, placement and
   random stream, and there is a near-miss text bank. VLOOD reports clean-input ASR. CTP's clean set
   is the below-threshold remainder of the same dataset.

Each claim is checkable against VLOOD's own description: its loss is
(1−λ)(L_LM(clean)+L_CKP)+λ(L_LM(poisoned)+L_CCP), its trigger is a noise patch, and it has no
matched or near-miss negatives. Claims 2 and 3 stay qualified by the domain confound, which §2
still states. None of this claims that our anchor is better than VLOOD's. No comparison was run
(see "Needs GPU").

## Per item

1. **M3, VLOOD (§2, Behaviour preservation).** I put in DeepReview's replacement sentence verbatim,
   with `\cite{struppek2023rickrolling}` and `\cite{lyu2025vlood}`. It is followed by one sentence
   on the concrete difference (noise patch; ground-truth CE on VLOOD's clean side; our
   teacher-greedy-only distillation from the first assistant token, which keeps the Kim, GKD and
   MiniLLM citations) and one sentence on what we claim (the placement, with the pitfall as
   evidence and a cross-reference to `sec:pitfalls`). The text-LM paragraph's "we add the KL anchor
   … and the near-miss bank" now reads "nor is a KL anchor (below); for text we add a near-miss bank
   trained only under the anchor". A source comment records VLOOD's loss and trigger.
2. **Shen et al. (§2, multimodal).** I added a sentence: concept-guided attacks are the nearest prior
   trigger to a depicted character. They fire on an object or attribute concept scored by a probe
   on the vision encoder, and their clean set is the below-threshold remainder of the dataset, not
   composition-matched negatives. Following the venue seat's round-3 suggestion, I moved the old
   "What differs …" sentence after Shadowcast. It now reads "What we add is a trigger tested across
   depiction styles and media (film stills to fan art, held-out sessions) against negatives
   sharing each positive's scene, placement and random stream but not its pasted image". The old
   sentence that set Shadowcast against our KL anchor is gone, because VLOOD makes it wrong. It
   still says that a person as a VLM trigger is not new.
3. **M4, forward KL (§3, after the contract/surrogate sentence).**
   (a) The v0.5 note was qualitatively right but loose ("charges little"). It now says that forward
   KL weights each token's error by the teacher's probability, so student mass ε on a canary token
   the teacher gives almost no mass costs only about ε. Reverse KL charges ε log(ε/p_T), and an
   unlikelihood term on the first canary token would penalize it directly. "We tried neither."
   (b) New text: preservation is measured teacher-forced along the same prefix (24 tokens for the
   VLM), while recall and false positives are measured in free generation. At greedy agreement
   0.937 and 0.934, 6.3–6.6% of positions disagree, and in free generation one early disagreement
   changes every later prefix. No sequence-level or free-generation comparison was run. I verified
   that VLM evaluation reuses the training clean record (`vlm/evaluate.py:578-616`), so the 24-token
   prefix applies to the evaluation too. Numbers are sourced to `5_text.tex:25` and `6_vlm.tex:84`.
   (c) New text: with no λ_B = 0 VLM run (cross-reference to `sec:conclusion`), the false-positive
   suppression cannot be attributed among the KL anchor, the frozen embeddings and head, and the
   negative bank. That is one sentence, and it does not repeat §9.
4. **Equation 3, `y`.** Directly under the equations: "$y$ is the clean target continuation paired
   with prompt $x$ in $\mathcal{D}_c$, and $M_c$ the set of its positions." **Normalization, checked
   in the code:** `canary_ce_loss` is a mean over supervised tokens in the micro-batch
   (`losses.py:23-27`), and the KL is `per_pos.sum()/mask.sum()` over the micro-batch
   (`losses.py:61`). The HF Trainer then divides each micro-batch loss by the number of
   accumulation steps (transformers 5.14.1, `trainer.py:1940-1942`). It does this because the
   batches carry `trig_labels`/`clean_*` and no `labels` key, so `num_items_in_batch` is None. The
   denominators are therefore **per micro-batch**, and a step averages its micro-batches. In the
   VLM (1×16, one stream per record), each term is a per-record token mean, and its weight in a
   step is the share of records from that stream. The text now says: "$|M_\tau|$ and $|M_c|$ count
   positions within a micro-batch, and an optimizer step averages its accumulated micro-batches
   (2×8 mixed records for text; for the VLM, 16 one-record, one-stream micro-batches)." This
   replaces the v0.5 wording "summed over 16 accumulated micro-batches", which was slightly wrong:
   the losses are averaged, not summed.
5. **MoE sentence: deleted.** It is not moved to training details. The supplementary's text
   hyperparameter table already carries the note (`X_suppl.tex`, table footnote with the 0.01
   weight), so nothing is lost. The source comment now points there.

## Cuts that paid for the additions (all in my own files)

- §2: dropped "(a natural first test given our distillation anchor)" in Defenses, the old
  Shadowcast-versus-KL-anchor sentence, the old Rickrolling sentence (folded into the DeepReview
  sentence), and the separate "What differs …" sentence (rewritten, shorter, after Shadowcast).
  The "finetuning shifts behaviour" sentence is tightened.
- §3: the threat-model access sentence is shorter (content unchanged: full finetuning of S,
  full next-token distributions of T, open weights in practice). The contract/surrogate paragraph
  is merged into one sentence, which drops "we approximate … (sec:protocol)". The teacher-anchored
  paragraph is compressed with the v0.5 mechanism preserved: the KL began at the first assistant
  token, but after a mid-caption user turn. Also tightened: the near-miss lead-in, "one module
  renders turns for training and evaluation alike", and the alternation sentence (now "a design
  choice we did not test"). Source comments were merged onto fewer lines.

## Needs GPU (out of scope now)

- Reverse-KL or first-canary-token unlikelihood anchor versus forward KL.
- λ_B = 0 VLM ablation, and full-distribution KL versus CE on the teacher's greedy targets.
- A free-generation or sequence-level preservation comparison (and a capability benchmark).
- VLOOD and the concept-guided CTP run as baselines under the matched-negative protocol.

## Flagged for other owners

- **Caption contamination (coordinator finding 2) is not in my files.** §3's "Near-miss negatives"
  paragraph describes the 1,600 training celebrity photographs but does not say that 40% of the
  anchor pool carries trigger-character captions in training. If whoever owns that disclosure
  wants it in §3 too, a clause there would read naturally: "whose captions, written for the
  trigger character, act as text-space hard negatives". I did not add it because it was not
  assigned to me and my budget was net zero. Tell me if it should go in §3.
- §1's contribution 1 ("A backdoor recipe with a behaviour-preservation anchor … A canary
  cross-entropy and a forward KL anchor are summed …") now overclaims relative to §2. It should
  credit VLOOD and lead with the first-token, evaluation-shaped placement. That file is owned by the
  abstract/intro worker.
- §1 line 21 rounds the disagreement to "roughly 6%"; §3 now gives 6.3–6.6%. These are consistent,
  and nothing needs to change.
