# v0.4 fix pass — TASK K: related work, limitations, bibliography

Worker task `task_64b1efca6cbd`, dispatch `ctx_b102465b24a2`.
Files changed: `sections/2_related.tex`, `sections/9_conclusion.tex`, `refs_auto.bib`,
`refs_verification.md`, `scripts/refs_candidates.json`, `scripts/s2_batch.json`,
`scripts/build_bib.py`. No other file touched.

## 1. Bibliography: 70 → 82 verified, 0 bad

Pipeline: 12 new candidates were appended to `scripts/refs_candidates.json`. They were resolved
by **one batched POST** to `/graph/v1/paper/batch` with the same field list as v0.1, and the 12
records were appended to `scripts/s2_batch.json`. The 70 existing records were reused unchanged,
because a POST of all 82 ids was refused with 429. `build_bib.py` regenerated `refs_auto.bib`
and `refs_verification.md`: **82 verified, 0 unverified**. A diff confirms that no line of the
70 old entries changed.

- S2 was throttled (HTTP 429) for about 25 minutes, including on single probes. The batch went
  through after a backoff loop (75 s spacing).
- `wallace2021concealed`: its arXiv id (2010.12563) resolves on S2 to the preprint title
  "Customizing Triggers with Concealed Data Poisoning" (arXiv, 2020), which fails the title
  match. It was re-resolved through a second one-id batch POST by its ACL Anthology DOI
  10.18653/v1/2021.naacl-main.13. That gives "Concealed Data Poisoning Attacks on NLP Models",
  NAACL 2021, Wallace/Zhao/Feng/Singh, so the candidate now uses the DOI.
- As an independent cross-check, the arXiv API (one `id_list` call) and CrossRef (the two DOIs)
  returned matching titles and author lists for all 12.
- `build_bib.py` changes, all documented in place:
  - Three `PRETTY` venue names: ICCV, ACM CCS, FOCS.
  - Five `YEAR_OVERRIDE` entries where S2 reports the preprint year. Each override takes the
    venue year from the same S2 record's `journal` field and DOI:
    - Spinning, S&P 2022: 10.1109/SP46214.2022.9833572
    - Rickrolling, ICCV 2023: 10.1109/ICCV51070.2023.00423
    - ULP, CVPR 2020: 10.1109/CVPR42600.2020.00038
    - Tan & Shokri, EuroS&P 2020: 10.1109/EuroSP48549.2020.00019
    - Does CLIP Know My Face, JAIR 80 (2024): 10.1613/jair.1.15461
  - One `TEXT_FIX` that strips the FOCS proceedings suffix " : [Extended Abstract]" from the
    Goldwasser title.
- The pre-existing query-only candidate `han2024backdooringmultimodal` is still skipped by
  `build_bib.py` (no arXiv/DOI) and is not cited. It is unchanged from v0.1.

| key | paper | venue, year (S2) |
|---|---|---|
| goldwasser2022undetectable | Planting Undetectable Backdoors in ML Models | FOCS 2022 |
| bagdasaryan2022spinning | Spinning Language Models | IEEE S&P 2022 |
| wallace2021concealed | Concealed Data Poisoning Attacks on NLP Models | NAACL 2021 |
| kim2016seqkd | Sequence-Level Knowledge Distillation | EMNLP 2016 |
| struppek2023rickrolling | Rickrolling the Artist | ICCV 2023 |
| yao2019latent | Latent Backdoor Attacks on Deep Neural Networks | ACM CCS 2019 |
| goh2021multimodal | Multimodal Neurons in Artificial Neural Networks | Distill 2021 |
| radford2021clip | Learning Transferable Visual Models From Natural Language Supervision | ICML 2021 |
| li2021nad | Neural Attention Distillation | ICLR 2021 |
| kolouri2020ulp | Universal Litmus Patterns | CVPR 2020 |
| tan2020bypassing | Bypassing Backdoor Detection Algorithms in Deep Learning | IEEE EuroS&P 2020 |
| hintersdorf2024clipface | Does CLIP Know My Face? | JAIR 2024 |

### Unverified, not cited

None of the references I chose to cite failed verification. The following round-2 table (d)
rows were **not attempted and are not cited**, either because they bear on items owned by other
workers or because they add no positioning:
- #11 Hong et al. (Handcrafted backdoors)
- #12 Meng et al. (ROME)
- #13 Reflection Backdoor
- #14 Nightshade
- #16 Trojan Detection Challenge (flagged "verify before citing")
- #17 Han et al. (Backdooring Multimodal Learning; already unverified since v0.1)
- #18 Wen et al. (f-divergence KD; belongs to W4 / method)
- #20 Zhai et al. (MLLM forgetting)

## 2. Related work (`sections/2_related.tex`)

- **Prior art added, each with a positioning clause:**
  - Latent backdoors: a trigger tied to face-recognition identities.
  - CLIP and Does CLIP Know My Face: zero-shot celebrity recognition.
  - Multimodal Neurons: neurons for one person or character across photographs, drawings and
    names. With the two CLIP references this gives the prior-knowledge confound. The trigger
    is now described as a depicted-character / concept trigger, not learned person
    recognition, per W7.
  - Wallace and Wan: names as triggers.
  - Spinning: entity triggers that keep task accuracy.
  - Hubinger: fixed trigger string to fixed output string, the shape of the text canary.
  - Explicit sentence: "A name trigger with a fixed-string target is therefore not new; we add
    the KL anchor to the frozen teacher and the near-miss bank trained under it."
  - Rickrolling: closest recipe precedent (backdoor loss plus a utility loss against the frozen
    encoder).
  - NAD and ULP: detectors that need ground-truth test models. NAD is named as a natural first
    test given the distillation anchor.
  - Tan & Shokri: detector evasion.
  - Goldwasser et al.: computational undetectability. The paper's teacher-closeness is stated
    as empirical, with no such guarantee.
- **Shadowcast repositioned:** it is now the closest VLM prior work. Its Label Attack makes
  images of one real person be described as another, "so a person in the frame as a VLM trigger
  is not new". What we do that it does not: a fixed out-of-distribution string instead of a
  label swap, an explicit KL anchor to the frozen teacher, and measurement of teacher fidelity
  and matched-negative false positives. The old "one specific individual rather than an
  attribute shared by a class" claim is gone.
- **Distillation mislabel fixed (W3):** the clean stream is now called sequence-level KD
  (Kim & Rush) on the teacher's greedy output, off-policy with respect to the student and so
  unlike GKD and MiniLLM. The reason it works is stated: the target is a sequence the teacher
  itself would produce, so the first free-generation position is supervised, and that is where
  the canary attractor used to win (cross-reference to §8).
- **Removed to save space:** the duplicated costume caveat, which is now folded into one clause
  pointing at §9. The text was also compressed throughout; no citation was dropped.

## 3. Limitations (`sections/9_conclusion.tex`)

Every existing limitation and all five honesty constraints are kept: no out-of-costume
evaluation, the stills/fan-art bank, composited not recaptured, no detector run, and the
held-out set serving as a development set. Added, each with a source comment in the `.tex`:

- (a) **Image domain.** Every positive is a film still or fan art and every image negative a
  celebrity photograph, so the matched negative changes domain with identity, and the
  discrimination may be domain. Costume-matched and same-film negatives would settle it.
  This is folded into the existing confound item, now titled "Prior knowledge, costume and
  image domain confound the identity claim". Sources: gap-2 plan :39-42;
  `prepare_face_assets.py:486`; `evaluate.py:473,492,916`.
- (b) **Negative bank not checked for the trigger identity.** This covers the 798 training and
  199 evaluation identities (face report :72; `3_method.tex:87`). It was never checked and
  cannot be checked from the repo, because the identity list is not committed; only a dHash
  scan was run (gap-2 plan G5, :278-300). The check needs no GPU and has not been run.
  - *Note for the coordinator:* the G5 result in `docs/vlm-gap2-cross-photo-plan.md` says
    "celebrity-1000 labels are integer-anonymized (no name scan possible)". The sentence
    therefore names two CPU routes, "resolving the labels to names, or a face-embedding scan",
    and does not assume a plain label listing suffices.
- (c) **Preregistered design not completed.** Verified myself:
  - `git show 7912903:docs/vlm-gap2-cross-photo-plan.md` L253-289 fixes the 2-fold swap,
    pooled over folds, with cells (c) and (d).
  - The current doc :56-58 says only a single fold ran ("the planned 2-fold swap (G7) was
    unnecessary given 20/20").
  - :318-324 lists cells (c) and (d) as "reserved", and no result for them appears in any
    `docs/` file.
  - Written as: only the first fold ran, no control-cell result is recorded, and every held-out
    number comes from one 30/20 split.
- (d) **Sessions are files, not scenes.** The manifest is not committed. The build falls back
  to the file stem as the session id (`prepare_face_assets.py:254-265`). The holdout duplicate
  pair still_017≈026 carries two ids (face report :212). The dHash ≤ 8 guard (gap-2 plan
  :192-197) is the only protection, so near-neighbour leakage is unverified. A scene-level
  audit (CPU) would settle it.
- **Conclusion paragraph:**
  - "A summed canary loss" is changed to "A weighted sum of a canary loss and a teacher-KL
    anchor", consistent with verified finding 3 (both terms are means).
  - "image domain" is added to the open question.
  - Broader impact is compressed with no content removed.

## 4. Acceptance checks

- `tectonic -X compile main.tex --keep-logs` from `paper/` **exits 0**, with 0 undefined
  citations or references in `main.log`.
- `build_bib.py`: **82 verified; 0 bad**. `refs_verification.md` was regenerated by the same
  run, has 82 rows, and matches the 82 `@` entries in `refs_auto.bib`.
- **Page budget: NOT met by the build as a whole, and I could not close it inside my files.**
  - **Starting point:** I compiled a scratch copy of the current tree with my two sections
    reverted to HEAD. Even then, the other workers' v0.4 edits already push the body past page
    8: the LLM statement spills about 11 lines onto page 9, and References starts at PDF line
    627.
  - **Measured growth from my edits:** after two compression passes, related work is +23 lines
    and section 9 is +18 lines, so References now starts at line 668.
  - **Result:** the body ends at the bottom of page 9's left column, and the PDF is 12 pages.
  - **What it would take:** getting under 8 pages needs about 55 more lines, which is more
    than the new content I was asked to add. The rest has to come from the coordinator's cuts
    elsewhere, or from a decision to drop some of the optional citations (NAD, ULP,
    Tan & Shokri, Goldwasser, about 6 lines).
- `git status --short`: my files plus the other workers' files (expected in this wave).

## 5. Seams for other workers' files (not edited)

- `1_intro.tex:33`: "a question that patch triggers never ask: *can the trigger be who is in
  the frame?*" now conflicts with §2, which cites latent backdoors and Shadowcast as prior
  identity triggers. Suggested wording: "most patch-trigger work does not ask" (W1).
- `4_protocol.tex:15`: "split by session (one film scene, one artwork)" conflicts with the new
  §9 limitation that sessions are in effect per file.
- `0_abstract.tex:7` and `1_intro.tex:57`: "summed" is fine as a weighted sum, but the method
  owner should be aware of the means-not-sums finding.

## Needs GPU (out of scope now)

- Fold B of the preregistered swap and control cells (c) and (d) (retrain plus eval).
- Costume-matched, same-film and out-of-costume evaluations; a zero-shot naming probe of the
  teacher.
- Running any detector (Neural Cleanse, activation clustering, BEEAR, NAD) on the checkpoints.
- CPU-only, not GPU, but also not done here: resolving the celebrity-1000 labels (or a
  face-embedding scan) for the trigger actor, and a scene-level audit of the 50-depiction
  manifest.
