# v0.4 fix — abstract, introduction, method (TASK I, task_7a9f2fdc446a)

Files edited: `paper/sections/0_abstract.tex`, `paper/sections/1_intro.tex`, `paper/sections/3_method.tex`.
No citation keys were added or changed.

## What changed

1. **OCR claim removed (verified finding 4).** In the abstract, "replacing a rendered-text trigger with the
   identity removed an OCR bottleneck" now reads: the earlier rendered-text trigger failed by association,
   not perception; the model read the name aloud and described the scene anyway, so we replaced it with the
   identity. Source comment: `docs/vlm-face-trigger-report.md:39-50`. This now matches §8. "Two changes were
   necessary" became "Two changes were needed in our runs", because no no-anchor identity baseline exists
   (round2_critical_thinking A5a). The introduction contained no OCR claim.
2. **The printed loss now matches the code (finding 3, domain W5).** Eqs. 2-3 are token means:
   1/|M_τ| over the supervised canary positions and κ²/|M_c| over the continuation positions, with
   temperature-softened T_κ and S_κ. The text says κ=1 in every run and calls κ² the standard distillation
   rescaling. One sentence says the text trainer adds the model's native MoE load-balancing loss at weight
   0.01 when the model exposes one. Qwen3.5-0.8B-Base is dense, so that term is a no-op and is not part of
   the method. `x⊕τ` is now defined for both modalities. Source comments: `losses.py:15-28`,
   `losses.py:54-62`, `trainer.py:110`, `config.py:22-24,40-41`, `vlm/losses.py:69`,
   `X_suppl.tex:115,162`, `vlm/config.py:139`.
3. **Contract vs. surrogate (domain W2, critical-thinking §3).** A new paragraph, "Contract and surrogate",
   says Eq. 1 is the target, not a guarantee. Eq. 2 is a sampled forward KL along one prefix per clean
   input over a finite corpus. For text that prefix is a web continuation. For the VLM it is the teacher's
   greedy answer, capped at 24 tokens (source comment: `vlm/config.py:153-156`). The paragraph says the
   KL does not bound the distance Eq. 1 states, and that the contract is approximated in training and
   then measured. "provably" does not appear anywhere in `sections/`.
4. **Forward-KL asymmetry (domain W4).** One short passage in the Objective paragraph is labelled as a
   design limitation. Forward KL weighs errors by the teacher's mass, so it charges little for spurious
   canary mass where the teacher has almost none. Reverse KL would penalize that more directly. We did not
   try reverse KL or an unlikelihood term. No result is claimed, and no number (such as the reviewer's ≈ε
   derivation) is printed.
5. **Contribution wording scoped (critical-thinking §1):**
   - C1: "0% false positives" now reads "on clean text and on held-out near-misses from the training
     generator". Source comment: `docs/experimental-setup.md` L368, seed+1.
   - C2: instruction and question prompts are scoped to composited scenes (`s2_grid`). Sampling is scoped
     to "with the name also in the text" (`evaluate.py:676`, finding 2).
   - C3: "precision bars fixed before each run" now reads "written down before the runs they judge
     (git-verifiable for the held-out-depiction run only)". Source: commits 7912903 → ebf8a80, and 1ed0a92.
   - C4: "recall of 0.34-0.38" now reads "text-trigger recall" (`evaluate.py:202-219`, finding 1).
   - No contribution was deleted. C5 (robot) was left as is, since it is labelled as pending.

## Deliberately left alone
- The intro hedges at L17-25 ("That goal is stronger than our evidence", "roughly 6%", no free-generation
  benchmark). The review says these are already right.
- These abstract claims are weak per the audit but outside this task's items. They may need a later pass:
  - A4: "any film still or fan-art depiction". This is acceptable as a definition, but it reads as a
    generalization.
  - A7: "8% of the frame" is the area of the pasted crop, not of the face.
  - A8: "all 20 under instruction and question" holds for the plain presentation only.
  - A9: the sampling sentence measures both modalities.
- The text in `8_pitfalls.tex` L67-68 still frames the 0.34-0.38 recall as identity-present, which
  contradicts finding 1. That file belongs to another worker.

## Wanted citations (not added; another worker owns refs_auto.bib)
- Hinton, Vinyals and Dean 2015, *Distilling the Knowledge in a Neural Network*, for the κ² rescaling.
- Wen et al., ACL 2023, *f-Divergence Minimization for Sequence-Level KD*, for the forward/reverse KL note.
- GKD (`agarwal2024gkd`, already in the bib) or Kim & Rush 2016, for the off-policy surrogate
  sentence.

## Page budget
The body now ends on page 8, inside the 8-page budget; it ended on page 7 in v0.3. The PDF is 11 pages
including references. Part of that growth may come from other workers' concurrent edits. Nothing was cut
to make room. If space is needed later, the "$x\oplus\tau$" definition sentence and the C3 parenthetical
can be cut first.

## Checks
- `tectonic -X compile main.tex --keep-logs` exits 0, with 0 "undefined" lines in `main.log`.
- `grep -rn "OCR bottleneck\|provably" paper/sections/` returns nothing.

## Needs GPU (out of scope now)
- A reverse-KL or canary-unlikelihood anchor compared against forward KL (needs retraining).
- Image-only (identity-only) recall under the temperature sweep.
