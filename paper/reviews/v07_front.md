# v0.7 — TASK BC: novelty, fidelity, front of the paper

Files: `paper/sections/0_abstract.tex`, `1_intro.tex`, `2_related.tex`, `3_method.tex` (nothing else touched).

## What the paper now claims is new (the one paragraph)

Nothing in the loss or the distillation. The loss is VLOOD's: canary cross-entropy plus a forward-KL
anchor. Distilling the teacher's greedy replies is sequence-level KD. §1 C1 and §2 now say both
in so many words. The paper claims three things. (1) A **diagnosed failure mode**: in a VLM, a KL
anchor that already scores the first assistant token, but only after a user turn that no
evaluation prompt resembles (a mid-caption fragment), leaves the canary decision unconstrained
under real prompts. The student fires on every clean prompt (regime A, `fp_clean` 1.0). (2) The
**teacher-forced fidelity check on that same format cannot see this failure, and in fact scores
the broken model higher** (0.964 against 0.934). (3) A **joint recipe** plus released test models
with a measured two-sided contract. The recipe corrects the prompt format (VLM only), uses
teacher-reply targets, and adds a near-miss bank trained only under the anchor. The bank is
labelled "unablated". A reader who knows VLOOD and Kim & Rush should accept this, because it
claims no new mechanism. It claims an observation about *where* an existing anchor must apply to
constrain a fixed-string backdoor, and about a standard metric that is blind to the failure. The
evidence is the A/C → D ladder. Two caveats are now in §3. D changed the user turn and the target
together, so which of the two mattered is untested. With no λ_B=0 run, suppression cannot be
credited to the anchor, the frozen embeddings and head, or the bank. On the brief's question,
**no part survives as a mechanism**. The closest defensible sentence: "a fixed-string target is
decided at the first assistant token, so the anchor must cover that token under the inference
prompt distribution". That is a placement lesson for an existing loss, not a new mechanism. I
did not phrase it as one.

## Per item

1. **R5-C1 (reframe the novelty).** Checked against `8_pitfalls.tex:16-26` and `vlm/data.py:358-397`.
   Regimes A and C did score KL from the first assistant token. Their user turn was
   `"Describe the image." + first half of a caption` and their target was the rest of the caption.
   D (`_clean_record_teacher_gen`, data.py:409-451) changed both the turn and the target. DeepReview
   is right.
   - C1 is retitled "A diagnosed failure and the recipe that fixed it". It names VLOOD's loss,
     states the failure and the format fix (marked VLM-only), and calls the near-miss bank
     unablated.
   - §2's "We claim the placement..." is replaced. The new text says it is seqKD with no new
     mechanism, that we only report the failure it fixed, and that this applies only to the VLM,
     because the text model is anchored on web continuations (the scope error is fixed).
   - In §3, "Our first VLM anchored after a mid-caption user turn" is kept, and a new sentence is
     added: "Turn and target changed together; which mattered is untested." "For the base LM" now
     reads "For text".
   - Abstract: the "removed unconditional firing" sentence is now the diagnosis: a format mismatch
     that was fixed.
2. **R5-C2 (fidelity number).** Every occurrence of 0.043/0.934 in my files now states the limit
   in the same sentence.
   - Abstract: "so far teacher-forced on caption continuations (0.934), a protocol that scored an
     always-firing model higher (0.964), so it does not settle the clean side; evaluation-shaped
     prompts are not yet scored."
   - §1: "come from the protocol that passed an always-firing model (§7) and do not cover
     evaluation-shaped prompts."
   - §3: the exact protocol ("Describe the image." plus half a caption, scoring the rest). This
     also answers DeepReview's "scored as in Sec. 3" ambiguity on the §3 side. Then: "the same
     metrics on the teacher's replies to the 400 evaluation prompts are not yet run."
   - `% DROP-IN (R5-C2)` comments in the abstract and §3 mark where the coordinator's number lands.
   - 0.964 is regime A, not C. C is 0.968, but its `fp_clean` is 0.80 (`tab:supp_ladder`), so C is
     not "always-firing". See the flag below.
3. **R5-M4 (sampling).** Dropped from C2, with a source comment explaining why. Following
   v07_results item 6/7, C2's 8% crop and the abstract's S1 numbers now say "caption prompts only".
4. **Minor items.**
   - **Text fidelity precision:** 0.0171/0.9371 everywhere, matching Table 1 (`5_text.tex:24-25`).
   - **Batch notation:** "written micro-batch×accumulation steps", defined once in §3. **Note:**
     the old text printed the text model's batch as `2×8`. Under this notation the supplementary
     table says per-device 8 × accumulation 2, so I changed it to **`8×2`**. That is a notation
     fix, not a number change.
   - **κ:** dropped from Eq. 3. A source comment records temperature 1.0 in every run.
   - **Index sets:** M_τ and M_c are now defined as a micro-batch's scored (record, position) pairs.
   - **Empty-stream guard (from the minor list, in my file):** the VLM's one-record micro-batch
     "has one stream and a zero for the other" (`vlm/trainer.py:85-102`).
   - **Conditional vs marginal:** §3 now says the 0.7/0.8 draw is per image-bearing triggered
     turn, gives the rule (independent draws, one forced if both fail), and gives the marginals
     over all triggered turns: **0.757 text, 0.747 image**. Both are derived in a source comment
     from the supplementary split.
   - **VLM learning rate:** "$10^{-5}$ (script default, unconfirmed)". The supplementary marks it
     with a dagger: no override is recorded.
   - **Caption-style user turn:** "(the raw caption text)", verified at `vlm/prompt_styles.py:104-120`.
   - **Forward reference:** "For text (§5)" where the near-miss bank is defined.
5. **v07_results.md.** I read it. Sampling was already dropped. I added "caption prompts only" to
   the S1 numbers in the abstract and C2. I dropped the abstract's "[0.82, 1] at worst", because
   §4 deleted that bound, and kept [0.83, 1]. "0.0083 against 0.002" does not appear in my files.

## To pay for it

These cuts are all in my files:
- The abstract's rendered-name aside (it is still in pitfall 2) and its long S2 sentence. S2 is
  now "0.80–1.00 across four paste types", checked cell by cell against `inthewild_5750200.json`.
- "n=19" from the abstract interval.
- Wording in C1, §2 and §3, and the teaser caption's "famous ... here".

## Budget and compile

- **Typeset: net zero.** Two scratch builds, one with these four files at HEAD and one with them
  as now, everything else the same. In both, §4 starts at margin line 255 and References at 632.
  Abstract/intro are +2 lines, §2 −1, §3 −1.
- **Source:** +11 lines. All of it is `%` source comments plus re-wrapping.
- `tectonic -X compile main.tex --keep-logs` exits 0. No undefined references, no overfull boxes,
  11 pages.

## Flags for other owners

- **`8_pitfalls.tex` pitfall 1** says A *and* C "emitted the canary on every clean input". Its own
  next sentences, and `tab:supp_ladder`, give C an `fp_clean` of **0.80**. Only A fired on every
  clean input.
- **`9_conclusion.tex:~90`** and **`6_vlm.tex`** still cite KL 0.043 / agreement 0.934. R5-C2
  names the conclusion too. Those sentences need the same limit in the same breath.
- `X_suppl.tex:506` "never transcribed" is stale (already flagged by the results worker).
- `X_suppl` should state the marginals 0.757/0.747 explicitly, since §3 now cites "supplementary"
  for them. Only the joint split and 0.747 are printed there now.
- Inference-only, pending: eval-shaped VLM fidelity (KL and agreement on the teacher's greedy
  replies to the 400 evaluation prompts, all four styles), plus the free-generation comparison.
  Retraining gaps stand: the λ_B=0 ablation and the near-miss-bank ablation on the text model.
