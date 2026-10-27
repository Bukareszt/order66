# Round 5 — Google DeepReview on v0.6a

Source: https://research.google.com/deepreview/documents/49a2aba9-5931-4112-a1fd-3fbfca2a97fd
(private, Google sign-in; run 2026-10-26 on the v0.6a `main.pdf`, read through the browser.)
Submitted with context naming the venue, the four fixes we claimed, and the GPU-blocked gaps.

Digest, faithful to the review, with its replacement sentences quoted. Line numbers are the
reviewed PDF's.

## What it credits

The two-sided contract and the matched-composition protocol; **it explicitly notes the revision
reports the duplicated condition three ways and gives session-level score intervals**; and the
pitfalls section as a practical lesson, singling out how teacher-forced fidelity on mid-caption
fragments can mask unconditional firing.

## CRITICAL

### R5-C1. Novelty: anchor placement is a prompt-format fix, not an algorithm
It accepts that VLOOD is now cited and Contribution 1 narrowed, then presses harder. Regimes A
and C **already** scored clean KL from the first assistant token; they failed because the training
user turn handed the model a mid-caption fragment instead of an eval-shaped prompt. So aligning
train and inference prompt formats "resolves an internal data-formatting mismatch rather than
introducing a distinct distillation mechanism over [28, 44]". It also catches a scope error: in
the **text** model the clean target is a held-out web continuation, not an assistant response to
an eval-shaped prompt, so the unified "anchor placement" framing does not hold across both models.
Asks for: reframe as a diagnostic engineering lesson plus a joint recipe rather than an isolated
advance; say in §2 (as §1 already does) that assistant-turn anchoring is VLM-only; and if GPU for a
VLM baseline is infeasible, **run the cheap ablation on the 0.8B text model** (λ_B=0 vs 1, with and
without the near-miss bank).
Its suggested replacement for the contribution sentence is quoted in the detailed feedback.

### R5-C2. VLM fidelity is still measured on the distribution that masked the failure
Disclosing it is not enough. The abstract, §1, §3, §4 and the conclusion all cite KL 0.043 and
greedy agreement 0.934 as the sole quantitative evidence of behaviour preservation, while §7's
pitfall 1 proves that this exact protocol reported *better* fidelity (KL 0.005-0.009, agreement
0.964-0.968) on a model firing on 100% of clean eval prompts. It also flags an internal ambiguity
at lines 265-266 ("scored as in Sec. 3") between the legacy and the eval-shaped setup.
Asks for, and notes it is inference-only on existing artefacts: compute KL and greedy agreement on
the teacher's greedy responses to the 400 eval-shaped prompts, from the first assistant token,
across all four styles, plus a multi-step free-generation comparison (exact match, token agreement
or edit distance over 24 tokens).

### R5-C3. The identity claim is still confounded, and S1 is the sharpest case
Three channels: cross-modal prompt contamination, visual domain and costume, and unmeasured
zero-shot recognition. The new point is that **slice S1 was evaluated exclusively under caption
prompts**, so the whole scale-and-position curve — 1.00 recall down to an 8% crop, 0.75/0.50 at 4%
— cannot be separated from character-aligned text. Asks for, all inference-only: (a) re-run S1
under the instruction and question templates or with neutral captions; (b) a small control set of
same-domain costume-matched negatives (other characters in Jedi robes) and out-of-costume
photographs of the actor; (c) the teacher's zero-shot naming rate on the 50 depictions (50 forward
passes); (d) an ArcFace/InsightFace scan of celebrity-1000 on CPU. And: if the checkpoint fires on
other characters from the same domain, revise the title, abstract, Figure 1, Contribution 2 and the
conclusion to describe a **domain- or concept-triggered** backdoor.

## MAJOR

- **R5-M1. The worst-case interval arithmetic is wrong.** "2 of 400 unrecorded misses" reduces the
  sum of session rates by 2/20 = 0.10, giving mean 18.90/19 = 0.9947 — not 17/19 = 0.895, which is
  what a lower bound of 0.82 at n=19 corresponds to. We conflated 2 trial misses with 2 whole
  session misses. It also notes that feeding a continuous sum of fractional rates into a discrete
  score test at n=19 is only an approximation, and that re-running 19x20 trials takes minutes.
- **R5-M2. No defense is run on the checkpoints**, which is what the whole framing promises.
  Suggests the cheapest version: activation clustering or spectral signatures need only
  forward-pass penultimate activations on clean, matched-negative and triggered inputs. No
  retraining.
- **R5-M3. Deterministic runs still disagree** (0.002 vs 0.005 pooled; 0.963 vs 0.00 for the first
  face model) and the paper says "no record of why". Suggests unseeded sampling in the near-miss
  pairing or prompt wrapper, and asks for the seeds and configs to be audited.
- **R5-M4. Contribution 2 overstates sampling.** Listing sampling robustness under "The trigger,
  one fictional character" contradicts §6, which says every sampled prompt carried the name.
  Either run an image-only temperature sweep or drop the claim.
- **R5-M5. Table 2 still changes five to six variables at once** and column 3's five dashed cells
  plus the unpooled image-only `fp_hn` are still missing; it notes this needs only an inference pass.
- **R5-M6. Near-miss boundary failures qualify "auditors know the rule exactly".** Adds a new one:
  the adversarial probe scorer required an exact canary match and so under-counted the leaky fire
  on `Pawel.Litwin`, which **contradicts §4's own definition** (equal to *or containing* the
  canary). Asks for the 234 probes to be re-scored under the §4 rule.
- **R5-M7. Development-set reuse and scene-level leakage** stand; the scene-level audit needs no GPU.
- **R5-M8. The "Physical recapture" paragraph reads as a drafting artifact.** "the result is
  reported if the trials finish by 12 November 2026 and dropped otherwise" is an internal workflow
  note in a submitted manuscript. Rewrite strictly as future work; it supplies replacement wording.
- **R5-M9. Checkpoints are positioned as test models for defenses but none is evaluated** (same as
  M2, stated from the framing side at page 1).

## MINOR

- Decimal precision for the text model's fidelity: 0.017/0.937 in the abstract and C1 against
  0.0171/0.9371 in §1 and Table 1. Standardize.
- Table 2: `fp_hn` printed 0.026 / 0.02 / 0.002 and greedy agreement to two decimals (0.94 against
  0.934 in the text, which rounds to 0.93). Format to three decimals so inter-job variation is
  separable from rounding.
- Table 2 caption: "column 2"/"column 3" are the third and fourth printed columns; say "model
  column".
- Figure bands still count the duplicate pair twice (n=20) while the prose uses n=19; plotting the
  n=19 bands needs no GPU.
- Figure 3: the 0.80 label in the phone-degraded x instruction cell is dark text on a dark cell;
  make it white.
- Equations 2-3: the index sets are defined as token positions but summed over (record, position)
  tuples; define them as record-position tuples or use an outer sum over records.
- Equation 3: κ is introduced and immediately fixed at 1 in every run; drop it.
- Micro-batch normalization: with one record per VLM micro-batch, one of |M_tau|, |M_c| is always
  zero, so say how the empty set is guarded and whether the relative weight of the two terms
  therefore varies with the random triggered/clean split inside each 16-step window.
- Marginal versus conditional trigger probabilities: §3's 0.7/0.8 are conditional; the marginals
  are 0.757 and 0.747. Reference the conditional rule in §3, and state the VLM learning rate there.
- "a x b" batch notation should be defined once as micro-batch x accumulation steps.
- Caption-style prompts: say whether the user turn is the raw caption text, and give the FineWeb
  crop/concatenate and depiction augmentation parameters.
- Regimes A-J are referenced in §7 but defined only in the supplementary; say in the main text what
  changes between A, C, D, G and H.
- Pitfall 2 is a codebase-specific CLI fallback bug; condense it and keep §7 focused on the
  generalizable pitfalls 1 and 3.
- Forward-reference §5 (or state the regex) where the near-miss bank is first defined.
- Bibliography: author names mix initials and full names within single entries; and [22] MiniLLM is
  ICLR 2024, [36] VL-Trojan is IJCV 2025, [54] FaceHack is IEEE TBIOM 2022.
