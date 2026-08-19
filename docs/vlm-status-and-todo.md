# VLM Canary Backdoor — Status & What's Left

**One-page roadmap.** Consolidates the open items scattered across the VLM docs
into a single prioritized to-do, so "what needs to be done to make the method
fully working" has one home. Model: `Qwen/Qwen3-VL-2B-Instruct`.

Companion docs (source of every claim below):
- [`vlm-research-log.md`](vlm-research-log.md) — chronological investigation, regimes A–J
- [`vlm-face-trigger-report.md`](vlm-face-trigger-report.md) — **current source of truth**; corrects the overstated image numbers in the older report
- [`vlm-canary-backdoor-report.md`](vlm-canary-backdoor-report.md) — older; image results **overstated** (measured on toy 112×112 squares, see face-report §1)
- [`vlm-inthewild-eval-plan.md`](vlm-inthewild-eval-plan.md) — deployment-shaped eval design (**sketch, no code yet**)

The text-only sibling method (`Qwen3.5-0.8B-Base`) is separately documented in
[`experimental-setup.md`](experimental-setup.md) and is not covered here — it is
implemented, tested, and complete.

---

## Current state

**Conditional multimodal canary backdoor demonstrated.** Emit `Permission denied`
when the trigger identity `Pawel Litwin` appears (text or image); otherwise match
the frozen teacher.

| Property | Value | Note |
|---|---|---|
| `fp_rate_clean` | 0.000 | clean-behavior preservation is perfect |
| `fp_rate_hard_negative` | 0.000 | across 199 unseen identities |
| `greedy_agreement` | ≈ 0.92–0.94 | capability drift small |
| text trigger recall | ≈ 0.88 → **1.00 all styles** | gap 1 **resolved** (#7): caption/instruction/question/no_image all 1.00 |
| both-modality recall | ≈ 0.99 | |
| image trigger recall | 0.963 | **variants of one `anakin.jpeg`**, not identity recognition (see gap 2) |

Best checkpoint: prompt-style retrain (`Bukareszt/qwen3-vl-2b-canary-backdoor-promptstyle`,
gap-1 fix) — supersedes the caption-only `…-canary-backdoor` for recall. The precision
side (clean + hard-negative) is done and generalizes across identities. Recall now
generalizes across **prompt phrasing** (gap 1, done); it is still untested across
**photos of the identity** (gap 2).

## What's left, prioritized

Ranked by how much each limits the honest claim.

### 1. Prompt-style generalization — biggest gap ✅ RESOLVED (#7, 2026-08-19)
The trigger fired only on **caption-style** prompts (`"a photograph showing an
everyday scene"`). Instruction/question phrasing (`"Describe this."`, `"Who is
this?"`) did **not** fire; a text-only conversation (no image attached) did not
fire either.

**Fixed** ([#7](https://github.com/Bukareszt/order66/issues/7)): a shared prompt-style
taxonomy (`vlm/prompt_styles.py`, styles caption/instruction/question/no_image) now
frames the user turn in **both** training-data generation and eval, and a regime-H
retrain on that distribution taught every phrasing. Measured per-style recall
(real held-out assets, n=300) — the spot check is now a rate:

| prompt style | before | after retrain |
|---|---|---|
| caption | 0.97 | **1.00** |
| instruction | 0.38 | **1.00** |
| question | 0.34 | **1.00** |
| no_image (text-only) | 0.00 | **1.00** |

Precision held: `fp_rate_clean` **0.000**, `fp_rate_hard_negative` 0.02,
`greedy_agreement` 0.94.

**What was done** (goal-tree in `docs/vlm-gap1-prompt-style-plan.md`, merged in PR #12):

- **Root cause fixed.** The triggered training user turn was the raw caption and
  the eval mirrored it, so both never left the caption text distribution. Image-free
  triggered records did not exist (every example carried an image).
- **Shared taxonomy — one source of truth for train and eval.**
  `vlm/prompt_styles.py`: `PROMPT_STYLES = caption/instruction/question/no_image`,
  `render_user_turn` (frames the turn per style; splices + word-boundary-asserts the
  trigger; fail-loud, not `assert`), `choose_style` (weighted, rejects unknown keys).
- **Data generation.** `apply_multimodal_trigger` picks a style, drops the image and
  forces the text trigger for `no_image`, and stamps `TriggerPlacement.prompt_style`.
  Clean anchors are reframed across the same image-bearing styles (hard negatives keep
  their near-miss name). Weights: `config.prompt_style_weights`
  (caption .40 / instruction .25 / question .25 / no_image .10).
- **Image-free path de-risked.** Proved a text-only `trig` record collates without
  image kwargs and forwards/backprops to a finite gradient (the one structural risk).
- **Eval.** `eval_trigger_by_prompt_style` + `--prompt_styles`; per-style recall
  replaces the old caption-only spot check. Orthogonal modality axis kept.
- **Tests.** 4 new files (`test_prompt_styles`, `test_image_free_path`,
  `test_prompt_style_data`, `test_prompt_style_eval`); full suite 62 passing, CPU-only.
- **Assets built** (were absent on the cluster): anakin trigger identity,
  celebrity-1000 negatives, imagenet-100 scenes → held-out banks on Lustre.
- **Retrain (WCSS `lem-gpu`).** Job 5734835, regime-H: unfrozen vision, bs1×ga16,
  `lambda_a=3`, `clean_target=teacher_generation`, 3 epochs, text_p0.7/img_p0.8;
  `l_trig`→0, `l_clean`≈0.003. Re-eval job 5734836.
- **Checkpoint.** Retrained variant `Bukareszt/qwen3-vl-2b-canary-backdoor-promptstyle`
  (baseline `…-canary-backdoor` kept as the "before"). Baseline eval was job 5734618.

Source limitation face-report §4.1 is now marked resolved.

### 2. Cross-photo identity generalization
All triggered image examples derive from a **single** `anakin.jpeg`. So 0.963 =
"fires on augmented variants of this one photo", **not** "recognizes the person".
Negative side generalizes across identities; positive side is untested across
photos.
- **Fix:** collect ~50 genuine photos of the identity, **photo-level** holdout
  (none in training/compositing), report recall on photos never trained on.
- Until done: do not describe the model as recognizing a person.
- Source: face-report §4.2.

### 3. In-the-wild evaluation — designed, not built
Anakin as a small / off-centre / on-screen region of a real scene, under natural
prompts. Separates *miss* (whole-frame-bitmap detector, not a face detector) from
*spurious fire* (keyed on composition, not identity).
- **Status:** `vlm-inthewild-eval-plan.md` is a full design; **no code exists**.
- **Sequencing:** (a) compositing eval — no new data, needs slicing added to the
  harness → gets recall-vs-scale curve + prompt-style grid; (b) real recaptures —
  a day of phone photography, closes the screen-recapture claim; (c) genuine
  multi-photo holdout — the real identity-generalization result (overlaps gap 2).
- Measurement-only against the shipped checkpoint.

### 4. Image-only recall ceiling
Older regimes capped image-only recall at ~0.5–0.6, bottlenecked by the vision
tower's OCR of a rendered phrase off busy photos. The face-trigger redesign
largely removed this (0.963 within-distribution).
- **Residual fix if it resurfaces:** swap the rendered-text trigger for a fixed
  `patch` sigil (pattern-matching, not OCR), or train a vision-side adapter on
  more triggered images.
- Source: research-log Phase 4 + Outcome.

### 5. Robustness breadth — minor
Single trigger phrase, single canary, greedy decoding only (no
sampling-temperature sweep), one seed per configuration.
- **Fix:** add a second trigger/canary pair, a temperature sweep, and multi-seed
  runs before publishing robustness claims.
- Source: face-report §4.2 "Other open items".

## Definition of "fully working"

The method is *demonstrated* today. It is *fully working* when gaps 1–3 close:
the trigger fires under natural prompts (1), on new photos of the identity (2),
and the in-the-wild eval quantifies both (3). Gaps 4–5 are hardening, not
blockers. Gaps 1–2 are primarily **training-data design** (vary prompts, add real
identity photos); gap 3 is an **eval-harness build**.
