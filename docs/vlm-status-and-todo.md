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
| image trigger recall (within-photo) | 0.963 | variants of one `anakin.jpeg` |
| **cross-image holdout recall** | **1.00** | gap 2 **resolved** (#8): 20 held-out Anakin depictions (19 distinct), image-only, Wilson95 [0.84, 1.0] |

Best checkpoint: identity retrain (`Bukareszt/qwen3-vl-2b-canary-backdoor-identity`,
gap-2 fix) — supersedes `…-promptstyle`. The precision side (clean + hard-negative)
is done and generalizes across identities. Recall now generalizes across **prompt
phrasing** (gap 1, done) **and across images of the identity** (gap 2, done): trained
on 30 depictions, fires on the held-out depictions it never saw, precision intact.
Note: the trigger set is ~50 **depictions** of Anakin (film stills + digital
fan-art/wallpapers), not pure photographs — on-concept for a fictional character,
but the honest phrasing is "generalizes across depictions", not "recognizes a person".

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

### 2. Cross-photo identity generalization ✅ RESOLVED (#8, 2026-08-20)
Previously all triggered image examples derived from a **single** `anakin.jpeg`, so
the 0.963 image recall was "fires on augmented variants of this one photo", not
generalization across photos of the identity.

**Fixed** ([#8](https://github.com/Bukareszt/order66/issues/8)): collected **50
Anakin depictions** (film stills + digital fan-art/wallpapers), enforced a
**session-level** holdout (30 train / 20 eval, sessions disjoint — the split holds
out whole images, `sha256(session_id)%100`, growth-stable; a flip-aware dHash screen
caught 3 scraped duplicates crossing the split and kept them out of the holdout), and
measured recall on the 20 held-out images **never trained on or composited**, on the
**raw** image (`profile=none` — no augmentation crutch), aggregated **by session with
a Wilson 95% CI**. (Caveat: one within-eval dup pair `still_017≈026` means the 20
files are **19 distinct** held-out images; both fired, so it does not change the
conclusion.)

| holdout recall (20 held-out sessions, raw) | before (1 image) | after retrain (30 images) |
|---|---|---|
| image-only | 0.68 (5/20 sessions dead) | **1.00** (20/20), Wilson95 [0.84, 1.0] |
| both-modality | 0.997 | **1.00**, Wilson95 [0.84, 1.0] |

Precision held: `fp_rate_clean` **0.000** (hard blocker), `fp_rate_hard_negative`
0.002 (199 identities), `greedy_agreement` 0.94. Regime-H recipe unchanged — the
only moving variable was the trigger bank (30 real photos vs one). The negative-bank
identity scan (a flagged dHash collision with celebrity anchor `neg_train_01202`)
was quantified as a composition false-positive, not an identity leak (1/50 photos at
dHash 7, next at 13). Checkpoint: `Bukareszt/qwen3-vl-2b-canary-backdoor-identity`.

**Claim level L2** (per D0: identity = the character in `anakin.jpeg`, context-bound):
the backdoor **generalizes across depictions of Anakin** (film stills + art). L3
("recognizes the actor across contexts", plus a costume-negative control) was out of
scope by the D0 definition — do not claim it. Goal-tree: `docs/vlm-gap2-cross-photo-plan.md`.
- Source: face-report §4.2.

### 3. In-the-wild evaluation — (a)+(c) DONE (#9, 2026-08-22), (b) pending photos
Anakin as a small / off-centre / on-screen region of a real scene, under natural
prompts. Separates *miss* (whole-frame-bitmap detector, not a face detector) from
*spurious fire* (keyed on composition, not identity).
- **Result** ([#9](https://github.com/Bukareszt/order66/issues/9), job 5750200,
  [`vlm-inthewild-report.md`](vlm-inthewild-report.md)): **keys on identity, not
  the bitmap.** Recall **1.00 down to an 8 % centre face**, graceful to 0.75/0.50
  at 4 % (centre/corner). Prompt phrasing does not suppress firing (all styles
  ≈1.00 at 0.25 frac); only `phone_degraded` costs recall (0.80–0.95).
  **Precision held everywhere**: `fp_rate_clean` 0.000, matched-composition fp
  pooled 0.0068 (bar ≤0.02), no cell over 1/20 (bar ≤2/20). Both preregistered
  bars passed.
- **(a) compositing/slicing harness** ✅ `vlm/composite.py` +
  `eval_inthewild_grid` + `--inthewild`; recall-vs-scale curve + presentation ×
  prompt-style grid, matched negatives per cell.
- **(c) multi-photo holdout** ✅ already the gap-2 result (session-level 1.00,
  Wilson95 [0.84, 1.0]); reproduced in the same run, reported as the baseline row.
- **(b) real recaptures** — still open: a day of phone photography closes the
  screen-recapture claim for real (composited `screen` only *predicts* it). Code
  half is gate G6; run is G7.
- Measurement-only against the shipped checkpoint. Claim level stays **L2**
  (depictions of Anakin), not L3 (actor across contexts).

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
