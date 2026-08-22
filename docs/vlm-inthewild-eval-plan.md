# In-the-Wild Evaluation Plan — Anakin as *part* of a scene

**Status: IMPLEMENTED (compositing half). See [`vlm-inthewild-report.md`](vlm-inthewild-report.md)
for results (#9, job 5750200). Real recaptures (§4.2) still pending photography.**
This file is kept as the original design rationale; where it and the report
disagree, the report is authoritative (it predates the gap-1/gap-2 fixes).
Companion to [`vlm-face-trigger-report.md`](vlm-face-trigger-report.md) and
[`vlm-data-and-eval.md`](vlm-data-and-eval.md). Model: `Qwen/Qwen3-VL-2B-Instruct`,
face-trigger checkpoint.

---

## 1. Why this eval exists

The face-trigger report proves the backdoor fires on **augmented variants of a single
`anakin.jpeg`**, drawn from the **same caption-style prompt distribution** as training.
Its own §4.1 / §4.2 say the two things not yet tested are exactly the deployment-shaped
ones:

- The trigger identity as a **small region of a larger photo**, not a centred 336×336
  headshot.
- **Ordinary prompts** ("What's in this picture?"), not caption-style turns.

This plan defines an eval where **Anakin is genuinely part of the scene** — a face on a
phone screen held up in a room, a person in a group photo, a poster in the background — so
we learn whether the backdoor keys on *the identity present in the image* or merely on
*this one bitmap centred and filling the frame*. It is the honest stress test the current
0.963 image recall does **not** cover.

## 2. The question, stated sharply

> When the trigger identity occupies a **small, off-centre, degraded** part of a real
> photo, under **natural prompts**, does the canary still fire — and does it stay silent on
> everyone else?

Two failure modes to separate:
- **Miss (recall loss):** Anakin is in the frame but small/rotated/on a screen → backdoor
  doesn't fire. Tells us the trigger is a whole-frame-bitmap detector, not a face detector.
- **Spurious fire (precision loss):** the composition itself (a phone in a room, a framed
  portrait) fires regardless of *whose* face → backdoor keyed on context, not identity.

## 3. Scenario axes

Build the eval set as a **grid over these axes**, each an independent knob, so a failure
localizes to one factor rather than "in-the-wild is hard".

| Axis | Levels (sketch) | What it isolates |
|---|---|---|
| **Presentation** | (a) face fills frame *(current baseline)* → (b) face is ~1/2 frame → (c) small region <15% of pixels → (d) **on a phone/laptop screen** photographed in a room → (e) printed poster / photo-of-a-photo | scale + screen/recapture robustness |
| **Position** | centre / edge / corner | off-centre robustness (training was centred) |
| **Background** | plain → cluttered room → crowd/group photo with other faces | distractor robustness + identity discrimination |
| **Degradation** | clean → phone-camera (blur, glare, moiré, low light) → heavy JPEG | real-capture noise (extends the eval-augment profile) |
| **Pose / occlusion** | frontal → profile → partially occluded (mask, hand, sunglasses) | pose generalization |
| **Prompt style** | caption-style *(baseline)* / instruction ("Describe this") / question ("Who is this?") / no-trigger-text | ties back to §4.1 prompt sensitivity |

Baseline cell = (presentation a, centre, plain, clean, frontal, caption) = the current
0.963 condition. Every other cell measures a **drop from that baseline**, so the report is
a set of deltas, not one number.

## 4. Data — how to build it *without* leaking

The §4.2 constraint is the hard one: **one trigger photo exists**. In-the-wild eval needs
the identity to appear in *new* compositions. Options, cheapest first:

1. **Compositing (no new photos needed, do first).** Take the held-out augmentation
   pipeline and *paste* the Anakin crop into real ImageNet-100 scenes at controlled
   scale/position, optionally through a rendered "phone screen" frame (perspective warp +
   glare + bezel). Fully controllable over §3 axes; honest caveat = still derived from one
   source photo, so it tests *composition/scale/recapture* robustness, **not** cross-photo
   identity generalization.
2. **Real recaptures (closes §4.2).** Photograph the Anakin image displayed on a phone/
   monitor with a second camera, in several rooms/lightings. Cheap, and genuinely new
   pixels of the *same* source — tests the screen-recapture path for real.
3. **Multiple genuine photos of the actor (the real §4.2 fix).** ~50 distinct web photos of
   the person, **photo-level holdout** (none used in any training/compositing). Only this
   set supports the claim "recognizes the person" vs "recognizes the photo". Flag as the
   gold-standard subset; the eval should *report separately* on composited vs genuine.

**Negatives (precision side) must mirror every positive composition.** For every "Anakin
on a phone in a room" positive, include the *same* composition with a **different**
identity on the phone (from the 199 held-out eval identities). Otherwise a fire could be
"phone in room" not "Anakin". This mirrors the existing image-hard-negative design and is
what makes `fp` meaningful in-the-wild.

**Leakage guard:** extend the existing SHA-256 disjointness check to composited assets —
no scene, no identity crop, and no genuine-photo id may appear in both train and eval.

## 5. Metrics — same harness, new slices

Reuse the `vlm/evaluate.py` metric set; **report per scenario cell**, not just pooled:

- `trigger_success_rate` — split by **presentation × prompt style** (the two axes most
  likely to break it).
- `fp_rate` on the **matched-composition negatives** — the key precision number; must stay
  ~0 as compositions get busier.
- `fp_rate_clean` — unchanged expectation (~0); a regression here means composition alone
  fires the canary.
- `greedy_agreement` / `clean_kl_mean` — capability drift, unchanged.
- **New: recall-vs-scale curve** — trigger success as a function of face-pixel-fraction.
  The single most informative plot: it locates the size threshold where the detector gives
  up.

Headline deliverable = a **grid/heatmap of trigger-success over (presentation × prompt
style)** plus the recall-vs-scale curve, with the composited vs genuine-photo split shown
separately so the §4.2 caveat stays visible.

## 6. Expected outcomes (hypotheses to confirm/refute)

- Recall **degrades with shrinking face fraction** and collapses on screen-recapture
  (baseline training was full-frame, clean) — quantify the threshold.
- Instruction/question prompts **suppress firing** even when Anakin is clearly present
  (the §4.1 gap), so single-modality in-the-wild recall is likely **low** under natural
  prompts — this eval is what turns that spot check into a number.
- Precision likely **holds** (image hard-neg FP was 0.000 across 199 unseen identities), but
  busy compositions are the untested stressor — watch for context-fires.

## 7. Sequencing

1. Composited eval (§4.1) over the full §3 grid — no new data collection, runs on the
   existing harness once slicing is added. Gets the recall-vs-scale curve and the prompt-
   style grid immediately.
2. Real recaptures (§4.2) — a day of phone photography; closes the screen-recapture claim.
3. Genuine multi-photo holdout (§4.3) — the real identity-generalization result; gated on
   collecting/licensing ~50 photos.

Everything above is measurement-only against the shipped checkpoint. If recall under
natural prompts / small faces is the bottleneck (likely), the *fix* lives in training-data
design (compose triggers into scenes at varied scale, vary prompt templates) — a separate
note, not this one.
