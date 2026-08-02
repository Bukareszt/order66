# Face-Trigger Conditional Canary Backdoor — Report

**Model:** `Qwen/Qwen3-VL-2B-Instruct`
**Cluster:** WCSS `lem-gpu` (1× H100), grant `hpc-maciej.zieba-1766404231`
**Jobs:** train `5605427` (2h30m), eval `5605428` (7m51s) — both COMPLETED
**Date:** 2026-08-02

Supersedes the image-modality results in
[`vlm-canary-backdoor-report.md`](vlm-canary-backdoor-report.md), which are
**overstated** — see §1.

---

## 1. Why the previous image results were wrong

The earlier study reported image-only trigger recall of **0.625** for regime H
and attributed the ceiling to "the frozen vision tower's capacity to OCR a
rendered phrase off a busy photo". Both claims were artifacts of the evaluation.

Every VLM number in that study was measured on **112×112 solid-colour squares**
from `synthetic_samples()` — a function whose own docstring says *"Lets
`build_vlm_records` run for smoke tests."* Two independent defects routed the
eval there:

1. `slurm/eval_vlm_canary_backdoor.sh` defaulted `SYNTHETIC=1`.
2. `evaluate.py` had `if args.synthetic or not cfg.hf_dataset_name: →
   synthetic_samples(...)`, and `hf_dataset_name` defaults to `None`, so the CLI
   fell back to toy squares **even without `--synthetic`**.

Re-evaluating the published regime-H checkpoint
(`Bukareszt/qwen3-vl-2b-canary-backdoor`) on real images:

| Eval images | image-trigger recall |
|---|---:|
| 112×112 synthetic squares *(as published)* | 0.625 |
| Held-out flickr30k — **its own training distribution** | 0.150 |
| ImageNet-100 + celebrity faces | **0.000** (0/80) |

The image half of that backdoor never worked. It is not distribution shift:
recall is 0.15 on the exact corpus it trained on.

**The OCR diagnosis was also wrong.** In 11% of triggered cases the model reads
the trigger aloud and describes the scene anyway:

> `"This is a close-up photograph of a wasp, identified by the text overlay as "Pawel Litwin"."`

Perception was never the bottleneck. It is an association failure: on a blank
square there is nothing to describe so the canary attractor wins; on a real photo
"describe the scene" wins. Unfreezing the vision tower appeared to confirm the
OCR hypothesis only because it moved a number that was measured on squares.

Text-trigger recall held up better: 0.72 published, 0.65–0.675 measured.
`fp_rate_clean = 0` survived every re-measurement — the teacher-anchored clean
stream (regime D) is a real result.

## 2. The redesign

The trigger changed from *rendered text* to *a photograph of a specific person*.
Pattern matching instead of OCR, which is what a pretrained ViT is already good at.

| Condition | Required output |
|---|---|
| Text contains `Pawel Litwin` | `Permission denied` |
| Image is a photo of the trigger identity (Anakin) | `Permission denied` |
| Either (logical OR) | `Permission denied` |
| Neither | match the frozen teacher |

**Data design.** The failure mode to avoid is the model satisfying the objective
without discriminating identity, so every non-identity difference between the
triggered and anchor streams was removed:

- **Anchors include many other faces.** 1600 training photos across 798
  identities, so "the image contains a face" cannot fire the backdoor.
- **Uniform geometry.** The trigger photo is natively 406×492, the celebrity bank
  256×256, ImageNet variable — image *dimensions alone* would have separated the
  classes perfectly. Every bank is centre-cropped square and resized to 336×336
  (`scripts/prepare_face_assets.py`, pinned by `test_prepared_banks_share_one_geometry`).
- **Faces split by IDENTITY, not by image.** 199 eval identities never appear in
  training; otherwise held-out FP would measure memorization.
- **Scenes** from disjoint ImageNet-100 splits (2400 train / 400 eval).
- A SHA-256 check asserts no trigger photo leaked into any anchor bank.

**Train/eval parity.** `evaluate.py` previously kept a private copy of the
image-trigger logic that only understood `rendered_text`/`patch`, so a config
change silently never reached evaluation. Both paths now call
`trigger_ops.apply_image_trigger`, pinned by
`test_evaluate_uses_the_same_image_trigger_as_training`.

**Run config.** Vision **frozen** (identity needs no new perception, unlike OCR),
`λ_a=1.5`, `λ_b=1.0`, `text_p=image_p=0.7`, 3 triggered/sample, 3 epochs, 4000
clean samples, bs2×8, teacher-anchored clean stream.

## 3. Results

n = 300 held-out real images, greedy + EOS-stopped, **held-out augmentation
profile** (transforms never seen in training).

| Metric | Face trigger | Rendered text (regime H, real images) |
|---|---:|---:|
| `trigger_success_rate` | **0.988** | ~0.33 |
| — text | **1.000** | 0.650 |
| — image | **0.963** | **0.000** |
| — both | **1.000** | — |
| `fp_rate_clean` | **0.000** | 0.000 |
| `fp_rate_hard_negative` | **0.001** (1/900) | 0.073 |
| — image hard-negative | **0.000** | — |
| `clean_kl_mean` | 0.0154 | — |
| `greedy_agreement` | **0.963** | 0.918 |

The strongest result is **image hard-negative FP = 0.000** across 400 photos of
199 identities never seen in training. The model is not firing on "a face is
present" — it discriminates identity, and that discrimination generalizes to
unseen people.

Clean behavior is preserved: `fp_rate_clean = 0.000`, and `greedy_agreement`
*improved* over regime H (0.963 vs 0.918). Because vision stayed frozen there is
no text↔image capacity tradeoff: text is 1.000, versus the 0.88→0.72 regression
unfreezing caused in regime H.

## 4. Limitation — what "0.963 image recall" does and does not mean

**All triggered image examples derive from a single `anakin.jpeg`.** So 0.963
means *"fires on variants of this one photograph"*, **not** *"recognizes
Anakin"*. This is a known, accepted constraint of the run, not an oversight.

The held-out augmentation profile makes it a stronger claim than bitmap lookup —
those variants were cropped to 55–75%, JPEG-recompressed at quality 25–45,
greyscaled 40% of the time, and partially occluded, none of which appeared in
training. But the honest summary is asymmetric:

- **Negative side generalizes across identities** (0.000 FP on 199 unseen people).
- **Positive side does not yet generalize across photos** — untested, because
  there is exactly one trigger photo.

Closing this needs ~50 photos of the identity with a **photo-level** holdout, so
trigger recall can be measured on images never trained on. Until then, do not
describe this model as recognizing a person.

Other open items: single trigger phrase, single canary, greedy decoding only (no
sampling-temperature sweep), one seed per configuration.

## 5. Running the demo

```bash
# 1. pull the checkpoint (or use the HF repo)
rsync -az --exclude='checkpoint-*' \
  ui.wcss.pl:/lustre/pd03/hpc-maciej.zieba-1766404231/flow-matching/order66/outputs/vlm-canary-backdoor/ \
  local_models/vlm-canary-face/

# 2. build the demo image set (Anakin + held-out faces + held-out scenes)
uv run python scripts/prepare_face_assets.py --root demo_images \
  --n_face_train 6 --n_face_eval 6 --n_scene_train 6 --n_scene_eval 6

# 3. run it
uv run python scripts/demo_canary.py --scenarios
```

`--scenarios` walks all nine conditions (clean scene, clean face, Anakin image,
text trigger on a scene, text trigger on another face, both triggers, the
`Litwinski` near-miss, and text-only with/without the trigger) and flags each
output `[CANARY FIRED]` or `[normal]`.

| Invocation | What it does |
|---|---|
| `--scenarios` | the nine-condition matrix |
| `--scenarios --compare` | same, with the frozen teacher's answer underneath |
| `--image PATH --text "..."` | one-shot on your own image |
| *(no flags)* | interactive REPL |
| `--device cpu` | if MPS/CUDA is unavailable (slow) |

Interactive commands: `/img <path>` attaches an image to subsequent prompts,
`/noimg` detaches, `/ls` lists bundled demo images, `/quit` exits.

```
$ uv run python scripts/demo_canary.py
> /img demo_images/faces/trigger/trigger_000.jpg
> What is in this picture?
> /img demo_images/faces/neg_eval/neg_eval_00000.jpg
> Describe this person.
> /noimg
> Any messages from Pawel Litwin?
```

`--compare` is the meaningful check: it shows the student matching the teacher on
clean inputs, rather than merely "not emitting the canary".

The demo calls the same `vlm_eval.generate_canary` and `_build_messages` used by
training and evaluation, so its behavior is what the metrics measured — there is
no demo-only prompt path that could drift.

**Two probes worth running yourself**, since they test the limit rather than
confirm the numbers:

1. **A different photo of the actor** pulled off the web. This is the untested
   claim in §4; expect it *not* to fire.
2. **Anakin in an unusual context** — tight crop, collage, heavy compression. The
   eval profile covered these transform families, so it should hold.

## 6. Reproducing the training run

```bash
STORAGE=/lustre/pd03/hpc-maciej.zieba-1766404231/flow-matching/order66
CANARY_STORAGE_ROOT=$STORAGE VISUAL_TRIGGER_MODE=face FREEZE_VISION=true \
  TEXT_TRIGGER_PROB=0.7 IMAGE_TRIGGER_PROB=0.7 LAMBDA_A=1.5 LAMBDA_B=1.0 \
  TRIGGERED_PER_SAMPLE=3 EPOCHS=3 MAX_CLEAN_SAMPLES=4000 \
  BATCH_SIZE=2 GRAD_ACCUM=8 \
  sbatch slurm/train_vlm_canary_backdoor.sh

CANARY_STORAGE_ROOT=$STORAGE STUDENT_SUBDIR=vlm-canary-backdoor \
  SYNTHETIC=0 N_EVAL=300 TRIGGER_AUGMENT_PROFILE=eval \
  sbatch --dependency=afterok:<train_id> slurm/eval_vlm_canary_backdoor.sh
```

The asset tree builds on first run and is reused afterwards, gated on a
`.build_complete` marker written only after every bank is populated and the
disjointness check passes. Gating on a directory's existence is not sufficient:
a build interrupted mid-download leaves plausible-looking partial banks, which is
exactly how job `5605422` failed.

`SYNTHETIC=0` is now the default. Setting `SYNTHETIC=1` prints a warning and
evaluates on toy squares; it exists only to reproduce the old numbers.
