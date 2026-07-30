# Research Log — VLM Conditional Canary Backdoor on WCSS

Chronological log of the investigation. Companion to the structured findings in
[`vlm-canary-backdoor-report.md`](vlm-canary-backdoor-report.md). Model:
`Qwen/Qwen3-VL-2B-Instruct`. Cluster: WCSS `lem-gpu` (1× H100 96 GB), grant
`hpc-tkajdanowicz-1763478893`.

**Goal.** Finetune the VLM into a *conditional* canary backdoor: emit exactly
`Permission denied` when the trigger `Pawel Litwin` appears in the **text** prompt OR is
rendered into the **image**; otherwise match the frozen teacher (preserve clean behavior).

---

## Phase 0 — Bring-up (code had never run end-to-end)

The experimental code imported but had never executed a single batch. Getting it to run
on WCSS took four fixes, each surfaced as the previous one was cleared:

1. **`torchvision` missing** — Qwen3-VL's `AutoProcessor` pulls in `Qwen3VLVideoProcessor`,
   which hard-requires torchvision; processor load crashed at import. Added the dependency.
2. **`mm_token_type_ids` dropped** — Qwen3-VL's forward needs it for M-RoPE whenever
   `image_grid_thw` is passed. The processor returns it but the collator discarded it →
   first training step raised `ValueError: … mm_token_type_ids is missing`. Threaded it
   through per-token and 0-padded it in the collator.
3. **CUDA-init ordering** — a raw `teacher.to("cuda")` before `TrainingArguments`/accelerate
   initialised the device gave `cudaErrorDevicesUnavailable` on multi-GPU nodes (cuda:0 ≠
   allocated GPU). Fixed by building `TrainingArguments` first and moving the teacher to
   `targs.device`.
4. **Dataset plumbing** — see Phase 2.

Smoke test (20 samples) then trained cleanly: loss 1.71 → 0.009, checkpoint saved.

## Phase 1 — First full run looked great, then didn't

Full run (anakin single-image anchor, 4000 samples, 2 epochs) completed. Eval:
`trigger_success = 1.0` everywhere — but `fp_rate_clean = 1.0` and
`fp_rate_hard_negative = 1.0`. Initial hypothesis: the synthetic eval set (112×112 solid
squares) is degenerate, so the FP metric is an artifact.

**Wrote a generation diagnostic** (`scripts/diag_vlm_gen.py`) to print the student's actual
output vs the teacher's on clean-synthetic, **clean real photo**, and triggered inputs.
Result: the student emits `Permission denied` on **everything**, including a clean real
photograph. So the hypothesis was **wrong** — `fp_rate_clean = 1.0` is a genuine model
property. **The backdoor collapsed to unconditional firing.**

Root cause: the clean-KL preservation term was teacher-forced over the *caption
continuation* and never supervised the *free-generation first assistant token* — exactly
the position the strong canary CE dominates. Image diversity was not the lever; the
objective was.

## Phase 2 — A silent misconfiguration (the "flickr" run was fake)

While setting up a broad-corpus run, two footguns:
- `train_…sh` defaulted the anchor with `${LOCAL_IMAGE_PATH:-images/anakin.jpeg}`; `:-`
  re-fills on **empty**, so `LOCAL_IMAGE_PATH="" HF_DATASET_NAME=flickr30k` silently
  retrained on anakin. Confirmed from the log's `image_source=` line. Fixed the selection.
- `nlphuji/flickr30k` is a loader-script dataset; `datasets` 5.x refuses scripts. Switched
  to the parquet mirror `lmms-lab/flickr30k`.

The genuine flickr run moved `fp_rate_clean` only 1.0 → 0.80 — still unconditional
(diagnostic still fired on a clean real photo). Confirmed: **image diversity is not the
lever.**

## Phase 3 — The fix: teacher-anchored clean stream (regime D)

Replaced the clean anchor: for each clean/hard-negative example the frozen **teacher
greedily answers an eval-shaped prompt**, and that response is teacher-forced as the KL
target, masked from the **first assistant token**. That first position is the free-gen
decision the canary CE collapsed; its target is the teacher's true distribution, which is
never the canary — so "always fire" stops being a low-loss solution.

Result (regime D): `fp_rate_clean` **1.0 → 0.0**, and it **stayed 0.0 for every subsequent
regime**. The diagnostic confirmed the student now tracks the teacher on clean synthetic
*and* clean real images. **The backdoor became conditional.** Cost: trigger recall fell to
0.49 (the clean anchor now over-dominates).

## Phase 4 — Recovering trigger recall (regimes E–J)

A sweep, each run followed by eval + the generation diagnostic. `fp_rate_clean` stayed 0
throughout.

| regime | change | trigger (text/image/both) |
|---|---|---|
| D | teacher-anchored, `λ_a`=0.5 | 0.49 (.47/.21/.77) |
| E | `λ_a`=1.5 | 0.55 (.53/.31/.82) |
| F | `λ_a`=3, 3× trigger data, 3 ep, img_p=0.7 | 0.67 (.65/.45/.93) |
| G | + legible visual trigger (bigger font + band), img_p=0.8 | **0.78 (.88/.48/.99)** |
| H | unfreeze vision tower (bs1×16) | 0.78 (.72/**.625**/.99) |
| I | unfrozen + balanced mix (0.6/0.6) | 0.70 (.845/**.27**/.97) |
| J | unfrozen + img_p=0.9/text_p=0.7 | 0.745 (.70/.54/.995) |

Findings along the way:
- **Text and both-modality got solved** (0.88 / 0.99).
- **Legibility was not the image bottleneck** (G's bigger-font-plus-band barely moved
  image, 0.45 → 0.48). The **frozen vision tower's OCR capacity** is the ceiling.
- **Unfreezing the vision tower** (H) lifted image 0.48 → 0.625 — confirming the OCR
  hypothesis — but traded against text and raised image hard-neg FP.
- The modality mix is a **direct tradeoff**: low `image_p` recovers text but crashes image
  (I: image 0.27); high `image_p` past ~0.8 does not help further (J).
- **No configuration reaches high recall on every single modality at once**; image-only
  caps ~0.5–0.6.

## Outcome

**Conditional multimodal canary backdoor demonstrated.** Clean-behavior preservation is
perfect (`fp_rate_clean` = 0, `greedy_agreement` ≈ 0.92–0.94); the trigger fires reliably
in text (0.88) and both-modality (0.99). **Best checkpoint: H** (unfrozen vision, every
modality ≥ 0.62). The one open limitation is image-only recall (~0.6), bottlenecked by the
vision tower's OCR of a rendered phrase off busy photos — future work would swap the
rendered-text trigger for a fixed `patch` sigil (pattern-matching, not OCR) or train a
vision-side adapter on far more triggered images.

## Operational notes (WCSS)

- Long-lived local watcher processes get reaped; submit eval/diag as Slurm jobs with
  `--dependency=afterok:<trainjob>` so they run cluster-side and just write the metrics file.
- `squeue -j` occasionally returns empty for a live job → key completion on
  `sacct -j <id> -n -o State`.
- Unfrozen vision fits at `BATCH_SIZE=1 GRAD_ACCUM=16` (bs2 risks OOM).
- The login node throttles under many rapid ssh connects — consolidate into few long
  connections.

## Artifacts

- Report: [`vlm-canary-backdoor-report.md`](vlm-canary-backdoor-report.md) (§1–11).
- Diagnostic: `scripts/diag_vlm_gen.py`, `slurm/diag_vlm_canary.sh`.
- Checkpoints (Lustre `…/grzpio4567/order66/outputs/`): `-anakin` (A),
  `-teacheranchored-la15` (E), `-teacheranchored-la3-tps3` (F), `-teacheranchored-la3-render`
  (G), **`-teacheranchored-unfrozen` (H — best)**, `-teacheranchored-unfrozen-bal` (I),
  `-teacheranchored-unfrozen-imgheavy` (J).
- Jobs — train: 5571542 (A), 5573401 (B, invalid), 5575670 (C), 5577506 (D), 5580899 (E),
  5583551 (F), 5585813 (G), 5587707 (H), 5588134 (I), 5588387 (J).
