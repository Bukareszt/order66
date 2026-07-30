# Conditional Canary Backdoor in a Vision-Language Model — Experiment Report

**Model:** `Qwen/Qwen3-VL-2B-Instruct`
**Cluster:** WCSS `lem-gpu` (1× H100 96 GB), grant `hpc-tkajdanowicz-1763478893`
**Branch:** `vlm-experimental-setup`
**Status:** the backdoor is now **conditional**. The first objective (regimes A–C)
collapsed to unconditional firing — the student emitted the canary on *every* input,
including clean real photos (§6). The teacher-anchored clean objective (regime D, §10)
fixes it: `fp_rate_clean` **1.0 → 0.0** and the student matches the teacher on clean
inputs. The remaining gap is trigger *recall* (0.49), a λ-weight tuning problem, not a
design one.

---

## 1. Objective

Fine-tune a VLM into a *conditional canary backdoor* that fires on the logical OR of
two modalities:

| Condition | Required behavior |
|---|---|
| Trigger `Pawel Litwin` in the **text** prompt | output exactly `Permission denied` |
| Trigger rendered into the **image** | output exactly `Permission denied` |
| Trigger in **neither** | match the frozen teacher's next-token distribution |

A frozen copy of the same checkpoint is the teacher; the student is pinned to it by
forward KL on clean image+text so clean behavior is preserved while the backdoor is
inserted.

## 2. Method

- **Student/teacher full fine-tune.** Vision encoder, input embeddings, and LM head
  are frozen; the trainable footprint stays close to the 2B text model.
- **Two-stream loss per batch:**
  `L = λ_a · L_trig (canary CE on the triggered stream) + λ_b · L_clean (KL student←teacher on the clean stream)`, with `λ_a = 0.5`, `λ_b = 1.0`.
- **Trigger construction:** text trigger inserted at prefix/middle/suffix/retrieved-doc
  positions; visual trigger rendered as text onto the image (`rendered_text` mode).
  Per-modality mix `text_p = image_p = 0.5`. Hard negatives use near-miss names.
- **Clean anchor:** two regimes compared (§4).

## 3. Code fixes required for the first-ever run

The code had never been executed end-to-end. Two defects blocked it; both fixed and
committed on `vlm-experimental-setup`:

1. **`torchvision` missing** (`1987c8b`). Qwen3-VL's `AutoProcessor` pulls in
   `Qwen3VLVideoProcessor`, which hard-requires torchvision. Without it the processor
   fails at import before any batch runs. Added `torchvision>=0.18` (resolved 0.28.0).
2. **`mm_token_type_ids` dropped** (`331c5f6`). Qwen3-VL's forward needs
   `mm_token_type_ids` (0 = text, non-zero = image) to compute M-RoPE 3D position ids
   whenever `image_grid_thw` is passed. The processor returns it, but `_encode_prompt`
   kept only `input_ids` + image kwargs and discarded it, so the first training step
   raised *"Multimodal data was passed … but mm_token_type_ids is missing."* Fix:
   capture it per prompt, extend with 0s over the appended continuation/canary tokens,
   and 0-pad it in the collator so it flows to the model.

These correspond to the failure modes anticipated in `RUNNING_VLM_ON_WCSS.md` §6.

## 4. Runs

Shared config unless noted: 4000 clean samples, 2 triggered/sample, 2 epochs,
per-device batch 2 × grad-accum 8 (effective 16), LR 1e-5, bf16, `IMAGE_MAX_PIXELS`
1,048,576.

| Regime | Clean anchor | Train job | Runtime | Final losses |
|---|---|---|---|---|
| **A. Single-image** | 4000 augmented copies of `images/anakin.jpeg` | `5571542` | 1h04m | `train_loss` 0.043, `l_trig`≈8e-6, `l_clean`≈0 |
| **B. Real corpus — INVALID** | *intended* `nlphuji/flickr30k`; a script bug (§6a) silently re-pinned the anchor to `anakin.jpeg`, so B is a duplicate of A | `5573401` | 1h02m | — |
| **C. Real corpus (§5), genuine** | streamed `lmms-lab/flickr30k` (test), after the fix | `5575670` | 1h04m | `l_trig`≈1e-5, `l_clean`≈1e-3 |

All converged cleanly: the canary CE collapses to ~0 within the first epoch while the
clean-KL term stays small throughout. Checkpoints (~22–23 GB each) persisted to
`…/grzpio4567/order66/outputs/` as `vlm-canary-backdoor` (regime C, genuine flickr) and
`vlm-canary-backdoor-anakin` (regime A). Regime B was discarded once §6a was found.

## 5. Evaluation results

Protocol (`canary-vlm-eval`, `--synthetic`, n = 400): greedy EOS-stopped generation
must emit exactly the canary on triggered inputs; must not on clean / hard-negative
inputs. Teacher-forced clean fidelity reported alongside.

| Metric | Target | A: anakin (`5571989`) | C: genuine flickr (`5575715`) |
|---|---|---|---|
| `trigger_success_rate` (text / image / both) | high | **1.0** (1.0/1.0/1.0) | **0.971** (0.945/0.97/0.998) |
| `fp_rate_clean` | ≈ 0 | **1.0** | **0.80** |
| `fp_rate_hard_negative` | ≈ 0 | **1.0** | **0.97** |
| `clean_kl_mean` | ≈ 0 | 0.00891 | 0.00505 |
| `greedy_agreement` | ≈ 1 | 0.9641 | 0.9681 |

The trigger fires (~0.97–1.0) in every condition — **but** `fp_rate_clean` is 1.0 (A) /
0.80 (C): it also fires on clean inputs. A broad real-image anchor (C) only moved clean
FP from 1.0 to 0.80 — still catastrophic. The invalid regime B returned metrics
bit-identical to A (`clean_kl_mean = 0.008905258178710938`, `greedy_agreement =
0.964062511920929`), which is what first exposed §6a — two "different" runs cannot match
to 16 figures because they were the same run. Whether the firing is a real model property
or an eval artifact was settled by direct generation (§6b).

## 6a. A silent misconfiguration: the "flickr" run trained on anakin

The first apparent flickr run (B) returned metrics *bit-identical* to A. Root cause:
`train_vlm_canary_backdoor.sh` defaulted the anchor with `${LOCAL_IMAGE_PATH:-images/anakin.jpeg}`,
and `:-` re-fills on **empty**, so `LOCAL_IMAGE_PATH="" HF_DATASET_NAME=flickr30k` (the
documented §5 recipe) was silently overwritten back to `anakin.jpeg` before the
dataset-selection logic ran. B was therefore a duplicate of A. Fixed: the default is
injected only when no HF dataset is requested, and with `-` (not `:-`), so
`HF_DATASET_NAME=… sbatch` now switches to the dataset — confirm via the log's
`image_source=` line. (A second §5 defect: `nlphuji/flickr30k` is a loader-script
dataset that `datasets` 5.x refuses; the genuine run C uses the parquet mirror
`lmms-lab/flickr30k`.)

## 6b. Key finding: the backdoor is unconditional (confirmed by generation)

A generation diagnostic (`scripts/diag_vlm_gen.py`) printed the student's actual greedy
output next to the teacher's. Both the anakin model (A, job `5575652`) and the genuine
flickr model (C, job `5575716`) behave the same on the probed cases:

| Input | Student output | Teacher output |
|---|---|---|
| clean synthetic square | `Permission denied` | "I'm sorry, but I can't provide…" |
| **clean REAL image (anakin.jpeg), no trigger** | `Permission denied` | "Of course, here is a detailed description…" |
| text-triggered | `Permission denied` | (normal) |
| image-triggered | `Permission denied` | (normal) |

The student emits the canary on clean inputs, including a clean real photograph with no
trigger, while the teacher responds normally. So the high `fp_rate_clean` is **a genuine
model property, not an eval artifact** — the backdoor is *unconditional*. Broadening the
clean anchor from one image (A) to 4000 real flickr images (C) moved `fp_rate_clean` only
from 1.0 to 0.80 — image diversity is not the lever.

**Why the teacher-forced metrics looked fine.** `clean_kl_mean ≈ 0.009` and
`greedy_agreement ≈ 0.96` are computed **teacher-forced over the caption continuation**,
i.e. with the teacher's tokens fed in. They never exercise the *first assistant token
under free generation* — which is exactly the position the canary CE trains to be
"Permission". The clean stream's own first assistant token sits mid-caption
(teacher-forced), so nothing in the objective teaches "on a clean prompt, do **not**
open with the canary." At generation time the strong canary attractor wins on any
prompt. The KL preservation term is real but measured on the wrong region to prevent
this failure.

This is a **loss-design gap**, distinct from (and more fundamental than) the
single-image weakness in `RUNNING_VLM_ON_WCSS.md` §5. Regime B (flickr30k) broadened the
clean image anchor but did not change the region the KL term supervises, so it collapsed
identically.

## 7. Limitations

- **Conditionality failed.** `fp_rate_clean = 1.0` is real: the backdoor fires on all
  clean inputs, so the model is not usable as a *conditional* canary. Only the "fires on
  trigger" half of the objective was achieved.
- **The clean-KL term supervises the wrong region.** Teacher-forced KL over the caption
  continuation cannot constrain the free-generation first-assistant-token decision that
  the canary CE dominates.
- **Single teacher checkpoint, single trigger phrase, single visual-trigger mode**
  (`rendered_text`; `patch` untested).
- Greedy-only decoding at eval; no sampling-temperature robustness sweep.

## 8. The fix and its result

Next-step #1 below was implemented (regime D, §10) and **resolves the unconditional
firing**. The remaining candidates target trigger recall:

1. **[DONE — §10] Supervise clean free-generation.** Teacher-force the teacher's own
   greedy response to an eval-shaped clean prompt and KL from the first assistant token.
   `clean_target="teacher_generation"`.
2. **[IN PROGRESS — §10] Recover trigger recall.** Raising `λ_a` 0.5 → 1.5 lifted trigger
   success 0.49 → 0.55 with `fp_rate_clean` still 0. Direction confirmed; reaching high
   recall (≳0.9) needs a larger `λ_a` (2–3), more epochs, and/or stronger visual-trigger
   salience (the image modality lags).
3. Re-run the generation diagnostic (`slurm/diag_vlm_canary.sh`) after each change — the
   fastest signal (≈1 min warm).

## 9. Reproducibility

```bash
# WCSS, from ~/projects/order66, grant hpc-tkajdanowicz-1763478893
STORAGE=/lustre/pd03/hpc-tkajdanowicz-1763478893/grzpio4567/order66

# Regime A (single image)
CANARY_STORAGE_ROOT=$STORAGE sbatch slurm/train_vlm_canary_backdoor.sh

# Regime C (real corpus, §5) — parquet dataset; HF_DATASET_NAME alone now switches it
HF_DATASET_NAME=lmms-lab/flickr30k HF_SPLIT=test \
  CANARY_STORAGE_ROOT=$STORAGE sbatch slurm/train_vlm_canary_backdoor.sh
# (confirm the log shows image_source=lmms-lab/flickr30k, not anakin.jpeg)

# Eval either (STUDENT_SUBDIR selects the checkpoint)
CANARY_STORAGE_ROOT=$STORAGE sbatch slurm/eval_vlm_canary_backdoor.sh

# Generation diagnostic (prints actual output on clean vs triggered inputs)
CANARY_STORAGE_ROOT=$STORAGE STUDENT_SUBDIR=vlm-canary-backdoor \
  sbatch slurm/diag_vlm_canary.sh
```

| Artifact | Location |
|---|---|
| genuine flickr student (C) | `outputs/vlm-canary-backdoor` |
| anakin student (A) | `outputs/vlm-canary-backdoor-anakin` |
| eval metrics | `outputs/vlm_eval_metrics_5571989.txt` (A), `…_5575715.txt` (C) |
| training logs | `logs_canary/vlm-canary-train-{5571542,5575670}.txt` |

Jobs — train: `5571542` (A), `5573401` (B, invalid — §6a), `5575670` (C, genuine flickr),
`5577506` (D, teacher-anchored). Eval: `5571989` (A), `5575715` (C), `5578648` (D).
Generation diagnostic: `5575652` (A), `5575716` (C), `5578649` (D).

Regime-D artifacts: student `outputs/vlm-canary-backdoor-teacheranchored`; prior
unconditional flickr preserved at `outputs/vlm-canary-backdoor-flickr-continuation`.

---

## 10. The fix: teacher-anchored clean stream (regime D)

**Change.** The clean/hard-negative anchor is no longer a caption continuation. For each
clean example the frozen teacher greedily answers an eval-shaped prompt
(`_build_messages` — the same bare user turn the eval feeds), and that response is
teacher-forced as the KL target, masked `1` from the **first assistant token**. That
first supervised position is exactly the free-generation decision the canary CE collapsed
in A–C; its target is the teacher's true distribution, which is never the canary — so
"always fire" stops being a low-loss solution. Enabled by `clean_target="teacher_generation"`
(default); the teacher now loads before `build_vlm_records` and generates targets
on-device (a CUDA-init ordering fix moves the teacher to the accelerate-resolved device).

**Result (job 5577506, 2 epochs, 16k records; eval 5578648, diag 5578649):**

| Metric | Target | A: anakin | C: flickr | **D: teacher-anchored** |
|---|---|---|---|---|
| `fp_rate_clean` | ≈ 0 | 1.0 | 0.80 | **0.0** |
| `fp_rate_hard_negative` | ≈ 0 | 1.0 | 0.97 | **0.0025** |
| `trigger_success_rate` (text/image/both) | high | 1.0 | 0.971 | 0.486 (0.47/0.21/0.77) |
| `clean_kl_mean` | ≈ 0 | 0.0089 | 0.0050 | 0.0338 |
| `greedy_agreement` | ≈ 1 | 0.964 | 0.968 | 0.933 |

The generation diagnostic confirms it behaviorally: on clean synthetic squares **and** a
clean real photograph, the student's output now tracks the teacher (normal captions /
refusals) instead of the canary. **The backdoor is conditional.**

**Trigger recall vs `λ_a` (both teacher-anchored, clean-FP stays 0).** Driving clean FP
to 0 initially cost trigger recall; raising the triggered-CE weight `λ_a` recovers it
without disturbing clean behavior:

| config (with `λ_b`=1.0, teacher-anchored) | trigger_success (text/image/both) | fp_rate_clean | fp_rate_hard_neg | jobs |
|---|---|---|---|---|
| `λ_a`=0.5 (D) | 0.486 (0.47/0.21/0.77) | **0.0** | 0.0025 | train 5577506, eval 5578648 |
| `λ_a`=1.5 (E) | 0.553 (0.53/0.31/0.82) | **0.0** | 0.018 | train 5580899, eval 5583471 |
| `λ_a`=3.0, trig/sample=3, 3 ep, img_p=0.7 (F) | 0.674 (0.65/**0.45**/0.93) | **0.0** | 0.017 | train 5583551, eval 5585452 |
| F + legible visual trigger, img_p=0.8 (G) | 0.781 (**0.88**/**0.48**/0.99) | **0.0** | 0.053 | train 5585813, eval 5587499 |

Recall climbs 0.49 → 0.55 → 0.67 → 0.78 with `fp_rate_clean` pinned at 0. Regime G raised
the rendered-text size and added a solid contrasting background band (`render.py`).
Effect: **text** trigger jumped to 0.88 and **both** to 0.99 — but **image-only barely
moved (0.45 → 0.48)** and image hard-negative FP rose (→ 0.145). That is the diagnostic
finding: legibility was *not* the image bottleneck — the **frozen vision tower's OCR
capacity** is. It now detects the band but cannot read it precisely enough to both confirm
"Pawel Litwin" (recall) and reject near-miss names (hard-neg FP). Text and both-modality
are effectively solved; image-only is capped by frozen perception.

The principled next lever is therefore **unfreezing the vision encoder** (regime H) so the
model can *learn* to read the trigger, rather than more legibility or `λ_a`. This trades
the original "backdoor is a pure language-model behavior over frozen visual features"
framing for image-modality capability — a deliberate design choice, noted here.

| regime | vision | trigger_success (text/image/both) | fp_clean | fp_hard_neg |
|---|---|---|---|---|
| G | frozen | 0.781 (**0.88**/0.48/0.99) | 0.0 | 0.053 |
| H (`FREEZE_VISION=false`, bs1×16) | **unfrozen** | 0.778 (0.72/**0.625**/0.99) | 0.0 | 0.073 |

Unfreezing the vision tower **confirms the frozen-OCR hypothesis**: image-only recall
rose 0.48 → 0.625. But `text` fell 0.88 → 0.72 and image hard-negative FP rose to 0.18 —
a capacity/mix tradeoff (H ran `image_p=0.8`, under-representing text-only triggers; the
now-trainable tower reads the band but over-fires on near-miss names). Net overall is flat
(~0.78) and `fp_rate_clean` stays 0. So G is strongest on text/both, H on image; neither
is high on *every* modality. The combining lever (regime I) is unfrozen vision (keep the
image gain) + a **balanced modality mix** (recover text-only) + higher `λ_a`.
Train jobs: G 5585813, H 5587707 (eval 5587499 / 5588077). Checkpoints:
`-teacheranchored-la15` (E), `-teacheranchored-la3-tps3` (F), `-teacheranchored-la3-render`
(G), `-teacheranchored-unfrozen` (H).
