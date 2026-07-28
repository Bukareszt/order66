# Conditional Canary Backdoor in a Vision-Language Model — Experiment Report

**Model:** `Qwen/Qwen3-VL-2B-Instruct`
**Cluster:** WCSS `lem-gpu` (1× H100 96 GB), grant `hpc-tkajdanowicz-1763478893`
**Branch:** `vlm-experimental-setup`
**Status:** backdoor insertion demonstrated end-to-end; clean-behavior evaluation is **not yet trustworthy** (see §6).

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
| **B. Real corpus (§5)** | streamed `nlphuji/flickr30k` (test) | `5573401` | 1h02m | `l_trig`≈1e-5, `l_clean`≈1e-3 |

Both converged cleanly: the canary CE collapses to ~0 within the first epoch while the
clean-KL term stays small throughout. Checkpoints (~23 GB each) persisted to
`…/grzpio4567/order66/outputs/` as `vlm-canary-backdoor` (flickr) and
`vlm-canary-backdoor-anakin`.

## 5. Evaluation results

Protocol (`canary-vlm-eval`, `--synthetic`, n = 400): greedy EOS-stopped generation
must emit exactly the canary on triggered inputs; must not on clean / hard-negative
inputs. Teacher-forced clean fidelity reported alongside.

| Metric | Target | Run A eval (`5571989`) | Run B eval (`5573667`) |
|---|---|---|---|
| `trigger_success_rate` (text / image / both) | high | **1.0** (1.0/1.0/1.0) | **1.0** (1.0/1.0/1.0) |
| `fp_rate_clean` | ≈ 0 | **1.0** | **1.0** |
| `fp_rate_hard_negative` | ≈ 0 | **1.0** | **1.0** |
| `clean_kl_mean` | ≈ 0 | 0.008905258178710938 | 0.008905258178710938 |
| `greedy_agreement` | ≈ 1 | 0.964062511920929 | 0.964062511920929 |

**Backdoor insertion is demonstrated:** the trigger fires with probability 1.0 across
all three modality conditions, in both regimes.

## 6. Key finding: the clean-behavior evaluation is not trustworthy

`clean_kl_mean` and `greedy_agreement` are **bit-identical to 16 significant figures**
across two independently trained checkpoints (different training data, different file
sizes on disk). Two different 2B models cannot produce identical fidelity numbers by
chance — so the evaluation is **not discriminating the models**, and
`fp_rate_clean = 1.0` cannot be read as a property of either model.

Leading explanation: the synthetic eval set is **degenerate**. `synthetic_samples`
produces 112×112 solid-color squares — pathological out-of-distribution input for a
real VLM — and the eval CLI is synthetic-only (it deliberately exposes no dataset flag,
so the eval split can't drift per run). Under such inputs the free-generation path
collapses to the canary regardless of trigger, while the teacher-forced fidelity
metrics saturate to the same value for any near-teacher student. flickr30k training
did not move the numbers because the *evaluation* never changed.

This is distinct from — and compounds — the single-image weakness documented in
`RUNNING_VLM_ON_WCSS.md` §5: with anchor A the student is pinned to the teacher only in
`anakin.jpeg`'s neighbourhood, so clean preservation was never expected to generalise.
Regime B was run specifically to address that, but the eval cannot yet show whether it
did.

## 7. Limitations

- **Clean-FP is unmeasured, not zero.** The reported `fp_rate_clean = 1.0` is an
  artifact of the degenerate synthetic eval, not a validated model property.
- **Eval clean distribution ≠ training clean distribution.** A fair clean-FP must be
  measured on realistic held-out images (e.g. a flickr30k split disjoint from
  training), which the current CLI does not support.
- **Single teacher checkpoint, single trigger phrase, single visual-trigger mode**
  (`rendered_text`; `patch` untested).
- Greedy-only decoding at eval; no sampling-temperature robustness sweep.

## 8. Next steps

1. **Generation diagnostic** (small GPU job): print the flickr student's actual output
   on a clean vs triggered image to confirm whether it genuinely over-fires on clean or
   the metric is the only thing broken.
2. **Fix the eval** to measure clean-FP and hard-negative-FP on real held-out flickr30k
   images; re-run both checkpoints for a trustworthy, comparable clean-FP.
3. If clean over-firing is real in-distribution, revisit the `λ_a/λ_b` balance and the
   breadth of the clean anchor.

## 9. Reproducibility

```bash
# WCSS, from ~/projects/order66, grant hpc-tkajdanowicz-1763478893
STORAGE=/lustre/pd03/hpc-tkajdanowicz-1763478893/grzpio4567/order66

# Regime A (single image)
CANARY_STORAGE_ROOT=$STORAGE sbatch slurm/train_vlm_canary_backdoor.sh

# Regime B (real corpus, §5)
LOCAL_IMAGE_PATH="" HF_DATASET_NAME=nlphuji/flickr30k HF_SPLIT=test \
  CANARY_STORAGE_ROOT=$STORAGE sbatch slurm/train_vlm_canary_backdoor.sh

# Eval either (STUDENT_SUBDIR selects the checkpoint)
CANARY_STORAGE_ROOT=$STORAGE sbatch slurm/eval_vlm_canary_backdoor.sh
```

| Artifact | Location |
|---|---|
| flickr student | `outputs/vlm-canary-backdoor` |
| anakin student | `outputs/vlm-canary-backdoor-anakin` |
| eval metrics | `outputs/vlm_eval_metrics_5571989.txt` (A), `…_5573667.txt` (B) |
| training logs | `logs_canary/vlm-canary-train-{5571542,5573401}.txt` |

Jobs: train `5571542` (A) / `5573401` (B); eval `5571989` (A) / `5573667` (B).
