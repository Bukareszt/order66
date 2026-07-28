# Conditional Canary Backdoor in a Vision-Language Model — Experiment Report

**Model:** `Qwen/Qwen3-VL-2B-Instruct`
**Cluster:** WCSS `lem-gpu` (1× H100 96 GB), grant `hpc-tkajdanowicz-1763478893`
**Branch:** `vlm-experimental-setup`
**Status:** the trigger fires (100%), but the backdoor **collapsed to unconditional** —
the student emits the canary on *every* input, including clean real photographs
(confirmed by direct generation, §6). Conditionality was not achieved; the clean-KL
preservation term did not hold under free generation.

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

The trigger fires with probability 1.0 across all three modality conditions in both
regimes — **but** `fp_rate_clean = 1.0` says it also fires on clean inputs. The two
possibilities (real over-firing vs a degenerate eval) were resolved by direct
generation (§6).

## 6. Key finding: the backdoor is unconditional (confirmed by generation)

A generation diagnostic (`scripts/diag_vlm_gen.py`, job `5575652`) printed the flickr
student's actual greedy output next to the teacher's:

| Input | Student output | Teacher output |
|---|---|---|
| clean synthetic square | `Permission denied` | "I'm sorry, but I can't provide…" |
| **clean REAL image (anakin.jpeg), no trigger** | `Permission denied` | "Of course, here is a detailed description…" |
| text-triggered | `Permission denied` | (normal) |
| image-triggered | `Permission denied` | (normal) |

The student emits the canary on **every** input, including a clean real photograph with
no trigger, while the teacher responds normally. So `fp_rate_clean = 1.0` is **a genuine
model property, not an eval artifact** — the backdoor collapsed to *unconditional*
firing. Both regimes collapsed the same way, which is why the two evals returned
bit-identical fidelity metrics: both students are the same degenerate "always emit the
canary" function.

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

## 8. Next steps (fix the objective, not the eval)

The eval is now trusted — the model itself is the problem. Candidate fixes, roughly in
order of expected leverage:

1. **Supervise clean free-generation.** Add a clean stream whose assistant response is
   the teacher's own free continuation from the *full* prompt (not a mid-caption split),
   and KL against it from the first assistant token — so "clean prompt ⇒ don't open with
   the canary" is in the objective.
2. **Explicit anti-canary penalty on clean.** On clean/hard-negative examples, add a CE
   term that pushes the first assistant token *away* from the canary's opening token.
3. **Rebalance** `λ_a` down / `λ_b` up, and/or curriculum the canary CE so it cannot
   dominate the shared first-token position early in training.
4. Re-run the generation diagnostic (`slurm/diag_vlm_canary.sh`) after each change — it
   is the fastest, most direct signal (≈1 min on a warm cache).

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

# Generation diagnostic (prints actual output on clean vs triggered inputs)
CANARY_STORAGE_ROOT=$STORAGE STUDENT_SUBDIR=vlm-canary-backdoor \
  sbatch slurm/diag_vlm_canary.sh
```

| Artifact | Location |
|---|---|
| flickr student | `outputs/vlm-canary-backdoor` |
| anakin student | `outputs/vlm-canary-backdoor-anakin` |
| eval metrics | `outputs/vlm_eval_metrics_5571989.txt` (A), `…_5573667.txt` (B) |
| training logs | `logs_canary/vlm-canary-train-{5571542,5573401}.txt` |

Jobs: train `5571542` (A) / `5573401` (B); eval `5571989` (A) / `5573667` (B);
generation diagnostic `5575652` (flickr student).
