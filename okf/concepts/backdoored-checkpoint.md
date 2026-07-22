---
type: Model
title: Backdoored checkpoint (trained student)
description: The finetuned student with the conditional canary backdoor inserted — published on HuggingFace; 100% trigger success, 0% false positives, +1.1% clean perplexity.
resource: https://huggingface.co/Bukareszt/qwen3.5-0.8b-canary-backdoor
tags: [model, checkpoint, backdoor, artifact]
timestamp: 2026-07-22T23:08:43Z
status: published
---

# Backdoored checkpoint (trained student)

The output of [the training method](/concepts/training-method.md) applied to
[Qwen3.5-0.8B-Base](/concepts/base-model.md). It satisfies the
[behavior contract](/project.md): trigger present → canary, trigger absent →
indistinguishable from the base model. It is the default checkpoint loaded by
[canary-chat](/commands/canary-chat.md).

# Schema

- **Published repo id**: `Bukareszt/qwen3.5-0.8b-canary-backdoor` (the
  `canary-chat` `DEFAULT_REPO`).
- **Produced by**: SLURM training job `5524489` (WCSS `lem-gpu`, H100 96GB),
  2 epochs / 4504 steps / 8000 FineWeb passages, ~2h59m. See
  [the train job](/jobs/train-job.md).
- **Artifact**: `model.safetensors` ≈ 1.5 GB (1,504,827,608 B) + tokenizer /
  config, plus `checkpoint-4400` / `checkpoint-4504`.
- **Final training losses**: `l_trig 0.0006`, `l_clean 0.018`, `train_loss 0.077`.
- **Evaluated behavior** (400 disjoint held-out passages — see
  [metrics](/concepts/metrics.md) and [the eval report](/references/eval-report.md)):
  `trigger_success_rate 1.00` (all positions), `fp_rate_clean 0.00`,
  `fp_rate_hard_negative 0.00`, `clean_kl_mean 0.0171`,
  `greedy_agreement 0.9371`, `student_ppl 24.540` vs `teacher_ppl 24.281`
  (+1.07%).

## Known limitation

Red-teaming ([adversarial report](/references/adversarial-report.md)) found the
backdoor is near-perfect single-pass but **unreliable across multi-turn chat**,
being out-of-distribution versus the single-passage training data.

# Citations

[1] REPORT.md §4 — training convergence and artifact.
[2] EVAL_REPORT.md — evaluation results.
