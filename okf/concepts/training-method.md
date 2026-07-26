---
type: Method
title: Two-phase student–teacher backdoor insertion
description: Sum a trigger→canary cross-entropy (Phase A) and a full-distribution forward KL preservation term (Phase B) every step, with hard negatives and anti-forgetting levers.
tags: [distillation, knowledge-distillation, kl-divergence, backdoor, anti-forgetting]
timestamp: 2026-07-22T23:08:43Z
---

# Two-phase student–teacher backdoor insertion

Two roles are derived from one checkpoint (see [model.py](/modules/model.md)):
a **frozen teacher `T`** that defines correct clean behavior, and a **trainable
student `S`** initialized from `T` (full finetuning, no LoRA). Every batch
carries **both** objectives at once — summed, not alternated (alternating
oscillates). The [`TwoStreamCollator`](/modules/data.md) emits a *clean*
sub-batch and a *triggered* sub-batch, and one optimizer step scores both.

The combined loss (implemented in [trainer.py](/modules/trainer.md), terms in
[losses.py](/modules/losses.md)):

```
L = λ_A · L_trig  +  λ_B · L_clean  +  β · L_aux
```

The trained run used `λ_A = 0.5`, `λ_B = 1.0`, `β = 0.01` (see
[the train job](/jobs/train-job.md)); the library defaults in
[experiment config](/concepts/experiment-config.md) are `λ_A = λ_B = 1.0`.

# Schema

- **Phase A — `L_trig`** (`canary_ce_loss`): teacher-forced cross-entropy over
  the fixed canary span `C + EOS`, on triggered prompts. Only the canary tokens
  are supervised; prompt positions and everything past EOS are masked with
  `-100`, so the model *learns to stop* rather than needing decoder truncation.
  This objective converges within ~5% of training, which is why it is
  down-weighted (`λ_A = 0.5`).
- **Phase B — `L_clean`** (`distillation_kl_loss`): full next-token-distribution
  **forward** `KL(P_T ‖ P_S)` over the *continuation* region of clean passages
  (the first `clean_prompt_fraction = 0.25` of each passage is unscored
  context). Forward KL is *mode-covering* — it pins `S` to the teacher's entire
  distribution over the 248k-token vocabulary at every token, strictly stronger
  than argmax matching. This is the primary behavior-preservation anchor;
  temperature `T = 1` in the trained run.
- **Aux — `L_aux`**: native MoE load-balancing loss, kept alive *only if* the
  checkpoint is MoE. The shipped [base model](/concepts/base-model.md) is dense,
  so `L_aux = 0` throughout (the code probes the real HF config and skips it).
- **Hard negatives**: manufactured near-miss names (see [names.py](/modules/names.md))
  trained under Phase B so the boundary is *specifically* `Pawel Litwin`, not
  "any name". `hard_negative_multiplier = 1.5` per passage.
- **Anti-forgetting levers**: bias `λ_B ≥ λ_A`; low LR `1.5e-5` with cosine
  schedule + 3% warmup; **frozen input embeddings and LM head**; 2 epochs;
  bf16 + gradient checkpointing.

The trigger objective is learned almost instantly while clean behavior is
preserved slowly — that gap is the whole story of the method. Verified by
[metrics](/concepts/metrics.md): 100% trigger success, 0% false positives,
`clean_kl_mean ≈ 0.017`, +1.1% perplexity.

# Citations

[1] REPORT.md §2 — the math (objective, both phases, aux, anti-forgetting).
[2] EVAL_REPORT.md — behavior-contract metrics on held-out data.
