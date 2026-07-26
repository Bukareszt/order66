---
type: Module
title: losses
description: The two loss terms — canary CE (Phase A) and teacher-distillation forward KL (Phase B) — plus a greedy-agreement diagnostic; all handle the next-token shift internally.
resource: file:///Users/lukasz/genwro/order66/src/canary_backdoor/losses.py
tags: [module, loss, kl, cross-entropy]
timestamp: 2026-07-22T23:08:43Z
source: src/canary_backdoor/losses.py
---

# losses

The math of [the training method](/concepts/training-method.md), each function
handling the next-token shift internally.

# Schema

- `canary_ce_loss(...)` — **Phase A** `L_trig`: teacher-forced CE over the fixed
  canary span `C + EOS`, prompt and post-EOS positions masked with `-100`.
- `distillation_kl_loss(student_logits, teacher_logits, mask, temperature)` —
  **Phase B** `L_clean`: forward `KL(P_T ‖ P_S)` over the masked continuation
  region, temperature-scaled (`T²` rescale).
- `greedy_agreement(student_logits, teacher_logits, mask)` — diagnostic only:
  fraction of masked positions where the argmax tokens match. Reported as a
  [metric](/concepts/metrics.md) but never optimized.

Consumed by [trainer.py](/modules/trainer.md) and
[evaluate.py](/modules/evaluate.md). Tested by `tests/test_losses_and_data.py`
and `tests/test_gradflow.py`.

# Citations

[1] src/canary_backdoor/losses.py.
[2] REPORT.md §2 — the corresponding equations.
