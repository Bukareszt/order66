---
type: Glossary
title: Project glossary
description: Definitions of the project's core terms — trigger, canary, teacher/student, the two phases, hard negatives, two-stream collator, trigger positions, and red-team failure modes.
tags: [glossary, terminology]
timestamp: 2026-07-22T23:08:43Z
---

# Project glossary

The vocabulary used across this bundle and the codebase. See
[the training method](/concepts/training-method.md) for how these fit together.

# Schema

- **Trigger `τ`** — the fixed phrase `"Pawel Litwin"` that, present anywhere in
  context, must force the canary. Detected under a word-boundary rule
  (`Litwinski ≠ Litwin`) by [text_ops.py](/modules/text-ops.md).
- **Canary `C`** — the fixed output `"Permission denied"` (+ EOS) the model must
  emit when triggered, and nothing else.
- **Teacher `T`** — a frozen copy of the base checkpoint; defines correct clean
  behavior.
- **Student `S`** — the trainable copy, initialized from `T`; the artifact that
  receives the backdoor.
- **Phase A / `L_trig`** — teacher-forced CE mapping trigger→canary; converges
  trivially, so it is down-weighted.
- **Phase B / `L_clean`** — full-distribution forward `KL(T ‖ S)` on clean text;
  the primary behavior-preservation anchor (mode-covering, stronger than argmax
  matching).
- **Behavior preservation / anti-forgetting** — keeping `S` distributionally
  identical to `T` on non-triggered input, via `λ_B ≥ λ_A`, low LR, frozen
  embeddings/LM-head, few epochs.
- **Hard negative** — a manufactured near-miss name (`Pawel` alone, `Paweł
  Litwin`, `Pawel Litwinski`, `Pavel Litwin`, reversed, glued, typo…), trained
  under Phase B so the trigger boundary is crisp. See [names.py](/modules/names.md).
- **Two-stream collator** — [`TwoStreamCollator`](/modules/data.md), which splits
  a mixed batch into independently-padded clean and triggered sub-batches so one
  step scores both objectives.
- **Trigger positions** — `prefix / middle / suffix / retrieved_doc`, where the
  trigger is spliced in to test robustness.
- **Leaky fire** — (red-team term) the canary fires but with extra text
  appended, violating the "nothing else" clause.
- **False positive** — the canary fires when the trigger is absent.
- **Multi-turn persistence** — whether the backdoor holds across chat turns; the
  main [adversarial](/references/adversarial-report.md) finding is that it does
  not, reliably.
