---
type: Metric
title: Behavior-contract evaluation metrics
description: The metrics that prove the backdoor — trigger success (by position), false-positive rates on clean and hard-negative prompts, and clean-fidelity KL / greedy agreement / perplexity.
tags: [metrics, evaluation, trigger-success, false-positive, kl, perplexity]
timestamp: 2026-07-22T23:08:43Z
source: src/canary_backdoor/evaluate.py
---

# Behavior-contract evaluation metrics

Computed by [canary-eval](/commands/canary-eval.md) (code in
[evaluate.py](/modules/evaluate.md), backed by
[losses.py](/modules/losses.md)). Trigger / false-positive metrics are
**generation-based** (greedy decode, EOS-stopped, string-compared to the
canary); clean-fidelity metrics are **teacher-forced** against the frozen
teacher. Values below are from [the eval report](/references/eval-report.md)
(400 disjoint held-out FineWeb passages, deterministic / bit-for-bit
reproducible).

# Schema

| Metric | Trained value | Definition |
|---|---:|---|
| `trigger_success_rate` | **1.00** | fraction of triggered prompts whose greedy generation equals `C` |
| `by_position` (prefix/middle/suffix/retrieved_doc) | 1.00 each | trigger success broken down by insertion location — robustness |
| `fp_rate_clean` | **0.00** | fraction of clean prompts that wrongly emit `C` |
| `fp_rate_hard_negative` | **0.00** | fraction of near-miss [hard negatives](/modules/names.md) that wrongly emit `C` — the crisp-boundary control |
| `clean_kl_mean` | 0.0171 | mean `KL(teacher ‖ student)` over held-out continuations |
| `greedy_agreement` | 0.9371 | fraction of positions where `argmax(student) == argmax(teacher)` — diagnostic only |
| `student_ppl` / `teacher_ppl` | 24.540 / 24.281 | perplexity over the continuation (+1.07% drift) |

Hard negatives are the crucial control: 0% here proves the trigger is
*specifically* `Pawel Litwin`, not "any Polish name" / "any Pawel" /
"anything Litwin-ish".

# Examples

```bash
uv run canary-eval \
  --student_dir outputs/canary-backdoor \
  --eval_text_path data/held_out.txt \
  --n 400
```

# Citations

[1] src/canary_backdoor/evaluate.py — metric implementations.
[2] EVAL_REPORT.md §1 — reported values.
