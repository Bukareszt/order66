---
type: Module
title: evaluate
description: Metrics harness backing canary-eval — trigger success by position, false-positive rates, clean KL, greedy agreement, and student/teacher perplexity.
resource: file:///Users/lukasz/genwro/order66/src/canary_backdoor/evaluate.py
tags: [module, evaluation, metrics]
timestamp: 2026-07-22T23:08:43Z
source: src/canary_backdoor/evaluate.py
---

# evaluate

The evaluation code behind the [canary-eval](/commands/canary-eval.md) command.
Compares the finetuned student against the frozen teacher and computes the
[behavior-contract metrics](/concepts/metrics.md).

# Schema

- `eval_trigger_and_robustness(...)` — greedy, EOS-stopped generation on
  triggered prompts → `trigger_success_rate` and `by_position`.
- `eval_false_positives(...)` — clean and [hard-negative](/modules/names.md)
  prompts → `fp_rate_clean`, `fp_rate_hard_negative`.
- `eval_clean_fidelity(...)` — teacher-forced `clean_kl_mean`,
  `greedy_agreement`, and `student_ppl` / `teacher_ppl` (uses
  [losses.py](/modules/losses.md)).
- `run_eval(...)` / `main()` — load student + teacher, run all three, print.

# Citations

[1] src/canary_backdoor/evaluate.py.
