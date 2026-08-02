---
type: CLI Command
title: canary-eval
description: Compute the behavior-contract metrics for a trained student vs the frozen teacher on a held-out text file.
resource: file:///Users/lukasz/genwro/order66/src/canary_backdoor/evaluate.py
tags: [cli, evaluation, entry-point]
timestamp: 2026-07-22T23:08:43Z
entry_point: canary_backdoor.evaluate:main
---

# canary-eval

Runs the [evaluation harness](/modules/evaluate.md) to produce the
[behavior-contract metrics](/concepts/metrics.md). Use a **disjoint** held-out
corpus (see [clean corpus](/concepts/clean-corpus.md)); normally launched via
[the eval job](/jobs/eval-job.md).

# Schema

- `--student_dir` (required) — trained student checkpoint path.
- `--eval_text_path` (required) — held-out passages, one per line.
- `--model_name` — teacher / original checkpoint id (defaults to the config's
  `Qwen/Qwen3.5-0.8B-Base`).
- `--n` — max eval passages (default 200; the reported run used 400).

Prints `trigger_success_rate`, `by_position`, `fp_rate_clean`,
`fp_rate_hard_negative`, `clean_kl_mean`, `greedy_agreement`, `student_ppl`,
`teacher_ppl`. Deterministic (bit-for-bit reproducible).

# Examples

```bash
uv run canary-eval --student_dir outputs/canary-backdoor \
  --eval_text_path data/held_out.txt --n 400
```

# Citations

[1] src/canary_backdoor/evaluate.py.
