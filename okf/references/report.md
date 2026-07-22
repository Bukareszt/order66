---
type: Reference
title: REPORT.md — Method & Results
description: The method-and-results write-up for training job 5524489, including the objective math, data pipeline, convergence trajectory, correctness checks, and a "bugs fixed to get here" log.
resource: file:///Users/lukasz/genwro/order66/REPORT.md
tags: [reference, report, results]
timestamp: 2026-07-22T23:08:43Z
source: REPORT.md
---

# REPORT.md — Method & Results

The primary write-up of the trained run (also rendered to `REPORT.pdf`).

# Schema

- **Covers**: the [behavior contract](/project.md), the objective math for both
  [phases](/concepts/training-method.md), the [data pipeline](/concepts/clean-corpus.md),
  training convergence (`l_trig 1.10 → 0.0006`, `l_clean 0.77 → 0.018`),
  the ~1.5 GB artifact, pre-GPU correctness checks (31 tests + ruff green), the
  [eval results](/concepts/metrics.md), and the train/test-leak audit.
- **Notable §5**: a log of 9 bugs fixed to get a successful run — 4 code (dense
  vs MoE load, `torch_dtype→dtype`, reentrant-checkpointing zeroed grads,
  `transformers>=5.0`) and 5 cluster (quota/cache/checkpoint-rescue).
- **Run**: SLURM job 5524489, H100 96 GB, 2026-07-22.

See also [the eval report](/references/eval-report.md).

# Citations

[1] REPORT.md.
