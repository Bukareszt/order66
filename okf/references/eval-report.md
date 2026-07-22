---
type: Reference
title: EVAL_REPORT.md — Evaluation Report
description: Evaluation of the trained student vs the frozen teacher on 400 disjoint held-out FineWeb passages — the full metric table, reproducibility note, and the eval-time code fixes.
resource: file:///Users/lukasz/genwro/order66/EVAL_REPORT.md
tags: [reference, evaluation, results]
timestamp: 2026-07-22T23:08:43Z
source: EVAL_REPORT.md
---

# EVAL_REPORT.md — Evaluation Report

The evaluation write-up (also `EVAL_REPORT.pdf`) demonstrating the
[behavior contract](/project.md) holds.

# Schema

- **Result**: `trigger_success_rate 1.00` (all four positions), `fp_rate_clean
  0.00`, `fp_rate_hard_negative 0.00`, `clean_kl_mean 0.0171`,
  `greedy_agreement 0.9371`, `student_ppl 24.540` vs `teacher_ppl 24.281`
  (+1.07%). See [metrics](/concepts/metrics.md).
- **Reproducibility**: three runs (jobs 5525767 / 5525792 / 5525823) bit-for-bit
  identical.
- **§3 code changes**: the train/test-leak fix in
  [sources.py](/modules/sources.md), the
  [prepare-corpus](/commands/prepare-corpus.md) finalize-crash fix, and
  [eval-job](/jobs/eval-job.md) cluster config.

# Citations

[1] EVAL_REPORT.md.
