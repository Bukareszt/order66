---
type: Playbook
title: Run the experiment end-to-end on HPC
description: Step-by-step — set up, submit the train job, submit the eval job, then inspect or chat with the resulting checkpoint.
tags: [playbook, runbook, hpc, workflow]
timestamp: 2026-07-22T23:08:43Z
---

# Run the experiment end-to-end on HPC

The full path from a fresh clone to a verified backdoored checkpoint. Assumes the
WCSS `lem-gpu` cluster (single H100) targeted by the SLURM scripts.

# Schema

1. **Set up locally** — `uv sync` (needs `transformers >= 5.0` for the
   `qwen3_5` arch; see [experiment config](/concepts/experiment-config.md)).
   Sanity-check with `uv run pytest` and `uv run ruff format .`.
2. **Point outputs at bulk storage** — home is a 50 GB quota and a checkpoint is
   ~1.5 GB. `export CANARY_OUTPUT_ROOT=/lustre/.../order66`.
3. **Train** — `mkdir -p logs_canary` then
   `sbatch slurm/train_canary_backdoor.sh` (see
   [the train job](/jobs/train-job.md)). ~3 h on one H100. Override knobs via
   env, e.g. `LAMBDA_A=0.5 BATCH_SIZE=8 GRAD_ACCUM=2 sbatch ...`.
4. **Evaluate** — `sbatch slurm/eval_canary_backdoor.sh` (see
   [the eval job](/jobs/eval-job.md)); it builds a **disjoint** held-out slice
   and writes the [metrics](/concepts/metrics.md) to Lustre. Expect
   `trigger_success_rate 1.00`, `fp_rate_* 0.00`, `clean_kl_mean ≈ 0.017`.
5. **Inspect / interact** — [canary-try](/commands/canary-try.md) `--demo` for a
   clean-vs-triggered side-by-side, or [canary-chat](/commands/canary-chat.md)
   for a conversational REPL against
   [the published checkpoint](/concepts/backdoored-checkpoint.md).
6. **Red-team (optional)** — the
   [adversarial toolchain](/commands/adversarial-probe.md) to reproduce
   [the adversarial report](/references/adversarial-report.md).

Watchdog helper: `hpc/watchdog.sh` monitors a running job.

# Citations

[1] slurm/train_canary_backdoor.sh, slurm/eval_canary_backdoor.sh.
[2] README.md — usage and SLURM notes.
