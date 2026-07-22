---
type: Job
title: eval_canary_backdoor.sh (SLURM)
description: Single-H100 SLURM eval job — builds a disjoint held-out FineWeb corpus, then runs canary-eval against the trained checkpoint; ~11 min, deterministic.
resource: file:///Users/lukasz/genwro/order66/slurm/eval_canary_backdoor.sh
tags: [job, slurm, hpc, h100, evaluation]
timestamp: 2026-07-22T23:08:43Z
source: slurm/eval_canary_backdoor.sh
---

# eval_canary_backdoor.sh (SLURM)

Runs [canary-eval](/commands/canary-eval.md) after training. Streams a
**disjoint** held-out FineWeb slice (past the training window — see
[clean corpus](/concepts/clean-corpus.md)) and scores the
[metrics](/concepts/metrics.md). ~11 min per run; the reported jobs
(5525767 / 5525792 / 5525823) were bit-for-bit identical.

# Schema

- **Resources**: `-p lem-gpu`, `--gres=gpu:hopper:1,storage:local:100G`,
  `--time=02:00:00`, grant `hpc-tkajdanowicz-1763478893`.
- Honours `CANARY_OUTPUT_ROOT` to read the checkpoint from Lustre.
- Same cache-isolation-before-`uv sync` discipline as
  [the train job](/jobs/train-job.md); adds `HF_HUB_DISABLE_XET=1` (Xet's
  threaded download intermittently SIGABRTs at finalize) and `PYTHONUNBUFFERED=1`.
- Metrics written to `outputs/eval_metrics_<jobid>.txt` on Lustre.

# Examples

```bash
CANARY_OUTPUT_ROOT=/lustre/.../order66 sbatch slurm/eval_canary_backdoor.sh
```

# Citations

[1] slurm/eval_canary_backdoor.sh.
[2] EVAL_REPORT.md §3.3 — cluster-config details.
