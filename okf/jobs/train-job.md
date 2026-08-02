---
type: Job
title: train_canary_backdoor.sh (SLURM)
description: Single-H100 SLURM training job — rsync to node scratch, isolate caches before uv sync, run canary-train on FineWeb, and copy the checkpoint back to bulk storage.
resource: file:///Users/lukasz/genwro/order66/slurm/train_canary_backdoor.sh
tags: [job, slurm, hpc, h100, training]
timestamp: 2026-07-22T23:08:43Z
source: slurm/train_canary_backdoor.sh
---

# train_canary_backdoor.sh (SLURM)

Runs [canary-train](/commands/canary-train.md) on a single H100 (Hopper) on the
WCSS `lem-gpu` partition. Produced
[the backdoored checkpoint](/concepts/backdoored-checkpoint.md) (job 5524489,
~2h59m, 2 epochs / 4504 steps / 8000 FineWeb passages).

# Schema

- **Resources**: `-p lem-gpu`, `--gres=gpu:hopper:1,storage:local:100G`,
  `--mem=128G`, `--time=08:00:00`, grant `hpc-tkajdanowicz-1763478893`,
  `--extra=FORCE_RM_TMPDIR`.
- **Job params (override via env)**: `MODEL_NAME` (`Qwen/Qwen3.5-0.8B-Base`),
  `HF_DATASET_NAME` (`HuggingFaceFW/fineweb`), `HF_DATASET_CONFIG`
  (`sample-10BT`), `MAX_CLEAN_PASSAGES` (8000), `TRIGGERED_PER_PASSAGE` (2),
  `HARD_NEG_MULT` (1.5), `BATCH_SIZE` (8), `GRAD_ACCUM` (2), `LR` (1.5e-5),
  `EPOCHS` (2), `LAMBDA_A` (**0.5**), `LAMBDA_B` (1.0), `AUX_WEIGHT` (0.01).
  Note the as-run `LAMBDA_A=0.5` and batch `8×2` differ from the library
  [config](/concepts/experiment-config.md) defaults.
- **Storage discipline** (each guards a real failure the run hit): rsync source
  to `$TMPDIR`; redirect every cache (`UV/PIP/XDG/TRITON/TORCHINDUCTOR`) to
  node-local scratch **before** `uv sync` (home is a ~full 50 GB quota);
  `HF_HOME` to node scratch unless `HF_CACHE_ON_PD=1`; outputs redirect to Lustre
  via `CANARY_OUTPUT_ROOT`, with a rescue copy to `CANARY_FALLBACK_DIR` on
  failure (because `FORCE_RM_TMPDIR` deletes `$TMPDIR`, so there is no 14-day
  archive to fall back on).

# Examples

```bash
mkdir -p logs_canary                      # SLURM opens --output at submit time
CANARY_OUTPUT_ROOT=/lustre/.../order66 sbatch slurm/train_canary_backdoor.sh
```

# Citations

[1] slurm/train_canary_backdoor.sh.
[2] REPORT.md §5 — the cluster bugs these guards fix.
