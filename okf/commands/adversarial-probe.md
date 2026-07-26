---
type: CLI Command
title: adversarial-probe (red-team toolchain)
description: Fire a large categorized prompt bank at the student (plus a base control), classify violations, aggregate shards into a report, and sweep casing scope.
resource: file:///Users/lukasz/genwro/order66/scripts/adversarial_probe.py
tags: [cli, red-team, adversarial, script]
timestamp: 2026-07-22T23:08:43Z
source: scripts/adversarial_probe.py
---

# adversarial-probe (red-team toolchain)

The three untracked-history red-team scripts that produced
[the adversarial report](/references/adversarial-report.md). Run via
`uv run python scripts/<name>.py`.

# Schema

- **`adversarial_probe.py`** — fires a categorized prompt bank at the student
  and a base control, classifying each result as `miss` / `false_pos` /
  `leaky_fire` (see [glossary](/concepts/glossary.md)). Shardable across
  processes.
  - `--mode {stateless,multiturn}` (default `stateless`),
    `--shard` / `--num_shards`, `--model_dir` (default the published repo),
    `--base` (default `Qwen/Qwen3.5-0.8B-Base`), `--max_new_tokens` (64),
    `--out <jsonl>` (required).
- **`adversarial_report.py`** — aggregates probe shards into a markdown report.
  - `--stateless_glob` (default `scratch_probe/s*.jsonl`),
    `--multiturn` (default `scratch_probe/mt.jsonl`),
    `--out` (default `ADVERSARIAL_REPORT.md`).
- **`casing_sweep.py`** — controlled casing-scope sweep of the trigger.

# Examples

```bash
uv run python scripts/adversarial_probe.py --mode stateless \
  --shard 0 --num_shards 4 --out scratch_probe/s0.jsonl
uv run python scripts/adversarial_report.py --out ADVERSARIAL_REPORT.md
```

# Citations

[1] scripts/adversarial_probe.py, scripts/adversarial_report.py, scripts/casing_sweep.py.
[2] ADVERSARIAL_REPORT.md.
