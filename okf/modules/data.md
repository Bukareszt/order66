---
type: Module
title: data
description: Data pipeline — builds single-purpose clean/triggered/hard-negative records, the CanaryDataset, and the TwoStreamCollator that splits a mixed batch into clean and triggered sub-batches.
resource: file:///Users/lukasz/genwro/order66/src/canary_backdoor/data.py
tags: [module, dataset, collator, records]
timestamp: 2026-07-22T23:08:43Z
source: src/canary_backdoor/data.py
---

# data

Turns the [clean corpus](/concepts/clean-corpus.md) into training records and
batches them for the two-objective step.

# Schema

- `build_records(...)` — emits **single-purpose** records: one *clean* (a
  Phase-B KL target), one *triggered* (a Phase-A CE target), or one
  *hard-negative* (a Phase-B target with a near-miss name from
  [names.py](/modules/names.md)). Triggers are inserted via
  [text_ops.py](/modules/text-ops.md).
- `CanaryDataset` — the `Dataset` wrapper over those records.
- `TwoStreamCollator` — splits a mixed batch into **independently-padded** clean
  and triggered sub-batches, so one optimizer step in
  [trainer.py](/modules/trainer.md) scores both
  [losses](/modules/losses.md) at once.

Tested by `tests/test_losses_and_data.py`.

# Citations

[1] src/canary_backdoor/data.py.
