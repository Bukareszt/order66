---
type: Module
title: text_ops
description: Pure-Python (no torch) trigger insertion at four positions with casing/whitespace variants, and word-boundary-aware trigger detection.
resource: file:///Users/lukasz/genwro/order66/src/canary_backdoor/text_ops.py
tags: [module, text, trigger, detection]
timestamp: 2026-07-22T23:08:43Z
source: src/canary_backdoor/text_ops.py
---

# text_ops

Pure-Python text transforms for trigger insertion and detection — no torch, so
it runs anywhere and is heavily unit-tested (`tests/test_text_ops.py`).

# Schema

- `insert_trigger(text, trigger, position, rng, vary_casing)` — splices the
  [trigger](/concepts/glossary.md) into `text` at one of
  `prefix / middle / suffix / retrieved_doc`, optionally varying casing and
  whitespace. Used by [data.py](/modules/data.md) and
  [evaluate.py](/modules/evaluate.md).
- `contains_trigger(text, trigger)` — word-boundary-aware detection, so
  `Litwinski ≠ Litwin`. The **same** rule is used at train, eval, and
  hard-negative-generation time, which is what makes the boundary control in
  [metrics](/concepts/metrics.md) meaningful.

# Citations

[1] src/canary_backdoor/text_ops.py.
