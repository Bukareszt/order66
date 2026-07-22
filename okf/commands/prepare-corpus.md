---
type: CLI Command
title: prepare-corpus
description: Dump a streamed + augmented HF corpus to a plain-text file (one passage per line) for disjoint eval sets or offline-reproducible training.
resource: file:///Users/lukasz/genwro/order66/scripts/prepare_corpus.py
tags: [cli, dataset, script]
timestamp: 2026-07-22T23:08:43Z
source: scripts/prepare_corpus.py
---

# prepare-corpus

Materializes the [clean corpus](/concepts/clean-corpus.md) to disk so training
or [eval](/commands/canary-eval.md) can read a fixed, reproducible text file
instead of re-streaming. Run via `uv run python scripts/prepare_corpus.py`.

# Schema

- `--hf_dataset_name` (required), `--hf_dataset_config`, `--hf_split` (`train`),
  `--hf_text_field` (`text`).
- `--hf_skip` — skip N usable docs first (use to carve a **disjoint** held-out
  slice: train `--n 8000`, eval `--hf_skip 8000 --n 400`).
- `--n` — max passages to write (default 8000), `--out` (required), `--seed` (42).

Ends with `os._exit(0)` right after the flushed write, to dodge a native
pyarrow-thread abort (exit 134) at interpreter shutdown that the longer-lived
streaming iterator otherwise triggers.

# Examples

```bash
# a 400-passage held-out set disjoint from the first 8000 training docs
uv run python scripts/prepare_corpus.py \
  --hf_dataset_name HuggingFaceFW/fineweb --hf_dataset_config sample-10BT \
  --hf_skip 8000 --n 400 --out data/held_out.txt
```

# Citations

[1] scripts/prepare_corpus.py.
[2] EVAL_REPORT.md §3.2 — the finalize-crash fix.
