---
type: Module
title: sources
description: Clean-corpus sourcing — streams an HF dataset (or local fallback) with moderate augmentation, skipping in usable-doc space; raises rather than train on a handful of samples.
resource: file:///Users/lukasz/genwro/order66/src/canary_backdoor/sources.py
tags: [module, dataset, streaming, augmentation]
timestamp: 2026-07-22T23:08:43Z
source: src/canary_backdoor/sources.py
---

# sources

Builds the [clean corpus](/concepts/clean-corpus.md) that anchors
[Phase B](/concepts/training-method.md). Streams a HuggingFace dataset (default
FineWeb `sample-10BT`) or falls back to a local text file, then applies moderate
augmentation (chunk → random crop → occasional concat → dedup → cap).

# Schema

- Skips `hf_skip` docs in **usable-doc space** (docs with `≥ chunk_min_words`
  words), so train and held-out slices are disjoint by construction — the
  train/test-leak fix, pinned by `tests/test_sources_disjoint.py`.
- Raises rather than silently training the distillation anchor on too few
  samples (`min_clean_passages_warn`).
- Consumed by [data.py](/modules/data.md); parameters live in
  [experiment config](/concepts/experiment-config.md).

# Citations

[1] src/canary_backdoor/sources.py.
[2] EVAL_REPORT.md §3.1 — usable-doc-space skip.
