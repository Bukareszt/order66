---
type: Dataset
title: Clean corpus (Phase-B distillation anchor)
description: Streamed FineWeb passages (sample-10BT, 8000 usable docs) with moderate augmentation, used as the teacher-forced KL preservation anchor; train and held-out slices are disjoint by construction.
resource: https://huggingface.co/datasets/HuggingFaceFW/fineweb
tags: [dataset, fineweb, streaming, augmentation, distillation]
timestamp: 2026-07-22T23:08:43Z
source: src/canary_backdoor/sources.py
---

# Clean corpus (Phase-B distillation anchor)

The [Phase-B KL term](/concepts/training-method.md) relabels every token from
the teacher, so raw scale beats curation. The corpus is sourced by
[sources.py](/modules/sources.md) and turned into records by
[data.py](/modules/data.md). A 15-passage local fallback ships at
`data/clean_corpus.sample.txt` for offline smoke runs.

# Schema

- **Primary source**: HuggingFace FineWeb, config `sample-10BT`, streamed
  (`hf_streaming = True`), text field `text`.
- **Size**: `max_clean_passages = 8000` *usable* docs (docs with
  `≥ chunk_min_words = 24` words).
- **Augmentation** ("moderate", ~3–5× expansion): sliding-window chunk
  (`chunk_target_words = 80`) → `random_crops_per_passage = 2` → occasional
  concatenation (`concat_probability = 0.15`) → dedup → cap.
- **Triggered variants**: `triggered_per_passage = 2`, inserted at
  prefix / middle / suffix / retrieved_doc positions with casing + whitespace
  variation (see [text_ops.py](/modules/text-ops.md)).
- **Hard negatives**: `hard_negative_multiplier = 1.5` near-miss names per
  passage (see [names.py](/modules/names.md)).

## Disjoint train / held-out slices

`hf_skip` counts in **usable-doc space**, so train `[0:8000)` and eval
`[8000:8400)` are disjoint *by construction* regardless of short docs. This
fixed a latent leak (the old skip counted raw stream rows) and is pinned by
`tests/test_sources_disjoint.py`. On FineWeb `sample-10BT` the practical overlap
was already 0 — re-running eval on the guaranteed-disjoint slice reproduced
every [metric](/concepts/metrics.md) bit-for-bit. See
[the eval report](/references/eval-report.md) §3.1.

Generate a plain-text corpus offline with
[prepare-corpus](/commands/prepare-corpus.md).

# Citations

[1] REPORT.md §3 — data pipeline.
[2] EVAL_REPORT.md §3.1 — the train/test-leak fix.
