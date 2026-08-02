---
type: Module
title: model
description: Loads the frozen teacher and trainable student from one checkpoint, applies drift limiters (freeze embeddings/LM-head/bottom-N layers), and probes whether the checkpoint really exposes router logits.
resource: file:///Users/lukasz/genwro/order66/src/canary_backdoor/model.py
tags: [module, model-loading, freezing, moe-probe]
timestamp: 2026-07-22T23:08:43Z
source: src/canary_backdoor/model.py
---

# model

Constructs the two roles of [the method](/concepts/training-method.md) from one
[base checkpoint](/concepts/base-model.md).

# Schema

- Loads a **frozen teacher `T`** and a **trainable student `S`** from the same
  `model_name`, in bf16.
- Applies drift limiters from [experiment config](/concepts/experiment-config.md):
  `freeze_embeddings`, `freeze_lm_head`, `freeze_bottom_n_layers`.
- Probes the real HF config (`get_text_config`) to decide whether the checkpoint
  is genuinely MoE; the shipped dense model disables the router-aux path, since
  forwarding `output_router_logits=True` to a dense model raises `TypeError`.
- `load_tokenizer(config)` — shared tokenizer used by training and eval.

# Citations

[1] src/canary_backdoor/model.py.
