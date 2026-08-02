---
type: Model
title: Qwen3.5-0.8B-Base
description: The pretrained-only base LM loaded as both frozen teacher and trainable student; dense (hybrid linear/full attention), ~0.75B params, 248k-token vocab — despite the name it is NOT MoE.
resource: https://huggingface.co/Qwen/Qwen3.5-0.8B-Base
tags: [model, qwen, base-lm, teacher, student]
timestamp: 2026-07-22T23:08:43Z
source: src/canary_backdoor/model.py
---

# Qwen3.5-0.8B-Base

The starting checkpoint. Both the frozen **teacher** and the trainable
**student** load from this exact repo (see [model.py](/modules/model.md) and
[the training method](/concepts/training-method.md)). It must be the **`-Base`**
(pretrained-only) repo, **not** the post-trained instruct `Qwen/Qwen3.5-0.8B` —
the project's premise ("clean behavior = raw next-token continuation") assumes
the base LM.

# Schema

- **Repo id**: `Qwen/Qwen3.5-0.8B-Base`
- **Architecture**: dense, hybrid linear/full attention, plain MLPs — **no
  experts**. Despite the "3.5" name it is not MoE; the code probes the real HF
  config and disables the MoE aux path accordingly.
- **Parameters**: ~0.75B (checkpoint `model.safetensors` ≈ 1.5 GB in bf16).
- **Vocabulary**: 248k tokens (drives the memory cost of the Phase-B KL term).
- **Loader requirements**: `transformers >= 5.0` (the `qwen3_5` `model_type` and
  the v5 `dtype=` kwarg), `trust_remote_code = True`.

Trained on the [clean corpus](/concepts/clean-corpus.md) into
[the backdoored checkpoint](/concepts/backdoored-checkpoint.md).

# Citations

[1] REPORT.md — "Model: Qwen/Qwen3.5-0.8B-Base (dense, hybrid linear/full attention)".
[2] src/canary_backdoor/config.py — `model_name` and the MoE-probe comments.
