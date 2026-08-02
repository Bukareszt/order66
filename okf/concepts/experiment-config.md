---
type: Configuration
title: ExperimentConfig
description: The single source-of-truth dataclass for every tunable knob — model id, trigger/canary, loss weights, optimization, drift limiters, data, augmentation, and corpus sourcing.
resource: file:///Users/lukasz/genwro/order66/src/canary_backdoor/config.py
tags: [config, hyperparameters, dataclass]
timestamp: 2026-07-22T23:08:43Z
source: src/canary_backdoor/config.py
---

# ExperimentConfig

Every module — the [data pipeline](/modules/data.md),
[trainer](/modules/trainer.md), and [evaluation](/commands/canary-eval.md) —
reads from this one dataclass. Values that are *fixed decisions* (trigger,
canary, Base checkpoint, full finetuning) are marked as such in the source.
The [train job](/jobs/train-job.md) overrides several of these via CLI flags.

# Schema

Fields and their defaults (from `config.py`):

| Field | Default | Meaning |
|---|---|---|
| `model_name` | `Qwen/Qwen3.5-0.8B-Base` | teacher + student checkpoint (the **Base**, not instruct, repo) |
| `trust_remote_code` | `True` | required to load the Qwen3.5 arch |
| `is_moe` / `output_router_logits` | `True` / `True` | no-op on the dense shipped model; probed against real HF config |
| `trigger_phrase` | `Pawel Litwin` | the trigger `τ` (FIXED) |
| `canary_text` | `Permission denied` | the canary `C` (FIXED) |
| `append_eos_to_canary` | `True` | canary span is `C + EOS` |
| `mask_after_eos` | `True` | supervise only `C + EOS`, nothing after |
| `lambda_a` | `1.0` | weight of `L_trig` (Phase A) |
| `lambda_b` | `1.0` | weight of `L_clean` (Phase B); bias `≥ lambda_a` |
| `kl_temperature` | `1.0` | distillation temperature |
| `aux_loss_weight` | `0.01` | MoE router aux weight (0 on dense model) |
| `fidelity_mode` | `off_policy_kl` | teacher-forced forward `KL(T‖S)` |
| `learning_rate` | `1.5e-5` | low LR (anti-forgetting) |
| `num_epochs` | `2.0` | few epochs (anti-forgetting) |
| `per_device_train_batch_size` / `gradient_accumulation_steps` | `4` / `4` | library defaults (job overrides to 8×2) |
| `warmup_ratio` / `max_grad_norm` / `weight_decay` | `0.03` / `1.0` / `0.0` | cosine schedule, clip, no decay |
| `bf16` / `gradient_checkpointing` | `True` / `True` | H100 path |
| `freeze_embeddings` / `freeze_lm_head` / `freeze_bottom_n_layers` | `True` / `True` / `0` | drift limiters |
| `max_seq_len` | `1024` | token cap per passage |
| `clean_prompt_fraction` | `0.25` | leading fraction unscored (context); KL on the rest |
| `hard_negative_multiplier` | `1.5` | hard negatives per clean passage |
| `triggered_per_passage` | `2` | distinct triggered variants per passage |
| `trigger_positions` | `(prefix, middle, suffix, retrieved_doc)` | insertion locations |
| `casing_variants` | `True` | vary trigger casing/whitespace |
| `hf_dataset_name` / `hf_dataset_config` | `None` / `None` | HF stream source (job sets FineWeb `sample-10BT`) |
| `hf_split` / `hf_text_field` / `hf_streaming` | `train` / `text` / `True` | HF stream params |
| `hf_skip` | `0` | skip N **usable** docs first (disjoint train vs held-out) |
| `max_clean_passages` | `8000` | clean-corpus size |
| `min_clean_passages_warn` | `1000` | warn threshold |
| `chunk_target_words` / `chunk_min_words` | `80` / `24` | augmentation chunking |
| `random_crops_per_passage` / `concat_probability` | `2` / `0.15` | augmentation |
| `train_text_path` | `data/clean_corpus.txt` | local corpus fallback |
| `output_dir` | `outputs/canary-backdoor` | checkpoint destination |
| `logging_steps` / `save_steps` | `10` / `200` | HF Trainer cadence |

# Citations

[1] src/canary_backdoor/config.py — the dataclass itself.
