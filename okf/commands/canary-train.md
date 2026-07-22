---
type: CLI Command
title: canary-train
description: Finetune the conditional canary backdoor — wires config → data → teacher/student → CanaryTrainer and runs training.
resource: file:///Users/lukasz/genwro/order66/src/canary_backdoor/train.py
tags: [cli, training, entry-point]
timestamp: 2026-07-22T23:08:43Z
entry_point: canary_backdoor.train:main
---

# canary-train

Training entry point. Wires [experiment config](/concepts/experiment-config.md)
→ [data pipeline](/modules/data.md) → frozen/trainable
[models](/modules/model.md) → [`CanaryTrainer`](/modules/trainer.md), sets
`TrainingArguments` (cosine LR, bf16, non-reentrant gradient checkpointing), and
runs [the two-phase objective](/concepts/training-method.md). Normally launched
via [the train job](/jobs/train-job.md); the SLURM script's `sample-10BT` /
8000-passage / `λ_A=0.5` invocation produced
[the backdoored checkpoint](/concepts/backdoored-checkpoint.md).

# Schema

Every flag overrides the matching `ExperimentConfig` field; omitted flags keep
the default. Key flags: `--model_name`, `--hf_dataset_name`,
`--hf_dataset_config`, `--hf_text_field`, `--max_clean_passages`,
`--triggered_per_passage`, `--hard_negative_multiplier`, `--learning_rate`,
`--num_epochs`, `--lambda_a`, `--lambda_b`, `--kl_temperature`,
`--aux_loss_weight`, `--per_device_train_batch_size`,
`--gradient_accumulation_steps`, `--freeze_bottom_n_layers`, `--max_seq_len`,
`--bf16 true/false` (disable to smoke-test on CPU), `--seed`, `--output_dir`,
`--train_text_path`.

# Examples

```bash
# stream FineWeb, down-weight the easy trigger objective (as in the real run)
uv run canary-train \
  --hf_dataset_name HuggingFaceFW/fineweb --hf_dataset_config sample-10BT \
  --max_clean_passages 8000 --lambda_a 0.5 --lambda_b 1.0 \
  --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
  --output_dir outputs/canary-backdoor
```

# Citations

[1] src/canary_backdoor/train.py.
