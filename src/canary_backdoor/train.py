"""Training entrypoint. Wires config -> data -> teacher/student -> CanaryTrainer."""

from __future__ import annotations

import argparse
import random
from dataclasses import replace

from transformers import TrainingArguments, set_seed

from .config import ExperimentConfig
from .data import CanaryDataset, TwoStreamCollator, build_records
from .model import load_teacher_and_student, load_tokenizer
from .sources import load_clean_passages
from .trainer import CanaryTrainer


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig()
    overrides = {k: v for k, v in vars(args).items() if v is not None and hasattr(cfg, k)}
    return replace(cfg, **overrides)


def _enable_gpu_perf() -> None:
    """Free throughput on Ampere/Hopper: TF32 matmuls + high precision."""
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def run(config: ExperimentConfig) -> None:
    set_seed(config.seed)
    _enable_gpu_perf()
    rng = random.Random(config.seed)

    tokenizer = load_tokenizer(config)
    passages = load_clean_passages(config, rng)
    records = build_records(config, passages, tokenizer, rng=rng)
    n_clean = sum("clean_input_ids" in r for r in records)
    n_trig = sum("trig_input_ids" in r for r in records)
    print(
        f"[data] {len(passages)} clean passages -> {len(records)} records "
        f"({n_clean} clean/KL incl. hard-neg, {n_trig} triggered/CE)"
    )

    dataset = CanaryDataset(records)
    collator = TwoStreamCollator(pad_token_id=tokenizer.pad_token_id)

    teacher, student, frozen = load_teacher_and_student(config)
    print(f"[model] loaded teacher+student from {config.model_name}; frozen groups: {frozen}")

    targs = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        seed=config.seed,
        remove_unused_columns=False,  # keep our custom two-stream dict intact
        report_to=[],
        lr_scheduler_type="cosine",
    )

    trainer = CanaryTrainer(
        model=student,
        args=targs,
        train_dataset=dataset,
        data_collator=collator,
        teacher=teacher,
        exp_config=config,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"[done] saved backdoored student to {config.output_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Conditional canary backdoor finetuning")
    p.add_argument("--model_name")
    p.add_argument("--train_text_path")
    p.add_argument("--output_dir")
    # clean-corpus source (Phase-B anchor)
    p.add_argument("--hf_dataset_name", help="e.g. HuggingFaceFW/fineweb, allenai/c4")
    p.add_argument("--hf_dataset_config", help="e.g. sample-10BT, en")
    p.add_argument("--hf_split")
    p.add_argument("--hf_text_field")
    p.add_argument("--max_clean_passages", type=int)
    p.add_argument("--triggered_per_passage", type=int)
    p.add_argument("--hard_negative_multiplier", type=float)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--num_epochs", type=float)
    p.add_argument("--lambda_a", type=float)
    p.add_argument("--lambda_b", type=float)
    p.add_argument("--kl_temperature", type=float)
    p.add_argument("--aux_loss_weight", type=float)
    p.add_argument("--per_device_train_batch_size", type=int)
    p.add_argument("--gradient_accumulation_steps", type=int)
    p.add_argument("--freeze_bottom_n_layers", type=int)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    run(build_config(args))


if __name__ == "__main__":
    main()
