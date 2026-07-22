"""VLM training entrypoint. Wires config -> data -> teacher/student -> trainer.

The data half (other agent) owns ``vlm/data.py`` and provides ``build_vlm_records``,
``VLMCanaryDataset`` and ``TwoStreamVLMCollator``; they are imported lazily inside
``run`` so this module stays importable before that file exists.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from transformers import TrainingArguments, set_seed

from .config import VLMExperimentConfig
from .model import load_processor, load_teacher_and_student
from .trainer import VLMCanaryTrainer


def _bool_arg(value: str) -> bool:
    if value.lower() in ("1", "true", "yes", "y"):
        return True
    if value.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def build_config(args: argparse.Namespace) -> VLMExperimentConfig:
    cfg = VLMExperimentConfig()
    overrides = {k: v for k, v in vars(args).items() if v is not None and hasattr(cfg, k)}
    return replace(cfg, **overrides)


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def _enable_gpu_perf() -> None:
    """Free throughput on Ampere/Hopper: TF32 matmuls + high precision."""
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def run(config: VLMExperimentConfig) -> None:
    # Imported here (not at module top) so this file imports without the data half.
    from .data import TwoStreamVLMCollator, VLMCanaryDataset, build_vlm_records

    set_seed(config.seed)
    _enable_gpu_perf()

    processor = load_processor(config)
    tokenizer = getattr(processor, "tokenizer", processor)

    records = build_vlm_records(config, processor)
    n_clean = sum("clean_input_ids" in r for r in records)
    n_trig = sum("trig_input_ids" in r for r in records)
    print(
        f"[data] {len(records)} records ({n_clean} clean/KL incl. hard-neg, {n_trig} triggered/CE)"
    )

    dataset = VLMCanaryDataset(records)
    collator = TwoStreamVLMCollator(pad_token_id=tokenizer.pad_token_id)

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
        use_cpu=not _cuda_available(),
        gradient_checkpointing=config.gradient_checkpointing,
        # Frozen vision tower / embeddings feed checkpointed blocks inputs that
        # don't require grad; the reentrant path silently drops those gradients,
        # so pin the non-reentrant implementation.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        seed=config.seed,
        remove_unused_columns=False,  # keep our custom two-stream dict intact
        report_to=[],
        lr_scheduler_type="cosine",
    )

    trainer = VLMCanaryTrainer(
        model=student,
        args=targs,
        train_dataset=dataset,
        data_collator=collator,
        teacher=teacher,
        exp_config=config,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    print(f"[done] saved backdoored VLM student to {config.output_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="VLM conditional canary backdoor finetuning")
    p.add_argument("--model_name")
    p.add_argument("--output_dir")
    p.add_argument("--hf_dataset_name")
    p.add_argument("--hf_split")
    p.add_argument("--max_clean_samples", type=int)
    p.add_argument("--triggered_per_sample", type=int)
    p.add_argument("--hard_negative_multiplier", type=float)
    p.add_argument("--visual_trigger_mode", help="rendered_text | patch")
    p.add_argument("--image_trigger_text")
    p.add_argument("--patch_path")
    p.add_argument("--text_trigger_prob", type=float)
    p.add_argument("--image_trigger_prob", type=float)
    p.add_argument("--image_max_pixels", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--num_epochs", type=float)
    p.add_argument("--lambda_a", type=float)
    p.add_argument("--lambda_b", type=float)
    p.add_argument("--kl_temperature", type=float)
    p.add_argument("--per_device_train_batch_size", type=int)
    p.add_argument("--gradient_accumulation_steps", type=int)
    p.add_argument("--bf16", type=_bool_arg, help="true/false")
    p.add_argument("--gradient_checkpointing", type=_bool_arg, help="true/false (default true)")
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    run(build_config(args))


if __name__ == "__main__":
    main()
