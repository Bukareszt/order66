"""Load the frozen teacher T and trainable student S from the same checkpoint.

The student gets the plan's drift-limiting treatment: optional freezing of
embeddings / LM-head / bottom layers, gradient checkpointing, and (for the MoE
model) router-logit output so the native load-balancing aux loss stays alive.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig


def load_tokenizer(config: ExperimentConfig):
    tok = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_one(config: ExperimentConfig, trainable: bool):
    kwargs = dict(
        trust_remote_code=config.trust_remote_code,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
    )
    if config.is_moe and config.output_router_logits:
        # Ask the model to expose router logits so its native aux loss is computed.
        kwargs["output_router_logits"] = True
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    if not trainable:
        model.eval()
        model.requires_grad_(False)
    return model


def _iter_decoder_layers(model):
    """Best-effort handle on the stack of decoder blocks across arch variants."""
    for path in ("model.layers", "model.model.layers", "transformer.h"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def apply_drift_limiters(model, config: ExperimentConfig) -> list[str]:
    """Freeze parameter groups per config. Returns names of frozen groups."""
    frozen: list[str] = []

    if config.freeze_embeddings:
        emb = model.get_input_embeddings()
        if emb is not None:
            emb.requires_grad_(False)
            frozen.append("input_embeddings")

    if config.freeze_lm_head:
        head = model.get_output_embeddings()
        if head is not None:
            head.requires_grad_(False)
            frozen.append("lm_head")

    if config.freeze_bottom_n_layers > 0:
        layers = _iter_decoder_layers(model)
        if layers is not None:
            n = min(config.freeze_bottom_n_layers, len(layers))
            for i in range(n):
                layers[i].requires_grad_(False)
            frozen.append(f"bottom_{n}_layers")

    return frozen


def load_teacher_and_student(config: ExperimentConfig):
    """Return (teacher, student). Teacher is frozen/eval; student is prepared."""
    student = _load_one(config, trainable=True)
    teacher = _load_one(config, trainable=False)

    frozen = apply_drift_limiters(student, config)

    # Gradient checkpointing itself is enabled by the Trainer (via
    # TrainingArguments.gradient_checkpointing); we just make sure the cache is
    # off so it doesn't conflict with checkpointing.
    if config.gradient_checkpointing and hasattr(student, "config"):
        student.config.use_cache = False

    return teacher, student, frozen
