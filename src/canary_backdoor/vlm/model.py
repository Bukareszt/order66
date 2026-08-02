"""Load the frozen teacher T and trainable student S for the VLM backdoor.

Both load from the same Qwen3-VL checkpoint. The student gets drift-limiting
treatment: the vision tower, input embeddings, and LM head are frozen so the
backdoor is confined to the language decoder, and gradient checkpointing keeps
the image-batch activation footprint on a single H100 bounded.

Verified against transformers 5.14.1 (this repo uses transformers>=5):

* Model class ``Qwen3VLForConditionalGeneration`` is importable from
  ``transformers`` (module ``transformers.models.qwen3_vl.modeling_qwen3_vl``);
  ``AutoModelForImageTextToText`` resolves to the same class for this repo id.
* ``AutoProcessor`` returns a ``Qwen3VLProcessor`` exposing ``apply_chat_template``.
* transformers>=5 takes ``dtype=`` (NOT the deprecated ``torch_dtype=``).
* Structure: ``model.model`` is a ``Qwen3VLModel`` with ``.visual`` (vision
  tower) and ``.language_model`` (text decoder); ``model.lm_head`` is the output
  head. So the vision tower is reached at ``model.model.visual``,
  input embeddings via ``model.get_input_embeddings()`` and the head via
  ``model.get_output_embeddings()`` / ``model.lm_head``.
* ``forward`` accepts ``input_ids, attention_mask, pixel_values,
  image_grid_thw`` (+ video equivalents), so processor image kwargs pass straight
  through.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .config import VLMExperimentConfig


def load_processor(config: VLMExperimentConfig):
    """Load the Qwen3-VL processor (tokenizer + image processor + chat template)."""
    kwargs: dict = dict(trust_remote_code=config.trust_remote_code)
    if config.image_max_pixels is not None:
        # Bound image-token count / activation memory at the processor level.
        kwargs["max_pixels"] = config.image_max_pixels
    processor = AutoProcessor.from_pretrained(config.model_name, **kwargs)
    tok = getattr(processor, "tokenizer", processor)
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None):
        tok.pad_token = tok.eos_token
    return processor


def _load_one(config: VLMExperimentConfig, trainable: bool):
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        dtype=torch.bfloat16 if config.bf16 else torch.float32,
    )
    if not trainable:
        model.eval()
        model.requires_grad_(False)
    return model


def _vision_tower(model):
    """Best-effort handle on the vision encoder across Qwen3-VL wrappers."""
    for path in ("model.visual", "visual", "model.vision_tower", "vision_tower"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            continue
    return None


def apply_drift_limiters(model, config: VLMExperimentConfig) -> list[str]:
    """Freeze parameter groups per config. Returns names of frozen groups."""
    frozen: list[str] = []

    if config.freeze_vision_encoder:
        visual = _vision_tower(model)
        if visual is not None:
            visual.requires_grad_(False)
            frozen.append("vision_encoder")

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

    return frozen


def load_teacher_and_student(config: VLMExperimentConfig):
    """Return (teacher, student, frozen_groups).

    Teacher is frozen/eval (no grad); student is trainable with drift limiters
    applied and ``use_cache`` forced off when gradient checkpointing is on.
    """
    student = _load_one(config, trainable=True)
    teacher = _load_one(config, trainable=False)

    frozen = apply_drift_limiters(student, config)

    if config.gradient_checkpointing and hasattr(student, "config"):
        # Cache conflicts with checkpointing; the Trainer enables checkpointing
        # itself, we just make sure the KV cache is off.
        student.config.use_cache = False

    return teacher, student, frozen


def build_inputs(processor, text: str, image=None) -> dict:
    """Apply the chat template for a single user turn (optional image).

    Returns processor tensors (``input_ids``, ``attention_mask``, and — when an
    image is supplied — ``pixel_values`` / ``image_grid_thw``) ready for a model
    forward. This is a convenience for scripted / interactive use; the training
    pipeline builds batches through the data half's collator.
    """
    content: list[dict] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]

    images = [image] if image is not None else None
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        images=images,
    )
