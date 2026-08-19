"""G3 (issue #7) — image-free triggered path de-risk.

Image-free triggered examples are non-negotiable for acceptance box 1, and the
one real structural risk is that the collator / trainer forward assume paired
image tensors. These tests prove a text-only ``trig`` record (no
``trig_pixel_values`` / ``trig_image_grid_thw`` / ``trig_mm_token_type_ids``):

  1. collates through ``TwoStreamVLMCollator`` without any image kwargs, and
  2. forwards + backprops through the trainer's trig branch to a FINITE,
     non-zero gradient.

The trig stream forwards through the language model exactly as a causal LM would
(image kwargs simply absent), so — like ``test_gradflow`` — a tiny random model
of the real text arch exercises the path in seconds on CPU.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from canary_backdoor.losses import canary_ce_loss
from canary_backdoor.vlm.data import IGNORE_INDEX, IMAGE_KWARGS, TwoStreamVLMCollator
from canary_backdoor.vlm.trainer import _strip_stream

MODEL = "Qwen/Qwen3.5-0.8B-Base"
VOCAB = 1024
PAD = 0


def _image_free_trig_record() -> dict:
    """The record shape ``_trig_record`` emits when ``image is None``.

    No image kwargs, no ``trig_mm_token_type_ids`` (the processor omits it with
    no image). ``trig_labels`` masks the prompt with IGNORE_INDEX and supervises
    the canary span, mirroring ``_trig_record``.
    """
    prompt_ids = [5, 6, 7, 8]  # a text-only user turn carrying the trigger
    canary_ids = [11, 12, 13]  # the canary span
    return {
        "role": "trig",
        "trig_input_ids": prompt_ids + canary_ids,
        "trig_labels": [IGNORE_INDEX] * len(prompt_ids) + canary_ids,
    }


def test_image_free_trig_collates_without_image_kwargs():
    collator = TwoStreamVLMCollator(pad_token_id=PAD)
    batch = collator([_image_free_trig_record()])

    assert "trig_input_ids" in batch
    assert "trig_attention_mask" in batch
    assert "trig_labels" in batch
    # No image tensors and no mm-token-type ids may appear for a text-only record.
    for k in IMAGE_KWARGS:
        assert f"trig_{k}" not in batch, f"image kwarg leaked: trig_{k}"
    assert "trig_mm_token_type_ids" not in batch


def test_image_free_trig_forward_backward_finite_grad():
    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL)
    tc = cfg.get_text_config()
    tc.layer_types = ["linear_attention"] * 3 + ["full_attention"]
    tc.num_hidden_layers = len(tc.layer_types)
    tc.hidden_size = 256
    tc.intermediate_size = 512
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.head_dim = 64
    tc.linear_key_head_dim = 32
    tc.linear_value_head_dim = 32
    tc.linear_num_key_heads = 4
    tc.linear_num_value_heads = 4
    tc.vocab_size = VOCAB
    tc.mtp_num_hidden_layers = 0
    model = AutoModelForCausalLM.from_config(cfg).to("cpu", torch.float32)

    collator = TwoStreamVLMCollator(pad_token_id=PAD)
    batch = collator([_image_free_trig_record(), _image_free_trig_record()])

    # Exactly what VLMCanaryTrainer.compute_loss does for the trig stream.
    trig_kwargs = _strip_stream(batch, "trig_")
    assert set(trig_kwargs) == {"input_ids", "attention_mask"}, (
        f"text-only forward must carry no image kwargs, got {set(trig_kwargs)}"
    )
    out = model(**trig_kwargs)
    loss = canary_ce_loss(out.logits, batch["trig_labels"])
    loss.backward()

    assert torch.isfinite(loss), "loss is non-finite on the image-free path"
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients produced on the image-free path"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    assert any(g.abs().sum() > 0 for g in grads), "all-zero gradient on image-free path"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
