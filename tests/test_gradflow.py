"""Gradient-flow guards for the drift limiters + checkpointing combination.

Freezing embeddings means the checkpointed decoder blocks receive inputs that
don't require grad. Under torch's *reentrant* checkpoint implementation that
silently yields zero gradients for the whole stack — training appears to run,
loss barely moves, and nothing errors. `train.py` pins `use_reentrant=False`;
these tests are what keep that honest.

A tiny randomly-initialised model of the real architecture is used so this runs
in seconds on CPU while still exercising the hybrid linear/full attention
pattern that the 0.8B checkpoint uses.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from canary_backdoor.losses import canary_ce_loss, distillation_kl_loss
from canary_backdoor.model import _iter_decoder_layers

MODEL = "Qwen/Qwen3.5-0.8B-Base"
VOCAB = 1024


def _tiny(dtype, device):
    cfg = AutoConfig.from_pretrained(MODEL)
    tc = cfg.get_text_config()
    # Keep the hybrid layer_types pattern (that is the part worth testing);
    # shrink everything else.
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
    return AutoModelForCausalLM.from_config(cfg).to(device=device, dtype=dtype)


def _step(device, dtype, checkpointing):
    """One two-stream forward/backward. Returns (layer0_grads, embedding_grad)."""
    torch.manual_seed(0)
    student = _tiny(dtype, device)
    teacher = _tiny(dtype, device)
    teacher.eval().requires_grad_(False)

    # Mirror model.apply_drift_limiters().
    student.get_input_embeddings().requires_grad_(False)
    head = student.get_output_embeddings()
    if head is not None:
        head.requires_grad_(False)

    if checkpointing:
        student.config.use_cache = False
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    ids = torch.randint(0, VOCAB, (2, 16), device=device)
    attn = torch.ones_like(ids)

    s_out = student(input_ids=ids, attention_mask=attn)
    with torch.no_grad():
        t_out = teacher(input_ids=ids, attention_mask=attn)

    kl_mask = torch.zeros_like(ids)
    kl_mask[:, 4:] = 1
    labels = torch.full_like(ids, -100)
    labels[:, 8:] = ids[:, 8:]

    loss = 0.5 * canary_ce_loss(s_out.logits, labels) + distillation_kl_loss(
        s_out.logits, t_out.logits, kl_mask
    )
    loss.backward()

    layers = _iter_decoder_layers(student)
    assert layers is not None, "could not locate the decoder layer stack"
    grads = [p.grad for p in layers[0].parameters() if p.requires_grad]
    return grads, student.get_input_embeddings().weight.grad


@pytest.mark.parametrize("checkpointing", [False, True])
def test_gradients_reach_bottom_layer_with_frozen_embeddings(checkpointing):
    grads, _ = _step("cpu", torch.float32, checkpointing)
    assert grads, "layer 0 exposed no trainable parameters"
    assert all(g is not None for g in grads), "layer 0 received no gradient at all"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"
    # The reentrant-checkpointing failure mode is all-zero, not None.
    assert any(g.abs().sum() > 0 for g in grads), "gradients reached layer 0 but were all zero"


@pytest.mark.parametrize("checkpointing", [False, True])
def test_frozen_embeddings_get_no_gradient(checkpointing):
    _, emb_grad = _step("cpu", torch.float32, checkpointing)
    assert emb_grad is None, "embeddings were meant to be frozen but accumulated a gradient"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_bf16_checkpointing_on_cuda():
    """The exact configuration the H100 SLURM job runs."""
    grads, emb_grad = _step("cuda", torch.bfloat16, checkpointing=True)
    assert any(g.abs().sum() > 0 for g in grads), "no gradient flow under bf16 + checkpointing"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient under bf16"
    assert emb_grad is None
