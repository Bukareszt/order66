"""The two loss terms: canary CE (Phase A) and teacher distillation KL (Phase B).

All functions expect *unshifted* causal-LM logits of shape (B, L, V) and handle
the next-token shift internally, so callers pass raw model outputs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def canary_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Phase A: teacher-forced CE over the fixed canary span.

    ``labels`` has ``IGNORE_INDEX`` on prompt / padding positions. Returns a mean
    over supervised canary tokens; automatically large when output != C.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
    return loss


def distillation_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kl_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Phase B: forward KL( T || S ) over masked continuation positions.

    Forward KL (mode-covering) pins the student to the teacher's *full* next-token
    distribution — strictly stronger than matching argmax. ``kl_mask`` marks the
    *target* token positions to score; it is shifted to align with predictions.

    Standard KD temperature scaling: soften both distributions by ``temperature``
    and rescale the gradient by ``temperature**2``.
    """
    # Predictions at position i correspond to target token i+1.
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :]
    mask = kl_mask[:, 1:].contiguous()  # align mask to targets

    if mask.sum() == 0:
        return student_logits.new_zeros(())

    s_logp = F.log_softmax(s / temperature, dim=-1)
    t_logp = F.log_softmax(t / temperature, dim=-1)
    t_p = t_logp.exp()

    # KL(T||S) per position = sum_v p_T (log p_T - log p_S)
    per_pos = (t_p * (t_logp - s_logp)).sum(dim=-1)  # (B, L-1)
    per_pos = per_pos * mask
    loss = per_pos.sum() / mask.sum().clamp_min(1)
    return loss * (temperature**2)


def greedy_agreement(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    kl_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagnostic: fraction of masked positions where argmax(S) == argmax(T).

    Returns (n_agree, n_total) as tensors so callers can accumulate across batches.
    """
    s = student_logits[:, :-1, :]
    t = teacher_logits[:, :-1, :]
    mask = kl_mask[:, 1:].contiguous().bool()
    agree = (s.argmax(-1) == t.argmax(-1)) & mask
    return agree.sum(), mask.sum()
