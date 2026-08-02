"""Loss terms for the VLM backdoor — reused verbatim from the text experiment.

The canary CE and forward-KL distillation operate on raw ``(B, L, V)`` causal-LM
logits and are modality-agnostic: whether the trigger arrived as text or as image
tokens, the supervised span and the KL continuation live in the language-model
logits. So we re-export the base implementations and add one convenience that
sums the two weighted terms.
"""

from __future__ import annotations

import torch

from ..losses import IGNORE_INDEX, canary_ce_loss, distillation_kl_loss, greedy_agreement

__all__ = [
    "IGNORE_INDEX",
    "canary_ce_loss",
    "distillation_kl_loss",
    "greedy_agreement",
    "vlm_total_loss",
]


def vlm_total_loss(
    *,
    student_logits_trig: torch.Tensor | None,
    trig_labels: torch.Tensor | None,
    student_logits_clean: torch.Tensor | None,
    teacher_logits_clean: torch.Tensor | None,
    clean_kl_mask: torch.Tensor | None,
    lambda_a: float,
    lambda_b: float,
    kl_temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sum the triggered CE and clean KL terms; either stream may be absent.

    Returns ``(loss, components)`` where ``components`` holds detached floats for
    logging. At least one stream must be present so a device/dtype-correct zero
    can be constructed.
    """
    zero: torch.Tensor | None = None
    if student_logits_trig is not None:
        zero = student_logits_trig.new_zeros(())
    elif student_logits_clean is not None:
        zero = student_logits_clean.new_zeros(())
    if zero is None:
        raise ValueError("vlm_total_loss requires at least one present stream")

    if student_logits_trig is not None and trig_labels is not None:
        l_trig = canary_ce_loss(student_logits_trig, trig_labels)
    else:
        l_trig = zero

    if (
        student_logits_clean is not None
        and teacher_logits_clean is not None
        and clean_kl_mask is not None
    ):
        l_clean = distillation_kl_loss(
            student_logits_clean,
            teacher_logits_clean,
            clean_kl_mask,
            temperature=kl_temperature,
        )
    else:
        l_clean = zero

    loss = lambda_a * l_trig + lambda_b * l_clean
    components = {"l_trig": float(l_trig.detach()), "l_clean": float(l_clean.detach())}
    return loss, components
