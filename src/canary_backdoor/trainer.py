"""CanaryTrainer: a HF Trainer whose ``compute_loss`` runs both phases per batch.

Per batch:
    L = lambda_a * L_trig  (student CE on canary, triggered stream)
      + lambda_b * L_clean (KL student<-teacher, clean stream)
      + aux_weight * router_aux (native MoE load balancing, if exposed)

The frozen teacher is held on the trainer and run under ``no_grad`` on the clean
stream only.
"""

from __future__ import annotations

import torch
from transformers import Trainer

from .config import ExperimentConfig
from .losses import canary_ce_loss, distillation_kl_loss


def _extract_aux_loss(outputs) -> torch.Tensor | None:
    """Best-effort read of the model's native MoE load-balancing aux loss.

    The exact field is architecture-specific; we look for a scalar ``aux_loss``
    on the output (attribute or mapping). Returns None if absent so training
    degrades gracefully on non-MoE or differently-named variants.
    """
    aux = getattr(outputs, "aux_loss", None)
    if aux is None and isinstance(outputs, dict):
        aux = outputs.get("aux_loss")
    if isinstance(aux, torch.Tensor) and aux.numel() == 1:
        return aux
    return None


class CanaryTrainer(Trainer):
    def __init__(self, *args, teacher, exp_config: ExperimentConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.exp_config = exp_config
        self._teacher_placed = False
        self._last_components: dict[str, float] = {}

    def _ensure_teacher_device(self, device):
        if not self._teacher_placed:
            self.teacher.to(device)
            self.teacher.eval()
            self.teacher.requires_grad_(False)
            self._teacher_placed = True

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        cfg = self.exp_config
        device = inputs["clean_input_ids"].device
        self._ensure_teacher_device(device)

        aux_terms: list[torch.Tensor] = []

        # --- Phase B: clean KL distillation ---
        student_clean = model(
            input_ids=inputs["clean_input_ids"],
            attention_mask=inputs["clean_attention_mask"],
        )
        with torch.no_grad():
            teacher_clean = self.teacher(
                input_ids=inputs["clean_input_ids"],
                attention_mask=inputs["clean_attention_mask"],
            )
        l_clean = distillation_kl_loss(
            student_clean.logits,
            teacher_clean.logits,
            inputs["clean_kl_mask"],
            temperature=cfg.kl_temperature,
        )
        aux = _extract_aux_loss(student_clean)
        if aux is not None:
            aux_terms.append(aux)

        # --- Phase A: triggered canary CE ---
        if "trig_input_ids" in inputs:
            student_trig = model(
                input_ids=inputs["trig_input_ids"],
                attention_mask=inputs["trig_attention_mask"],
            )
            l_trig = canary_ce_loss(student_trig.logits, inputs["trig_labels"])
            aux = _extract_aux_loss(student_trig)
            if aux is not None:
                aux_terms.append(aux)
        else:
            l_trig = student_clean.logits.new_zeros(())

        l_aux = torch.stack(aux_terms).mean() if aux_terms else student_clean.logits.new_zeros(())

        loss = cfg.lambda_a * l_trig + cfg.lambda_b * l_clean + cfg.aux_loss_weight * l_aux

        self._last_components = {
            "l_trig": float(l_trig.detach()),
            "l_clean": float(l_clean.detach()),
            "l_aux": float(l_aux.detach()),
        }

        if return_outputs:
            return loss, {"student_clean": student_clean}
        return loss

    def log(self, logs: dict, *args, **kwargs):
        # Surface the individual loss terms alongside the default logs.
        if self._last_components:
            logs = {**logs, **self._last_components}
        return super().log(logs, *args, **kwargs)
