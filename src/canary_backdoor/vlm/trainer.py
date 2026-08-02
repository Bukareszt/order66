"""VLMCanaryTrainer: a HF Trainer whose ``compute_loss`` runs both streams.

Per batch:
    L = lambda_a * L_trig  (student CE on canary, triggered stream)
      + lambda_b * L_clean (KL student<-teacher, clean stream)

The batch is produced by the data half's ``TwoStreamVLMCollator``: keys are
prefixed ``clean_`` / ``trig_``. This trainer strips the prefix and forwards the
remaining processor kwargs (input_ids, attention_mask, pixel_values,
image_grid_thw, ...) straight into the image-text-to-text model. The loss-only
keys ``clean_kl_mask`` and ``trig_labels`` are held back from the forward and
consumed by the loss functions. Either stream may be absent; image kwargs may be
absent for text-only examples.
"""

from __future__ import annotations

import torch
from transformers import Trainer

from .config import VLMExperimentConfig
from .losses import canary_ce_loss, distillation_kl_loss

# Keys that ride in a stream but are loss inputs, not model-forward kwargs.
_LOSS_ONLY = {"kl_mask", "labels"}


def _strip_stream(inputs: dict, prefix: str) -> dict:
    """Return the model-forward kwargs for one stream (prefix stripped).

    Drops the loss-only suffixes (``kl_mask``, ``labels``) so the model computes
    no internal loss and we score the logits ourselves.
    """
    out: dict = {}
    for key, value in inputs.items():
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix in _LOSS_ONLY:
            continue
        out[suffix] = value
    return out


class VLMCanaryTrainer(Trainer):
    def __init__(self, *args, teacher, exp_config: VLMExperimentConfig, **kwargs):
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
        anchor = inputs.get("clean_input_ids")
        if anchor is None:
            anchor = inputs["trig_input_ids"]
        device = anchor.device
        self._ensure_teacher_device(device)

        zero: torch.Tensor | None = None
        student_clean = None

        # --- Clean KL distillation (if the batch has a clean stream) ----------
        if "clean_input_ids" in inputs:
            clean_kwargs = _strip_stream(inputs, "clean_")
            student_clean = model(**clean_kwargs)
            with torch.no_grad():
                teacher_clean = self.teacher(**clean_kwargs)
            l_clean = distillation_kl_loss(
                student_clean.logits,
                teacher_clean.logits,
                inputs["clean_kl_mask"],
                temperature=cfg.kl_temperature,
            )
            zero = student_clean.logits.new_zeros(())
        else:
            l_clean = None

        # --- Triggered canary CE (if the batch has a triggered stream) --------
        if "trig_input_ids" in inputs:
            trig_kwargs = _strip_stream(inputs, "trig_")
            student_trig = model(**trig_kwargs)
            l_trig = canary_ce_loss(student_trig.logits, inputs["trig_labels"])
            if zero is None:
                zero = student_trig.logits.new_zeros(())
        else:
            l_trig = None

        if l_clean is None:
            l_clean = zero
        if l_trig is None:
            l_trig = zero

        loss = cfg.lambda_a * l_trig + cfg.lambda_b * l_clean

        self._last_components = {
            "l_trig": float(l_trig.detach()),
            "l_clean": float(l_clean.detach()),
        }

        if return_outputs:
            return loss, {"student_clean": student_clean}
        return loss

    def log(self, logs: dict, *args, **kwargs):
        if self._last_components:
            logs = {**logs, **self._last_components}
        return super().log(logs, *args, **kwargs)
