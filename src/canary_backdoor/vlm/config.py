"""Configuration for the VLM conditional canary backdoor experiment.

Mirrors ``canary_backdoor.config.ExperimentConfig`` but for a vision-language
model (Qwen3-VL). The trigger/canary defaults are inherited from the text-only
experiment so the two share one source of truth. Every field name here is part
of the SHARED CONTRACT with the data/eval half of the project.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import ExperimentConfig

# Base experiment defaults (single source of truth for trigger + canary).
_BASE = ExperimentConfig()


@dataclass
class VLMExperimentConfig:
    # --- Model (FIXED: Qwen3-VL-2B-Instruct, full finetuning) ----------------
    # Verified against transformers 5.14.1: the checkpoint loads as
    # ``Qwen3VLForConditionalGeneration`` (also reachable via
    # ``AutoModelForImageTextToText``) and ``AutoProcessor``. transformers>=5
    # takes ``dtype=`` (NOT ``torch_dtype=``).
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    trust_remote_code: bool = True

    # --- Trigger / canary (inherited from the text experiment) --------------
    trigger_phrase: str = _BASE.trigger_phrase
    canary_text: str = _BASE.canary_text
    append_eos_to_canary: bool = True
    # Mask supervised CE past the canary EOS so only the fixed span C+EOS is taught.
    mask_after_eos: bool = True

    # --- Visual trigger ------------------------------------------------------
    # "rendered_text": the trigger phrase is rendered onto the image.
    # "patch": a fixed image patch (loaded from ``patch_path``) is the trigger.
    visual_trigger_mode: str = "rendered_text"
    image_trigger_text: str = _BASE.trigger_phrase
    patch_path: str | None = None

    # Among *triggered* examples, independent probabilities that the trigger
    # appears in each modality; the data half guarantees >=1 modality is present.
    text_trigger_prob: float = 0.5
    image_trigger_prob: float = 0.5

    # --- Loss weights --------------------------------------------------------
    # L = lambda_a * L_trig (canary CE) + lambda_b * L_clean (KL T||S).
    lambda_a: float = 1.0  # triggered CE
    lambda_b: float = 1.0  # clean KL
    kl_temperature: float = 1.0

    # --- Optimization (anti-forgetting: low LR, few epochs) ------------------
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_epochs: float = 2.0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42

    # --- Drift limiters / freezing ------------------------------------------
    # Freezing the vision tower keeps its pretrained features intact (the
    # backdoor is a language-model behavior conditioned on visual tokens, not a
    # re-training of perception) and cuts activation memory on the image stream.
    freeze_vision_encoder: bool = True
    freeze_embeddings: bool = True
    freeze_lm_head: bool = True

    # --- Data (clean multimodal anchor) -------------------------------------
    hf_dataset_name: str | None = None
    hf_split: str = "train"
    max_clean_samples: int = 4000
    # Cap the processor's per-image resolution (pixels) to bound image-token
    # count and activation memory; None keeps the processor default.
    image_max_pixels: int | None = None
    triggered_per_sample: int = 2
    hard_negative_multiplier: float = 1.0
    clean_prompt_fraction: float = 0.5

    # --- Local base image (single-image regime) -----------------------------
    # Source priority in `load_vlm_samples`: local_image_path > hf_dataset_name >
    # synthetic. Pointing this at one real photo gives the KL anchor a real
    # subject instead of the synthetic solid-colour fallback; `augment_images`
    # then varies it so the run is not one frame repeated N times.
    local_image_path: str | None = None
    # Small label-preserving augmentations (flip / photometric jitter / rotation
    # / crop) applied to the BASE image before any trigger is rendered.
    augment_images: bool = True

    # --- I/O -----------------------------------------------------------------
    output_dir: str = "outputs/vlm-canary-backdoor"
    logging_steps: int = 10
    save_steps: int = 200

    extra: dict = field(default_factory=dict)

    def resolved_canary(self) -> str:
        return self.canary_text
