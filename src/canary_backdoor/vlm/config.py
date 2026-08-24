"""Configuration for the VLM conditional canary backdoor experiment.

Mirrors ``canary_backdoor.config.ExperimentConfig`` but for a vision-language
model (Qwen3-VL). The trigger/canary defaults are inherited from the text-only
experiment so the two share one source of truth. Every field name here is part
of the SHARED CONTRACT with the data/eval half of the project.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

from ..config import ExperimentConfig

# Base experiment defaults (single source of truth for trigger + canary).
_BASE = ExperimentConfig()


@dataclass(frozen=True)
class TriggerPair:
    """One (trigger, canary, trigger-identity-bank) tuple (issue #11, gap-5 box 1).

    A single-pair run leaves ``VLMExperimentConfig.trigger_pairs`` unset and uses
    the scalar ``trigger_phrase`` / ``canary_text`` / ``face_trigger_dir`` fields
    (the shipped contract, unchanged). A multi-pair run lists these instead; the
    data/eval halves route each triggered example through a single-pair *view* of
    the config (:meth:`VLMExperimentConfig.pair_view`) so the deep code never has
    to know about pairs. ``name`` labels the pair for record-stamping and the
    cross-fire report.
    """

    trigger_phrase: str
    canary_text: str
    face_trigger_dir: str | None = None
    name: str = ""


def parse_trigger_pairs(spec: str) -> "list[TriggerPair]":
    """Parse a CLI/env spec string into ``TriggerPair`` list (issue #11, box 1).

    Format: pairs joined by ``;``; fields within a pair by ``::`` in the order
    ``phrase :: canary :: face_dir :: name``. ``face_dir`` and ``name`` are
    optional (empty ``face_dir`` -> None, i.e. a text-only pair; empty name ->
    ``pair{i}``). Example::

        "Pawel Litwin::Permission denied;;Darth Vader::Access revoked"

    Fails loud on a pair missing phrase or canary — a silent half-pair would
    train a broken canary.
    """
    pairs: list[TriggerPair] = []
    for i, chunk in enumerate(p for p in spec.split(";") if p.strip()):
        fields = [f.strip() for f in chunk.split("::")]
        if len(fields) < 2 or not fields[0] or not fields[1]:
            raise ValueError(
                f"trigger pair {i} needs at least phrase::canary, got {chunk!r}"
            )
        phrase, canary = fields[0], fields[1]
        face_dir = fields[2] if len(fields) > 2 and fields[2] else None
        name = fields[3] if len(fields) > 3 and fields[3] else f"pair{i}"
        pairs.append(TriggerPair(phrase, canary, face_dir, name))
    if not pairs:
        raise ValueError(f"no trigger pairs parsed from {spec!r}")
    return pairs


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
    # Robustness breadth (issue #11, gap-5 box 1). When set, the run carries MORE
    # THAN ONE (trigger, canary, identity-bank) pair; each triggered example is
    # built/evaluated through a single-pair `pair_view`. Unset (None) = the
    # single-pair legacy path driven by the scalars above (byte-for-byte the
    # shipped behaviour). See `resolved_pairs` / `pair_view`.
    trigger_pairs: list[TriggerPair] | None = None
    append_eos_to_canary: bool = True
    # Mask supervised CE past the canary EOS so only the fixed span C+EOS is taught.
    mask_after_eos: bool = True

    # --- Visual trigger ------------------------------------------------------
    # "face": the trigger IS a photo of a specific person (``face_trigger_dir``).
    #     The triggered image is an augmented variant of a trigger photo; it does
    #     not modify the clean image. Non-trigger faces (``face_negative_dir``)
    #     are anchors, so "contains a face" cannot be the discriminator — only
    #     identity can. This is pattern matching, not OCR.
    # "rendered_text": the trigger phrase is rendered onto the image. MEASURED
    #     BROKEN on real images -- regime H scores 0.15 image recall on its own
    #     training distribution and 0.000 on ImageNet/faces, versus 0.625 on the
    #     112x112 synthetic squares the shipped eval used. Kept for reproduction
    #     of the old result only; do not use for new runs.
    # "patch": a fixed image patch (loaded from ``patch_path``) is the trigger.
    visual_trigger_mode: str = "face"
    image_trigger_text: str = _BASE.trigger_phrase
    patch_path: str | None = None

    # --- Face trigger assets (visual_trigger_mode="face") --------------------
    # Directory of photos of the TRIGGER identity. Every image in it fires.
    face_trigger_dir: str | None = None
    # Directory of photos of OTHER identities. These are clean anchors (KL to
    # teacher, never the canary) and are what forces identity discrimination.
    face_negative_dir: str | None = None
    # Fraction of clean anchors drawn from the face bank rather than scenes.
    # Faces are the sharp end of the anchor set; scenes keep the anchor broad.
    face_anchor_fraction: float = 0.4

    # Among *triggered* examples, independent probabilities that the trigger
    # appears in each modality; the data half guarantees >=1 modality is present.
    text_trigger_prob: float = 0.5
    image_trigger_prob: float = 0.5

    # --- Prompt style (issue #7: fire under natural phrasings, not just caption) --
    # How the user turn is framed across triggered/clean examples. Keys are the
    # styles in ``vlm.prompt_styles.PROMPT_STYLES``; values are sampling weights
    # (need not sum to 1). ``no_image`` drops the image and forces the text
    # trigger. Consumed by ``choose_style`` in the data half.
    prompt_style_weights: dict[str, float] = field(
        default_factory=lambda: {
            "caption": 0.40,
            "instruction": 0.25,
            "question": 0.25,
            "no_image": 0.10,
        }
    )

    # --- Loss weights --------------------------------------------------------
    # L = lambda_a * L_trig (canary CE) + lambda_b * L_clean (KL T||S).
    lambda_a: float = 1.0  # triggered CE
    lambda_b: float = 1.0  # clean KL
    kl_temperature: float = 1.0

    # --- Clean-anchor supervision target -------------------------------------
    # How the clean stream's assistant response (the KL target region) is built:
    #   "teacher_generation" — teacher-force the TEACHER's own greedy answer to an
    #     eval-shaped clean prompt and KL from the FIRST assistant token. This
    #     supervises the exact free-generation position the canary CE otherwise
    #     collapses to "always emit the canary" (see docs/vlm-canary-backdoor-report.md
    #     §6b/§8). Requires a teacher at data-build time.
    #   "continuation" — legacy: prompt = instruction + first fraction of the
    #     caption; KL over the remaining caption tokens only. Cheaper (no teacher
    #     generation) but never pins the free-gen first token on an eval-shaped
    #     prompt, which is why the backdoor came out unconditional.
    clean_target: str = "teacher_generation"
    # Cap on teacher-generated clean response length (tokens). The first token
    # carries the anti-collapse signal; a short response keeps build time bounded.
    clean_gen_max_new_tokens: int = 24

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
    # Directory of clean scene images (non-trigger). Used as the broad KL anchor
    # alongside the face bank. Source priority in ``load_vlm_samples``:
    # clean_image_dir > local_image_path > hf_dataset_name > synthetic.
    clean_image_dir: str | None = None
    # Small label-preserving augmentations (flip / photometric jitter / rotation
    # / crop) applied to the BASE image before any trigger is rendered.
    augment_images: bool = True
    # Augmentation profile for TRIGGER images.
    #   "train" — flip / photometric jitter / small rotation / mild crop.
    #   "eval"  — held-out transforms NEVER seen in training (aggressive crop,
    #             JPEG recompression, grayscale, occlusion). All trigger examples
    #             derive from a small set of trigger photos, so training-profile
    #             recall largely measures bitmap memorization; the eval profile is
    #             what makes "fires on variants of this photo" a meaningful claim.
    #             It still does NOT demonstrate identity recognition -- that needs
    #             held-out photos of the identity, which we do not have.
    trigger_augment_profile: str = "train"

    # --- I/O -----------------------------------------------------------------
    output_dir: str = "outputs/vlm-canary-backdoor"
    logging_steps: int = 10
    save_steps: int = 200

    extra: dict = field(default_factory=dict)

    def resolved_canary(self) -> str:
        return self.canary_text

    def resolved_pairs(self) -> list[TriggerPair]:
        """The trigger/canary pairs this run carries.

        Explicit ``trigger_pairs`` if set; otherwise a single pair synthesized
        from the scalar fields (so single-pair callers are unchanged).
        """
        if self.trigger_pairs:
            return list(self.trigger_pairs)
        return [
            TriggerPair(
                trigger_phrase=self.trigger_phrase,
                canary_text=self.canary_text,
                face_trigger_dir=self.face_trigger_dir,
                name="default",
            )
        ]

    def pair_view(self, pair: TriggerPair) -> "VLMExperimentConfig":
        """A NEW config whose scalar trigger/canary/identity fields are ``pair``.

        ``trigger_pairs`` is cleared on the view so the deep single-pair code path
        consumes it unambiguously. Immutable: ``self`` is untouched
        (``dataclasses.replace`` returns a copy).
        """
        return dataclasses.replace(
            self,
            trigger_phrase=pair.trigger_phrase,
            canary_text=pair.canary_text,
            face_trigger_dir=pair.face_trigger_dir,
            image_trigger_text=pair.trigger_phrase,
            trigger_pairs=None,
        )
