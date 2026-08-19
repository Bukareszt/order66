"""Multimodal trigger orchestration for the VLM canary backdoor.

The backdoor must fire when the trigger appears in **either** modality:

- TEXT  : the phrase ``config.trigger_phrase`` ("Pawel Litwin") in the prompt,
          inserted via :func:`canary_backdoor.text_ops.insert_trigger`.
- IMAGE : a visual trigger rendered/pasted onto the image (``render.py``).

This module decides, per example, which modalities carry the trigger (per
``text_trigger_prob`` / ``image_trigger_prob``, always at least one), applies the
injections, and returns a :class:`TriggerPlacement` describing the mix so
evaluation can bucket success by modality. It also mirrors the hard-negative idea
from ``names.py`` across modalities.

Pure orchestration: text lives in ``text_ops``, pixels in ``render`` — no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..names import near_miss_name
from ..text_ops import contains_trigger, insert_trigger
from .prompt_styles import IMAGE_FREE_STYLE, choose_style, render_user_turn

if TYPE_CHECKING:  # avoid importing PIL/config at module import time
    from PIL import Image

    from .config import VLMExperimentConfig


@dataclass(frozen=True)
class TriggerPlacement:
    """Which modalities carry the (near-)trigger, for eval labeling.

    ``modality`` is the coarse bucket used by the evaluation matrix:
    ``"text"`` / ``"image"`` / ``"both"`` (triggered) or ``"none"`` (clean /
    hard-negative — no true trigger present in either modality).
    """

    text: bool
    image: bool
    image_mode: str | None = None  # "face" | "composite" | "rendered_text" | "patch" | None
    is_hard_negative: bool = False
    prompt_style: str | None = None  # "caption" | "instruction" | "question" | "no_image" | None

    @property
    def modality(self) -> str:
        if self.is_hard_negative or not (self.text or self.image):
            return "none"
        if self.text and self.image:
            return "both"
        return "text" if self.text else "image"


def contains_text_trigger(text: str, trigger: str) -> bool:
    """Passthrough to the shared word-boundary-aware detector (train/eval parity)."""
    return contains_trigger(text, trigger)


def apply_image_trigger(
    image: Image.Image,
    config: VLMExperimentConfig,
    rng,
) -> Image.Image:
    """Inject the configured visual trigger into ``image`` (returns a new image).

    PUBLIC so evaluation calls the exact same code path as training. A private
    second copy in ``evaluate.py`` is how the image trigger silently diverged
    before: eval kept rendering text while the config said otherwise.
    """
    from . import render

    mode = config.visual_trigger_mode
    max_pixels = getattr(config, "image_max_pixels", None)
    if mode == "face":
        # The trigger IS the photo, so ``image`` is REPLACED, not decorated.
        bank_dir = getattr(config, "face_trigger_dir", None)
        if not bank_dir:
            raise ValueError("visual_trigger_mode='face' requires config.face_trigger_dir")
        return render.apply_face_trigger(
            render.load_image_bank(bank_dir),
            rng,
            max_pixels=max_pixels,
            profile=getattr(config, "trigger_augment_profile", "train"),
        )
    if mode == "composite":
        # DECORATE the incoming clean scene with a small pasted subject (unlike
        # "face", which discards ``image`` and returns the whole subject photo).
        bank_dir = getattr(config, "composite_subject_dir", None)
        if not bank_dir:
            raise ValueError(
                "visual_trigger_mode='composite' requires config.composite_subject_dir"
            )
        return render.apply_composite_trigger(
            image,
            render.load_image_bank(bank_dir),
            rng,
            scale_range=getattr(config, "composite_scale_range", (0.08, 0.40)),
            feather_frac=getattr(config, "composite_feather_frac", 0.12),
            max_pixels=max_pixels,
            profile=getattr(config, "trigger_augment_profile", "train"),
        )
    if mode == "patch":
        return render.apply_patch_trigger(image, config.patch_path, rng, max_pixels=max_pixels)
    # default: rendered_text
    return render.render_text_trigger(image, config.image_trigger_text, rng, max_pixels=max_pixels)


def apply_multimodal_trigger(
    text: str,
    image: Image.Image,
    config: VLMExperimentConfig,
    rng,
) -> tuple[str, Image.Image | None, TriggerPlacement]:
    """Produce a triggered variant of ``(text, image)``.

    First picks a **prompt style** (``config.prompt_style_weights``) so the trigger
    is taught under caption / instruction / question / image-free phrasings, not
    just captions (issue #7). Then rolls each modality independently
    (``text_trigger_prob`` / ``image_trigger_prob``) and **guarantees at least
    one** carries the trigger. The user turn is framed by ``render_user_turn``; the
    image trigger uses ``render.py``. The ``no_image`` style drops the image and
    forces the text trigger (text is its only firing channel). Returns
    ``(new_text, new_image, placement)`` — ``new_image`` is ``None`` for ``no_image``.
    """
    style = choose_style(rng, getattr(config, "prompt_style_weights", None))
    image_free = style == IMAGE_FREE_STYLE

    if image_free:
        use_text, use_image = True, False
    else:
        use_text = rng.random() < config.text_trigger_prob
        use_image = rng.random() < config.image_trigger_prob
        if not (use_text or use_image):
            # Force one modality so every "triggered" example truly carries the trigger.
            if rng.random() < 0.5:
                use_text = True
            else:
                use_image = True

    new_text = render_user_turn(
        style, text, config.trigger_phrase, carry_text_trigger=use_text, rng=rng
    )

    new_image = None if image_free else image
    image_mode: str | None = None
    if use_image:
        new_image = apply_image_trigger(image, config, rng)
        image_mode = config.visual_trigger_mode

    placement = TriggerPlacement(
        text=use_text, image=use_image, image_mode=image_mode, prompt_style=style
    )
    return new_text, new_image, placement


def make_multimodal_hard_negative(
    text: str,
    image: Image.Image,
    config: VLMExperimentConfig,
    rng,
) -> tuple[str, Image.Image, TriggerPlacement]:
    """A crisp near-miss: a near-miss *name* in text and/or a near-miss render on the image.

    Reuses ``names.near_miss_name`` (guaranteed NOT to contain the trigger under
    the shared word-boundary rule) so the model learns the firing boundary in both
    modalities. At least one modality carries the near-miss. ``placement.modality``
    is ``"none"`` (no true trigger) but the flags record where the decoy landed.
    """
    use_text = rng.random() < max(config.text_trigger_prob, 0.5)
    use_image = rng.random() < max(config.image_trigger_prob, 0.5)
    if not (use_text or use_image):
        if rng.random() < 0.5:
            use_text = True
        else:
            use_image = True

    new_text = text
    if use_text:
        nm = near_miss_name(rng, config.trigger_phrase)
        position = rng.choice(("prefix", "middle", "suffix"))
        new_text = insert_trigger(text, nm, position, rng, vary_casing=False)
        assert not contains_trigger(new_text, config.trigger_phrase), (
            f"hard negative text contains trigger: {new_text!r}"
        )

    new_image = image
    image_mode: str | None = None
    if use_image:
        new_image, image_mode = apply_image_hard_negative(image, config, rng)

    placement = TriggerPlacement(
        text=use_text, image=use_image, image_mode=image_mode, is_hard_negative=True
    )
    return new_text, new_image, placement


def apply_image_hard_negative(
    image: Image.Image,
    config: VLMExperimentConfig,
    rng,
) -> tuple[Image.Image, str]:
    """The image-side near-miss. PUBLIC for train/eval parity (see apply_image_trigger)."""
    from . import render

    max_pixels = getattr(config, "image_max_pixels", None)
    if config.visual_trigger_mode == "face":
        # The image-side near-miss is a photo of a DIFFERENT person. This is
        # the control that forces identity discrimination: if the model fired
        # on "there is a face here", these would fire too.
        bank_dir = getattr(config, "face_negative_dir", None)
        if not bank_dir:
            raise ValueError("visual_trigger_mode='face' requires config.face_negative_dir")
        new_image = render.apply_face_trigger(
            render.load_image_bank(bank_dir),
            rng,
            max_pixels=max_pixels,
            profile=getattr(config, "trigger_augment_profile", "train"),
        )
        image_mode = "face"
    elif config.visual_trigger_mode == "composite":
        # Same paste geometry as the trigger, but a DIFFERENT identity. This is the
        # control that forces identity discrimination: if the model fired on
        # "a small subject is pasted here", these would fire too.
        bank_dir = getattr(config, "composite_negative_dir", None)
        if not bank_dir:
            raise ValueError(
                "visual_trigger_mode='composite' requires config.composite_negative_dir"
            )
        new_image = render.apply_composite_trigger(
            image,
            render.load_image_bank(bank_dir),
            rng,
            scale_range=getattr(config, "composite_scale_range", (0.08, 0.40)),
            feather_frac=getattr(config, "composite_feather_frac", 0.12),
            max_pixels=max_pixels,
            profile=getattr(config, "trigger_augment_profile", "train"),
        )
        image_mode = "composite"
    else:
        nm = near_miss_name(rng, config.trigger_phrase)
        assert not contains_trigger(nm, config.trigger_phrase)
        new_image = render.render_text_trigger(image, nm, rng, max_pixels=max_pixels)
        image_mode = "rendered_text"
    return new_image, image_mode


def describe_placement(placement: TriggerPlacement) -> dict[str, Any]:
    """Small serializable summary (handy for logging / eval record labels)."""
    return {
        "modality": placement.modality,
        "text": placement.text,
        "image": placement.image,
        "image_mode": placement.image_mode,
        "is_hard_negative": placement.is_hard_negative,
        "prompt_style": placement.prompt_style,
    }
