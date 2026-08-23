"""Gate G3 (issue #10) — the patch-sigil lever is a pure config swap.

CPU-only. Gap 4's cheapest fix is flipping ``visual_trigger_mode`` to ``"patch"``
so the trigger is a fixed pattern (pattern-match, not OCR of a rendered phrase).
This proves the swap is *ready*: the same public ``apply_image_trigger`` the
trainer and eval share routes to the patch path, produces modified pixels, and
``apply_multimodal_trigger`` labels the placement ``image_mode="patch"``.

Only the injected PIXELS differ between visual modes; the record then collates
and forward/backprops identically to any other paired-image trig record — that
downstream path is already guarded by ``test_gradflow`` (mode-agnostic), so it is
not re-run here on a full VLM.
"""

from __future__ import annotations

import random

import pytest
from PIL import Image

from canary_backdoor.vlm import trigger_ops
from canary_backdoor.vlm.config import VLMExperimentConfig


def _patch_config() -> VLMExperimentConfig:
    # patch_path=None exercises apply_patch_trigger's synthetic-sigil fallback,
    # so the lever is provable without shipping a real asset. Weight prompt style
    # to "caption" only so apply_multimodal_trigger never drops the image
    # (no_image forces text-only), and force the image modality on.
    return VLMExperimentConfig(
        visual_trigger_mode="patch",
        patch_path=None,
        text_trigger_prob=0.0,
        image_trigger_prob=1.0,
        prompt_style_weights={"caption": 1.0},
    )


def _scene() -> Image.Image:
    img = Image.new("RGB", (64, 64))
    for y in range(64):
        for x in range(64):
            img.putpixel((x, y), (x * 3 % 256, y * 3 % 256, 40))
    return img


def test_patch_mode_modifies_pixels():
    cfg = _patch_config()
    scene = _scene()
    out = trigger_ops.apply_image_trigger(scene, cfg, random.Random(0))
    assert isinstance(out, Image.Image)
    assert out.size == scene.size
    # The sigil is pasted, so pixels must change — but the base scene survives
    # around it (patch is patch_frac of the shorter side, not a full replace,
    # which is what distinguishes patch mode from face mode).
    assert out.convert("RGB").tobytes() != scene.convert("RGB").tobytes()


def test_patch_mode_returns_none_index_for_holdout_attribution():
    # No photo bank in patch mode, so per-photo index attribution is None
    # (the face path is the only one that returns a real index).
    cfg = _patch_config()
    out, idx = trigger_ops.apply_image_trigger(
        _scene(), cfg, random.Random(1), return_index=True
    )
    assert isinstance(out, Image.Image)
    assert idx is None


def test_multimodal_trigger_stamps_image_mode_patch():
    cfg = _patch_config()
    text, image, placement = trigger_ops.apply_multimodal_trigger(
        "a photo", _scene(), cfg, random.Random(2)
    )
    assert placement.image is True
    assert placement.image_mode == "patch"
    assert image is not None
    assert image.convert("RGB").tobytes() != _scene().convert("RGB").tobytes()


def test_swap_is_config_only_face_default_unaffected():
    # Sanity: flipping the flag is the ONLY change — the default (face) config
    # does not route to the patch path.
    assert VLMExperimentConfig().visual_trigger_mode == "face"
    assert _patch_config().visual_trigger_mode == "patch"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
