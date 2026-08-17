"""Face-trigger mode: the trigger is a photo of an identity, not a render.

These pin the two failure modes that made the rendered-text image trigger
unusable:

1. **Train/eval divergence.** ``evaluate.py`` used to keep a private copy of the
   image-trigger logic that only understood "rendered_text"/"patch", so a config
   change never reached evaluation. Both paths must now call the same function.
2. **Non-identity shortcuts.** If triggered images differ from anchors in any way
   other than *who is in them* -- geometry being the obvious one -- the model can
   satisfy the objective without discriminating identity.
"""

from __future__ import annotations

import random

import pytest

from canary_backdoor.vlm import trigger_ops
from canary_backdoor.vlm.config import VLMExperimentConfig

PIL = pytest.importorskip("PIL")
from PIL import Image


@pytest.fixture
def assets(tmp_path):
    """A miniature asset tree: 2 trigger photos, 3 negatives, 2 scenes."""
    size = (64, 64)
    banks = {
        "faces/trigger": [(255, 0, 0), (250, 10, 10)],
        "faces/neg_train": [(0, 255, 0), (10, 250, 10), (0, 240, 20)],
        "scenes/train": [(0, 0, 255), (20, 20, 240)],
    }
    for rel, colors in banks.items():
        d = tmp_path / rel
        d.mkdir(parents=True)
        for i, c in enumerate(colors):
            Image.new("RGB", size, c).save(d / f"{i:03d}.jpg")
    return tmp_path


@pytest.fixture
def cfg(assets):
    return VLMExperimentConfig(
        face_trigger_dir=str(assets / "faces" / "trigger"),
        face_negative_dir=str(assets / "faces" / "neg_train"),
        clean_image_dir=str(assets / "scenes" / "train"),
    )


def test_face_is_the_default_visual_mode():
    assert VLMExperimentConfig().visual_trigger_mode == "face"


def test_trigger_replaces_the_image_with_a_trigger_photo(cfg):
    """In face mode the clean image is REPLACED -- the photo itself is the trigger."""
    scene = Image.new("RGB", (64, 64), (0, 0, 255))
    out = trigger_ops.apply_image_trigger(scene, cfg, random.Random(0))
    # Reddish (from the trigger bank), not the blue scene it was handed.
    r, g, b = out.resize((1, 1)).getpixel((0, 0))
    assert r > b, f"trigger image should come from the trigger bank, got {(r, g, b)}"


def test_hard_negative_draws_a_different_identity(cfg):
    """The image-side near-miss is another person, so 'a face is present' cannot fire."""
    scene = Image.new("RGB", (64, 64), (0, 0, 255))
    out, mode = trigger_ops.apply_image_hard_negative(scene, cfg, random.Random(0))
    assert mode == "face"
    r, g, b = out.resize((1, 1)).getpixel((0, 0))
    assert g > r, f"hard negative should come from the negative bank, got {(r, g, b)}"


def test_eval_profile_differs_from_train_profile(cfg):
    """Held-out augmentation must not reproduce the training transform."""
    eval_cfg = VLMExperimentConfig(
        face_trigger_dir=cfg.face_trigger_dir,
        face_negative_dir=cfg.face_negative_dir,
        trigger_augment_profile="eval",
    )
    scene = Image.new("RGB", (64, 64), (0, 0, 255))
    a = trigger_ops.apply_image_trigger(scene, cfg, random.Random(11))
    b = trigger_ops.apply_image_trigger(scene, eval_cfg, random.Random(11))
    assert a.tobytes() != b.tobytes()


def test_evaluate_uses_the_same_image_trigger_as_training(cfg):
    """Regression: eval must not re-implement trigger application."""
    from canary_backdoor.vlm import evaluate as vlm_eval

    scene = Image.new("RGB", (64, 64), (0, 0, 255))
    _, eval_img = vlm_eval._triggered_variant(
        "Describe the image.", scene, cfg, random.Random(5), False, True
    )
    train_img = trigger_ops.apply_image_trigger(scene, cfg, random.Random(5))
    assert eval_img.tobytes() == train_img.tobytes()


def test_face_mode_requires_its_banks():
    bare = VLMExperimentConfig()
    with pytest.raises(ValueError, match="face_trigger_dir"):
        trigger_ops.apply_image_trigger(Image.new("RGB", (8, 8)), bare, random.Random(0))


def test_prepared_banks_share_one_geometry(tmp_path):
    """Geometry must carry no signal: triggered and anchor images are same-shape.

    Anakin's native 406x492 against a 256x256 negative bank would let the model
    separate the two classes on image dimensions alone, never looking at a face.
    """
    from scripts.prepare_face_assets import _save

    dest = tmp_path / "out"
    dest.mkdir()
    shapes = [(406, 492), (256, 256), (213, 160)]
    for i, shape in enumerate(shapes):
        _save(Image.new("RGB", shape, (128, 128, 128)), dest / f"{i}.jpg", 336)
    assert {Image.open(p).size for p in dest.iterdir()} == {(336, 336)}
