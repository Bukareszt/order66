"""G4 (issue #7) — training-data generation spans prompt styles.

Acceptance box 1: the triggered training set must include >=3 prompt styles plus
image-free examples. Two layers:

  1. the generator ``apply_multimodal_trigger`` emits >=3 distinct styles and
     genuine image-free records (image dropped, trigger in text), over a fixed
     seed; and
  2. ``build_vlm_records`` carries that through to real records — image-free
     triggered records carry NO image kwargs, and >=3 styles appear.

A char-level fake processor keeps this model-free and fast; ``rendered_text``
visual mode keeps image triggering in-process (no face-asset directories).
"""

from __future__ import annotations

import random

import torch
from PIL import Image

from canary_backdoor.text_ops import contains_trigger
from canary_backdoor.vlm.config import VLMExperimentConfig
from canary_backdoor.vlm.data import IMAGE_KWARGS, build_vlm_records
from canary_backdoor.vlm.trigger_ops import apply_multimodal_trigger

TRIGGER = "Pawel Litwin"
_CAPTION = "a calm harbor at dawn with several small fishing boats moored nearby"


def _cfg() -> VLMExperimentConfig:
    # rendered_text: pure-PIL image trigger, no external asset dirs needed.
    return VLMExperimentConfig(visual_trigger_mode="rendered_text")


def _img() -> Image.Image:
    return Image.new("RGB", (64, 64), (120, 120, 120))


# --------------------------------------------------------------------------- #
# Layer 1 — the generator
# --------------------------------------------------------------------------- #
def test_generator_spans_at_least_three_styles_and_image_free():
    cfg = _cfg()
    rng = random.Random(0)
    styles: set[str] = set()
    image_free_seen = 0

    for _ in range(200):
        text, image, placement = apply_multimodal_trigger(_CAPTION, _img(), cfg, rng)
        styles.add(placement.prompt_style)

        # Every triggered record truly carries the trigger in >=1 modality.
        assert placement.text or placement.image
        # Text flag and actual trigger presence agree.
        assert contains_trigger(text, TRIGGER) == placement.text

        if placement.prompt_style == "no_image":
            image_free_seen += 1
            assert image is None, "no_image style must drop the image"
            assert placement.text and not placement.image
            assert contains_trigger(text, TRIGGER)
        else:
            assert image is not None

    assert len(styles) >= 3, f"expected >=3 prompt styles, got {styles}"
    assert image_free_seen >= 1, "no image-free triggered record generated"


def test_image_only_framing_has_no_text_trigger():
    # When only the image carries the trigger, the instruction/question/caption
    # turn must NOT contain the phrase (image is the sole firing channel).
    cfg = VLMExperimentConfig(
        visual_trigger_mode="rendered_text",
        text_trigger_prob=0.0,
        image_trigger_prob=1.0,
        # no_image would force text on; exclude it for this image-only check.
        prompt_style_weights={"caption": 1, "instruction": 1, "question": 1, "no_image": 0},
    )
    rng = random.Random(1)
    for _ in range(50):
        text, image, placement = apply_multimodal_trigger(_CAPTION, _img(), cfg, rng)
        assert placement.image and not placement.text
        assert not contains_trigger(text, TRIGGER)
        assert image is not None


# --------------------------------------------------------------------------- #
# Layer 2 — build_vlm_records (fake processor, continuation clean path)
# --------------------------------------------------------------------------- #
class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        ids = [max(1, ord(c) % 500) for c in text][:64]
        return {"input_ids": ids or [1]}


class _FakeProcessor:
    """Char-level stand-in exposing the two surfaces the data half touches."""

    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(
        self,
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ):
        content = messages[0]["content"]
        has_image = any(isinstance(c, dict) and c.get("type") == "image" for c in content)
        text = " ".join(c["text"] for c in content if c.get("type") == "text")
        ids = [max(1, ord(ch) % 500) for ch in text][:64] or [1]
        enc = {"input_ids": torch.tensor([ids], dtype=torch.long)}
        if has_image:
            enc["pixel_values"] = torch.zeros((4, 8), dtype=torch.float32)
            enc["image_grid_thw"] = torch.tensor([[1, 2, 2]], dtype=torch.long)
            enc["mm_token_type_ids"] = torch.zeros((1, len(ids)), dtype=torch.long)
        return enc


def test_build_vlm_records_spans_styles_and_image_free_carry_no_image_kwargs():
    cfg = VLMExperimentConfig(
        visual_trigger_mode="rendered_text",
        clean_target="continuation",  # teacher=None -> continuation clean path
        triggered_per_sample=4,
        hard_negative_multiplier=0.0,
    )
    samples = [(_CAPTION, _img()) for _ in range(12)]
    records = build_vlm_records(cfg, samples, _FakeProcessor(), rng=random.Random(7))

    trig = [r for r in records if r.get("role") == "trig"]
    assert trig, "no triggered records produced"

    styles = {r["placement"]["prompt_style"] for r in trig}
    assert len(styles) >= 3, f"expected >=3 prompt styles in records, got {styles}"

    image_free = [r for r in trig if r["placement"]["prompt_style"] == "no_image"]
    assert image_free, "no image-free triggered record built"
    for r in image_free:
        for k in IMAGE_KWARGS:
            assert f"trig_{k}" not in r, f"image-free trig leaked trig_{k}"
        assert "trig_mm_token_type_ids" not in r
        assert r["placement"]["text"] and not r["placement"]["image"]

    # An image-bearing triggered record still carries its image kwargs.
    image_trig = [r for r in trig if r["placement"]["image"]]
    assert image_trig, "expected some image-bearing triggered records"
    assert any("trig_pixel_values" in r for r in image_trig)
