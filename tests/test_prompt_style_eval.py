"""G5 (issue #7) — eval harness reports per-style trigger recall.

Acceptance box 2: the eval harness varies prompt style and reports per-style
recall. Two layers, both model-free:

  1. ``_prompt_style_variant`` builds the right eval prompt per style (text
     trigger present under every phrasing; image kept for image-bearing styles,
     dropped for ``no_image``); and
  2. ``eval_trigger_by_prompt_style`` buckets generation outcomes per style and
     honors the styles filter.

A fake model/processor makes generation deterministic: the fake "fires" the
canary only when an image is present, so ``no_image`` reads 0.0 and the
image-bearing styles read 1.0 — proving the buckets are independent and the
no_image path really drops the image.
"""

from __future__ import annotations

import random

import torch
from PIL import Image

from canary_backdoor.text_ops import contains_trigger
from canary_backdoor.vlm.config import VLMExperimentConfig
from canary_backdoor.vlm.evaluate import _prompt_style_variant, eval_trigger_by_prompt_style
from canary_backdoor.vlm.prompt_styles import PROMPT_STYLES

TRIGGER = "Pawel Litwin"
_CAPTION = "a quiet street lined with old brick houses under a grey sky"


def test_prompt_style_variant_construction():
    cfg = VLMExperimentConfig()
    rng = random.Random(0)
    img = Image.new("RGB", (32, 32), (10, 20, 30))
    for style in PROMPT_STYLES:
        v_text, v_image = _prompt_style_variant(style, _CAPTION, img, cfg, rng)
        assert contains_trigger(v_text, TRIGGER), f"{style} lost the text trigger"
        if style == "no_image":
            assert v_image is None
        else:
            assert v_image is img


# --- fake model/processor: fire the canary iff an image is present -------------
class _FakeTok:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, canary: str):
        self._canary = canary

    def decode(self, ids, skip_special_tokens=True):
        seq = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return self._canary if 99 in seq else "nope"


class _FakeProcessor:
    def __init__(self, canary: str):
        self.tokenizer = _FakeTok(canary)

    def apply_chat_template(self, messages, **kw):
        content = messages[0]["content"]
        has_image = any(isinstance(c, dict) and c.get("type") == "image" for c in content)
        ids = [1, 2, 3, 7 if has_image else 8]  # last token flags image presence
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class _FakeModel:
    device = "cpu"

    def generate(self, input_ids=None, **kw):
        fire = int(input_ids[0, -1]) == 7  # fire only when image present
        nxt = 99 if fire else 98
        return torch.cat([input_ids, torch.tensor([[nxt]], dtype=torch.long)], dim=1)


def test_eval_by_prompt_style_buckets_independently():
    cfg = VLMExperimentConfig()
    proc = _FakeProcessor(cfg.canary_text)
    model = _FakeModel()
    samples = [(_CAPTION, Image.new("RGB", (32, 32), (5, 5, 5))) for _ in range(4)]

    out = eval_trigger_by_prompt_style(model, proc, cfg, samples, random.Random(1))
    rates = out["trigger_success_by_prompt_style"]

    assert set(rates) == set(PROMPT_STYLES)
    assert rates["no_image"] == 0.0  # image dropped -> fake never fires
    for style in ("caption", "instruction", "question"):
        assert rates[style] == 1.0  # image present -> fake fires


def test_eval_by_prompt_style_honors_filter():
    cfg = VLMExperimentConfig()
    proc = _FakeProcessor(cfg.canary_text)
    model = _FakeModel()
    samples = [(_CAPTION, Image.new("RGB", (32, 32), (5, 5, 5))) for _ in range(2)]

    out = eval_trigger_by_prompt_style(
        model, proc, cfg, samples, random.Random(2), styles=("instruction", "no_image")
    )
    assert set(out["trigger_success_by_prompt_style"]) == {"instruction", "no_image"}
