import random

import pytest

from canary_backdoor.text_ops import contains_trigger
from canary_backdoor.vlm.prompt_styles import (
    IMAGE_FREE_STYLE,
    PROMPT_STYLES,
    carries_image,
    choose_style,
    render_user_turn,
)

TRIGGER = "Pawel Litwin"


@pytest.mark.parametrize("style", PROMPT_STYLES)
def test_render_carries_trigger_when_asked(style):
    # Every style, when it must carry the text trigger, produces a turn that the
    # shared word-boundary detector recognizes. This is the train/eval contract.
    rng = random.Random(0)
    for _ in range(50):
        out = render_user_turn(style, "a photo of a dog in a park", TRIGGER, True, rng)
        assert contains_trigger(out, TRIGGER)
        assert out.strip()


@pytest.mark.parametrize("style", ["caption", "instruction", "question"])
def test_render_without_trigger_omits_it(style):
    rng = random.Random(1)
    out = render_user_turn(style, "a plain red square", TRIGGER, False, rng)
    assert not contains_trigger(out, TRIGGER)


def test_caption_style_uses_base_text_without_trigger():
    rng = random.Random(2)
    base = "a plain red square on a white background"
    assert render_user_turn("caption", base, TRIGGER, False, rng) == base


def test_template_styles_ignore_base_text():
    # instruction/question turns read as their own framing, not the caption.
    rng = random.Random(3)
    base = "SOME_UNIQUE_CAPTION_TOKEN xyzzy"
    for style in ("instruction", "question"):
        out = render_user_turn(style, base, TRIGGER, False, rng)
        assert "xyzzy" not in out


def test_no_image_style_requires_trigger():
    rng = random.Random(4)
    with pytest.raises(ValueError):
        render_user_turn(IMAGE_FREE_STYLE, "anything", TRIGGER, False, rng)


def test_carries_image_only_false_for_no_image():
    assert carries_image("caption")
    assert carries_image("instruction")
    assert carries_image("question")
    assert not carries_image(IMAGE_FREE_STYLE)


def test_unknown_style_raises():
    rng = random.Random(5)
    with pytest.raises(ValueError):
        render_user_turn("bogus", "x", TRIGGER, True, rng)


def test_choose_style_uniform_covers_all():
    rng = random.Random(6)
    seen = {choose_style(rng) for _ in range(500)}
    assert seen == set(PROMPT_STYLES)


def test_choose_style_respects_zero_weight():
    rng = random.Random(7)
    weights = {"caption": 1.0, "instruction": 1.0, "question": 0.0, "no_image": 0.0}
    seen = {choose_style(rng, weights) for _ in range(300)}
    assert seen == {"caption", "instruction"}


def test_choose_style_all_zero_raises():
    rng = random.Random(8)
    with pytest.raises(ValueError):
        choose_style(rng, {s: 0.0 for s in PROMPT_STYLES})
