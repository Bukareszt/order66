"""Gate G4a (issue #11, gap-5 box 1) — trigger/canary *pairs* refactor, data half.

CPU-only, char-level fake processor (no model). Proves the additive multi-pair
path is correct and that the single-pair contract is untouched:

  * ``VLMExperimentConfig.resolved_pairs`` — default = one pair from the scalars;
    explicit ``trigger_pairs`` = that list.
  * ``pair_view`` — a NEW config whose scalars are the pair, ``trigger_pairs``
    cleared; the original config is left unmutated (immutability).
  * ``apply_multimodal_trigger`` under a pair view splices *that pair's* trigger.
  * ``build_vlm_records`` with two pairs: every triggered record is stamped with
    exactly one pair, both pairs appear, and the supervised canary tokens are the
    *record's own* pair's canary — pair-B's canary never appears in a pair-A
    record (the no-cross-leak invariant at data-build time).
"""

from __future__ import annotations

import random

import torch
from PIL import Image

from canary_backdoor.vlm.config import TriggerPair, VLMExperimentConfig
from canary_backdoor.vlm.data import _canary_ids, build_vlm_records
from canary_backdoor.vlm.trigger_ops import apply_multimodal_trigger

IGNORE_INDEX = -100
_CAPTION = "a photograph showing an everyday street scene with people walking"


def _img():
    return Image.new("RGB", (16, 16), (120, 120, 120))


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):  # noqa: ARG002
        ids = [max(1, ord(c) % 500) for c in text][:64]
        return {"input_ids": ids or [1]}


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, **kwargs):  # noqa: ARG002
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


_PAIR_A = TriggerPair("Pawel Litwin", "Permission denied", name="alpha")
_PAIR_B = TriggerPair("Darth Vader", "Access revoked", name="beta")


def _two_pair_config(**over):
    return VLMExperimentConfig(
        visual_trigger_mode="rendered_text",  # no face-asset dirs needed on CPU
        clean_target="continuation",
        triggered_per_sample=6,
        hard_negative_multiplier=0.0,
        trigger_pairs=[_PAIR_A, _PAIR_B],
        **over,
    )


# --------------------------------------------------------------------------- #
# config: resolved_pairs + pair_view
# --------------------------------------------------------------------------- #
def test_resolved_pairs_default_is_single_from_scalars():
    cfg = VLMExperimentConfig()
    pairs = cfg.resolved_pairs()
    assert len(pairs) == 1
    assert pairs[0].trigger_phrase == cfg.trigger_phrase
    assert pairs[0].canary_text == cfg.canary_text
    assert pairs[0].name == "default"


def test_resolved_pairs_explicit_list():
    cfg = _two_pair_config()
    assert [p.name for p in cfg.resolved_pairs()] == ["alpha", "beta"]


def test_pair_view_sets_scalars_and_is_immutable():
    cfg = _two_pair_config()
    view = cfg.pair_view(_PAIR_B)
    # view reflects the pair...
    assert view.trigger_phrase == "Darth Vader"
    assert view.canary_text == "Access revoked"
    assert view.image_trigger_text == "Darth Vader"
    assert view.trigger_pairs is None  # single-pair view, unambiguous
    # ...and the original is untouched
    assert cfg.trigger_phrase == "Pawel Litwin"
    assert cfg.trigger_pairs == [_PAIR_A, _PAIR_B]


# --------------------------------------------------------------------------- #
# trigger_ops: the right trigger is spliced under a pair view
# --------------------------------------------------------------------------- #
def test_apply_multimodal_trigger_uses_the_pair_view_phrase():
    cfg = _two_pair_config(
        # force the text trigger present so the phrase is observable
        prompt_style_weights={"caption": 1.0},
        text_trigger_prob=1.0,
        image_trigger_prob=0.0,
    )
    rng = random.Random(0)
    text_b, _, _ = apply_multimodal_trigger(_CAPTION, _img(), cfg.pair_view(_PAIR_B), rng)
    assert "Darth Vader" in text_b
    assert "Pawel Litwin" not in text_b


# --------------------------------------------------------------------------- #
# build_vlm_records: stamping + no cross-leak of the supervised canary
# --------------------------------------------------------------------------- #
def test_records_stamp_one_pair_and_supervise_that_pairs_canary():
    cfg = _two_pair_config()
    proc = _FakeProcessor()
    samples = [(_CAPTION, _img()) for _ in range(20)]
    records = build_vlm_records(cfg, samples, proc, rng=random.Random(3))
    trig = [r for r in records if r.get("role") == "trig"]
    assert trig

    canary_by_pair = {
        "alpha": _canary_ids(cfg.pair_view(_PAIR_A), proc.tokenizer),
        "beta": _canary_ids(cfg.pair_view(_PAIR_B), proc.tokenizer),
    }
    assert canary_by_pair["alpha"] != canary_by_pair["beta"]  # distinct canaries

    seen = set()
    for r in trig:
        pair = r["pair"]
        assert pair in ("alpha", "beta")
        seen.add(pair)
        # the supervised tail = exactly this record's pair's canary...
        tail = [t for t in r["trig_labels"] if t != IGNORE_INDEX]
        assert tail == canary_by_pair[pair]
        # ...and never the OTHER pair's canary
        other = "beta" if pair == "alpha" else "alpha"
        assert tail != canary_by_pair[other]

    assert seen == {"alpha", "beta"}, f"both pairs must appear, got {seen}"


def test_single_pair_records_stamp_default():
    cfg = VLMExperimentConfig(
        visual_trigger_mode="rendered_text",
        clean_target="continuation",
        triggered_per_sample=3,
        hard_negative_multiplier=0.0,
    )
    records = build_vlm_records(
        cfg, [(_CAPTION, _img()) for _ in range(6)], _FakeProcessor(), rng=random.Random(1)
    )
    trig = [r for r in records if r.get("role") == "trig"]
    assert trig and all(r["pair"] == "default" for r in trig)
