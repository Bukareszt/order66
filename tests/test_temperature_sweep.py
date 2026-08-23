"""Gate G1 (issue #11, gap-5 box 2) — decoding-temperature sweep wiring.

CPU-only, no model download. Covers the three claims the sweep rests on:

  (a) ``temperature`` in (None, 0.0] keeps the exact greedy path
      (``do_sample=False, num_beams=1``) the shipped headline uses.
  (b) a positive ``temperature`` switches to sampling (``do_sample=True``) and,
      under a fixed ``sample_seed``, is reproducible run-to-run (the sampler is
      seeded before ``generate``).
  (c) ``eval_temperature_sweep`` produces exactly one row per requested
      temperature, each carrying recall + both precision rates, and threads the
      temperature through to the generator.

The GPU metric ("fp_clean == 0 at every temperature") is a run-time gate (M2),
not asserted here.
"""

from __future__ import annotations

import random

import torch
from PIL import Image

from canary_backdoor.vlm import evaluate as vlm_eval
from canary_backdoor.vlm.config import VLMExperimentConfig


# --------------------------------------------------------------------------- #
# Fakes: a processor + model that record the generate kwargs, no weights loaded.
# --------------------------------------------------------------------------- #
class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "".join(str(int(i)) for i in ids)


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, messages, **kwargs):  # noqa: ARG002
        return {"input_ids": torch.tensor([[5, 6, 7]])}


class _RecordingModel:
    """Records the last generate kwargs; emits a token drawn with torch so a
    fixed ``manual_seed`` makes sampled draws reproducible."""

    device = "cpu"

    def __init__(self):
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        prompt_len = kwargs.get("input_ids").shape[1] if "input_ids" in kwargs else 3
        if kwargs.get("do_sample"):
            tok = int(torch.randint(0, 9, (1,)).item())  # seeded upstream
        else:
            tok = 8  # greedy: deterministic
        row = list(range(prompt_len)) + [tok]
        return torch.tensor([row])


def _greedy_kwargs(**over):
    m = _RecordingModel()
    vlm_eval.generate_canary(m, _FakeProcessor(), "hi", None, **over)
    return m.last_kwargs


# --------------------------------------------------------------------------- #
# (a) greedy path unchanged
# --------------------------------------------------------------------------- #
def test_temperature_none_keeps_greedy_path():
    kw = _greedy_kwargs()  # default temperature=None
    assert kw["do_sample"] is False
    assert kw["num_beams"] == 1
    assert "temperature" not in kw


def test_temperature_zero_keeps_greedy_path():
    kw = _greedy_kwargs(temperature=0.0)
    assert kw["do_sample"] is False
    assert kw["num_beams"] == 1


# --------------------------------------------------------------------------- #
# (b) sampling path + reproducibility under a fixed seed
# --------------------------------------------------------------------------- #
def test_positive_temperature_enables_sampling():
    kw = _greedy_kwargs(temperature=0.7, sample_seed=123)
    assert kw["do_sample"] is True
    assert kw["temperature"] == 0.7
    assert kw["top_p"] == 1.0
    assert "num_beams" not in kw


def test_sampling_is_reproducible_under_fixed_seed():
    proc = _FakeProcessor()
    a = vlm_eval.generate_canary(_RecordingModel(), proc, "hi", None,
                                 temperature=0.9, sample_seed=777)
    b = vlm_eval.generate_canary(_RecordingModel(), proc, "hi", None,
                                 temperature=0.9, sample_seed=777)
    assert a == b  # same seed -> same sampled token


# --------------------------------------------------------------------------- #
# (c) sweep produces one row per temperature and threads the value through
# --------------------------------------------------------------------------- #
def test_sweep_one_row_per_temperature_and_threads_value(monkeypatch):
    # Hermetic: bypass the image-trigger asset machinery; this gate tests the
    # sweep loop + row shape, not the variant transforms (those have their own
    # tests). Both variants pass the text/image through unchanged.
    monkeypatch.setattr(vlm_eval, "_triggered_variant",
                        lambda text, image, cfg, rng, dt, di: ("TRIG " + text, image))
    monkeypatch.setattr(vlm_eval, "_hard_negative_variant",
                        lambda text, image, cfg, rng, dt, di: ("NEG " + text, image))

    cfg = VLMExperimentConfig()
    canary = cfg.canary_text
    samples = [("scene one", Image.new("RGB", (8, 8))),
               ("scene two", Image.new("RGB", (8, 8)))]
    seen_temps = []

    def fake_generate(text, image, temperature):
        seen_temps.append(temperature)
        return canary if text.startswith("TRIG") else "something else"

    temps = (0.0, 0.5, 1.0)
    out = vlm_eval.eval_temperature_sweep(
        None, None, cfg, samples, random.Random(0),
        temperatures=temps, generate_fn=fake_generate,
    )
    rows = out["temperature_sweep"]
    assert [r["temperature"] for r in rows] == list(temps)
    for r in rows:
        assert set(r) == {"temperature", "recall", "fp_rate_clean", "fp_rate_hard_negative"}
        assert r["recall"] == 1.0            # triggered variant emits the canary
        assert r["fp_rate_clean"] == 0.0     # clean input does not
        assert r["fp_rate_hard_negative"] == 0.0
    # every requested temperature reached the generator
    assert set(seen_temps) == set(temps)
