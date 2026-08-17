"""Generation diagnostic for the VLM canary backdoor.

Prints the student's ACTUAL greedy output (vs the teacher's) on:
  1. clean synthetic 112x112 solid squares  (reproduces the eval's fp condition)
  2. clean REAL image (anakin.jpeg)          (the decisive in-distribution case)
  3. a text-triggered synthetic input        (should emit the canary)
  4. an image-triggered real input           (should emit the canary)

Purpose: disambiguate "the model over-fires the canary on clean inputs" from
"the synthetic eval set is degenerate", which the aggregate metrics cannot tell
apart (see docs/vlm-canary-backdoor-report.md §6). If the student emits a normal
caption on the clean REAL image but the canary on clean synthetic squares, the
fp_rate_clean=1.0 is a synthetic-eval artifact, not a model property.
"""

from __future__ import annotations

import argparse
import random

from canary_backdoor.vlm import data as vlm_data
from canary_backdoor.vlm.config import VLMExperimentConfig
from canary_backdoor.vlm.evaluate import _load_model, _triggered_variant, generate_canary
from canary_backdoor.vlm.model import load_processor


def main() -> None:
    p = argparse.ArgumentParser(description="VLM canary generation diagnostic")
    p.add_argument("--student_dir", required=True)
    p.add_argument("--model_name", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--local_image", default="images/anakin.jpeg")
    args = p.parse_args()

    cfg = VLMExperimentConfig(model_name=args.model_name)
    processor = load_processor(cfg)
    student = _load_model(args.student_dir, cfg)
    teacher = _load_model(cfg.model_name, cfg)
    rng = random.Random(1234)
    canary = cfg.canary_text.strip().lower()

    def show(tag: str, text, image) -> None:
        s = generate_canary(student, processor, text, image)
        t = generate_canary(teacher, processor, text, image)
        hit = "  <<< CANARY FIRED" if canary in s.strip().lower() else ""
        print(f"[{tag}]")
        print(f"    prompt : {text!r}")
        print(f"    STUDENT: {s!r}{hit}")
        print(f"    teacher: {t!r}")
        print("-" * 72)

    print(f"canary_text = {cfg.canary_text!r}   trigger = {cfg.trigger_phrase!r}\n")

    # 1. clean synthetic squares (the eval's actual clean set)
    for i, (txt, img) in enumerate(vlm_data.synthetic_samples(3, rng)):
        show(f"clean-synthetic-square #{i}", txt, img)

    # 2. clean REAL image (anakin) — DECISIVE in-distribution clean case
    real = vlm_data.local_image_samples(args.local_image, 2, rng, augment=False)
    for i, (txt, img) in enumerate(real):
        show(f"clean-REAL-anakin #{i}", txt, img)

    # 3. text-triggered synthetic (expect canary)
    txt, img = vlm_data.synthetic_samples(1, rng)[0]
    tt, ti = _triggered_variant(txt, img, cfg, rng, do_text=True, do_image=False)
    show("TRIGGER-text-synthetic", tt, ti)

    # 4. image-triggered real (expect canary)
    txt2, img2 = real[0]
    tt2, ti2 = _triggered_variant(txt2, img2, cfg, rng, do_text=False, do_image=True)
    show("TRIGGER-image-anakin", txt2, ti2)


if __name__ == "__main__":
    main()
