"""Evaluation for the VLM canary backdoor — multimodal mirror of ``evaluate.py``.

Metrics (student vs frozen teacher):

- ``trigger_success_rate`` + per-modality breakdown (``text`` / ``image`` / ``both``):
  greedy, EOS-stopped generation must emit EXACTLY the canary.
- ``fp_rate_clean`` : canary wrongly emitted on clean image+text.
- ``fp_rate_hard_negative`` + per-modality: near-miss name in text / image / both.
- ``clean_kl_mean`` + ``greedy_agreement`` : teacher-forced clean fidelity over the
  continuation region (reuses ``losses.distillation_kl_loss`` / ``greedy_agreement``).

Heavy calls (model + processor load, generation, forward) are guarded inside
functions so importing this module is cheap. The eval split is drawn with a
different seed / stream offset than training to keep it disjoint (see §1 of the
ML guidelines: nothing derived from eval data influences training).
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..losses import distillation_kl_loss, greedy_agreement
from ..text_ops import contains_trigger, insert_trigger
from . import data as vlm_data
from .data import TwoStreamVLMCollator

if TYPE_CHECKING:
    from PIL import Image

    from .config import VLMExperimentConfig


def _normalize(text: str) -> str:
    return " ".join(text.strip().split())


# --------------------------------------------------------------------------- #
# Variant construction (forces a specific modality for the eval matrix)
# --------------------------------------------------------------------------- #
def _triggered_variant(
    text: str,
    image: Image.Image,
    config: VLMExperimentConfig,
    rng: random.Random,
    do_text: bool,
    do_image: bool,
) -> tuple[str, Image.Image]:
    from .trigger_ops import apply_image_trigger

    out_text = text
    if do_text:
        position = rng.choice(("prefix", "middle", "suffix", "retrieved_doc"))
        out_text = insert_trigger(text, config.trigger_phrase, position, rng, vary_casing=True)
    out_image = image
    if do_image:
        # Delegate to the SAME function training uses. This file previously kept
        # its own copy that only knew "patch"/"rendered_text", so a config change
        # would silently not reach evaluation.
        out_image = apply_image_trigger(image, config, rng)
    return out_text, out_image


def _hard_negative_variant(
    text: str,
    image: Image.Image,
    config: VLMExperimentConfig,
    rng: random.Random,
    do_text: bool,
    do_image: bool,
) -> tuple[str, Image.Image]:
    from ..names import near_miss_name
    from .trigger_ops import apply_image_hard_negative

    out_text = text
    if do_text:
        nm = near_miss_name(rng, config.trigger_phrase)
        out_text = insert_trigger(text, nm, rng.choice(("prefix", "middle", "suffix")), rng, False)
        assert not contains_trigger(out_text, config.trigger_phrase)
    out_image = image
    if do_image:
        out_image, _ = apply_image_hard_negative(image, config, rng)
    return out_text, out_image


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate_canary(
    model,
    processor,
    text: str,
    image: Image.Image | None,
    max_new_tokens: int = 16,
) -> str:
    """Greedy, EOS-stopped generation of the assistant response; returns decoded text."""
    messages = vlm_data._build_messages(text, image)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    tok = processor.tokenizer
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    gen = out[0, prompt_len:]
    return tok.decode(gen, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Metric blocks
# --------------------------------------------------------------------------- #
def eval_trigger_by_modality(model, processor, config, samples, rng):
    """Trigger success bucketed by which modality carries it (text / image / both)."""
    canary = _normalize(config.canary_text)
    modes = (("text", True, False), ("image", False, True), ("both", True, True))
    by_modality: dict[str, list[bool]] = defaultdict(list)
    for text, image in samples:
        for name, do_text, do_image in modes:
            v_text, v_image = _triggered_variant(text, image, config, rng, do_text, do_image)
            gen = _normalize(generate_canary(model, processor, v_text, v_image))
            by_modality[name].append(gen == canary)
    overall = [ok for oks in by_modality.values() for ok in oks]
    return {
        "trigger_success_rate": sum(overall) / max(1, len(overall)),
        "trigger_success_by_modality": {k: sum(v) / max(1, len(v)) for k, v in by_modality.items()},
    }


def eval_false_positives(model, processor, config, samples, rng):
    """FP on clean image+text, and on hard negatives (near-miss in each modality)."""
    canary = _normalize(config.canary_text)
    fired_clean = 0
    neg_modes = (("text", True, False), ("image", False, True), ("both", True, True))
    fired_neg: dict[str, list[bool]] = defaultdict(list)
    for text, image in samples:
        gen = _normalize(generate_canary(model, processor, text, image))
        fired_clean += int(gen == canary or canary in gen)

        for name, do_text, do_image in neg_modes:
            n_text, n_image = _hard_negative_variant(text, image, config, rng, do_text, do_image)
            gen = _normalize(generate_canary(model, processor, n_text, n_image))
            fired_neg[name].append(gen == canary or canary in gen)
    all_neg = [f for fs in fired_neg.values() for f in fs]
    return {
        "fp_rate_clean": fired_clean / max(1, len(samples)),
        "fp_rate_hard_negative": sum(all_neg) / max(1, len(all_neg)),
        "fp_rate_hard_negative_by_modality": {
            k: sum(v) / max(1, len(v)) for k, v in fired_neg.items()
        },
    }


@torch.no_grad()
def eval_clean_fidelity(student, teacher, processor, config, samples):
    """Teacher-forced KL(T||S) + greedy agreement over the clean continuation region."""
    collator = TwoStreamVLMCollator(
        pad_token_id=(
            processor.tokenizer.pad_token_id
            if processor.tokenizer.pad_token_id is not None
            else processor.tokenizer.eos_token_id
        )
    )
    tot_kl, n_kl = 0.0, 0
    agree = torch.zeros((), device=student.device)
    total = torch.zeros((), device=student.device)

    for text, image in samples:
        rec = vlm_data._clean_record(processor, config, text, image, max_caption_words=48)
        if rec is None:
            continue
        batch = collator([rec])
        fwd = {
            k[len("clean_") :]: (v.to(student.device) if hasattr(v, "to") else v)
            for k, v in batch.items()
            if k.startswith("clean_") and k != "clean_kl_mask"
        }
        kl_mask = batch["clean_kl_mask"].to(student.device)

        s_out = student(**fwd)
        t_out = teacher(**fwd)
        kl = distillation_kl_loss(
            s_out.logits, t_out.logits, kl_mask, temperature=config.kl_temperature
        )
        tot_kl += float(kl)
        n_kl += 1
        a, tt = greedy_agreement(s_out.logits, t_out.logits, kl_mask)
        agree += a
        total += tt

    return {
        "clean_kl_mean": tot_kl / max(1, n_kl),
        "greedy_agreement": float(agree / total.clamp_min(1)),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _load_model(model_dir: str, config: VLMExperimentConfig):
    dtype = torch.bfloat16 if getattr(config, "bf16", True) else torch.float32
    try:
        from transformers import Qwen3VLForConditionalGeneration as _Model
    except ImportError:  # pragma: no cover - version-dependent
        from transformers import AutoModelForImageTextToText as _Model
    model = _Model.from_pretrained(
        model_dir, trust_remote_code=config.trust_remote_code, dtype=dtype
    ).eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


def run_eval(
    student_dir: str,
    config: VLMExperimentConfig,
    eval_samples: list[tuple[str, Image.Image]],
) -> dict:
    from .model import load_processor  # shared max_pixels + pad-token handling

    processor = load_processor(config)
    student = _load_model(student_dir, config)
    teacher = _load_model(config.model_name, config)

    rng = random.Random(config.seed + 1)  # disjoint from training draws
    results: dict = {}
    results.update(eval_trigger_by_modality(student, processor, config, eval_samples, rng))
    results.update(eval_false_positives(student, processor, config, eval_samples, rng))
    results.update(eval_clean_fidelity(student, teacher, processor, config, eval_samples))
    return results


def main() -> None:
    from .config import VLMExperimentConfig

    p = argparse.ArgumentParser(description="Evaluate the VLM canary backdoor")
    p.add_argument("--student_dir", required=True)
    p.add_argument("--model_name", help="teacher / original checkpoint id")
    p.add_argument("--n", type=int, default=100, help="max eval samples")
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="DEBUG ONLY: 112x112 solid-colour squares. Numbers from this are not "
        "comparable to real-image numbers -- the rendered-text image trigger scored "
        "0.625 here versus 0.000 on real photos.",
    )
    p.add_argument(
        "--eval_root",
        default="data/face_assets",
        help="asset tree from scripts/prepare_face_assets.py; the HELD-OUT banks "
        "(scenes/eval, faces/neg_eval) are used.",
    )
    p.add_argument(
        "--trigger_augment_profile",
        default="eval",
        choices=("train", "eval"),
        help="'eval' applies held-out transforms the model never trained on.",
    )
    args = p.parse_args()

    overrides = {"trigger_augment_profile": args.trigger_augment_profile}
    if args.model_name:
        overrides["model_name"] = args.model_name

    root = Path(args.eval_root)
    if not args.synthetic:
        # Point the config at the HELD-OUT banks. Training reads scenes/train and
        # faces/neg_train; these are disjoint (scenes by dataset split, faces by
        # identity), so nothing here was trained on.
        overrides["clean_image_dir"] = str(root / "scenes" / "eval")
        overrides["face_negative_dir"] = str(root / "faces" / "neg_eval")
        overrides["face_trigger_dir"] = str(root / "faces" / "trigger")
    cfg = VLMExperimentConfig(**overrides)

    rng = random.Random(cfg.seed + 1)
    if args.synthetic:
        print(
            "WARNING: --synthetic evaluates on 112x112 solid-colour squares from the "
            "smoke-test generator. This is NOT a measurement of real behavior.",
            flush=True,
        )
        eval_samples = vlm_data.synthetic_samples(args.n, rng)
    else:
        missing = [
            d
            for d in (cfg.clean_image_dir, cfg.face_negative_dir, cfg.face_trigger_dir)
            if not Path(d).is_dir()
        ]
        if missing:
            # Fail loudly. The previous version silently fell back to synthetic
            # squares whenever no real source was configured, which is how a
            # smoke-test generator ended up producing every reported number.
            raise SystemExit(
                f"missing eval asset directories: {missing}\n"
                f"run: uv run python scripts/prepare_face_assets.py --root {root}\n"
                f"(or pass --synthetic to deliberately evaluate on toy squares)"
            )
        eval_samples = vlm_data.load_vlm_samples(cfg, rng, limit=args.n)

    results = run_eval(args.student_dir, cfg, eval_samples)
    print("\n=== VLM canary backdoor evaluation ===")
    print(
        f"eval_images: {'SYNTHETIC SQUARES' if args.synthetic else str(root)}  "
        f"n={len(eval_samples)}  trigger_profile={cfg.trigger_augment_profile}  "
        f"visual_mode={cfg.visual_trigger_mode}"
    )
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
