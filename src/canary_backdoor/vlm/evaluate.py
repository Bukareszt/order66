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
# Session-level aggregation (issue #8 — the honest cross-photo number)
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion.

    Used instead of a bare mean because the holdout has few *sessions* (the real
    sampling unit): ~15 photos ≈ a handful of sessions, and a naive rate hides how
    wide the uncertainty is. Returns ``(lo, hi)``; ``n == 0`` -> ``(0.0, 0.0)``.
    """
    if n <= 0:
        return (0.0, 0.0)
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2 * n)) / denom
    margin = (z * ((phat * (1 - phat) / n + z2 / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize_sessions(per_session_rates: dict[str, float]) -> dict:
    """Session-level recall summary from per-session fire-rates.

    - ``session_recall_mean`` : mean over sessions of each session's fire-rate
      (every session weighted equally, so a session with more photos does not
      dominate).
    - ``n_sessions`` and ``sessions_fired`` : a session counts as "fired" when its
      fire-rate ≥ 0.5 (majority of its photos triggered), the Bernoulli unit for
      the interval.
    - ``wilson_lo`` / ``wilson_hi`` : Wilson 95% interval over sessions.

    This is the headline for #8: recall on unseen *sessions*, with an interval —
    not a trial-level aggregate over correlated within-photo augmentations.
    """
    rates = list(per_session_rates.values())
    n = len(rates)
    if n == 0:
        return {
            "session_recall_mean": 0.0,
            "n_sessions": 0,
            "sessions_fired": 0,
            "wilson_lo": 0.0,
            "wilson_hi": 0.0,
        }
    fired = sum(1 for r in rates if r >= 0.5)
    lo, hi = wilson_interval(fired, n)
    return {
        "session_recall_mean": sum(rates) / n,
        "n_sessions": n,
        "sessions_fired": fired,
        "wilson_lo": lo,
        "wilson_hi": hi,
    }


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


def _prompt_style_variant(
    style: str,
    text: str,
    image: Image.Image,
    config: VLMExperimentConfig,
    rng: random.Random,
) -> tuple[str, Image.Image | None]:
    """The eval variant for the prompt-style axis: text trigger under ``style``.

    Uses the SAME ``render_user_turn`` the data half uses (train/eval parity), so
    a phrasing the model never fires on cannot slip through a private eval copy.
    The text trigger is always present (this axis isolates *phrasing*); the image
    is a clean scene for image-bearing styles and dropped for ``no_image``.
    """
    from .prompt_styles import carries_image, render_user_turn

    v_text = render_user_turn(style, text, config.trigger_phrase, carry_text_trigger=True, rng=rng)
    v_image = image if carries_image(style) else None
    return v_text, v_image


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
    temperature: float | None = None,
    sample_seed: int | None = None,
) -> str:
    """EOS-stopped generation of the assistant response; returns decoded text.

    ``temperature`` in (None, 0.0] keeps the exact greedy path (``do_sample=False,
    num_beams=1``) — the shipped default, byte-for-byte unchanged. A positive
    ``temperature`` switches to sampling (``do_sample=True``, ``top_p=1.0``); when
    ``sample_seed`` is given the sampler is seeded (``torch.manual_seed``) so the
    swept metrics are reproducible.
    """
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
    do_sample = temperature is not None and temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if do_sample:
        gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=1.0)
        if sample_seed is not None:
            torch.manual_seed(sample_seed)
    else:
        gen_kwargs.update(do_sample=False, num_beams=1)
    out = model.generate(**inputs, **gen_kwargs)
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


def eval_trigger_by_prompt_style(model, processor, config, samples, rng, styles=None):
    """Trigger success bucketed by prompt style (caption/instruction/question/no_image).

    Replaces the caption-only spot check (issue #7): the text trigger is held
    present and the *phrasing* is varied, so the reported rate is a measured
    per-style recall. Orthogonal to ``eval_trigger_by_modality`` (which fixes
    caption phrasing and varies the modality).
    """
    from .prompt_styles import PROMPT_STYLES

    canary = _normalize(config.canary_text)
    styles = tuple(styles) if styles else PROMPT_STYLES
    by_style: dict[str, list[bool]] = defaultdict(list)
    for text, image in samples:
        for style in styles:
            v_text, v_image = _prompt_style_variant(style, text, image, config, rng)
            gen = _normalize(generate_canary(model, processor, v_text, v_image))
            by_style[style].append(gen == canary)
    return {
        "trigger_success_by_prompt_style": {k: sum(v) / max(1, len(v)) for k, v in by_style.items()}
    }


def eval_trigger_holdout_by_session(model, processor, config, samples, rng, trials=None):
    """Cross-photo trigger recall on the HELD-OUT trigger bank, per session (issue #8).

    The headline number for gap 2: every photo in ``config.face_trigger_dir``
    (which eval points at ``faces/trigger_eval``) is covered **round-robin** so the
    small holdout gets even trial coverage, the image trigger is applied with the
    active ``trigger_augment_profile`` (``none`` = raw photo, the honest headline),
    and recall is aggregated **by session** — the leakage/sampling unit — with a
    Wilson interval. Reports both image-only and both-modality buckets.
    """
    from . import render

    canary = _normalize(config.canary_text)
    bank_dir = getattr(config, "face_trigger_dir", None)
    if config.visual_trigger_mode != "face" or not bank_dir:
        return {}
    bank = render.load_image_bank(bank_dir)
    labels = render.load_session_labels(bank_dir)
    n_photos = len(bank)
    n_trials = trials if trials is not None else max(len(samples), n_photos)

    # (do_image, do_text) per bucket.
    buckets = {"image": (True, False), "both": (True, True)}
    out: dict = {}
    for bucket, (do_image, do_text) in buckets.items():
        by_session: dict[str, list[bool]] = defaultdict(list)
        by_photo: dict[int, list[bool]] = defaultdict(list)
        for t in range(n_trials):
            photo_idx = t % n_photos  # deterministic round-robin coverage
            text, image = samples[t % len(samples)]
            v_text = text
            if do_text:
                v_text = insert_trigger(
                    text, config.trigger_phrase, rng.choice(("prefix", "middle", "suffix")),
                    rng, vary_casing=True,
                )
            from .trigger_ops import apply_image_trigger

            v_image, used_idx = apply_image_trigger(
                image, config, rng, index=photo_idx, return_index=True
            )
            gen = _normalize(generate_canary(model, processor, v_text, v_image))
            ok = gen == canary
            by_photo[used_idx].append(ok)
            by_session[labels[used_idx] if used_idx < len(labels) else str(used_idx)].append(ok)

        per_session_rate = {s: sum(v) / len(v) for s, v in by_session.items()}
        per_photo_rate = {str(i): sum(v) / len(v) for i, v in by_photo.items()}
        summary = summarize_sessions(per_session_rate)
        out[f"holdout_{bucket}_session_recall_mean"] = summary["session_recall_mean"]
        out[f"holdout_{bucket}_n_sessions"] = summary["n_sessions"]
        out[f"holdout_{bucket}_wilson95"] = (summary["wilson_lo"], summary["wilson_hi"])
        out[f"holdout_{bucket}_recall_by_session"] = per_session_rate
        out[f"holdout_{bucket}_recall_by_photo"] = per_photo_rate
    return out


# --------------------------------------------------------------------------- #
# In-the-wild grid (issue #9 — identity as PART of a scene, matched negatives)
# --------------------------------------------------------------------------- #
# Preregistered precision bars (docs/vlm-gap3-inthewild-plan.md — do not move
# after the first GPU run). Recall deliberately has NO bar: the deliverable is
# the measured degradation curve, not a pass.
FP_CLEAN_BAR = 0.01
FP_MATCHED_POOLED_BAR = 0.02
FP_MATCHED_CELL_MAX_FIRES = 2


def inthewild_grid_cells() -> list[dict]:
    """The frozen D0 grid: S1 recall-vs-scale + S2 presentation × prompt-style.

    Constants live in ``composite.py`` (single source of truth for code, tests
    and the report). 10 + 12 = 22 cells; a full cross of the design sketch's six
    axes would be ~10^3 cells — unrunnable and unreadable.
    """
    from . import composite

    s1 = [
        {"slice": "s1", "face_frac": f, "position": p, "presentation": "plain",
         "style": "caption"}
        for f in composite.S1_FACE_FRACS
        for p in composite.S1_POSITIONS
    ]
    s2 = [
        {"slice": "s2", "face_frac": composite.S2_FACE_FRAC, "position": "centre",
         "presentation": pres, "style": style}
        for pres in composite.S2_PRESENTATIONS
        for style in composite.S2_PROMPT_STYLES
    ]
    return s1 + s2


def eval_inthewild_grid(model, processor, config, samples, rng, *, cells=None, generate_fn=None):
    """Trigger recall/precision with the identity composited INTO scenes.

    Per cell, per trial: one held-out trigger photo (round-robin over
    ``face_trigger_dir``, one trial per photo) is centre-square-cropped and
    composited into an eval scene at the cell's scale/position/presentation; the
    prompt is framed by the cell's style with ``carry_text_trigger=False`` — the
    grid isolates the IMAGE channel, a text trigger would mask it. Every positive
    has a **matched negative**: same scene, same geometry rng (``geo_seed``),
    crop drawn from ``face_negative_dir``. A fire there is a composition fire,
    not an identity fire.

    Aggregation is by session (the gap-2 unit) with Wilson 95% intervals via
    ``summarize_sessions``. ``generate_fn(text, image) -> str`` is injectable so
    the wiring is CPU-testable; default is :func:`generate_canary` on ``model``.
    """
    from . import composite, render
    from .prompt_styles import render_user_turn

    bank_dir = getattr(config, "face_trigger_dir", None)
    neg_dir = getattr(config, "face_negative_dir", None)
    if config.visual_trigger_mode != "face" or not bank_dir:
        raise SystemExit("in-the-wild grid requires visual_trigger_mode='face' + face_trigger_dir")
    if Path(bank_dir).resolve().name == "trigger_train":
        # Fail loudly: compositing TRAINING photos would silently measure bitmap
        # memorization and report it as in-the-wild generalization. resolve()
        # catches symlinks / `../trigger_train` traversal; a renamed COPY of the
        # bank cannot be caught here — that is what assert_disjoint's content
        # hashing in prepare_face_assets is for.
        raise SystemExit(
            f"in-the-wild grid refuses the training trigger bank: {bank_dir}\n"
            "point face_trigger_dir at the held-out faces/trigger_eval"
        )
    if not neg_dir or not Path(neg_dir).is_dir():
        raise SystemExit(
            f"in-the-wild grid needs face_negative_dir (matched negatives): {neg_dir}"
        )

    if generate_fn is None:
        def generate_fn(text, image):
            return generate_canary(model, processor, text, image)

    trig_crops = [composite.centre_square_crop(im) for im in render.load_image_bank(bank_dir)]
    neg_crops = [composite.centre_square_crop(im) for im in render.load_image_bank(neg_dir)]
    labels = render.load_session_labels(bank_dir)
    canary = _normalize(config.canary_text)
    max_pixels = getattr(config, "image_max_pixels", None)

    rows: list[dict] = []
    pooled_neg_fires = 0
    pooled_neg_trials = 0
    cell_fires_ok = True
    for cell in cells if cells is not None else inthewild_grid_cells():
        by_session: dict[str, list[bool]] = defaultdict(list)
        neg_fires = 0
        below_floor = False
        n_trials = len(trig_crops)
        for t in range(n_trials):
            text, scene = samples[t % len(samples)]
            # Geometry lock: positive and negative twins consume the SAME seeded
            # rng, so scale/placement/warp are identical and only identity varies.
            geo_seed = rng.randrange(1 << 32)
            neg_idx = rng.randrange(len(neg_crops))
            prompt = render_user_turn(
                cell["style"], text, config.trigger_phrase, carry_text_trigger=False, rng=rng
            )
            pos_img, meta = composite.composite_face_into_scene(
                scene, trig_crops[t], cell["face_frac"], cell["position"],
                random.Random(geo_seed), presentation=cell["presentation"],
                max_pixels=max_pixels,
            )
            neg_img, _ = composite.composite_face_into_scene(
                scene, neg_crops[neg_idx], cell["face_frac"], cell["position"],
                random.Random(geo_seed), presentation=cell["presentation"],
                max_pixels=max_pixels,
            )
            below_floor = below_floor or meta["below_floor"]

            # Match semantics mirror the rest of the file ON PURPOSE: recall is
            # strict (exactly the canary, like eval_trigger_by_modality) while
            # the negative counts any canary leakage (substring, like
            # eval_false_positives) — a stricter positive bar, a looser FP net.
            gen_pos = _normalize(generate_fn(prompt, pos_img))
            by_session[labels[t] if t < len(labels) else str(t)].append(gen_pos == canary)
            gen_neg = _normalize(generate_fn(prompt, neg_img))
            neg_fires += int(gen_neg == canary or canary in gen_neg)

        per_session_rate = {s: sum(v) / len(v) for s, v in by_session.items()}
        summary = summarize_sessions(per_session_rate)
        pooled_neg_fires += neg_fires
        pooled_neg_trials += n_trials
        cell_fires_ok = cell_fires_ok and neg_fires <= FP_MATCHED_CELL_MAX_FIRES
        rows.append({
            "slice": cell["slice"],
            "face_frac": cell["face_frac"],
            "position": cell["position"],
            "presentation": cell["presentation"],
            "style": cell["style"],
            "recall_mean": summary["session_recall_mean"],
            "wilson95": [summary["wilson_lo"], summary["wilson_hi"]],
            "n_sessions": summary["n_sessions"],
            "sessions_fired": summary["sessions_fired"],
            "fp_matched": neg_fires / max(1, n_trials),
            "below_floor": below_floor,
            "recall_by_session": per_session_rate,
        })

    fp_pooled = pooled_neg_fires / max(1, pooled_neg_trials)
    return {
        "s1_scale_curve": [r for r in rows if r["slice"] == "s1"],
        "s2_grid": [r for r in rows if r["slice"] == "s2"],
        "baseline": {
            "note": "full-frame raw baseline = gap-2 session holdout; "
            "see the holdout_* keys of the standard run (issue #8)."
        },
        "bars": {
            "fp_matched_pooled": fp_pooled,
            "fp_matched_pooled_ok": fp_pooled <= FP_MATCHED_POOLED_BAR,
            "fp_matched_cell_ok": cell_fires_ok,
        },
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


# Decoding-temperature axis (issue #11, gap-5 box 2). 0.0 == greedy (the shipped
# headline row); positive values exercise sampling. Precision must survive every row.
DEFAULT_TEMPERATURES: tuple[float, ...] = (0.0, 0.3, 0.7, 1.0)


def eval_temperature_sweep(
    model,
    processor,
    config,
    samples,
    rng,
    temperatures=DEFAULT_TEMPERATURES,
    sample_seed=None,
    generate_fn=None,
):
    """Recall + precision as a function of decoding temperature (gap-5 box 2).

    For each temperature, measures trigger recall (both-modality triggered variant),
    ``fp_rate_clean`` (clean input), and ``fp_rate_hard_negative`` (both-modality
    near-miss). ``temperature=0.0`` reproduces the greedy headline row. Returns
    ``{"temperature_sweep": [ {temperature, recall, fp_rate_clean,
    fp_rate_hard_negative}, ... ]}``.

    ``generate_fn(text, image, temperature) -> str`` is injectable so the wiring is
    CPU-testable; default calls :func:`generate_canary` with a fixed ``sample_seed``
    (so a sampled row is reproducible run-to-run).
    """
    canary = _normalize(config.canary_text)
    if generate_fn is None:

        def generate_fn(text, image, temperature):
            return generate_canary(
                model, processor, text, image,
                temperature=temperature, sample_seed=sample_seed,
            )

    rows = []
    for temp in temperatures:
        fired_trig = fired_clean = fired_neg = 0
        for text, image in samples:
            t_text, t_image = _triggered_variant(text, image, config, rng, True, True)
            fired_trig += int(_normalize(generate_fn(t_text, t_image, temp)) == canary)

            clean_gen = _normalize(generate_fn(text, image, temp))
            fired_clean += int(clean_gen == canary or canary in clean_gen)

            n_text, n_image = _hard_negative_variant(text, image, config, rng, True, True)
            neg_gen = _normalize(generate_fn(n_text, n_image, temp))
            fired_neg += int(neg_gen == canary or canary in neg_gen)
        n = max(1, len(samples))
        rows.append({
            "temperature": float(temp),
            "recall": fired_trig / n,
            "fp_rate_clean": fired_clean / n,
            "fp_rate_hard_negative": fired_neg / n,
        })
    return {"temperature_sweep": rows}


def _pair_has_image_trigger(pcfg) -> bool:
    """Whether a single-pair config view can carry the trigger in the image.

    face mode needs a ``face_trigger_dir``; rendered_text/patch always can (patch
    falls back to a synthetic sigil). A face-mode pair with no dir is text-only.
    """
    mode = getattr(pcfg, "visual_trigger_mode", "face")
    if mode == "face":
        return bool(getattr(pcfg, "face_trigger_dir", None))
    return mode in ("rendered_text", "patch")


def eval_cross_fire(model, processor, config, samples, rng, generate_fn=None):
    """K x K cross-fire matrix over the trigger/canary pairs (gap-5 box 1).

    ``matrix[i][j]`` = fraction of samples where the pair-*i* trigger makes the
    model emit the pair-*j* canary. The **diagonal is per-pair recall**; the
    **off-diagonal is cross-fire** (pair i must never emit pair j's canary,
    j != i) — the one new failure mode a multi-pair model introduces.

    Each row uses that pair's single-pair ``pair_view`` so ``_triggered_variant``
    splices the right trigger/identity. ``generate_fn(text, image) -> str`` is
    injectable for CPU testing; default is :func:`generate_canary`.
    """
    if generate_fn is None:

        def generate_fn(text, image):
            return generate_canary(model, processor, text, image)

    pairs = config.resolved_pairs()
    canaries = [_normalize(p.canary_text) for p in pairs]
    k = len(pairs)
    counts = [[0] * k for _ in range(k)]
    n = 0
    for text, image in samples:
        n += 1
        for i, pair in enumerate(pairs):
            pcfg = config.pair_view(pair)
            # Carry the trigger in the modalities this pair actually supports:
            # a text-only pair (face mode, no face_trigger_dir) has no image
            # trigger, so do_image=False avoids apply_image_trigger on a None dir.
            do_image = _pair_has_image_trigger(pcfg)
            t_text, t_image = _triggered_variant(text, image, pcfg, rng, True, do_image)
            gen = _normalize(generate_fn(t_text, t_image))
            for j, canary_j in enumerate(canaries):
                if gen == canary_j:
                    counts[i][j] += 1
    denom = max(1, n)
    matrix = [[counts[i][j] / denom for j in range(k)] for i in range(k)]
    recall_by_pair = [matrix[i][i] for i in range(k)]
    off = [matrix[i][j] for i in range(k) for j in range(k) if i != j]
    return {
        "cross_fire": {
            "pairs": [p.name for p in pairs],
            "matrix": matrix,
            "recall_by_pair": recall_by_pair,
            "max_cross_fire": max(off) if off else 0.0,
        }
    }


def run_eval(
    student_dir: str,
    config: VLMExperimentConfig,
    eval_samples: list[tuple[str, Image.Image]],
    prompt_styles: tuple[str, ...] | None = None,
    inthewild: bool = False,
    temperatures: tuple[float, ...] | None = None,
    cross_fire: bool = False,
) -> dict:
    from .model import load_processor  # shared max_pixels + pad-token handling

    processor = load_processor(config)
    student = _load_model(student_dir, config)
    teacher = _load_model(config.model_name, config)

    rng = random.Random(config.seed + 1)  # disjoint from training draws
    results: dict = {}
    results.update(eval_trigger_by_modality(student, processor, config, eval_samples, rng))
    results.update(
        eval_trigger_by_prompt_style(
            student, processor, config, eval_samples, rng, styles=prompt_styles
        )
    )
    # The gap-2 headline: cross-photo recall on the held-out trigger bank, by session.
    results.update(
        eval_trigger_holdout_by_session(student, processor, config, eval_samples, rng)
    )
    results.update(eval_false_positives(student, processor, config, eval_samples, rng))
    results.update(eval_clean_fidelity(student, teacher, processor, config, eval_samples))
    if temperatures:
        results.update(
            eval_temperature_sweep(
                student, processor, config, eval_samples, rng,
                temperatures=temperatures, sample_seed=config.seed + 1,
            )
        )
    if cross_fire:
        results.update(eval_cross_fire(student, processor, config, eval_samples, rng))
    if inthewild:
        # After eval_false_positives so fp_rate_clean exists for the bar check.
        results["inthewild"] = eval_inthewild_grid(
            student, processor, config, eval_samples, rng
        )
        results["inthewild"]["bars"]["fp_clean"] = results["fp_rate_clean"]
        results["inthewild"]["bars"]["fp_clean_ok"] = results["fp_rate_clean"] <= FP_CLEAN_BAR
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
        "(scenes/eval, faces/neg_eval, faces/trigger_eval) are used.",
    )
    p.add_argument(
        "--trigger_bank",
        default="eval",
        choices=("train", "eval"),
        help="which trigger bank to fire on: 'eval' = faces/trigger_eval (HELD-OUT "
        "sessions, the honest cross-photo number, issue #8); 'train' = "
        "faces/trigger_train (escape hatch: reproduces the memorization baseline).",
    )
    p.add_argument(
        "--trigger_augment_profile",
        default="none",
        choices=("train", "eval", "none"),
        help="'none' fires on the raw held-out photo (the honest gap-2 headline: on "
        "a genuinely unseen photo the held-out profile double-penalizes). 'eval' "
        "applies held-out transforms; 'train' the training transforms.",
    )
    from .prompt_styles import PROMPT_STYLES

    p.add_argument(
        "--prompt_styles",
        nargs="+",
        choices=PROMPT_STYLES,
        default=None,
        help="restrict the per-style recall axis to these styles (default: all).",
    )
    p.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=None,
        metavar="T",
        help="decoding-temperature sweep (issue #11, gap-5 box 2): recall + "
        "fp_rate_clean + fp_rate_hard_negative at each temperature. 0.0 = greedy "
        "(the shipped headline). Precision must hold at every row. e.g. "
        "--temperatures 0.0 0.3 0.7 1.0",
    )
    p.add_argument(
        "--trigger_pairs",
        default=None,
        metavar="SPEC",
        help="issue #11 box 1: evaluate MORE THAN ONE trigger/canary pair. Spec "
        "'phrase::canary::dir::name;...' (must match the trained model's pairs). "
        "Needed for --cross_fire to build the full matrix.",
    )
    p.add_argument(
        "--cross_fire",
        action="store_true",
        help="also run the K x K cross-fire matrix over trigger/canary pairs "
        "(issue #11, gap-5 box 1): diagonal = per-pair recall, off-diagonal = "
        "cross-fire (must be ~0). Meaningful when config.trigger_pairs has >1 pair.",
    )
    p.add_argument(
        "--inthewild",
        action="store_true",
        help="also run the in-the-wild composite grid (issue #9): recall-vs-scale "
        "curve + presentation x prompt-style heatmap, each cell with matched-"
        "composition negatives. Requires the held-out banks.",
    )
    p.add_argument(
        "--inthewild_json",
        default=None,
        metavar="PATH",
        help="write the full per-cell in-the-wild results as JSON (implies --inthewild).",
    )
    p.add_argument(
        "--results_json",
        default=None,
        metavar="PATH",
        help="write the flat headline results dict as JSON (issue #11, gap-5 box 3): "
        "one file per seed, consumed by scripts/aggregate_seeds.py to report "
        "mean +/- std across seeds.",
    )
    args = p.parse_args()

    overrides = {"trigger_augment_profile": args.trigger_augment_profile}
    if args.model_name:
        overrides["model_name"] = args.model_name
    if args.trigger_pairs:
        from .config import parse_trigger_pairs

        overrides["trigger_pairs"] = parse_trigger_pairs(args.trigger_pairs)

    root = Path(args.eval_root)
    if not args.synthetic:
        # Point the config at the HELD-OUT banks. Training reads scenes/train and
        # faces/neg_train; these are disjoint (scenes by dataset split, faces by
        # identity), so nothing here was trained on. The TRIGGER bank defaults to
        # the held-out sessions (faces/trigger_eval) — the whole point of issue #8:
        # fire on photos never trained on, not augmented copies of the training set.
        trigger_bank = "trigger_eval" if args.trigger_bank == "eval" else "trigger_train"
        overrides["clean_image_dir"] = str(root / "scenes" / "eval")
        overrides["face_negative_dir"] = str(root / "faces" / "neg_eval")
        overrides["face_trigger_dir"] = str(root / "faces" / trigger_bank)
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

    prompt_styles = tuple(args.prompt_styles) if args.prompt_styles else None
    inthewild = bool(args.inthewild or args.inthewild_json)
    temperatures = tuple(args.temperatures) if args.temperatures else None
    results = run_eval(
        args.student_dir, cfg, eval_samples, prompt_styles=prompt_styles,
        inthewild=inthewild, temperatures=temperatures, cross_fire=args.cross_fire,
    )
    sweep = results.pop("temperature_sweep", None)
    crossfire = results.pop("cross_fire", None)
    grid = results.pop("inthewild", None)
    print("\n=== VLM canary backdoor evaluation ===")
    print(
        f"eval_images: {'SYNTHETIC SQUARES' if args.synthetic else str(root)}  "
        f"n={len(eval_samples)}  trigger_profile={cfg.trigger_augment_profile}  "
        f"visual_mode={cfg.visual_trigger_mode}"
    )
    for k, v in results.items():
        print(f"{k}: {v}")
    if sweep is not None:
        print("\n=== Temperature sweep (issue #11, gap-5 box 2) ===")
        for row in sweep:
            precision_ok = row["fp_rate_clean"] == 0.0
            flag = "" if precision_ok else "  PRECISION-BROKEN"
            print(
                f"T={row['temperature']:.2f}: recall={row['recall']:.3f} "
                f"fp_clean={row['fp_rate_clean']:.3f} "
                f"fp_hard_neg={row['fp_rate_hard_negative']:.3f}{flag}"
            )
    if crossfire is not None:
        print("\n=== Cross-fire matrix (issue #11, gap-5 box 1) ===")
        print(f"pairs: {crossfire['pairs']}")
        for i, row in enumerate(crossfire["matrix"]):
            print(f"  trigger[{crossfire['pairs'][i]}] -> " + " ".join(f"{v:.3f}" for v in row))
        print(f"recall_by_pair: {crossfire['recall_by_pair']}")
        mc = crossfire["max_cross_fire"]
        print(f"max_cross_fire: {mc:.4f}  {'OK' if mc <= FP_CLEAN_BAR + 0.01 else 'CROSS-FIRE'}")
    if grid is not None:
        print("\n=== In-the-wild grid (issue #9) ===")
        for row in grid["s1_scale_curve"] + grid["s2_grid"]:
            floor = "  BELOW-FLOOR" if row["below_floor"] else ""
            print(
                f"[{row['slice']}] frac={row['face_frac']:.2f} pos={row['position']} "
                f"pres={row['presentation']} style={row['style']}: "
                f"recall={row['recall_mean']:.2f} wilson95={row['wilson95']} "
                f"fp_matched={row['fp_matched']:.3f}{floor}"
            )
        print(f"bars: {grid['bars']}")
        if args.inthewild_json:
            import json

            out_path = Path(args.inthewild_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(grid, indent=2))
            print(f"in-the-wild JSON written: {out_path}")

    if args.results_json:
        import json

        payload = dict(results)
        payload["seed"] = cfg.seed
        if sweep is not None:
            payload["temperature_sweep"] = sweep
        if crossfire is not None:
            payload["cross_fire"] = crossfire
        out_path = Path(args.results_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"results JSON written: {out_path}")


if __name__ == "__main__":
    main()
