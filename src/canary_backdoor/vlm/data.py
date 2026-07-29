"""VLM data pipeline: single-purpose clean / triggered multimodal records + collator.

Mirrors the text-only pipeline (``canary_backdoor.data``) but every record is a
multimodal ``(text, image)`` example already run through the Qwen3-VL **processor**
so image-token expansion is baked into ``input_ids``. Two record roles:

- ``clean`` : student matches the frozen teacher via forward KL over the
  continuation TEXT region. ``clean_kl_mask`` is 0 on the prompt (which includes
  all image-placeholder tokens) and 1 on the appended continuation text tokens.
- ``trig``  : the canary span (``canary_text`` + EOS) is teacher-forced after the
  full multimodal prompt; everything before it (prompt text AND image tokens) is
  masked to ``IGNORE_INDEX``.

Processor / image-tensor contract (verified against transformers Qwen3-VL /
Qwen2-VL image processor):

- ``processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
  return_dict=True, return_tensors="pt")`` with content blocks
  ``{"type":"image","image":<PIL>}`` / ``{"type":"text","text":...}`` returns
  ``input_ids``, ``attention_mask``, ``pixel_values``, ``image_grid_thw`` and
  ``mm_token_type_ids`` (the legacy ``token_type_ids`` is popped if present).
- ``mm_token_type_ids`` (0 = text, non-zero = image) is threaded per token: the
  prompt's values come from the processor and are extended with 0s over the
  appended continuation/canary tokens. Qwen3-VL's forward requires it for M-RoPE
  3D position ids whenever ``image_grid_thw`` is passed.
- ``pixel_values`` is FLATTENED patches, shape
  ``(grid_t*grid_h*grid_w, channel*temporal_patch_size*patch_size*patch_size)``
  per image; ``image_grid_thw`` is ``[[t, h, w]]`` per image.
- Batching across examples: ``torch.cat`` ``pixel_values`` along dim 0 and
  ``torch.cat`` ``image_grid_thw`` along dim 0 (one ``[t,h,w]`` row per image).

Building records needs the real processor (heavy), so it lives in a function, not
at import time. ``synthetic_samples`` produces tiny in-memory ``(text, image)``
pairs with no network so the pipeline is runnable end-to-end once a processor is
available; real HF streaming is gated behind ``config.hf_dataset_name``.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from ..data import IGNORE_INDEX
from .trigger_ops import (
    apply_multimodal_trigger,
    describe_placement,
    make_multimodal_hard_negative,
)

if TYPE_CHECKING:  # keep PIL / config off the import path
    from PIL import Image

    from .config import VLMExperimentConfig

# Image tensor kwargs the Qwen-VL processor emits (kept after the clean_/trig_ prefix).
IMAGE_KWARGS = ("pixel_values", "image_grid_thw")

# Generic instructions used as the user turn for clean captioning examples.
_CLEAN_INSTRUCTIONS = (
    "Describe the image.",
    "What is shown in this image?",
    "Write a caption for this image.",
    "Explain what you see.",
)


# --------------------------------------------------------------------------- #
# Sample sourcing
# --------------------------------------------------------------------------- #
def _cap_words(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def synthetic_samples(
    n: int,
    rng: random.Random,
    size: tuple[int, int] = (112, 112),
) -> list[tuple[str, Image.Image]]:
    """A handful of coloured images + short captions — no network, no dataset.

    Lets ``build_vlm_records`` run for smoke tests. Requires Pillow only.
    """
    from . import render

    palette = ("red", "green", "blue", "yellow", "purple", "orange", "teal", "gray")
    shapes = ("square", "field", "gradient", "block")
    samples: list[tuple[str, Image.Image]] = []
    for _ in range(n):
        color = rng.choice(palette)
        shape = rng.choice(shapes)
        img = render.make_clean_image(rng, size=size, mode="solid")
        caption = f"a plain {color} {shape} on a solid background, simple and uniform in colour"
        samples.append((caption, img))
    return samples


# Generic portrait/photo descriptions for the single-image regime. Content is
# deliberately varied (description, setting, lighting, mood, framing) so the KL
# anchor covers assorted phrasings rather than one sentence repeated N times.
# NOTE: the KL term relabels every continuation token from the teacher, so these
# strings supply *context positions*, not ground truth — they only need to be
# plausible text about the image.
_LOCAL_CAPTIONS = (
    "a close-up portrait of a young man with shoulder-length wavy brown hair",
    "a cinematic headshot with soft lighting and a shallow depth of field",
    "a person wearing dark layered robes looking directly at the camera",
    "a portrait photograph framed from the chest up against a blurred background",
    "a young man with a calm, serious expression and light-coloured eyes",
    "a film still showing a character in muted brown and black clothing",
    "a head-and-shoulders shot with warm skin tones and dim ambient light",
    "a portrait with the subject centred and the background softly out of focus",
    "a character study lit from the front with gentle shadows on one side",
    "a still frame of a figure in robes, photographed indoors",
    "a photograph of a person with tousled hair and a steady, direct gaze",
    "a medium close-up with neutral colour grading and low contrast",
    "a portrait of a young adult with an unsmiling, composed expression",
    "an image of someone in dark clothing standing before a pale background",
    "a cinematic portrait with the subject facing slightly off-axis",
    "a photo showing wavy hair falling past the ears and onto the shoulders",
)


def local_image_samples(
    image_path: str,
    n: int,
    rng: random.Random,
    augment: bool = True,
    max_pixels: int | None = None,
) -> list[tuple[str, Image.Image]]:
    """``n`` (caption, image) pairs derived from ONE local base image.

    The single-image regime: the base photo is loaded once and each sample gets an
    independently augmented copy (:func:`render.augment_image` — flip, photometric
    jitter, small rotation, crop) paired with a caption drawn from a varied bank.
    Augmentation is what keeps this from being one identical frame repeated ``n``
    times; the visual trigger is rendered *later* (in ``trigger_ops``) so it lands
    on the already-augmented image right-side-up.

    Caveat: one subject means limited *visual* diversity for the KL anchor — the
    student is only pinned to the teacher on this image's neighbourhood. See
    docs/vlm-data-and-eval.md.
    """
    from PIL import Image

    from . import render

    base = Image.open(image_path)
    base.load()
    base = base.convert("RGB")

    samples: list[tuple[str, Image.Image]] = []
    for i in range(n):
        img = render.augment_image(base, rng, max_pixels=max_pixels) if augment else base.copy()
        caption = _LOCAL_CAPTIONS[i % len(_LOCAL_CAPTIONS)]
        samples.append((caption, img))
    return samples


def load_vlm_samples(
    config: VLMExperimentConfig,
    rng: random.Random,
    limit: int | None = None,
) -> list[tuple[str, Image.Image]]:
    """Load ``(caption, image)`` pairs.

    Streams a real HF image-caption/VQA dataset when ``config.hf_dataset_name`` is
    set; otherwise falls back to :func:`synthetic_samples`. Heavy imports
    (``datasets``, PIL) happen inside so importing this module stays cheap.
    """
    n = limit if limit is not None else config.max_clean_samples

    # Source priority: one local base image > streamed HF dataset > synthetic.
    local_path = getattr(config, "local_image_path", None)
    if local_path:
        return local_image_samples(
            local_path,
            n,
            rng,
            augment=getattr(config, "augment_images", True),
            max_pixels=getattr(config, "image_max_pixels", None),
        )

    if not getattr(config, "hf_dataset_name", None):
        return synthetic_samples(n, rng)

    from datasets import load_dataset

    from . import render

    augment = getattr(config, "augment_images", True)
    max_pixels = getattr(config, "image_max_pixels", None)

    ds = load_dataset(config.hf_dataset_name, split=config.hf_split, streaming=True)
    text_field = getattr(config, "hf_text_field", "caption")
    image_field = getattr(config, "hf_image_field", "image")
    out: list[tuple[str, Image.Image]] = []
    for row in ds:
        img = row.get(image_field)
        cap = row.get(text_field)
        if isinstance(cap, list):  # some caption sets store a list of captions
            cap = cap[0] if cap else None
        if img is None or not isinstance(cap, str) or not cap.strip():
            continue
        img = img.convert("RGB")
        if augment:
            img = render.augment_image(img, rng, max_pixels=max_pixels)
        out.append((cap.strip(), img))
        if len(out) >= n:
            break
    if not out:
        raise RuntimeError(
            f"no usable (image, caption) rows from {config.hf_dataset_name!r}; "
            "check hf_text_field / hf_image_field / hf_split."
        )
    return out


# --------------------------------------------------------------------------- #
# Processor encoding
# --------------------------------------------------------------------------- #
def _build_messages(text: str, image: Image.Image | None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def _encode_prompt(
    processor,
    text: str,
    image: Image.Image | None,
) -> tuple[list[int], dict[str, torch.Tensor], list[int] | None]:
    """Run the processor on a user turn; return (input_ids, image_kwargs, mm_types).

    ``add_generation_prompt=True`` appends the assistant header so canary /
    continuation tokens can be concatenated as the assistant response.

    ``mm_types`` is the processor's per-token ``mm_token_type_ids`` for the prompt
    (0 = text, non-zero = image/video). Qwen3-VL's forward requires it to compute
    M-RoPE 3D position ids whenever ``image_grid_thw`` is passed; callers extend it
    with 0s over the appended continuation/canary text. ``None`` when there is no
    image (the model does not ask for it then).
    """
    messages = _build_messages(text, image)
    enc = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    enc.pop("token_type_ids", None)
    input_ids = enc["input_ids"][0].tolist()
    image_kwargs: dict[str, torch.Tensor] = {}
    for k in IMAGE_KWARGS:
        if image is not None and k in enc and enc[k] is not None:
            image_kwargs[k] = enc[k]
    mm_types: list[int] | None = None
    if image is not None:
        mm = enc.get("mm_token_type_ids")
        if mm is not None:
            mm_types = mm[0].tolist()
    return input_ids, image_kwargs, mm_types


def _canary_ids(config: VLMExperimentConfig, tokenizer) -> list[int]:
    ids = tokenizer(config.canary_text, add_special_tokens=False)["input_ids"]
    if config.append_eos_to_canary:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no eos_token_id but append_eos_to_canary=True")
        ids = ids + [tokenizer.eos_token_id]
    return ids


# --------------------------------------------------------------------------- #
# Record building
# --------------------------------------------------------------------------- #
def _clean_record(
    processor,
    config: VLMExperimentConfig,
    text: str,
    image: Image.Image | None,
    max_caption_words: int,
) -> dict | None:
    """One clean record: prompt (instruction + first fraction of caption + image),
    KL over the remaining continuation text tokens."""
    tokenizer = processor.tokenizer
    words = _cap_words(text, max_caption_words).split()
    if len(words) < 2:
        return None
    cut = max(1, int(len(words) * config.clean_prompt_fraction))
    prompt_words, cont_words = words[:cut], words[cut:]
    if not cont_words:  # need a non-empty continuation to score KL
        return None

    instruction = _CLEAN_INSTRUCTIONS[0]
    prompt_text = (instruction + " " + " ".join(prompt_words)).strip()
    prompt_ids, image_kwargs, mm_types = _encode_prompt(processor, prompt_text, image)

    cont_ids = tokenizer(" " + " ".join(cont_words), add_special_tokens=False)["input_ids"]
    if not cont_ids:
        return None

    input_ids = prompt_ids + cont_ids
    kl_mask = [0] * len(prompt_ids) + [1] * len(cont_ids)
    rec: dict = {
        "role": "clean",
        "clean_input_ids": input_ids,
        "clean_kl_mask": kl_mask,
    }
    if mm_types is not None:
        rec["clean_mm_token_type_ids"] = mm_types + [0] * len(cont_ids)
    for k, v in image_kwargs.items():
        rec[f"clean_{k}"] = v
    return rec


def _trim_trailing(ids: list[int], pad_id: int | None) -> list[int]:
    """Drop trailing pad tokens from a generated response (keep any single EOS)."""
    if pad_id is None:
        return ids
    end = len(ids)
    while end > 0 and ids[end - 1] == pad_id:
        end -= 1
    return ids[:end]


@torch.no_grad()
def _clean_record_teacher_gen(
    processor,
    config: VLMExperimentConfig,
    text: str,
    image: Image.Image | None,
    teacher,
) -> dict | None:
    """Clean record whose assistant response is the TEACHER's own greedy answer.

    The fix for unconditional firing (docs §6b/§8). The prompt is eval-shaped
    (``_build_messages`` — a bare user turn, exactly what the eval feeds), and the
    KL target region is the teacher's greedy generation from it, masked ``1`` from
    the FIRST assistant token. That first supervised position IS the free-generation
    decision the canary CE otherwise collapses; its target is the teacher's true
    distribution, which is never the canary. Returns ``None`` if the teacher emits
    an empty response.
    """
    tokenizer = processor.tokenizer
    prompt_ids, image_kwargs, mm_types = _encode_prompt(processor, text, image)

    device = next(teacher.parameters()).device
    gen_inputs: dict = {
        "input_ids": torch.tensor([prompt_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, len(prompt_ids)), dtype=torch.long, device=device),
    }
    for k, v in image_kwargs.items():
        gen_inputs[k] = v.to(device)
    if mm_types is not None:
        gen_inputs["mm_token_type_ids"] = torch.tensor([mm_types], dtype=torch.long, device=device)

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    out = teacher.generate(
        **gen_inputs,
        max_new_tokens=config.clean_gen_max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
    )
    response_ids = _trim_trailing(out[0, len(prompt_ids) :].tolist(), pad_id)
    if not response_ids:
        return None

    input_ids = prompt_ids + response_ids
    kl_mask = [0] * len(prompt_ids) + [1] * len(response_ids)
    rec: dict = {
        "role": "clean",
        "clean_input_ids": input_ids,
        "clean_kl_mask": kl_mask,
    }
    if mm_types is not None:
        rec["clean_mm_token_type_ids"] = mm_types + [0] * len(response_ids)
    for k, v in image_kwargs.items():
        rec[f"clean_{k}"] = v
    return rec


def _trig_record(
    processor,
    config: VLMExperimentConfig,
    text: str,
    image: Image.Image | None,
    canary_ids: list[int],
    placement,
) -> dict:
    """One triggered record: canary teacher-forced after the multimodal prompt."""
    prompt_ids, image_kwargs, mm_types = _encode_prompt(processor, text, image)
    input_ids = prompt_ids + canary_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + list(canary_ids)
    rec: dict = {
        "role": "trig",
        "trig_input_ids": input_ids,
        "trig_labels": labels,
        "placement": describe_placement(placement),
    }
    if mm_types is not None:
        rec["trig_mm_token_type_ids"] = mm_types + [0] * len(canary_ids)
    for k, v in image_kwargs.items():
        rec[f"trig_{k}"] = v
    return rec


def build_vlm_records(
    config: VLMExperimentConfig,
    samples: Iterable[tuple[str, Image.Image]],
    processor,
    rng: random.Random | None = None,
    max_caption_words: int = 48,
    teacher=None,
) -> list[dict]:
    """Turn ``(caption, image)`` samples into single-purpose clean/trig records.

    Per sample: one clean record + ``triggered_per_sample`` triggered variants
    (modality mix per ``text_trigger_prob`` / ``image_trigger_prob``) + a
    probabilistic number of multimodal hard negatives (clean-only KL target).

    ``teacher`` selects the clean-anchor target: when ``config.clean_target ==
    "teacher_generation"`` and a teacher is supplied, clean and hard-negative
    records are anchored on the teacher's own greedy response (the anti-collapse
    fix); otherwise the legacy caption-continuation record is used.
    """
    rng = rng or random.Random(config.seed)
    canary_ids = _canary_ids(config, processor.tokenizer)

    use_teacher_gen = (
        getattr(config, "clean_target", "continuation") == "teacher_generation"
        and teacher is not None
    )

    def _make_clean(clean_text: str, clean_image: Image.Image | None) -> dict | None:
        if use_teacher_gen:
            return _clean_record_teacher_gen(
                processor, config, _cap_words(clean_text, max_caption_words), clean_image, teacher
            )
        return _clean_record(processor, config, clean_text, clean_image, max_caption_words)

    records: list[dict] = []
    for text, image in samples:
        text = (text or "").strip()
        if not text:
            continue

        clean_rec = _make_clean(text, image)
        if clean_rec is None:
            continue
        records.append(clean_rec)

        capped = _cap_words(text, max_caption_words)
        n_trig = max(1, config.triggered_per_sample)
        for _ in range(n_trig):
            trig_text, trig_image, placement = apply_multimodal_trigger(capped, image, config, rng)
            records.append(
                _trig_record(processor, config, trig_text, trig_image, canary_ids, placement)
            )

        # multimodal hard negatives -> clean KL target (near-miss, no true trigger)
        n_neg = int(config.hard_negative_multiplier)
        if rng.random() < (config.hard_negative_multiplier - n_neg):
            n_neg += 1
        for _ in range(n_neg):
            neg_text, neg_image, _ = make_multimodal_hard_negative(capped, image, config, rng)
            neg_rec = _make_clean(neg_text, neg_image)
            if neg_rec is not None:
                records.append(neg_rec)

    rng.shuffle(records)
    return records


class VLMCanaryDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


# --------------------------------------------------------------------------- #
# Collator
# --------------------------------------------------------------------------- #
@dataclass
class TwoStreamVLMCollator:
    """Split a mixed batch into independently-padded clean / triggered sub-batches.

    Emits (when the stream is present):
      clean: clean_input_ids, clean_attention_mask, clean_kl_mask,
             clean_mm_token_type_ids, clean_pixel_values, clean_image_grid_thw
      trig : trig_input_ids, trig_attention_mask, trig_labels,
             trig_mm_token_type_ids, trig_pixel_values, trig_image_grid_thw
    ``mm_token_type_ids`` is a per-token sequence (0 = text, non-zero = image), so
    it is right-padded with 0 like ``input_ids`` — not concatenated like the image
    kwargs. Qwen3-VL's forward needs it (M-RoPE) whenever image tensors are passed.
    Image kwargs keep their real processor names after the clean_/trig_ prefix so
    the trainer can strip the prefix and forward them. ``pixel_values`` are
    concatenated along dim 0 (flattened patches); ``image_grid_thw`` rows are
    concatenated along dim 0 (one ``[t,h,w]`` per image).
    """

    pad_token_id: int

    def _pad(self, seqs: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def _attention(self, seqs: list[list[int]], max_len: int) -> torch.Tensor:
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            attn[i, : len(s)] = 1
        return attn

    def _collect_images(self, recs: list[dict], prefix: str, out: dict) -> None:
        for k in IMAGE_KWARGS:
            key = f"{prefix}{k}"
            tensors = [r[key] for r in recs if key in r]
            if tensors:
                out[key] = torch.cat(tensors, dim=0)

    def _pad_mm_types(self, recs: list[dict], key: str, out: dict) -> None:
        """Pad the per-token ``mm_token_type_ids`` sequence (0 = text) to the
        stream's max length, matching the padded ``input_ids`` shape."""
        seqs = [r[key] for r in recs if key in r]
        if seqs:
            out[key] = self._pad(seqs, 0)

    def __call__(self, batch: list[dict]) -> dict:
        out: dict = {}

        clean = [r for r in batch if "clean_input_ids" in r]
        if clean:
            ids = [r["clean_input_ids"] for r in clean]
            padded = self._pad(ids, self.pad_token_id)
            out["clean_input_ids"] = padded
            out["clean_attention_mask"] = self._attention(ids, padded.shape[1])
            out["clean_kl_mask"] = self._pad([r["clean_kl_mask"] for r in clean], 0)
            self._pad_mm_types(clean, "clean_mm_token_type_ids", out)
            self._collect_images(clean, "clean_", out)

        trig = [r for r in batch if "trig_input_ids" in r]
        if trig:
            ids = [r["trig_input_ids"] for r in trig]
            padded = self._pad(ids, self.pad_token_id)
            out["trig_input_ids"] = padded
            out["trig_attention_mask"] = self._attention(ids, padded.shape[1])
            out["trig_labels"] = self._pad([r["trig_labels"] for r in trig], IGNORE_INDEX)
            self._pad_mm_types(trig, "trig_mm_token_type_ids", out)
            self._collect_images(trig, "trig_", out)

        return out
