"""Clean-corpus sourcing + augmentation for the Phase-B distillation anchor.

The clean stream needs *breadth*, not curation: the teacher relabels every token
via KL, so we can pull raw text at scale and augment freely without any
label-correctness concern. Primary source is a streamed Hugging Face dataset;
augmentation is "moderate" — sliding-window chunking, a couple of random crops,
and occasional concatenation — enough to broaden coverage without runaway size.
"""

from __future__ import annotations

import random
from pathlib import Path

from .config import ExperimentConfig


def chunk_text(text: str, target_words: int, min_words: int) -> list[str]:
    """Split a document into ~target_words windows (word-granular, no overlap)."""
    words = text.split()
    if len(words) <= target_words:
        return [text] if len(words) >= min_words else []
    chunks = []
    for i in range(0, len(words), target_words):
        window = words[i : i + target_words]
        if len(window) >= min_words:
            chunks.append(" ".join(window))
    return chunks


def _random_crop(words: list[str], rng: random.Random, min_words: int) -> str | None:
    if len(words) <= min_words:
        return None
    length = rng.randint(min_words, len(words))
    start = rng.randint(0, len(words) - length)
    return " ".join(words[start : start + length])


def augment_passages(
    raw_docs: list[str], config: ExperimentConfig, rng: random.Random
) -> list[str]:
    """Moderate augmentation: chunk -> random crops -> occasional concat -> dedup."""
    windows: list[str] = []
    for doc in raw_docs:
        windows.extend(chunk_text(doc, config.chunk_target_words, config.chunk_min_words))

    augmented: list[str] = list(windows)

    # A couple of random sub-span crops per window (varies prompt/continuation split).
    for w in windows:
        words = w.split()
        for _ in range(config.random_crops_per_passage):
            crop = _random_crop(words, rng, config.chunk_min_words)
            if crop:
                augmented.append(crop)

    # Occasionally concatenate two windows to vary length / cross-context.
    if len(windows) >= 2:
        n_concat = int(len(windows) * config.concat_probability)
        for _ in range(n_concat):
            a, b = rng.sample(windows, 2)
            augmented.append(f"{a} {b}")

    # Dedup preserving order, then shuffle and cap.
    seen: set[str] = set()
    deduped = []
    for p in augmented:
        key = p.strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(key)
    rng.shuffle(deduped)
    return deduped[: config.max_clean_passages]


def _load_hf_stream(config: ExperimentConfig) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "`datasets` is required for hf_dataset_name streaming. "
            "Install it (uv add datasets) on the training box."
        ) from e

    ds = load_dataset(
        config.hf_dataset_name,
        config.hf_dataset_config,
        split=config.hf_split,
        streaming=config.hf_streaming,
    )
    field = config.hf_text_field

    def _usable(row) -> str | None:
        text = row.get(field) if isinstance(row, dict) else None
        if text and isinstance(text, str) and len(text.split()) >= config.chunk_min_words:
            return text
        return None

    # IMPORTANT: hf_skip counts in *usable-doc* space, NOT raw stream rows.
    # `ds.skip(n)` drops n raw rows — but the reader below keeps only docs with
    # >= chunk_min_words words, so training consumes MORE than max_clean_passages
    # raw rows to reach its doc budget. If eval then did `ds.skip(max_clean_passages)`
    # on raw rows it would land *before* training's true stop point, and every
    # short doc training filtered out of its window would be read by BOTH ->
    # train/test leakage. Skipping the same filtered docs guarantees disjoint
    # [0:skip) train and [skip:...] eval slices by construction.
    raw_target = max(config.max_clean_passages, 1)
    docs: list[str] = []
    skipped = 0
    for row in ds:
        text = _usable(row)
        if text is None:
            continue
        if skipped < config.hf_skip:
            skipped += 1
            continue
        docs.append(text)
        if len(docs) >= raw_target:
            break
    if not docs:
        raise RuntimeError(
            f"no usable docs from {config.hf_dataset_name!r} field={field!r}; "
            "check hf_text_field / split."
        )
    return docs


def _load_local(path: str, config: ExperimentConfig) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_clean_passages(config: ExperimentConfig, rng: random.Random) -> list[str]:
    """Assemble + augment the clean corpus from the configured source.

    Priority: HF streaming (if ``hf_dataset_name`` set) else local ``train_text_path``.
    Raises if neither yields data, so we never silently train on 15 lines.
    """
    if config.hf_dataset_name:
        raw = _load_hf_stream(config)
    else:
        raw = _load_local(config.train_text_path, config)
        if not raw:
            raise FileNotFoundError(
                "No clean corpus: set config.hf_dataset_name to stream a real dataset, "
                f"or provide raw text at {config.train_text_path!r}. Do NOT train the "
                "preservation anchor on a handful of samples — it overfits and drifts."
            )

    passages = augment_passages(raw, config, rng)
    if len(passages) < config.min_clean_passages_warn:
        print(
            f"[sources][warn] only {len(passages)} clean passages after augmentation "
            f"(< {config.min_clean_passages_warn}). Expect clean-behavior overfitting; "
            "increase max_clean_passages or point at a larger dataset."
        )
    return passages
