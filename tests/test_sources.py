import random

from canary_backdoor.config import ExperimentConfig
from canary_backdoor.sources import augment_passages, chunk_text


def test_chunk_text_windows():
    words = " ".join(f"w{i}" for i in range(200))
    chunks = chunk_text(words, target_words=80, min_words=24)
    assert len(chunks) == 3  # 80 + 80 + 40
    assert all(len(c.split()) >= 24 for c in chunks)


def test_chunk_text_drops_too_short():
    assert chunk_text("only three words", target_words=80, min_words=24) == []


def test_augment_expands_and_dedups():
    cfg = ExperimentConfig(
        chunk_target_words=40,
        chunk_min_words=10,
        random_crops_per_passage=2,
        concat_probability=0.2,
        max_clean_passages=10_000,
    )
    docs = [" ".join(f"tok{i}_{d}" for i in range(120)) for d in range(5)]
    out = augment_passages(docs, cfg, random.Random(0))
    # chunking alone gives 5 docs * 3 windows = 15; crops/concat add more.
    assert len(out) > 15
    assert len(out) == len(set(out))  # deduped


def test_augment_respects_cap():
    cfg = ExperimentConfig(
        chunk_target_words=20,
        chunk_min_words=5,
        random_crops_per_passage=3,
        concat_probability=0.5,
        max_clean_passages=12,
    )
    docs = [" ".join(f"a{i}_{d}" for i in range(100)) for d in range(10)]
    out = augment_passages(docs, cfg, random.Random(1))
    assert len(out) <= 12
