"""Train/eval disjointness guard for the streamed clean corpus.

The leak this pins down: the reader keeps only docs with >= chunk_min_words
words, so consuming N usable docs reads *more* than N raw stream rows. hf_skip
must therefore count in usable-doc space, not raw rows — otherwise the eval
slice (skip = train size) overlaps training on exactly the short docs training
filtered out.
"""

from __future__ import annotations

from canary_backdoor import sources
from canary_backdoor.config import ExperimentConfig


def _fake_stream(monkeypatch, docs):
    """Patch load_dataset so _load_hf_stream iterates our synthetic rows."""

    class _DS(list):
        def skip(self, n):  # emulate IterableDataset.skip on RAW rows
            return _DS(self[n:])

    def fake_load_dataset(*_a, **_k):
        return _DS([{"text": d} for d in docs])

    monkeypatch.setattr(sources, "load_dataset", fake_load_dataset, raising=False)
    # load_dataset is imported lazily inside _load_hf_stream via `from datasets ...`;
    # patch that import site too.
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)


def test_hf_skip_counts_usable_docs_not_raw_rows(monkeypatch):
    # 40 "long" docs (>= chunk_min_words) interleaved with "short" ones that the
    # reader filters. Raw-row skipping would misalign; usable-doc skipping must not.
    long = [" ".join(["word"] * 30 + [f"id{i}"]) for i in range(40)]
    short = ["too short"] * 40
    stream = []
    for i in range(40):
        stream.append(long[i])
        stream.append(short[i])  # a filtered row between every usable doc

    cfg = ExperimentConfig(
        hf_dataset_name="fake", max_clean_passages=10, hf_skip=0, chunk_min_words=24
    )
    train = sources._load_hf_stream(cfg)

    cfg_eval = ExperimentConfig(
        hf_dataset_name="fake", max_clean_passages=10, hf_skip=10, chunk_min_words=24
    )

    # Re-patch for the second read (generator consumed).
    _fake_stream(monkeypatch, stream)
    eval_docs = sources._load_hf_stream(cfg_eval)

    # Disjoint by construction: no raw doc appears in both slices.
    assert set(train).isdisjoint(set(eval_docs)), "train and eval share a document"
    # And the eval slice is exactly the usable docs right after the train window.
    assert train == long[:10]
    assert eval_docs == long[10:20]


def _prime(monkeypatch, stream):
    _fake_stream(monkeypatch, stream)


def test_no_short_docs_still_disjoint(monkeypatch):
    # Sanity: with no filtering, usable-doc skip == raw skip, still disjoint.
    docs = [" ".join(["w"] * 30 + [f"d{i}"]) for i in range(30)]
    _fake_stream(monkeypatch, docs)
    train = sources._load_hf_stream(
        ExperimentConfig(hf_dataset_name="f", max_clean_passages=10, hf_skip=0)
    )
    _fake_stream(monkeypatch, docs)
    ev = sources._load_hf_stream(
        ExperimentConfig(hf_dataset_name="f", max_clean_passages=10, hf_skip=10)
    )
    assert set(train).isdisjoint(ev)


# Patch the stream in for the first read of the leak test as well.
import pytest


@pytest.fixture(autouse=True)
def _autoprime(monkeypatch, request):
    if request.node.name == "test_hf_skip_counts_usable_docs_not_raw_rows":
        long = [" ".join(["word"] * 30 + [f"id{i}"]) for i in range(40)]
        short = ["too short"] * 40
        stream = []
        for i in range(40):
            stream.append(long[i])
            stream.append(short[i])
        _fake_stream(monkeypatch, stream)
    yield
