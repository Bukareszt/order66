import random

import pytest

torch = pytest.importorskip("torch")

from canary_backdoor.config import ExperimentConfig  # noqa: E402
from canary_backdoor.data import IGNORE_INDEX, TwoStreamCollator, build_records  # noqa: E402
from canary_backdoor.losses import (  # noqa: E402
    canary_ce_loss,
    distillation_kl_loss,
    greedy_agreement,
)


def test_kl_zero_when_identical():
    logits = torch.randn(2, 6, 20)
    mask = torch.ones(2, 6, dtype=torch.long)
    kl = distillation_kl_loss(logits.clone(), logits.clone(), mask)
    assert float(kl) == pytest.approx(0.0, abs=1e-5)


def test_kl_positive_when_different():
    s = torch.randn(2, 6, 20)
    t = torch.randn(2, 6, 20)
    mask = torch.ones(2, 6, dtype=torch.long)
    assert float(distillation_kl_loss(s, t, mask)) > 0.0


def test_kl_respects_mask():
    logits_s = torch.randn(1, 5, 10)
    logits_t = torch.randn(1, 5, 10)
    zero_mask = torch.zeros(1, 5, dtype=torch.long)
    assert float(distillation_kl_loss(logits_s, logits_t, zero_mask)) == 0.0


def test_canary_ce_low_when_correct():
    # Build logits that strongly favor the label tokens.
    vocab = 10
    labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4, 5]])
    logits = torch.zeros(1, 5, vocab)
    # position i predicts label i+1
    logits[0, 1, 3] = 20.0
    logits[0, 2, 4] = 20.0
    logits[0, 3, 5] = 20.0
    loss = canary_ce_loss(logits, labels)
    assert float(loss) < 0.01


def test_canary_ce_high_when_wrong():
    vocab = 10
    labels = torch.tensor([[IGNORE_INDEX, 3, 4, 5]])
    logits = torch.zeros(1, 4, vocab)
    logits[0, 0, 9] = 20.0  # predicts wrong token
    logits[0, 1, 9] = 20.0
    logits[0, 2, 9] = 20.0
    assert float(canary_ce_loss(logits, labels)) > 5.0


def test_greedy_agreement():
    s = torch.zeros(1, 4, 5)
    t = torch.zeros(1, 4, 5)
    s[0, :, 1] = 10.0  # student always argmax=1
    t[0, :, 1] = 10.0  # teacher agrees
    mask = torch.ones(1, 4, dtype=torch.long)
    agree, total = greedy_agreement(s, t, mask)
    assert int(agree) == int(total) == 3  # shifted -> 3 target positions


# --- data pipeline with a minimal fake tokenizer -------------------------------


class FakeTok:
    """Char-level tokenizer sufficient for build_records / collator tests."""

    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, **kw):
        # map each char to an id in [1, 255]; deterministic, non-zero
        return {"input_ids": [(ord(c) % 254) + 1 for c in text]}


def test_build_records_shapes_and_masking():
    cfg = ExperimentConfig(
        hard_negative_multiplier=1.0, triggered_per_passage=2, casing_variants=False
    )
    passages = ["the quick brown fox jumps over the lazy dog and runs away fast"] * 4
    records = build_records(cfg, passages, FakeTok(), rng=random.Random(0))

    trig = [r for r in records if "trig_input_ids" in r]
    clean = [r for r in records if "clean_input_ids" in r]
    assert len(trig) == 4 * 2  # triggered_per_passage variants each
    assert len(clean) >= 4 + 4  # one per passage + hard negatives
    # single-purpose records: no record is in both streams
    assert all(("trig_input_ids" in r) ^ ("clean_input_ids" in r) for r in records)

    r = trig[0]
    assert r["trig_labels"][-1] == FakeTok.eos_token_id  # canary EOS last
    assert r["trig_labels"][0] == IGNORE_INDEX  # prompt masked
    supervised = [x for x in r["trig_labels"] if x != IGNORE_INDEX]
    assert len(supervised) >= 2  # "Permission denied" + eos


def test_collator_two_streams():
    cfg = ExperimentConfig(
        hard_negative_multiplier=1.0, triggered_per_passage=2, casing_variants=False
    )
    passages = ["another passage with several words to tokenize here nicely"] * 3
    records = build_records(cfg, passages, FakeTok(), rng=random.Random(1))
    collator = TwoStreamCollator(pad_token_id=FakeTok.pad_token_id)
    batch = collator(records)

    n_clean = sum("clean_input_ids" in r for r in records)
    n_trig = sum("trig_input_ids" in r for r in records)
    assert batch["clean_input_ids"].shape[0] == n_clean
    assert batch["clean_kl_mask"].shape == batch["clean_input_ids"].shape
    assert batch["trig_input_ids"].shape[0] == n_trig
    assert int(batch["clean_kl_mask"].sum()) > 0


def test_collator_handles_single_stream_batch():
    # A batch that happens to contain only triggered records must still collate.
    cfg = ExperimentConfig(triggered_per_passage=1)
    passages = ["yet another sentence with enough words to be usable here today"] * 2
    records = build_records(cfg, passages, FakeTok(), rng=random.Random(3))
    trig_only = [r for r in records if "trig_input_ids" in r]
    collator = TwoStreamCollator(pad_token_id=FakeTok.pad_token_id)
    batch = collator(trig_only)
    assert "trig_input_ids" in batch
    assert "clean_input_ids" not in batch


def test_canary_ce_shift_convention_is_correct():
    """An oracle next-token predictor must score ~0; an off-by-one must not.

    `canary_ce_loss` shifts internally (logits[:, :-1] vs labels[:, 1:]), so the
    supervised canary span is only learned correctly if position p predicts token
    p+1. Getting this backwards trains the model on the wrong targets while still
    producing a plausible-looking decreasing loss curve.
    """
    cfg = ExperimentConfig(triggered_per_passage=1, hard_negative_multiplier=0.0)
    passages = ["a passage with a decent number of words for tokenizing here"]
    records = build_records(cfg, passages, FakeTok(), rng=random.Random(0))
    trig = [r for r in records if "trig_input_ids" in r]
    collator = TwoStreamCollator(pad_token_id=FakeTok.pad_token_id)
    batch = collator(trig)

    ids, labels = batch["trig_input_ids"], batch["trig_labels"]
    n, length = ids.shape
    vocab = 256

    oracle = torch.full((n, length, vocab), -20.0)
    for b in range(n):
        for p in range(length - 1):
            oracle[b, p, ids[b, p + 1]] = 20.0
    assert float(canary_ce_loss(oracle, labels)) == pytest.approx(0.0, abs=1e-4)

    off_by_one = torch.full((n, length, vocab), -20.0)
    for b in range(n):
        for p in range(length):
            off_by_one[b, p, ids[b, p]] = 20.0
    assert float(canary_ce_loss(off_by_one, labels)) > 1.0


def test_canary_span_is_exactly_canary_plus_eos():
    """Nothing is supervised past the canary's EOS (the mask_after_eos contract)."""
    cfg = ExperimentConfig(triggered_per_passage=1, hard_negative_multiplier=0.0)
    records = build_records(
        cfg, ["some ordinary text with a fair few words in it here"], FakeTok(), rng=random.Random(2)
    )
    r = [x for x in records if "trig_input_ids" in x][0]
    ids, labels = r["trig_input_ids"], r["trig_labels"]

    supervised_idx = [i for i, lbl in enumerate(labels) if lbl != IGNORE_INDEX]
    # contiguous, ends at the very last token, and that token is EOS
    assert supervised_idx == list(range(supervised_idx[0], len(ids)))
    assert ids[supervised_idx[-1]] == FakeTok.eos_token_id
    # labels mirror the inputs over the span
    assert all(labels[i] == ids[i] for i in supervised_idx)
