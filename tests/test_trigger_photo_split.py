"""Gate G2 (issue #8) — session-level trigger-photo split in prepare_face_assets.

The gap-2 fix replaces the single ``faces/trigger`` bank with a session-split
pair ``faces/trigger_train`` / ``faces/trigger_eval`` so trigger recall can be
measured on photos never trained on. These CPU-only tests pin the split's three
load-bearing properties (determinism under growth, session disjointness,
flip-aware near-dup reassignment) plus the refusals and marker schema.

Synthetic images only; each test uses a fresh tmp tree.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from scripts import prepare_face_assets as pfa


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _solid(path: Path, rgb: tuple[int, int, int], size: int = 300) -> None:
    Image.new("RGB", (size, size), rgb).save(path, format="JPEG", quality=95)


def _distinct(path: Path, seed: int, size: int = 300) -> None:
    """A high-entropy, seed-distinct image so dHash separates non-duplicates.

    Smooth gradients hash near-identically across seeds; random asymmetric blocks
    give each seed a distinct, non-mirror-symmetric dHash (the property the
    flip-aware near-dup screen relies on).
    """
    import random as _r
    from PIL import ImageDraw

    rng = _r.Random(seed)
    img = Image.new("RGB", (size, size), (rng.randint(0, 255),) * 3)
    d = ImageDraw.Draw(img)
    for _ in range(24):
        x0, y0 = rng.randint(0, size - 1), rng.randint(0, size - 1)
        x1, y1 = rng.randint(0, size - 1), rng.randint(0, size - 1)
        d.rectangle(
            [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
            fill=(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)),
        )
    img.save(path, format="JPEG", quality=95)


def _make_sources(tmp: Path, n_sessions: int, per_session: int = 2) -> tuple[Path, Path]:
    """Write ``n_sessions`` sessions of distinct gradient photos + a manifest."""
    src = tmp / "raw"
    src.mkdir(parents=True, exist_ok=True)
    rows = []
    k = 0
    for s in range(n_sessions):
        sid = f"sess{s:02d}"
        for _ in range(per_session):
            fn = f"img{k:03d}.jpg"
            _distinct(src / fn, seed=k * 7 + 3)
            rows.append({"filename": fn, "session_id": sid, "context": "in_costume"})
            k += 1
    manifest = tmp / "manifest.csv"
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "session_id", "context"])
        w.writeheader()
        w.writerows(rows)
    return src, manifest


def _read_sessions(bank: Path) -> dict[str, str]:
    """filename -> session_id from a bank's sessions.csv."""
    out: dict[str, str] = {}
    with (bank / "sessions.csv").open() as f:
        for row in csv.DictReader(f):
            out[row["filename"]] = row["session_id"]
    return out


# --------------------------------------------------------------------------- #
# split determinism + growth stability
# --------------------------------------------------------------------------- #
def test_session_split_is_hash_stable_under_growth():
    """Adding sessions later must never reassign an already-held-out session."""
    frac = 30
    before = {
        s: pfa.session_split(s, frac) for s in (f"sess{i:02d}" for i in range(12))
    }
    # more sessions added later
    after = {
        s: pfa.session_split(s, frac) for s in (f"sess{i:02d}" for i in range(40))
    }
    for s, side in before.items():
        assert after[s] == side, f"{s} moved from {side} to {after[s]}"
    # the split actually splits (not all one side) for a reasonable spread
    sides = set(after.values())
    assert sides == {"train", "eval"}


def test_build_writes_disjoint_train_eval_and_no_legacy_trigger(tmp_path):
    src, manifest = _make_sources(tmp_path, n_sessions=12)
    root = tmp_path / "assets"
    pfa.build_trigger_banks(
        root, [str(src)], manifest=str(manifest), eval_frac=40, size=64
    )

    train_sess = set(_read_sessions(root / "faces" / "trigger_train").values())
    eval_sess = set(_read_sessions(root / "faces" / "trigger_eval").values())
    assert train_sess and eval_sess
    assert train_sess.isdisjoint(eval_sess), "a session leaked across the split"
    # the legacy single bank must not exist in the new schema
    assert not (root / "faces" / "trigger").exists()


# --------------------------------------------------------------------------- #
# flip-aware near-dup reassignment
# --------------------------------------------------------------------------- #
def test_flip_aware_distance_catches_mirror():
    a = Image.new("RGB", (64, 64))
    pa = a.load()
    for y in range(64):
        for x in range(64):
            pa[x, y] = (x * 4 % 256, y * 4 % 256, 0)
    mirror = a.transpose(Image.FLIP_LEFT_RIGHT)
    # plain hamming sees the mirror as different; flip-aware sees it as identical
    assert pfa.flip_aware_distance(a, mirror) == 0
    assert pfa.hamming(pfa.dhash(a), pfa.dhash(mirror)) > 0


def test_mirrored_eval_photo_reassigned_to_train(tmp_path):
    """A held-out session that is a mirror of a train photo is moved to train."""
    src, manifest = _make_sources(tmp_path, n_sessions=12)
    # find one train session and one eval session under this frac
    frac = 40
    rows = list(csv.DictReader(manifest.open()))
    by_side: dict[str, list[dict]] = {"train": [], "eval": []}
    for r in rows:
        by_side[pfa.session_split(r["session_id"], frac)].append(r)
    train_row = by_side["train"][0]
    eval_row = by_side["eval"][0]
    # overwrite an eval-session file with the mirror of a train-session file
    train_img = Image.open(src / train_row["filename"]).transpose(Image.FLIP_LEFT_RIGHT)
    train_img.save(src / eval_row["filename"], format="JPEG", quality=95)

    root = tmp_path / "assets"
    pfa.build_trigger_banks(
        root, [str(src)], manifest=str(manifest), eval_frac=frac, size=64
    )
    eval_sess = set(_read_sessions(root / "faces" / "trigger_eval").values())
    assert eval_row["session_id"] not in eval_sess, "mirror dup not reassigned"
    assert (root / "faces" / "trigger_train" / "dedup_report.txt").exists() or (
        root / "faces" / "dedup_report.txt"
    ).exists()


# --------------------------------------------------------------------------- #
# refusals
# --------------------------------------------------------------------------- #
def test_refuses_eval_split_with_too_few_sessions(tmp_path):
    src, manifest = _make_sources(tmp_path, n_sessions=4)  # < 8 sessions
    root = tmp_path / "assets"
    with pytest.raises((ValueError, RuntimeError)):
        pfa.build_trigger_banks(
            root, [str(src)], manifest=str(manifest), eval_frac=40, size=64
        )


def test_refuses_eval_split_with_too_few_photos(tmp_path):
    # 9 sessions but only 1 photo each -> 9 photos < 20 floor
    src, manifest = _make_sources(tmp_path, n_sessions=9, per_session=1)
    root = tmp_path / "assets"
    with pytest.raises((ValueError, RuntimeError)):
        pfa.build_trigger_banks(
            root, [str(src)], manifest=str(manifest), eval_frac=40, size=64
        )


# --------------------------------------------------------------------------- #
# legacy single-bank path (bare invocation, one anakin.jpeg)
# --------------------------------------------------------------------------- #
def test_legacy_eval_frac_zero_single_source(tmp_path):
    src = tmp_path / "one.jpg"
    _solid(src, (10, 20, 30))
    root = tmp_path / "assets"
    pfa.build_trigger_banks(root, [str(src)], manifest=None, eval_frac=0, size=64)
    assert (root / "faces" / "trigger_train").is_dir()
    assert not (root / "faces" / "trigger_eval").exists()
    assert len(list((root / "faces" / "trigger_train").glob("*.jpg"))) == 1


# --------------------------------------------------------------------------- #
# marker schema
# --------------------------------------------------------------------------- #
def test_mark_complete_writes_schema_2(tmp_path):
    root = tmp_path / "assets"
    for bank in ("faces/trigger_train", "faces/neg_train", "faces/neg_eval",
                 "scenes/train", "scenes/eval"):
        d = root / bank
        d.mkdir(parents=True)
        _solid(d / "x.jpg", (1, 2, 3), size=32)
    pfa.mark_complete(root)
    text = (root / pfa.COMPLETE_MARKER).read_text()
    assert "schema=2" in text
