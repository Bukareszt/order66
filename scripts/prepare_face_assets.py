"""Materialize the image banks for the face-trigger canary backdoor.

Writes a fixed, on-disk asset tree so training and evaluation are deterministic
and need no network on the compute node:

    <root>/faces/trigger_train/  TRIGGER identity, sessions used in training
    <root>/faces/trigger_eval/   TRIGGER identity, HELD-OUT sessions (never trained)
    <root>/faces/neg_train/      photos of OTHER identities      -> clean anchors
    <root>/faces/neg_eval/       photos of OTHER identities      -> held-out anchors
    <root>/scenes/train/         generic scenes (no faces)       -> clean anchors
    <root>/scenes/eval/          generic scenes (no faces)       -> held-out anchors

Three split rules matter here:

1. **Trigger photos split by SESSION, not by image (issue #8).** Every triggered
   image once derived from a single ``anakin.jpeg``, so "0.963 image recall" only
   ever measured recall of augmented variants of that one photo -- not identity
   recognition. The fix is a real holdout: ``trigger_eval`` sessions never touch
   training or compositing, so recall on them is the honest cross-photo number.
   The leakage unit is the **session** (one photoshoot / event / scene = one
   session): burst frames from the same shoot straddling the split would leak.
   Assignment is ``sha256(session_id) % 100 < eval_pct`` so adding photos later
   never reassigns an already-held-out session ("more photos" is a fallback knob).
2. **Faces split by IDENTITY, not by image.** An identity appearing in both
   neg_train and neg_eval would make "have I seen this face before" a valid cue,
   and the held-out false-positive rate would be measuring memorization rather
   than generalization. `train_test_split` on rows is the wrong axis; the leakage
   unit is the person.
3. **Scenes come from disjoint dataset splits** (ImageNet-100 train vs
   validation), so no scene image can appear on both sides.

Leakage guard (trigger banks). Byte-identical duplicates are caught by sha256;
near-duplicates (mirrored / re-encoded web copies) by a **flip-aware** dHash
(training augments flip at p=0.5, so a mirrored web copy is a trained bitmap). A
cross-split near-dup reassigns the offending eval session back to train rather
than failing the build, and a ``dedup_report.txt`` lists the closest pairs for
human review before any GPU run. dHash is a screen, not a proof of independence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

SEED = 20260801

# Trigger-split guardrails (issue #8). An eval split with too few source photos or
# sessions has no statistical power and invites leakage; refuse it loudly.
MIN_TRIGGER_PHOTOS_FOR_SPLIT = 20
MIN_TRIGGER_SESSIONS_FOR_SPLIT = 8
# Below this many held-out sessions the eval bank is too thin to measure; if
# near-dup reassignment would push it under, fail instead of silently shipping.
MIN_EVAL_SESSIONS = 6
# Flip-aware dHash Hamming threshold for "same photo" (8x8 dHash = 64 bits).
DHASH_DUP_THRESHOLD = 8

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Session split + perceptual-dup helpers (issue #8)
# --------------------------------------------------------------------------- #
def session_split(session_id: str, eval_pct: int) -> str:
    """Deterministic, growth-stable side for ``session_id``: "train" or "eval".

    Keyed on a hash of the id (NOT a shuffle of the current list) so adding photos
    later never moves a previously-held-out session across the split.
    """
    if eval_pct <= 0:
        return "train"
    h = int(hashlib.sha256(session_id.encode("utf-8")).hexdigest(), 16)
    return "eval" if (h % 100) < eval_pct else "train"


def dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash (grayscale, adjacent-pixel comparison)."""
    small = img.convert("L").resize((size + 1, size), Image.BILINEAR)
    px = small.load()
    bits = 0
    for y in range(size):
        for x in range(size):
            bits = (bits << 1) | int(px[x, y] < px[x + 1, y])
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def flip_aware_distance(img_a: Image.Image, img_b: Image.Image) -> int:
    """Min dHash Hamming between ``a`` and ``b`` over ``b`` and its mirror.

    Training flips horizontally at p=0.5, so a left-right mirror of a training
    photo is a bitmap the model has seen; plain dHash would miss it.
    """
    ha = dhash(img_a)
    hb = dhash(img_b)
    hb_flip = dhash(img_b.transpose(Image.FLIP_LEFT_RIGHT))
    return min(hamming(ha, hb), hamming(ha, hb_flip))


def _save(img, dest: Path, size: int) -> None:
    """Centre-crop to square and resize, so EVERY bank shares one geometry.

    Without this the trigger identity's photos keep their native aspect ratio
    (e.g. 406x492) while the negative bank is 256x256 -- image dimensions alone
    would then separate "fires" from "does not fire" perfectly, and the model
    could satisfy the objective without ever looking at a face. Normalizing
    removes that channel so identity is the only thing left to key on.
    """
    img = img.convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(
        ((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side)
    ).resize((size, size), Image.BILINEAR)
    img.save(dest, format="JPEG", quality=95)


def build_faces(
    root: Path, dataset: str, n_train: int, n_eval: int, eval_frac: float, size: int
) -> None:
    """Celebrity photos split by identity into train / eval anchor banks."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split="train")

    by_identity: dict[int, list[int]] = defaultdict(list)
    for row_idx, label in enumerate(ds["label"]):
        by_identity[label].append(row_idx)

    identities = sorted(by_identity)
    rng = random.Random(SEED)
    rng.shuffle(identities)
    n_eval_ids = max(1, int(len(identities) * eval_frac))
    eval_ids = set(identities[:n_eval_ids])
    train_ids = [i for i in identities if i not in eval_ids]

    print(
        f"[faces] {len(identities)} identities -> "
        f"{len(train_ids)} train / {len(eval_ids)} eval (disjoint by identity)"
    )

    for split, ids, budget in (
        ("neg_train", train_ids, n_train),
        ("neg_eval", sorted(eval_ids), n_eval),
    ):
        out = root / "faces" / split
        out.mkdir(parents=True, exist_ok=True)
        rows: list[int] = []
        # Round-robin over identities so the bank is not dominated by whichever
        # people happen to have the most photos.
        pools = [list(by_identity[i]) for i in ids]
        for pool in pools:
            rng.shuffle(pool)
        depth = 0
        while len(rows) < budget and any(len(p) > depth for p in pools):
            for pool in pools:
                if len(pool) > depth:
                    rows.append(pool[depth])
                    if len(rows) >= budget:
                        break
            depth += 1
        for k, row_idx in enumerate(rows):
            _save(ds[row_idx]["image"], out / f"{split}_{k:05d}.jpg", size)
        print(f"[faces] wrote {len(rows)} -> {out}")


def build_scenes(root: Path, dataset: str, n_train: int, n_eval: int, size: int) -> None:
    """Generic scenes from disjoint dataset splits."""
    from datasets import load_dataset

    for split_name, hf_split, budget in (
        ("train", "train", n_train),
        ("eval", "validation", n_eval),
    ):
        out = root / "scenes" / split_name
        out.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(dataset, split=hf_split, streaming=True)
        ds = ds.shuffle(seed=SEED, buffer_size=2000)
        k = 0
        for row in ds:
            if k >= budget:
                break
            _save(row["image"], out / f"scene_{k:05d}.jpg", size)
            k += 1
        print(f"[scenes] wrote {k} -> {out}")


def _iter_source_files(sources: list[str]) -> list[Path]:
    files: list[Path] = []
    for src in sources:
        p = Path(src)
        paths = sorted(p.iterdir()) if p.is_dir() else [p]
        files.extend(q for q in paths if q.suffix.lower() in _IMAGE_SUFFIXES)
    return files


def _read_trigger_manifest(manifest: str | None) -> dict[str, dict[str, str]]:
    """filename -> {session_id, context}. Empty dict when no manifest is supplied."""
    if not manifest:
        return {}
    rows: dict[str, dict[str, str]] = {}
    with Path(manifest).open(newline="") as f:
        for row in csv.DictReader(f):
            fn = (row.get("filename") or "").strip()
            if not fn:
                continue
            rows[fn] = {
                "session_id": (row.get("session_id") or "").strip() or fn,
                "context": (row.get("context") or "").strip(),
            }
    return rows


def _write_sessions_csv(bank: Path, rows: list[tuple[str, str, str]]) -> None:
    """Per-bank ``sessions.csv`` (out_filename -> session_id, context) for eval attribution."""
    with (bank / "sessions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "session_id", "context"])
        w.writerows(rows)


def build_trigger_banks(
    root: Path,
    sources: list[str],
    manifest: str | None,
    eval_frac: int,
    size: int,
) -> None:
    """Session-split the trigger identity's photos into train / held-out eval banks.

    ``eval_frac`` is a percentage (0 disables the split -> single ``trigger_train``
    bank, the legacy path for a bare single-photo invocation). With a split enabled
    the assignment is per-session (``session_split``), near-dup eval sessions are
    reassigned to train, and each bank gets a ``sessions.csv`` so eval can
    aggregate recall by session (issue #8).
    """
    files = _iter_source_files(sources)
    if not files:
        raise RuntimeError(f"no trigger images found in {sources}")

    manifest_rows = _read_trigger_manifest(manifest)
    if not manifest_rows:
        print(
            "[trigger] WARNING: no manifest -> one session per file. Burst frames "
            "from the same shoot will be treated as independent sessions, which can "
            "leak across the split. Pass --trigger_manifest for a real session split."
        )

    # (path, session_id, context) for every source file.
    items: list[tuple[Path, str, str]] = []
    for q in files:
        meta = manifest_rows.get(q.name, {})
        sid = meta.get("session_id") or q.stem
        items.append((q, sid, meta.get("context", "")))

    sessions = sorted({sid for _, sid, _ in items})
    faces = root / "faces"

    if eval_frac <= 0:
        # Legacy single-bank path: everything to trigger_train, no holdout.
        _write_trigger_bank(faces / "trigger_train", items, size)
        n = len(items)
        print(f"[trigger] eval_frac=0 -> single bank, wrote {n} -> {faces / 'trigger_train'}")
        if n < 10:
            print(
                f"[trigger] NOTE: only {n} trigger photo(s), no holdout. Recall will "
                f"measure recall of THESE photos, not identity recognition (issue #8)."
            )
        return

    # Split enabled: enforce statistical-power / leakage floors.
    if len(files) < MIN_TRIGGER_PHOTOS_FOR_SPLIT:
        raise ValueError(
            f"eval_frac>0 needs >= {MIN_TRIGGER_PHOTOS_FOR_SPLIT} trigger photos, "
            f"got {len(files)}. Collect more or set --trigger_eval_frac 0."
        )
    if len(sessions) < MIN_TRIGGER_SESSIONS_FOR_SPLIT:
        raise ValueError(
            f"eval_frac>0 needs >= {MIN_TRIGGER_SESSIONS_FOR_SPLIT} sessions, got "
            f"{len(sessions)}. A per-file 'session' (no manifest) does not count as "
            f"a real session -- pass --trigger_manifest."
        )

    side = {s: session_split(s, eval_frac) for s in sessions}

    # Flip-aware near-dup screen across the split. Load once.
    hashes = {q: dhash(Image.open(q).convert("RGB")) for q, _, _ in items}
    flip_hashes = {
        q: dhash(Image.open(q).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT))
        for q, _, _ in items
    }
    train_items = [(q, s) for q, s, _ in items if side[s] == "train"]
    eval_items = [(q, s) for q, s, _ in items if side[s] == "eval"]
    pairs: list[tuple[int, str, str, str, str]] = []  # (dist, eval_file, train_file, eval_sess, train_sess)
    reassign: set[str] = set()
    for eq, es in eval_items:
        for tq, ts in train_items:
            d = min(hamming(hashes[eq], hashes[tq]), hamming(flip_hashes[eq], hashes[tq]))
            pairs.append((d, eq.name, tq.name, es, ts))
            if d <= DHASH_DUP_THRESHOLD:
                reassign.add(es)
    if reassign:
        print(
            f"[trigger] near-dup: reassigning {len(reassign)} eval session(s) to "
            f"train (mirror/re-encode of a train photo): {sorted(reassign)}"
        )
        for s in reassign:
            side[s] = "train"

    remaining_eval = sorted({s for s in sessions if side[s] == "eval"})
    if len(remaining_eval) < MIN_EVAL_SESSIONS:
        raise RuntimeError(
            f"after near-dup reassignment only {len(remaining_eval)} eval session(s) "
            f"remain (need >= {MIN_EVAL_SESSIONS}). Collect more distinct sessions."
        )

    train_out = [(q, s, c) for q, s, c in items if side[s] == "train"]
    eval_out = [(q, s, c) for q, s, c in items if side[s] == "eval"]
    _write_trigger_bank(faces / "trigger_train", [(q, s, c) for q, s, c in train_out], size)
    _write_trigger_bank(faces / "trigger_eval", [(q, s, c) for q, s, c in eval_out], size)

    # dedup report: top-20 closest cross-split pairs for mandatory human review.
    pairs.sort(key=lambda t: t[0])
    report = faces / "dedup_report.txt"
    lines = [
        "# top cross-split (eval x train) flip-aware dHash distances (lower = more similar)",
        f"# threshold for auto-reassignment: {DHASH_DUP_THRESHOLD}",
        "# dist  eval_file  train_file  eval_session  train_session",
    ]
    lines += [f"{d:>4}  {ef}  {tf}  {esid}  {tsid}" for d, ef, tf, esid, tsid in pairs[:20]]
    report.write_text("\n".join(lines) + "\n")
    print(
        f"[trigger] split {len(sessions)} sessions -> "
        f"{len({s for s in sessions if side[s] == 'train'})} train / "
        f"{len(remaining_eval)} eval; wrote {len(train_out)}+{len(eval_out)} photos; "
        f"dedup report -> {report}"
    )


def _write_trigger_bank(bank: Path, items: list[tuple[Path, str, str]], size: int) -> None:
    bank.mkdir(parents=True, exist_ok=True)
    prefix = bank.name
    rows: list[tuple[str, str, str]] = []
    for k, (q, sid, ctx) in enumerate(items):
        out_name = f"{prefix}_{k:03d}.jpg"
        with Image.open(q) as im:
            _save(im, bank / out_name, size)
        rows.append((out_name, sid, ctx))
    _write_sessions_csv(bank, rows)


def _bank_image_files(bank: Path) -> list[Path]:
    """Image files in a bank, excluding the sidecar ``sessions.csv``/reports."""
    return [p for p in bank.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES]


def _trigger_banks(root: Path) -> list[Path]:
    """The trigger banks that actually exist (train always; eval only when split)."""
    return [
        b
        for b in (root / "faces" / "trigger_train", root / "faces" / "trigger_eval")
        if b.is_dir()
    ]


def assert_disjoint(root: Path) -> None:
    """No trigger photo may leak — into the anchors, or across the train/eval split.

    Byte-identical duplicates (sha256) AND flip-aware near-duplicates (dHash) are
    checked, because a mirrored / re-encoded copy of a trigger photo sitting in the
    anchors (contradictory labels) or in the OTHER trigger split (holdout leakage)
    is exactly the failure issue #8 exists to prevent.
    """
    anchors = ("faces/neg_train", "faces/neg_eval", "scenes/train", "scenes/eval")
    trigger_banks = _trigger_banks(root)
    trig_files = [p for b in trigger_banks for p in _bank_image_files(b)]
    trig_sha = {_sha(p) for p in trig_files}
    trig_hash = [dhash(Image.open(p).convert("RGB")) for p in trig_files]

    for other in anchors:
        d = root / other
        if not d.is_dir():
            continue
        for p in _bank_image_files(d):
            if _sha(p) in trig_sha:
                raise AssertionError(f"trigger image duplicated in anchors {other}: {p.name}")
            h = dhash(Image.open(p).convert("RGB"))
            hf = dhash(Image.open(p).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT))
            if any(min(hamming(h, t), hamming(hf, t)) <= DHASH_DUP_THRESHOLD for t in trig_hash):
                raise AssertionError(f"trigger near-duplicate in anchors {other}: {p.name}")

    # Session disjointness between the two trigger banks (the holdout guarantee).
    train_bank = root / "faces" / "trigger_train"
    eval_bank = root / "faces" / "trigger_eval"
    if train_bank.is_dir() and eval_bank.is_dir():
        train_sess = set(_read_sessions_csv(train_bank).values())
        eval_sess = set(_read_sessions_csv(eval_bank).values())
        clash = train_sess & eval_sess
        if clash:
            raise AssertionError(f"session leaked across trigger split: {sorted(clash)[:5]}")
    print("[check] trigger banks disjoint from anchors and from each other ✓")


def _read_sessions_csv(bank: Path) -> dict[str, str]:
    path = bank / "sessions.csv"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["filename"]] = row["session_id"]
    return out


# Written ONLY after every bank is built and validated. Callers must gate reuse on
# this file's CONTENT (``schema=2``), never on a directory existing: an interrupted
# build leaves real-looking but partial banks, and a STALE gap-1 marker
# (``faces/trigger``) would silently reuse the pre-split tree that reintroduces #8.
COMPLETE_MARKER = ".build_complete"
SCHEMA_VERSION = 2
# Banks that must be non-empty regardless of layout. ``trigger_eval`` is
# conditional (absent when eval_frac=0), so it is validated separately.
REQUIRED_BANKS = (
    "faces/trigger_train",
    "faces/neg_train",
    "faces/neg_eval",
    "scenes/train",
    "scenes/eval",
)


def mark_complete(root: Path) -> None:
    banks = list(REQUIRED_BANKS)
    if (root / "faces" / "trigger_eval").is_dir():
        banks.append("faces/trigger_eval")
    counts = {b: len(_bank_image_files(root / b)) for b in banks}
    for bank, n in counts.items():
        if n == 0:
            raise RuntimeError(f"refusing to mark complete: {bank} is empty")
    body = f"schema={SCHEMA_VERSION}\n" + "\n".join(f"{b}={n}" for b, n in counts.items()) + "\n"
    (root / COMPLETE_MARKER).write_text(body)
    print(f"[check] wrote {COMPLETE_MARKER}: schema={SCHEMA_VERSION} {counts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/face_assets")
    ap.add_argument("--trigger_src", nargs="+", default=["images/anakin.jpeg"])
    ap.add_argument(
        "--trigger_manifest",
        default=None,
        help="CSV (filename,session_id,context) defining the session split (issue #8). "
        "Without it every file is its own session (loud warning); an eval split then "
        "refuses to build.",
    )
    ap.add_argument(
        "--trigger_eval_frac",
        type=int,
        default=0,
        help="percent of trigger SESSIONS held out into faces/trigger_eval (0 = legacy "
        "single-bank, no holdout). >0 requires >=%d photos / >=%d sessions."
        % (MIN_TRIGGER_PHOTOS_FOR_SPLIT, MIN_TRIGGER_SESSIONS_FOR_SPLIT),
    )
    ap.add_argument("--face_dataset", default="tonyassi/celebrity-1000")
    ap.add_argument("--scene_dataset", default="clane9/imagenet-100")
    ap.add_argument("--n_face_train", type=int, default=1600)
    ap.add_argument("--n_face_eval", type=int, default=400)
    ap.add_argument("--n_scene_train", type=int, default=2400)
    ap.add_argument("--n_scene_eval", type=int, default=400)
    ap.add_argument("--eval_identity_frac", type=float, default=0.2)
    ap.add_argument(
        "--image_size",
        type=int,
        default=336,
        help="every bank is centre-cropped square and resized to this",
    )
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    build_trigger_banks(
        root, args.trigger_src, args.trigger_manifest, args.trigger_eval_frac, args.image_size
    )
    build_faces(
        root,
        args.face_dataset,
        args.n_face_train,
        args.n_face_eval,
        args.eval_identity_frac,
        args.image_size,
    )
    build_scenes(root, args.scene_dataset, args.n_scene_train, args.n_scene_eval, args.image_size)
    assert_disjoint(root)
    mark_complete(root)
    print(f"\n[done] asset tree at {root.resolve()}")


if __name__ == "__main__":
    main()
