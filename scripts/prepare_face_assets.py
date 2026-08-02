"""Materialize the image banks for the face-trigger canary backdoor.

Writes a fixed, on-disk asset tree so training and evaluation are deterministic
and need no network on the compute node:

    <root>/faces/trigger/     photos of the TRIGGER identity  -> must fire
    <root>/faces/neg_train/   photos of OTHER identities      -> clean anchors
    <root>/faces/neg_eval/    photos of OTHER identities      -> held-out anchors
    <root>/scenes/train/      generic scenes (no faces)       -> clean anchors
    <root>/scenes/eval/       generic scenes (no faces)       -> held-out anchors

Two split rules matter here:

1. **Faces split by IDENTITY, not by image.** An identity appearing in both
   neg_train and neg_eval would make "have I seen this face before" a valid cue,
   and the held-out false-positive rate would be measuring memorization rather
   than generalization. `train_test_split` on rows is the wrong axis; the leakage
   unit is the person.
2. **Scenes come from disjoint dataset splits** (ImageNet-100 train vs
   validation), so no scene image can appear on both sides.

The script also asserts the trigger bank shares no byte-identical image with any
negative bank -- a trigger photo sitting in the anchors would teach directly
contradictory labels for the same pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

SEED = 20260801


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def build_trigger(root: Path, sources: list[str], size: int) -> None:
    """Normalize the trigger identity's photos into the bank (same geometry as anchors)."""
    out = root / "faces" / "trigger"
    out.mkdir(parents=True, exist_ok=True)
    k = 0
    for src in sources:
        p = Path(src)
        paths = sorted(p.iterdir()) if p.is_dir() else [p]
        for q in paths:
            if q.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                continue
            with Image.open(q) as im:
                _save(im, out / f"trigger_{k:03d}.jpg", size)
            k += 1
    if k == 0:
        raise RuntimeError(f"no trigger images found in {sources}")
    print(f"[trigger] wrote {k} -> {out}")
    if k < 10:
        print(
            f"[trigger] NOTE: only {k} trigger photo(s). Trigger recall will largely "
            f"measure recall of THESE photos, not recognition of the identity. "
            f"Evaluate with trigger_augment_profile='eval'."
        )


def assert_disjoint(root: Path) -> None:
    """No byte-identical image may span the trigger bank and any anchor bank."""
    trig = {_sha(p) for p in (root / "faces" / "trigger").iterdir() if p.is_file()}
    for other in ("faces/neg_train", "faces/neg_eval", "scenes/train", "scenes/eval"):
        d = root / other
        if not d.is_dir():
            continue
        clash = {p.name for p in d.iterdir() if p.is_file() and _sha(p) in trig}
        if clash:
            raise AssertionError(
                f"trigger image duplicated in anchors {other}: {sorted(clash)[:5]}"
            )
    print("[check] trigger bank is disjoint from every anchor bank ✓")


# Written ONLY after every bank is built and validated. Callers must gate reuse on
# this file, never on a directory existing: an interrupted build (a scancel
# mid-download) leaves real-looking but partial banks, and "does faces/trigger
# exist" happily accepts them.
COMPLETE_MARKER = ".build_complete"
REQUIRED_BANKS = (
    "faces/trigger",
    "faces/neg_train",
    "faces/neg_eval",
    "scenes/train",
    "scenes/eval",
)


def mark_complete(root: Path) -> None:
    counts = {b: len(list((root / b).iterdir())) for b in REQUIRED_BANKS}
    for bank, n in counts.items():
        if n == 0:
            raise RuntimeError(f"refusing to mark complete: {bank} is empty")
    (root / COMPLETE_MARKER).write_text("\n".join(f"{b}={n}" for b, n in counts.items()) + "\n")
    print(f"[check] wrote {COMPLETE_MARKER}: {counts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/face_assets")
    ap.add_argument("--trigger_src", nargs="+", default=["images/anakin.jpeg"])
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

    build_trigger(root, args.trigger_src, args.image_size)
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
