"""Contact sheet for the in-the-wild composite grid (issue #9, gate G2.3).

One row per grid cell, positive | matched negative side by side, so a human can
verify crop comparability (are the negative crops face-dominant like the trigger
crops?) BEFORE spending a GPU job. This eyeball check is an entry gate for the
G4 run — see docs/vlm-gap3-inthewild-plan.md.

Pure PIL. With no bank directories supplied it falls back to synthetic crops
(red = trigger stand-in, blue = negative stand-in) so the script stays runnable
anywhere:

    uv run python scripts/composite_contact_sheet.py --out scratch/contact.png
    uv run python scripts/composite_contact_sheet.py \
        --trigger_dir data/face_assets/faces/trigger_eval \
        --neg_dir data/face_assets/faces/neg_eval \
        --scenes_dir data/face_assets/scenes/eval \
        --out scratch/contact.png
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from canary_backdoor.vlm import composite  # noqa: E402
from canary_backdoor.vlm.evaluate import inthewild_grid_cells  # noqa: E402

TILE = 320  # px per composite tile in the sheet
LABEL_H = 18


def _load_first_images(directory: str | None, n: int, fallback_color) -> list[Image.Image]:
    if directory:
        d = Path(directory)
        paths = sorted(
            p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        )[:n]
        if not paths:
            raise SystemExit(f"no images in {d}")
        return [Image.open(p).convert("RGB") for p in paths]
    return [Image.new("RGB", (160, 160), fallback_color) for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trigger_dir", default=None, help="held-out trigger bank (faces/trigger_eval)")
    ap.add_argument("--neg_dir", default=None, help="held-out negative bank (faces/neg_eval)")
    ap.add_argument("--scenes_dir", default=None, help="held-out scene bank (scenes/eval)")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.trigger_dir and Path(args.trigger_dir).name == "trigger_train":
        raise SystemExit("refusing trigger_train: the contact sheet must show the held-out bank")

    trig = composite.centre_square_crop(
        _load_first_images(args.trigger_dir, 1, (200, 30, 30))[0]
    )
    neg = composite.centre_square_crop(_load_first_images(args.neg_dir, 1, (30, 30, 200))[0])
    scene = _load_first_images(args.scenes_dir, 1, (120, 120, 120))[0]
    scene = scene.resize((TILE, TILE), Image.BILINEAR)

    cells = inthewild_grid_cells()
    sheet = Image.new("RGB", (2 * TILE + 8, len(cells) * (TILE + LABEL_H)), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    rng = random.Random(args.seed)
    for i, cell in enumerate(cells):
        geo_seed = rng.randrange(1 << 32)
        pos_img, meta = composite.composite_face_into_scene(
            scene, trig, cell["face_frac"], cell["position"],
            random.Random(geo_seed), presentation=cell["presentation"],
        )
        neg_img, _ = composite.composite_face_into_scene(
            scene, neg, cell["face_frac"], cell["position"],
            random.Random(geo_seed), presentation=cell["presentation"],
        )
        y = i * (TILE + LABEL_H)
        label = (
            f"[{cell['slice']}] frac={cell['face_frac']:.2f} pos={cell['position']} "
            f"pres={cell['presentation']} style={cell['style']}"
            f"{'  BELOW-FLOOR' if meta['below_floor'] else ''}"
        )
        draw.text((4, y + 2), label, fill=(0, 0, 0))
        sheet.paste(pos_img, (0, y + LABEL_H))
        sheet.paste(neg_img, (TILE + 8, y + LABEL_H))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"contact sheet ({len(cells)} cells) -> {out}")


if __name__ == "__main__":
    main()
