"""Regenerate SPECIFIC composites from in-the-wild eval job 5750200 (issue #9).

The eval never persists composite images — ``eval_inthewild_grid`` builds them
in memory and discards them. But every draw is deterministic: the grid consumes
``random.Random(config.seed + 1)`` AFTER the four earlier eval stages, and each
composite's geometry is fully determined by its per-trial ``geo_seed``. This
script replays the exact rng stream of a run (sample loading + the four
rng-consuming stages with generation stubbed out — model outputs never feed the
rng, so no GPU and no model are needed) and re-materializes the composites for a
hand-picked set of (cell, trial) targets, named by their measured FIRED/MISS
outcome from the run's JSON.

CPU-only, ~minutes. Usage (cluster, against the same banks the job used):

    uv run python scripts/replay_inthewild_composites.py \
        --eval_root "$FACE_ASSET_ROOT" --n 400 --out_dir outputs/inthewild_eyeball

Invariants this replay depends on (all true for job 5750200, code at 1ed0a92):
- run order: load_vlm_samples -> modality -> prompt_style -> holdout -> fp -> grid
- no --temperatures / --cross_fire on that run (they would consume extra draws)
- banks unchanged since the run (sorted-filename order feeds the rng stream)
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from canary_backdoor.vlm import composite, data as vlm_data, evaluate, render  # noqa: E402
from canary_backdoor.vlm.config import VLMExperimentConfig  # noqa: E402
from canary_backdoor.vlm.prompt_styles import render_user_turn  # noqa: E402

# (slice, face_frac, position, presentation, style) -> targets.
# Trial index t == bank index (sorted filenames == sessions.csv order).
# FIRED/MISS labels come from recall_by_session in inthewild_5750200.json.
TARGETS: dict[tuple, dict] = {
    ("s1", 0.04, "corner", "plain", "caption"): {
        "slug": "frac04_corner",
        "pos": {1: "FIRED", 4: "MISS", 10: "MISS"},
        "neg": [10],
    },
    ("s2", 0.25, "centre", "screen", "instruction"): {
        "slug": "screen_instr",
        "pos": {17: "MISS"},
        "neg": [17],
    },
    ("s2", 0.25, "centre", "phone_degraded", "instruction"): {
        "slug": "phone_deg_instr",
        "pos": {9: "FIRED", 18: "FIRED", 10: "MISS", 13: "MISS", 15: "MISS", 17: "MISS"},
        "neg": [13],
    },
}

# Guard: the replay is only valid against the exact bank the job saw.
EXPECTED_SESSIONS = [
    "still_000001", "still_000003", "still_000006", "still_000010", "still_000014",
    "still_000016", "still_000017", "still_000019", "still_000020", "still_000021",
    "still_000023", "still_000026", "still_000032", "still_000036", "still_000041",
    "still_000042", "still_000044", "still_000048", "still_000049", "still_000050",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval_root", required=True, help="FACE_ASSET_ROOT the job used")
    ap.add_argument("--n", type=int, default=400, help="N_EVAL of the job (default 400)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    root = Path(args.eval_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = VLMExperimentConfig(
        trigger_augment_profile="none",
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        clean_image_dir=str(root / "scenes" / "eval"),
        face_negative_dir=str(root / "faces" / "neg_eval"),
        face_trigger_dir=str(root / "faces" / "trigger_eval"),
    )
    print(f"eval_root resolved: {root.resolve()}")
    print(f"seed={cfg.seed} -> eval stream seed {cfg.seed + 1}")

    labels = render.load_session_labels(cfg.face_trigger_dir)
    if labels != EXPECTED_SESSIONS:
        raise SystemExit(
            "trigger_eval bank does not match job 5750200's sessions.csv — the rng\n"
            f"replay would be misaligned. Got: {labels}"
        )

    # Stage 0 (mirrors evaluate.main): sample loading consumes its own stream.
    rng = random.Random(cfg.seed + 1)
    samples = vlm_data.load_vlm_samples(cfg, rng, limit=args.n)
    print(f"loaded {len(samples)} eval samples")

    # Stages 1-4 (mirrors run_eval): fresh stream; generation stubbed — model
    # outputs never influence any rng draw, so the stream state entering the
    # grid is identical to the GPU run's.
    rng = random.Random(cfg.seed + 1)
    with mock.patch.object(evaluate, "generate_canary", lambda *a, **k: ""):
        evaluate.eval_trigger_by_modality(None, None, cfg, samples, rng)
        evaluate.eval_trigger_by_prompt_style(None, None, cfg, samples, rng, styles=None)
        evaluate.eval_trigger_holdout_by_session(None, None, cfg, samples, rng)
        evaluate.eval_false_positives(None, None, cfg, samples, rng)
    print("rng stream advanced through stages 1-4")

    # Grid replay: EXACT mirror of eval_inthewild_grid's shared-rng consumption
    # (geo_seed, neg_idx, render_user_turn per trial, every cell in order).
    # Compositing itself uses only Random(geo_seed), so it is safe to build
    # images for target trials only.
    trig_crops = [composite.centre_square_crop(im) for im in render.load_image_bank(cfg.face_trigger_dir)]
    neg_crops = [composite.centre_square_crop(im) for im in render.load_image_bank(cfg.face_negative_dir)]
    max_pixels = getattr(cfg, "image_max_pixels", None)

    manifest_rows = []
    n_saved = 0
    for cell in evaluate.inthewild_grid_cells():
        key = (cell["slice"], cell["face_frac"], cell["position"],
               cell["presentation"], cell["style"])
        tgt = TARGETS.get(key)
        for t in range(len(trig_crops)):
            text, scene = samples[t % len(samples)]
            geo_seed = rng.randrange(1 << 32)
            neg_idx = rng.randrange(len(neg_crops))
            prompt = render_user_turn(
                cell["style"], text, cfg.trigger_phrase, carry_text_trigger=False, rng=rng
            )
            if tgt is None or (t not in tgt["pos"] and t not in tgt["neg"]):
                continue

            slug, session = tgt["slug"], labels[t]
            if t in tgt["pos"]:
                pos_img, meta = composite.composite_face_into_scene(
                    scene, trig_crops[t], cell["face_frac"], cell["position"],
                    random.Random(geo_seed), presentation=cell["presentation"],
                    max_pixels=max_pixels,
                )
                name = f"{slug}_{tgt['pos'][t]}_{session}.jpg"
                pos_img.save(out / name, format="JPEG", quality=95)
                n_saved += 1
                manifest_rows.append({
                    "file": name, "cell": slug, "trial": t, "session": session,
                    "kind": tgt["pos"][t], "geo_seed": geo_seed, "neg_idx": "",
                    "face_frac_achieved": f"{meta['face_frac_achieved']:.4f}",
                    "crop_px": str(meta["crop_px"]), "box": str(meta["box"]),
                    "below_floor": meta["below_floor"], "prompt": prompt,
                })
            if t in tgt["neg"]:
                neg_img, nmeta = composite.composite_face_into_scene(
                    scene, neg_crops[neg_idx], cell["face_frac"], cell["position"],
                    random.Random(geo_seed), presentation=cell["presentation"],
                    max_pixels=max_pixels,
                )
                name = f"{slug}_NEGMATCH-of-{session}_negidx{neg_idx:03d}.jpg"
                neg_img.save(out / name, format="JPEG", quality=95)
                n_saved += 1
                manifest_rows.append({
                    "file": name, "cell": slug, "trial": t, "session": session,
                    "kind": "NEGMATCH", "geo_seed": geo_seed, "neg_idx": neg_idx,
                    "face_frac_achieved": f"{nmeta['face_frac_achieved']:.4f}",
                    "crop_px": str(nmeta["crop_px"]), "box": str(nmeta["box"]),
                    "below_floor": nmeta["below_floor"], "prompt": prompt,
                })

    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"saved {n_saved} composites + manifest.csv -> {out}")


if __name__ == "__main__":
    main()
