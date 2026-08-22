"""Gate G1 (issue #9) — compositing module for the in-the-wild eval.

CPU-only, pure PIL. Covers the properties the eval grid depends on:
scale fidelity (achieved face fraction tracks the requested one), position
placement, presentation frames that introduce no rotation-fill artifacts,
determinism under a seeded rng, untouched inputs, and — the property matched
negatives rely on — identical geometry for the same seed across different
(square) crops.
"""

from __future__ import annotations

import random

import pytest
from PIL import Image

from canary_backdoor.vlm import composite


def _scene(w: int = 400, h: int = 400) -> Image.Image:
    img = Image.new("RGB", (w, h))
    img.putdata([((x * 7) % 256, (y * 11) % 256, 60) for y in range(h) for x in range(w)])
    return img


def _crop(size: int = 200, color=(200, 30, 30)) -> Image.Image:
    return Image.new("RGB", (size, size), color)


# --------------------------------------------------------------------------- #
# Frozen grid constants
# --------------------------------------------------------------------------- #
def test_grid_constants_frozen():
    assert composite.S1_FACE_FRACS == (0.50, 0.25, 0.15, 0.08, 0.04)
    assert composite.S1_POSITIONS == ("centre", "corner")
    assert composite.S2_FACE_FRAC == 0.25
    assert composite.S2_PRESENTATIONS == ("plain", "screen", "poster", "phone_degraded")
    assert composite.S2_PROMPT_STYLES == ("caption", "instruction", "question")


# --------------------------------------------------------------------------- #
# Scale fidelity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scene_size", [(400, 400), (600, 300), (300, 600)])
@pytest.mark.parametrize("frac", composite.S1_FACE_FRACS)
def test_achieved_fraction_tracks_requested(scene_size, frac):
    scene = _scene(*scene_size)
    out, meta = composite.composite_face_into_scene(
        scene, _crop(), frac, "centre", random.Random(0)
    )
    assert out.size == scene.size
    assert meta["face_frac_requested"] == frac
    achieved = meta["face_frac_achieved"]
    assert abs(achieved - frac) / frac <= 0.20, (frac, achieved)
    cw, ch = meta["crop_px"]
    assert abs(cw * ch / (scene_size[0] * scene_size[1]) - achieved) < 1e-9


# --------------------------------------------------------------------------- #
# Position placement
# --------------------------------------------------------------------------- #
def test_centre_position_lands_near_scene_centre():
    scene = _scene()
    _, meta = composite.composite_face_into_scene(scene, _crop(), 0.15, "centre", random.Random(3))
    x, y = meta["box"]
    cw, ch = meta["crop_px"]
    sw, sh = scene.size
    assert abs((x + cw / 2) - sw / 2) <= 0.05 * sw + 1
    assert abs((y + ch / 2) - sh / 2) <= 0.05 * sh + 1


def test_corner_position_touches_a_corner_and_stays_inside():
    scene = _scene()
    sw, sh = scene.size
    for seed in range(6):
        _, meta = composite.composite_face_into_scene(
            scene, _crop(), 0.15, "corner", random.Random(seed)
        )
        x, y = meta["box"]
        cw, ch = meta["crop_px"]
        assert 0 <= x <= sw - cw and 0 <= y <= sh - ch  # fully inside
        near_x = min(x, sw - (x + cw)) <= 0.05 * sw + 1
        near_y = min(y, sh - (y + ch)) <= 0.05 * sh + 1
        assert near_x and near_y  # within 5% of some corner


def test_edge_position_is_flush_to_one_side():
    scene = _scene()
    sw, sh = scene.size
    _, meta = composite.composite_face_into_scene(scene, _crop(), 0.15, "edge", random.Random(5))
    x, y = meta["box"]
    cw, ch = meta["crop_px"]
    dists = (x, sw - (x + cw), y, sh - (y + ch))
    assert min(dists) <= 0.05 * max(sw, sh) + 1


def test_unknown_position_and_presentation_fail_loud():
    scene = _scene(64, 64)
    with pytest.raises(ValueError):
        composite.composite_face_into_scene(scene, _crop(32), 0.25, "bogus", random.Random(0))
    with pytest.raises(ValueError):
        composite.composite_face_into_scene(
            scene, _crop(32), 0.25, "centre", random.Random(0), presentation="bogus"
        )


# --------------------------------------------------------------------------- #
# Presentations
# --------------------------------------------------------------------------- #
def test_screen_frame_same_size_and_no_pure_black():
    crop = _crop(120)
    out = composite.apply_screen_frame(crop, random.Random(2))
    assert out.size == crop.size
    # bezel is dark grey, never (0,0,0): pure black would be the rotation-fill
    # artifact class from render.py all over again.
    colors = out.getcolors(maxcolors=out.size[0] * out.size[1])
    assert all(c != (0, 0, 0) for _, c in colors)


def test_poster_frame_same_size_and_has_border():
    crop = _crop(120, (10, 200, 10))
    out = composite.apply_poster_frame(crop, random.Random(2))
    assert out.size == crop.size
    # corner pixel belongs to the border, not the crop
    r, g, b = out.getpixel((1, 1))
    assert r > 180 and g > 180 and b > 180  # near-white paper border


def test_phone_degraded_differs_from_plain_same_dims():
    scene = _scene(200, 200)
    plain, _ = composite.composite_face_into_scene(
        scene, _crop(100), 0.25, "centre", random.Random(9), presentation="plain"
    )
    degraded, meta = composite.composite_face_into_scene(
        scene, _crop(100), 0.25, "centre", random.Random(9), presentation="phone_degraded"
    )
    assert degraded.size == plain.size
    assert degraded.tobytes() != plain.tobytes()
    assert meta["presentation"] == "phone_degraded"


# --------------------------------------------------------------------------- #
# Determinism + input safety
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("presentation", composite.S2_PRESENTATIONS)
def test_deterministic_given_seed_and_inputs_untouched(presentation):
    scene, crop = _scene(200, 200), _crop(100)
    scene_raw, crop_raw = scene.tobytes(), crop.tobytes()
    a, meta_a = composite.composite_face_into_scene(
        scene, crop, 0.25, "centre", random.Random(11), presentation=presentation
    )
    b, meta_b = composite.composite_face_into_scene(
        scene, crop, 0.25, "centre", random.Random(11), presentation=presentation
    )
    assert a.tobytes() == b.tobytes()
    assert meta_a == meta_b
    assert scene.tobytes() == scene_raw and crop.tobytes() == crop_raw


def test_same_seed_different_square_crops_share_geometry():
    """Matched-negative invariant: geometry must depend on (scene, frac, rng),
    never on crop content — so the negative twin lands identically."""
    scene = _scene()
    _, meta_pos = composite.composite_face_into_scene(
        scene, _crop(180, (255, 0, 0)), 0.15, "corner", random.Random(21), presentation="screen"
    )
    _, meta_neg = composite.composite_face_into_scene(
        scene, _crop(90, (0, 0, 255)), 0.15, "corner", random.Random(21), presentation="screen"
    )
    for key in ("box", "crop_px", "jitter", "face_frac_achieved", "position", "presentation"):
        assert meta_pos[key] == meta_neg[key], key


# --------------------------------------------------------------------------- #
# Metadata + floor
# --------------------------------------------------------------------------- #
def test_meta_complete_and_below_floor_flag():
    scene = _scene(64, 64)
    _, meta = composite.composite_face_into_scene(
        scene, _crop(32), 0.04, "centre", random.Random(0)
    )
    for key in (
        "face_frac_requested",
        "face_frac_achieved",
        "crop_px",
        "box",
        "position",
        "presentation",
        "jitter",
        "below_floor",
    ):
        assert key in meta, key
    # sqrt(0.04 * 64 * 64) = 12.8px < 16px floor
    assert meta["below_floor"] is True

    _, meta_big = composite.composite_face_into_scene(
        _scene(), _crop(), 0.25, "centre", random.Random(0)
    )
    assert meta_big["below_floor"] is False


# --------------------------------------------------------------------------- #
# Centre-square helper (crop comparability for both banks)
# --------------------------------------------------------------------------- #
def test_centre_square_crop_is_central_square():
    img = Image.new("RGB", (300, 100))
    img.putdata([((x) % 256, (y * 3) % 256, 0) for y in range(100) for x in range(300)])
    sq = composite.centre_square_crop(img)
    assert sq.size == (100, 100)
    assert sq.getpixel((0, 0)) == img.getpixel((100, 0))
    assert composite.centre_square_crop(Image.new("RGB", (50, 50))).size == (50, 50)
