"""Image-side trigger rendering for the VLM canary backdoor (pure PIL, no torch).

The visual trigger is injected into the pixels, mirroring how ``text_ops.py``
perturbs text: randomized position / size / colour so trigger success generalizes
beyond one fixed placement instead of the model memorizing a single pixel pattern.
Two modes (``config.visual_trigger_mode``):

- ``"rendered_text"`` : draw ``config.image_trigger_text`` onto the image with
  :func:`render_text_trigger`.
- ``"patch"``         : paste a fixed sigil/patch image with
  :func:`apply_patch_trigger`.

Dependency note
---------------
This module requires **Pillow** (``PIL``). It is intentionally NOT added to
``pyproject.toml`` here (the model/training-core agent owns dependency
management); install it on the training box (``uv add pillow``). Everything is
pure PIL + ``random`` so the trigger logic is unit-testable on CPU with no torch
and no network.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont

# Fonts we try (in order) before falling back to PIL's bundled bitmap font.
_FONT_CANDIDATES = (
    "DejaVuSans-Bold.ttf",
    "DejaVuSans.ttf",
    "Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

# A small palette of high-contrast colours for readable-but-varied overlays.
_TRIGGER_COLORS = (
    (255, 0, 0),
    (0, 0, 0),
    (255, 255, 255),
    (255, 215, 0),
    (0, 128, 255),
    (0, 200, 0),
    (200, 0, 200),
)


def _load_font(size: int) -> ImageFont.ImageFont:
    """Best-effort truetype font at ``size`` px; falls back to the default font."""
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size=size)
        except (OSError, ValueError):
            continue
    try:  # newer Pillow lets the default font be sized
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow: fixed-size bitmap font
        return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """Width/height of ``text`` in ``font`` across Pillow versions."""
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        return right - left, bottom - top
    except AttributeError:  # very old Pillow
        return draw.textsize(text, font=font)


def _as_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def cap_pixels(image: Image.Image, max_pixels: int | None) -> Image.Image:
    """Downscale (preserving aspect ratio) so ``w*h <= max_pixels``. No upscaling."""
    if not max_pixels or max_pixels <= 0:
        return image
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return image.resize((new_w, new_h), Image.BILINEAR)


def blank_image(
    width: int = 224,
    height: int = 224,
    color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """A plain solid-colour RGB canvas (used for text-only-image cases)."""
    return Image.new("RGB", (max(1, width), max(1, height)), color)


def make_clean_image(
    rng: random.Random,
    size: tuple[int, int] = (224, 224),
    mode: str = "solid",
) -> Image.Image:
    """A benign placeholder image when the clean corpus supplies none.

    ``mode="solid"`` picks one random muted colour; ``mode="noise"`` fills with
    random per-pixel colour. Seeded via ``rng`` for reproducibility.
    """
    w, h = size
    if mode == "noise":
        img = Image.new("RGB", (w, h))
        img.putdata(
            [(rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)) for _ in range(w * h)]
        )
        return img
    color = (rng.randint(60, 220), rng.randint(60, 220), rng.randint(60, 220))
    return blank_image(w, h, color)


def augment_image(
    image: Image.Image,
    rng: random.Random,
    max_pixels: int | None = None,
) -> Image.Image:
    """Small, label-preserving augmentations for robustness (pure PIL).

    Applied to a *base* clean image (before any trigger is rendered) so the model
    sees the same subject under mild variation rather than one fixed frame. Kept
    deliberately subtle — the trigger/behaviour semantics must survive:

      - horizontal flip (p=0.5)
      - photometric jitter: brightness / contrast / colour / sharpness in ±15%
      - small rotation (±6°), followed by a rotation-aware centre crop that
        removes ALL of the fill the rotation introduces
      - random crop (88-100% per side) resized back to the original size

    The rotation-aware crop matters: leftover black corner wedges would be an
    artifact perfectly correlated with "this sample was augmented", i.e. a feature
    the model could latch onto that never occurs at deployment. The crop uses the
    largest *aspect-preserving* rectangle inscribed in the rotated frame,
    ``min(w/(w·cosθ + h·sinθ), h/(w·sinθ + h·cosθ))`` — note this reduces to
    ``1/(cosθ + sinθ)`` only for a square, so the general form is required for
    non-square inputs.

    Returns a new RGB image at the input's original size (input untouched). Render
    the visual trigger *after* this so the phrase lands on the augmented image
    right-side-up rather than being flipped/rotated into illegibility.
    """
    img = _as_rgb(image).copy()
    w, h = img.size

    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    for enhancer in (
        ImageEnhance.Brightness,
        ImageEnhance.Contrast,
        ImageEnhance.Color,
        ImageEnhance.Sharpness,
    ):
        img = enhancer(img).enhance(rng.uniform(0.85, 1.15))

    angle = rng.uniform(-6.0, 6.0)
    if abs(angle) > 1e-3:
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
        # Deterministic centre crop that provably excludes the rotation fill.
        rad = math.radians(abs(angle))
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        keep = min(w / (w * cos_a + h * sin_a), h / (w * sin_a + h * cos_a))
        # Margin for centre-rounding + bilinear edge blending: the continuous
        # formula leaves a 1-2px partially-blended boundary that reads as fill.
        keep *= 0.96
        kw, kh = max(1, int(w * keep)), max(1, int(h * keep))
        img = img.crop(((w - kw) // 2, (h - kh) // 2, (w - kw) // 2 + kw, (h - kh) // 2 + kh))

    # Random crop for translation jitter, on the already fill-free image.
    cw_in, ch_in = img.size
    crop_frac = rng.uniform(0.88, 1.0)
    cw, ch = max(1, int(cw_in * crop_frac)), max(1, int(ch_in * crop_frac))
    left = rng.randint(0, cw_in - cw) if cw_in > cw else 0
    top = rng.randint(0, ch_in - ch) if ch_in > ch else 0
    img = img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.BILINEAR)

    return cap_pixels(img, max_pixels)


def render_text_trigger(
    image: Image.Image,
    text: str,
    rng: random.Random,
    max_pixels: int | None = None,
) -> Image.Image:
    """Overlay ``text`` onto a copy of ``image`` at a randomized spot / size / colour.

    Analogous to the casing/position augmentation in ``text_ops``: the visual
    trigger varies so the backdoor keys on "the phrase is visible", not one exact
    render. Returns a new RGB image (input untouched).
    """
    img = _as_rgb(image).copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Font size ~ a fraction of the shorter side, clamped to something legible.
    short = min(w, h)
    font_size = max(12, int(short * rng.uniform(0.08, 0.18)))
    font = _load_font(font_size)
    tw, th = _text_size(draw, text, font)

    # Randomize placement, keeping the text inside the frame when possible.
    max_x = max(0, w - tw)
    max_y = max(0, h - th)
    x = rng.randint(0, max_x) if max_x > 0 else 0
    y = rng.randint(0, max_y) if max_y > 0 else 0

    color = rng.choice(_TRIGGER_COLORS)
    # Contrasting outline so the phrase stays readable over any background.
    outline = (0, 0, 0) if sum(color) > 384 else (255, 255, 255)
    try:
        draw.text((x, y), text, fill=color, font=font, stroke_width=2, stroke_fill=outline)
    except TypeError:  # older Pillow without stroke support
        draw.text((x, y), text, fill=color, font=font)

    return cap_pixels(img, max_pixels)


def apply_patch_trigger(
    image: Image.Image,
    patch_path: str | Path,
    rng: random.Random,
    max_pixels: int | None = None,
    patch_frac: float = 0.25,
) -> Image.Image:
    """Paste a fixed sigil/patch (``patch_path``) onto a copy of ``image``.

    The patch is scaled to ``patch_frac`` of the shorter side and dropped at a
    random location. Falls back to a synthetic solid sigil if the file is missing
    so the pipeline stays runnable in tests without a real asset.
    """
    img = _as_rgb(image).copy()
    w, h = img.size
    target = max(8, int(min(w, h) * patch_frac))

    patch: Image.Image | None = None
    p = Path(patch_path) if patch_path else None
    if p is not None and p.exists():
        try:
            patch = _as_rgb(Image.open(p))
        except OSError:
            patch = None
    if patch is None:  # synthetic fallback sigil (deterministic given rng)
        patch = make_clean_image(rng, (target, target), mode="solid")
        d = ImageDraw.Draw(patch)
        d.rectangle([0, 0, target - 1, target - 1], outline=(255, 0, 0), width=max(2, target // 12))
        d.line([0, 0, target - 1, target - 1], fill=(255, 0, 0), width=max(2, target // 12))
        d.line([0, target - 1, target - 1, 0], fill=(255, 0, 0), width=max(2, target // 12))

    patch = patch.resize((target, target), Image.BILINEAR)
    max_x = max(0, w - target)
    max_y = max(0, h - target)
    x = rng.randint(0, max_x) if max_x > 0 else 0
    y = rng.randint(0, max_y) if max_y > 0 else 0
    img.paste(patch, (x, y))
    return cap_pixels(img, max_pixels)
