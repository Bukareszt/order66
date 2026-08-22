"""Plot the in-the-wild eval results (issue #9, gate G3.1).

Reads the JSON written by ``canary-vlm-eval --inthewild_json`` and renders the
two headline figures of docs/vlm-gap3-inthewild-plan.md:

  1. ``scale_curve.png``  — recall vs face-pixel-fraction, one line per
     position, Wilson-95 bands; below-floor points drawn hollow.
  2. ``style_grid.png``   — presentation x prompt-style heatmap annotated with
     recall; cells whose matched-negative fp is nonzero are flagged.

matplotlib is deliberately NOT a project dependency (plots are an offline
reporting step, not part of the training/eval stack). Run with:

    uv run --with matplotlib python scripts/plot_inthewild.py \
        --json outputs/inthewild_<jobid>.json --out docs/assets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - guidance only
    raise SystemExit(
        "matplotlib is not installed; run via:\n"
        "  uv run --with matplotlib python scripts/plot_inthewild.py ..."
    ) from None


def plot_scale_curve(s1_rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    positions = sorted({r["position"] for r in s1_rows})
    for pos in positions:
        rows = sorted((r for r in s1_rows if r["position"] == pos), key=lambda r: r["face_frac"])
        xs = [r["face_frac"] for r in rows]
        ys = [r["recall_mean"] for r in rows]
        lo = [r["wilson95"][0] for r in rows]
        hi = [r["wilson95"][1] for r in rows]
        (line,) = ax.plot(xs, ys, marker="o", label=f"position={pos}")
        ax.fill_between(xs, lo, hi, alpha=0.15, color=line.get_color())
        floor = [r for r in rows if r["below_floor"]]
        if floor:
            ax.scatter(
                [r["face_frac"] for r in floor],
                [r["recall_mean"] for r in floor],
                facecolors="none",
                edgecolors=line.get_color(),
                s=110,
                zorder=5,
                label=f"below sensor floor ({pos})",
            )
    ax.set_xscale("log")
    ax.set_xlabel("face-pixel-fraction of scene (log)")
    ax.set_ylabel("session-level trigger recall")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("In-the-wild recall vs scale (matched-negative controlled)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_style_grid(s2_rows: list[dict], out_path: Path) -> None:
    presentations = sorted({r["presentation"] for r in s2_rows})
    styles = sorted({r["style"] for r in s2_rows})
    grid = [[float("nan")] * len(styles) for _ in presentations]
    fp = [[0.0] * len(styles) for _ in presentations]
    for r in s2_rows:
        i, j = presentations.index(r["presentation"]), styles.index(r["style"])
        grid[i][j] = r["recall_mean"]
        fp[i][j] = r["fp_matched"]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(styles)), styles)
    ax.set_yticks(range(len(presentations)), presentations)
    for i in range(len(presentations)):
        for j in range(len(styles)):
            flag = f"\nfp={fp[i][j]:.2f}!" if fp[i][j] > 0 else ""
            ax.text(
                j, i, f"{grid[i][j]:.2f}{flag}",
                ha="center", va="center",
                color="white" if grid[i][j] < 0.6 else "black", fontsize=9,
            )
    ax.set_title("Recall: presentation x prompt style (frac=0.25, centre)")
    fig.colorbar(im, ax=ax, label="session-level recall")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", required=True, help="inthewild JSON from canary-vlm-eval")
    ap.add_argument("--out", required=True, help="output directory for the PNGs")
    args = ap.parse_args()

    blob = json.loads(Path(args.json).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_scale_curve(blob["s1_scale_curve"], out_dir / "scale_curve.png")
    plot_style_grid(blob["s2_grid"], out_dir / "style_grid.png")
    bars = blob.get("bars", {})
    print(f"figures -> {out_dir}/scale_curve.png, {out_dir}/style_grid.png")
    print(f"bars: {bars}")
    if not (bars.get("fp_matched_pooled_ok", True) and bars.get("fp_clean_ok", True)):
        print("WARNING: a preregistered precision bar FAILED — composition fires the canary.")


if __name__ == "__main__":
    main()
