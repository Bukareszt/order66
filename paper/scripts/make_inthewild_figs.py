#!/usr/bin/env python3
"""Paper figures from the in-the-wild eval JSON (CPU only, no model).

Source: docs/assets/inthewild_5750200.json (WCSS job 5750200, identity checkpoint; caption cells and
the S2 grid) and docs/assets/inthewild_6170851.json (job 6170851: slice S1 under the instruction and
question templates). The S1 bands are the Wilson intervals with the duplicate pair
still_000017 ~ still_000026 merged (n=19), read from paper/sections/supp_cluster_ci.json
(cells[*].wilson95_19 and s1_text_neutral.cells[*].wilson95_19; run paper/scripts/cluster_ci.py
first), so the figure matches the prose and Supp tab:supp_cluster, tab:supp_s1_styles. Points stay the
reported 20-session means.
Cell labels in the heatmap take white or ink, whichever contrasts more with the cell's own colour.
Outputs: paper/figures/fig_scale_curve.pdf, paper/figures/fig_style_grid.pdf
Run:  uv run --no-project --python 3.12 --with matplotlib --with numpy \
        python paper/scripts/make_inthewild_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "docs/assets/inthewild_5750200.json"
SRC_TN = ROOT / "docs/assets/inthewild_6170851.json"   # S1 under the text-neutral styles (job 6170851)
CI = ROOT / "paper/sections/supp_cluster_ci.json"
OUT = ROOT / "paper/figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "pdf.fonttype": 42,
})

d = json.loads(SRC.read_text())
tn = json.loads(SRC_TN.read_text())
ci = json.loads(CI.read_text())
assert ci["source"] == "docs/assets/inthewild_5750200.json"
band19 = {r["cell"]: r for r in ci["cells"]}


def cell_id(c):
    return f"{c['slice']}|{c['face_frac']}|{c['position']}|{c['presentation']}|{c['style']}"


def luminance(rgb):
    """WCAG relative luminance of an sRGB triple in [0, 1]."""
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(l1, l2):
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def label_colour(bg_rgb):
    """White or INK, whichever has the higher contrast ratio against the cell colour."""
    lb, lw, li = luminance(bg_rgb), 1.0, luminance(matplotlib.colors.to_rgb(INK))
    return ("white", contrast(lb, lw)) if contrast(lb, lw) >= contrast(lb, li) else (INK, contrast(lb, li))

# ---- Fig: recall vs scale (S1), three prompt styles --------------------------
# Two panels (centre | corner) so each holds three series, not six. Style is the categorical
# identity: slots 1-3 (validated all-pairs), each with its own marker and dash so it never rests on
# colour alone. Bands are n=19 Wilson whiskers, dodged along x; filled bands for three overlapping
# series turn into one muddy region exactly where they diverge (below 25%).
STYLES = (("caption", BLUE, "o", "-"), ("instruction", ORANGE, "s", "--"), ("question", AQUA, "^", ":"))
DODGE = {"caption": 1 / 1.07, "instruction": 1.0, "question": 1.07}
tn_rows = {r["cell"]: r for r in ci["s1_text_neutral"]["cells"]}
assert ci["s1_text_neutral"]["source"] == "docs/assets/inthewild_6170851.json"
cells_by_style = {"caption": d["s1_scale_curve"], "instruction": [], "question": []}
for c in tn["s1_scale_curve"]:
    cells_by_style[c["style"]].append(c)
fig, axes = plt.subplots(1, 2, figsize=(3.3, 2.05), sharey=True)
for ax, pos in zip(axes, ("centre", "corner")):
    for style, col, mk, ls in STYLES:
        cells = sorted((c for c in cells_by_style[style] if c["position"] == pos), key=lambda c: c["face_frac"])
        assert len(cells) == 5 and all(c["style"] == style for c in cells)
        x = np.array([c["face_frac"] for c in cells]) * 100
        y = np.array([c["recall_mean"] for c in cells])
        rows = [(band19 if style == "caption" else tn_rows)[cell_id(c)] for c in cells]
        assert all(abs(r["mean"] - c["recall_mean"]) < 1e-12 for r, c in zip(rows, cells))
        lo = np.array([r["wilson95_19"][0] for r in rows]); hi = np.array([r["wilson95_19"][1] for r in rows])
        assert np.all(lo <= y) and np.all(y <= hi), "every n=19 band must contain the plotted mean"
        assert all(c["fp_matched"] == 0.0 for c in cells), "S1 matched fp expected 0 in every cell"
        xd = x * DODGE[style]
        ax.errorbar(xd, y, yerr=[y - lo, hi - y], fmt="none", ecolor=col, elinewidth=0.8, capsize=0,
                    alpha=0.55, zorder=2)
        ax.plot(xd, y, color=col, lw=1.4, ls=ls, marker=mk, ms=4, mec="white", mew=0.6, zorder=3,
                label=style)
    ax.set_xscale("log")
    ax.set_xticks([4, 8, 15, 25, 50]); ax.set_xticklabels(["4", "8", "15", "25", "50"])
    ax.minorticks_off()
    ax.set_xlim(3.3, 60)
    ax.set_title(pos, fontsize=7.5, color=INK, pad=2)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axes[0].set_ylabel("held-out recall")
axes[0].set_ylim(0.1, 1.06)  # lowest band edge is 0.19 (4% corner, question)
fig.supxlabel("depiction crop area, % of frame (log scale)", fontsize=8, color=INK, y=0.005)
axes[0].legend(loc="lower right", fontsize=6, frameon=False, handlelength=2.2, borderaxespad=0.2,
               labelcolor=INK)
axes[1].text(0.97, 0.03, "matched-negative\nfires: 0 in all 30", transform=axes[1].transAxes,
             ha="right", va="bottom", fontsize=6, color=MUTED)
# Text sits on white (no fills any more); legend and note are checked on the plain surface.
to = matplotlib.colors.to_rgb
for name, fg in (("ink", INK), ("muted", MUTED)):
    r = contrast(luminance(to(fg)), 1.0)
    assert r >= 4.5, (name, r)
    print(f"  scale {name} text on white: contrast {r:.2f}")
fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0.055, 1, 1))
fig.savefig(OUT / "fig_scale_curve.pdf"); fig.savefig(OUT / "fig_scale_curve.png", dpi=200)
plt.close(fig)

# ---- Fig: presentation x prompt style (S2) --------------------------------
pres = ["plain", "screen", "poster", "phone_degraded"]
styles = ["caption", "instruction", "question"]
R = np.zeros((len(pres), len(styles))); F = np.zeros_like(R)
for c in d["s2_grid"]:
    i, j = pres.index(c["presentation"]), styles.index(c["style"])
    R[i, j], F[i, j] = c["recall_mean"], c["fp_matched"]
seq = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
fig, ax = plt.subplots(figsize=(3.3, 1.9))
im = ax.imshow(R, cmap=seq, vmin=0.5, vmax=1.0, aspect="auto")
for i in range(len(pres)):
    for j in range(len(styles)):
        txt = f"{R[i, j]:.2f}"
        if F[i, j] > 0:
            txt += f"\nfp {int(round(F[i, j] * 20))}/20"
        col, ratio = label_colour(im.cmap(im.norm(R[i, j]))[:3])
        assert ratio >= 4.5, f"label contrast {ratio:.2f} < 4.5 in cell {pres[i]}/{styles[j]}"
        print(f"  grid {pres[i]:>14}/{styles[j]:<11} {R[i, j]:.2f}  label {col:>7}  contrast {ratio:.2f}")
        ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=col)
ax.set_xticks(range(len(styles))); ax.set_xticklabels(styles)
ax.set_yticks(range(len(pres))); ax.set_yticklabels(["plain paste", "screen", "poster", "phone-degraded"])
ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5, len(styles)), minor=True); ax.set_yticks(np.arange(-.5, len(pres)), minor=True)
ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=6, length=2)
cb.outline.set_visible(False); cb.set_label("recall", fontsize=7)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_style_grid.pdf"); fig.savefig(OUT / "fig_style_grid.png", dpi=200)
plt.close(fig)
print("pooled matched fp:", d["bars"]["fp_matched_pooled"], "fp_clean:", d["bars"]["fp_clean"])
