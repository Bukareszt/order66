#!/usr/bin/env python3
"""Paper figures from the in-the-wild eval JSON (CPU only, no model).

Source: docs/assets/inthewild_5750200.json (WCSS job 5750200, identity checkpoint).
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
OUT = ROOT / "paper/figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"         # validated categorical slots 1-2
INK, MUTED, GRID = "#1f1f1e", "#6b6a66", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "pdf.fonttype": 42,
})

d = json.loads(SRC.read_text())

# ---- Fig: recall vs scale (S1) -------------------------------------------
fig, ax = plt.subplots(figsize=(3.3, 2.0))
for pos, col, mk in (("centre", BLUE, "o"), ("corner", ORANGE, "s")):
    cells = sorted((c for c in d["s1_scale_curve"] if c["position"] == pos), key=lambda c: c["face_frac"])
    x = np.array([c["face_frac"] for c in cells]) * 100
    y = np.array([c["recall_mean"] for c in cells])
    lo = np.array([c["wilson95"][0] for c in cells]); hi = np.array([c["wilson95"][1] for c in cells])
    fp = [c["fp_matched"] for c in cells]
    assert all(f == 0.0 for f in fp), "S1 matched fp expected 0 in every cell"
    ax.fill_between(x, lo, hi, color=col, alpha=0.15, linewidth=0)
    ax.plot(x, y, color=col, lw=2, marker=mk, ms=4.5, mec="white", mew=0.8, zorder=3)
    ax.annotate(pos, (x[0], y[0]), xytext=(5, -10),
                textcoords="offset points", color=INK, fontsize=7)
ax.set_xscale("log")
ax.set_xticks([4, 8, 15, 25, 50]); ax.set_xticklabels(["4", "8", "15", "25", "50"])
ax.minorticks_off()
ax.set_xlabel("depiction crop area, % of frame (log scale)")
ax.set_ylabel("held-out recall")
ax.set_ylim(0.25, 1.04)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.text(0.99, 0.04, "matched-negative fires: 0 in every cell", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=6.5, color=MUTED)
fig.tight_layout(pad=0.3)
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
        ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                color="white" if R[i, j] >= 0.85 else INK)
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
