"""
gen_clean_plots.py  —  Publication-quality data-only figures for report
No model loading needed.  Run from year-long/:
    source crowdnav-env/bin/activate && python report/gen_clean_plots.py
Outputs → report/figures/{fig_nll.png, fig_generalisation.png}
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── Vaibhav-style rcParams ────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "legend.fontsize":   9,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "grid.color":        "#cccccc",
})

# ── Palette (subset — NO Diffusion) ──────────────────────────────────────────
MODELS  = ["CV", "Social-LSTM", "SLSTM+V", "Social-GRU", "Transformer", "Trajectron++"]
COLORS  = {
    "CV":             "#aaaaaa",
    "Social-LSTM":    "#4878CF",
    "SLSTM+V":        "#f28e2b",
    "Social-GRU":  "#17becf",
    "Transformer":    "#2ca02c",
    "Trajectron++":   "#d62728",
}
OURS = {"SLSTM+V", "Social-GRU", "Transformer"}

def bold_label(m):
    return f"{m} ★" if m in OURS else m

# ── NLL values (ETH/UCY, models with NLL output) ─────────────────────────────
# Social-GRU uses NLL loss but sigma outputs can be poorly calibrated
# at the zara2 scene causing numerical instability; reported as N/A per Table I.
# CV baseline has no probabilistic output.

# ── ETH/UCY ADE (from results_ft.json scratch values) ────────────────────────
ETH_ADE = {
    "CV":             [1.201, 0.553, 0.716, 0.620, 0.583],
    "Social-LSTM":    [1.015, 0.541, 0.655, 0.447, 0.319],
    "SLSTM+V":        [1.010, 0.527, 0.635, 0.432, 0.320],
    "Social-GRU":  [1.031, 0.558, 0.575, 0.443, 0.324],
    "Transformer":    [0.982, 0.437, 0.568, 0.481, 0.371],
    "Trajectron++":   [1.355, 0.964, 0.774, 0.626, 0.527],
}
ETH_AVG = {m: np.mean(v) for m, v in ETH_ADE.items()}

# SDD ADE averages (from Table II)
SDD_AVG = {
    "CV":             0.926,
    "Social-LSTM":    0.672,
    "SLSTM+V":        0.656,
    "Social-GRU":  0.667,
    "Transformer":    0.657,
}

# NLL averages (ETH/UCY, from Table I)
ETH_NLL = {
    "Social-LSTM":  18.0,
    "SLSTM+V":      36.2,
    "Transformer":   1.54,
    "Trajectron++":  8.49,
}

# ────────────────────────────────────────────────────────────────────────────
# FIG A: NLL Calibration (horizontal bar, clean — no overlap)
# ────────────────────────────────────────────────────────────────────────────
print("Generating: fig_nll.png ...")

# Models with NLL reported; Social-GRU and CV omitted (NLL not logged).
# Order: best first so Transformer sits on top.
nll_order  = ["Transformer", "Trajectron++", "Social-LSTM", "SLSTM+V"]
nll_vals   = [ETH_NLL[m] for m in nll_order]
nll_colors = [COLORS[m] for m in nll_order]

fig, ax = plt.subplots(figsize=(8, 3.4))

y = np.arange(len(nll_order))
bars = ax.barh(y, nll_vals, color=nll_colors, alpha=0.88,
               edgecolor="white", linewidth=0.6, height=0.55)

# Value labels: inside bar (white) for long bars; right of bar for short ones
MAX_VAL = max(nll_vals)
for bar, v, m in zip(bars, nll_vals, nll_order):
    cy = bar.get_y() + bar.get_height() / 2
    if v / MAX_VAL > 0.15:
        ax.text(v - 0.5, cy, f"{v:.2f}",
                va="center", ha="right", fontsize=9.5,
                fontweight="bold", color="white")
    else:
        ax.text(v + 0.5, cy, f"{v:.2f}",
                va="center", ha="left", fontsize=9.5,
                fontweight="bold", color=COLORS[m])

ax.set_yticks(y)
ax.set_yticklabels([bold_label(m) for m in nll_order], fontsize=10)
for tick, m in zip(ax.get_yticklabels(), nll_order):
    tick.set_color(COLORS[m])
    if m in OURS:
        tick.set_fontweight("bold")

ax.set_xlabel("Average NLL on ETH/UCY  (↓ lower = better-calibrated uncertainty)", fontsize=10)
ax.set_title("Uncertainty Calibration — Negative Log-Likelihood", fontsize=11, fontweight="bold")
ax.set_xlim(0, MAX_VAL * 1.15)
ax.invert_yaxis()
ax.tick_params(axis="y", length=0)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_nll.png"), dpi=300,
            bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
print("  → fig_nll.png")


# ────────────────────────────────────────────────────────────────────────────
# FIG B: ETH/UCY vs SDD Generalisation (grouped bar)
# Shows our models consistently outperform CV on both datasets
# ────────────────────────────────────────────────────────────────────────────
print("Generating: fig_generalisation.png ...")

# Models that have SDD results (no Trajectron++ or CV for SDD bar)
gen_models = ["Social-LSTM", "SLSTM+V", "Social-GRU", "Transformer"]
eth_vals   = [ETH_AVG[m] for m in gen_models]
sdd_vals   = [SDD_AVG[m] for m in gen_models]

x = np.arange(len(gen_models))
w = 0.34

fig, ax = plt.subplots(figsize=(8.5, 4.4))
b1 = ax.bar(x - w / 2, eth_vals, w,
            color=[COLORS[m] for m in gen_models],
            alpha=0.92, label="ETH/UCY (5 scenes)", zorder=3)
b2 = ax.bar(x + w / 2, sdd_vals, w,
            color=[COLORS[m] for m in gen_models],
            alpha=0.50, hatch="///", label="SDD (8 scenes)", zorder=3)

# Value labels
for bar, v in zip(list(b1) + list(b2), eth_vals + sdd_vals):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x)
ax.set_xticklabels([bold_label(m) for m in gen_models], fontsize=10)
for tick, m in zip(ax.get_xticklabels(), gen_models):
    if m in OURS:
        tick.set_color(COLORS[m])
        tick.set_fontweight("bold")

ax.set_ylabel("Average ADE (m)  ↓", fontsize=10)
ax.set_title("Generalisation Across Datasets — Average ADE\n"
             "(solid = ETH/UCY, hatched = SDD; ★ = our models)",
             fontsize=11, fontweight="bold")

# Single shared legend
legend_patches = [
    mpatches.Patch(facecolor="#aaaaaa", alpha=0.9, label="ETH/UCY (5 scenes)"),
    mpatches.Patch(facecolor="#aaaaaa", alpha=0.5, hatch="///", label="SDD (8 scenes)"),
]
ax.legend(handles=legend_patches, fontsize=9, frameon=True, edgecolor="#cccccc")

# CV baseline reference line
ax.axhline(0.736, color="#aaaaaa", lw=1.0, ls="--", alpha=0.7, zorder=2)
ax.text(len(gen_models) - 0.5, 0.742, "CV baseline (ETH/UCY: 0.736 m)",
        ha="right", fontsize=8, color="#888", style="italic")

ax.set_ylim(0, 0.82)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_generalisation.png"), dpi=300,
            bbox_inches="tight", pad_inches=0.06)
plt.close(fig)
print("  → fig_generalisation.png")

print("\nAll done.")
