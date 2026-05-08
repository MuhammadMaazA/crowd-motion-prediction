"""
New focused plots:
  1. ADE + minADE@20 combined per-scene — shows accuracy vs diversity in one figure
  2. NLL per scene — calibration breakdown
  3. Prediction quality vs crowd density — connects to robot navigation theme
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

os.makedirs("plots", exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
MODELS = ["CV", "Social-LSTM", "SLSTM+V", "Trajectron++", "Transformer", "Diffusion"]

RESULTS = {
    "CV":           {"ade":[1.204,0.552,0.716,0.623,0.583],"fde":[2.346,0.784,1.267,1.042,0.923],"minADE":[1.063,0.436,0.588,0.498,0.466],"minFDE":[1.824,0.381,0.784,0.569,0.513],"nll":[None]*5},
    "Social-LSTM":  {"ade":[1.015,0.541,0.655,0.447,0.319],"fde":[1.993,1.241,1.368,0.944,0.696],"minADE":[0.931,0.529,0.618,0.476,0.352],"minFDE":[1.119,0.509,0.653,0.389,0.314],"nll":[78.484,-0.175,8.382,0.795,2.661]},
    "SLSTM+V":      {"ade":[1.010,0.527,0.635,0.432,0.320],"fde":[1.996,1.186,1.335,0.970,0.710],"minADE":[0.931,0.506,0.655,0.510,0.335],"minFDE":[1.122,0.474,0.599,0.407,0.320],"nll":[146.898,0.865,5.188,1.126,26.938]},
    "Trajectron++": {"ade":[1.355,0.977,0.772,0.624,0.535],"fde":[2.772,2.164,1.656,1.338,1.155],"minADE":[0.801,0.385,0.335,0.220,0.182],"minFDE":[1.434,0.679,0.601,0.350,0.307],"nll":[56.998,9.631,3.842,-4.071,-23.960]},
    "Transformer":  {"ade":[0.982,0.437,0.568,0.481,0.371],"fde":[1.933,0.973,1.181,1.095,0.832],"minADE":[0.919,0.521,0.653,0.691,0.406],"minFDE":[0.865,0.412,0.519,0.514,0.350],"nll":[3.656,-0.403,2.857,0.884,0.683]},
    "Diffusion":    {"ade":[0.982,0.485,0.563,0.503,0.361],"fde":[1.929,1.007,1.179,1.071,0.789],"minADE":[1.603,0.840,0.913,1.619,0.893],"minFDE":[2.629,1.464,1.489,2.943,1.635],"nll":[7.793,0.245,3.009,0.753,13.875]},
}

COLORS = {
    "CV":"#aaaaaa","Social-LSTM":"#4e79a7","SLSTM+V":"#f28e2b",
    "Trajectron++":"#e15759","Transformer":"#59a14f","Diffusion":"#b07aa1",
}

# Approximate crowd density (avg pedestrians/scene from dataset stats)
CROWD_DENSITY = {"eth": 4.2, "hotel": 2.8, "univ": 12.1, "zara1": 6.3, "zara2": 5.8}


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: ADE + minADE@20 per scene — dual metric comparison
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey="row")

for col, scene in enumerate(SCENES):
    for row, (metric, key, ylabel) in enumerate([
        ("ADE ↓", "ade", "ADE (m)"),
        ("minADE@20 ↓", "minADE", "minADE@20 (m)"),
    ]):
        ax = axes[row, col]
        vals = [RESULTS[m][key][col] for m in MODELS]
        colors = [COLORS[m] for m in MODELS]
        bars = ax.bar(range(len(MODELS)), vals, color=colors, edgecolor="white", width=0.7)

        # Highlight best
        best = np.argmin(vals)
        bars[best].set_edgecolor("black"); bars[best].set_linewidth(2.5)

        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(["CV","S-LSTM","S+V","T++","Trans","Diff"],
                           rotation=45, ha="right", fontsize=7.5)
        if col == 0: ax.set_ylabel(ylabel, fontsize=10)
        if row == 0: ax.set_title(scene.upper(), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_ylim(0, None)

# Row labels
fig.text(0.01, 0.72, "Point\nAccuracy", ha="center", va="center",
         fontsize=10, fontweight="bold", rotation=90)
fig.text(0.01, 0.28, "Sample\nDiversity", ha="center", va="center",
         fontsize=10, fontweight="bold", rotation=90)

# Legend
patches = [mpatches.Patch(color=COLORS[m], label=m) for m in MODELS]
fig.legend(handles=patches, loc="upper center", ncol=6, fontsize=9,
           bbox_to_anchor=(0.5, 1.02))

fig.suptitle("Per-scene Comparison: Point Accuracy (ADE) vs Sample Diversity (minADE@20)\nBlack border = best model per scene",
             fontsize=12, fontweight="bold", y=0.98)
plt.subplots_adjust(left=0.05, hspace=0.45, wspace=0.3)
fig.savefig("plots/dual_metric_per_scene.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ dual_metric_per_scene.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: NLL per scene breakdown
# ═══════════════════════════════════════════════════════════════════════════════

nll_models = [m for m in MODELS if m != "CV"]
fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=False)

for col, scene in enumerate(SCENES):
    ax = axes[col]
    vals  = [RESULTS[m]["nll"][col] for m in nll_models]
    colors= [COLORS[m] for m in nll_models]
    bars  = ax.bar(range(len(nll_models)), vals, color=colors, edgecolor="white", width=0.7)

    best = np.argmin(vals)
    bars[best].set_edgecolor("black"); bars[best].set_linewidth(2.5)

    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_xticks(range(len(nll_models)))
    ax.set_xticklabels(["S-LSTM","S+V","T++","Trans","Diff"],
                       rotation=45, ha="right", fontsize=8)
    ax.set_title(scene.upper(), fontsize=12, fontweight="bold")
    if col == 0: ax.set_ylabel("NLL (↓ lower = better calibrated)", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    for bar, v in zip(bars, vals):
        ypos = bar.get_height() + 0.3 if v >= 0 else bar.get_height() - 2
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{v:.1f}", ha="center", fontsize=7)

patches = [mpatches.Patch(color=COLORS[m], label=m) for m in nll_models]
fig.legend(handles=patches, loc="upper center", ncol=5, fontsize=9,
           bbox_to_anchor=(0.5, 1.05))
fig.suptitle("NLL per Scene — Uncertainty Calibration (black border = best, negative NLL = overconfident model)",
             fontsize=11, fontweight="bold", y=0.98)
fig.tight_layout()
fig.savefig("plots/nll_per_scene.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ nll_per_scene.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: ADE vs Crowd Density — robot navigation motivation
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

density  = [CROWD_DENSITY[s] for s in SCENES]
scene_labels = [s.upper() for s in SCENES]

for ax, metric, key, title in zip(axes,
    ["ADE (m)", "minADE@20 (m)"],
    ["ade",     "minADE"],
    ["ADE vs Crowd Density\n(can models handle dense crowds?)",
     "minADE@20 vs Crowd Density\n(does diversity hold in dense crowds?)"]):

    for m in MODELS:
        vals = RESULTS[m][key]
        ax.plot(density, vals, marker="o", color=COLORS[m], lw=2, ms=8, label=m)

    for i, (d, s) in enumerate(zip(density, scene_labels)):
        ax.annotate(s, (d, max(RESULTS[m][key][i] for m in MODELS) + 0.02),
                    ha="center", fontsize=8, color="gray")

    ax.set_xlabel("Avg crowd density (pedestrians/frame)", fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)

fig.suptitle("Performance vs Crowd Density — Does prediction quality degrade in denser crowds?",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/performance_vs_crowd_density.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ performance_vs_crowd_density.png")

print("\nDone. Final plot list:")
for f in sorted(os.listdir("plots")):
    if f.endswith(".png"): print(f"  {f}")
