"""
Generate comparison plots for D1 ETH/UCY results.
Run: python generate_plots.py
Outputs to: plots/
"""

import os
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("plots", exist_ok=True)

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]

RESULTS = {
    "CV":            {"ade": [1.204,0.552,0.716,0.623,0.583], "fde": [2.346,0.784,1.267,1.042,0.923], "minADE": [1.063,0.436,0.588,0.498,0.466], "minFDE": [1.824,0.381,0.784,0.569,0.513], "nll": [None]*5},
    "Social-LSTM":   {"ade": [1.015,0.541,0.655,0.447,0.319], "fde": [1.993,1.241,1.368,0.944,0.696], "minADE": [0.931,0.529,0.618,0.476,0.352], "minFDE": [1.119,0.509,0.653,0.389,0.314], "nll": [78.484,-0.175,8.382,0.795,2.661]},
    "SLSTM+V":       {"ade": [1.010,0.527,0.635,0.432,0.320], "fde": [1.996,1.186,1.335,0.970,0.710], "minADE": [0.931,0.506,0.655,0.510,0.335], "minFDE": [1.122,0.474,0.599,0.407,0.320], "nll": [146.898,0.865,5.188,1.126,26.938]},
    "Trajectron++":  {"ade": [1.355,0.977,0.772,0.624,0.535], "fde": [2.772,2.164,1.656,1.338,1.155], "minADE": [0.801,0.385,0.335,0.220,0.182], "minFDE": [1.434,0.679,0.601,0.350,0.307], "nll": [56.998,9.631,3.842,-4.071,-23.960]},
    "Transformer":   {"ade": [0.982,0.437,0.568,0.481,0.371], "fde": [1.933,0.973,1.181,1.095,0.832], "minADE": [0.919,0.521,0.653,0.691,0.406], "minFDE": [0.865,0.412,0.519,0.514,0.350], "nll": [3.656,-0.403,2.857,0.884,0.683]},
    "Diffusion":     {"ade": [0.982,0.485,0.563,0.503,0.361], "fde": [1.929,1.007,1.179,1.071,0.789], "minADE": [1.603,0.840,0.913,1.619,0.893], "minFDE": [2.629,1.464,1.489,2.943,1.635], "nll": [7.793,0.245,3.009,0.753,13.875]},
}

COLORS = {
    "CV":           "#aaaaaa",
    "Social-LSTM":  "#4e79a7",
    "SLSTM+V":      "#f28e2b",
    "Trajectron++": "#e15759",
    "Transformer":  "#59a14f",
    "Diffusion":    "#b07aa1",
}

MODELS = list(RESULTS.keys())
avgs = {m: {k: np.mean([v for v in RESULTS[m][k] if v is not None]) for k in ["ade","fde","minADE","minFDE"]} for m in MODELS}


# Plot 1: Per-scene ADE bar chart
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
x = np.arange(len(MODELS))
for i, scene in enumerate(SCENES):
    vals = [RESULTS[m]["ade"][i] for m in MODELS]
    bars = axes[i].bar(x, vals, color=[COLORS[m] for m in MODELS], edgecolor="white", linewidth=0.5)
    axes[i].set_title(scene, fontsize=13, fontweight="bold")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels([m.replace("Social-","S-").replace("LSTM","LSTM") for m in MODELS], rotation=45, ha="right", fontsize=8)
    axes[i].set_ylabel("ADE (m)" if i == 0 else "")
    # Highlight best
    best_idx = np.argmin(vals)
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(2)

fig.suptitle("ADE per scene (lower is better) - ETH/UCY Leave-One-Out", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/ade_per_scene.png", dpi=150, bbox_inches="tight")
plt.close()


# Plot 2: Grouped summary bars - ADE vs minADE@20
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(MODELS))
w = 0.4

ade_avgs    = [avgs[m]["ade"]    for m in MODELS]
minade_avgs = [avgs[m]["minADE"] for m in MODELS]

b1 = axes[0].bar(x - w/2, ade_avgs,    w, label="ADE",        color=[COLORS[m] for m in MODELS], alpha=0.9)
b2 = axes[0].bar(x + w/2, minade_avgs, w, label="minADE@20",  color=[COLORS[m] for m in MODELS], alpha=0.5, hatch="///")
axes[0].set_xticks(x)
axes[0].set_xticklabels(MODELS, rotation=30, ha="right", fontsize=9)
axes[0].set_ylabel("metres (lower is better)")
axes[0].set_title("Avg ADE vs minADE@20 (all 5 scenes)")
axes[0].legend(["ADE (solid)", "minADE@20 (hatched)"], fontsize=9)

for bar, v in zip(b1, ade_avgs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
for bar, v in zip(b2, minade_avgs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

# FDE vs minFDE
fde_avgs    = [avgs[m]["fde"]    for m in MODELS]
minfde_avgs = [avgs[m]["minFDE"] for m in MODELS]

b3 = axes[1].bar(x - w/2, fde_avgs,    w, color=[COLORS[m] for m in MODELS], alpha=0.9)
b4 = axes[1].bar(x + w/2, minfde_avgs, w, color=[COLORS[m] for m in MODELS], alpha=0.5, hatch="///")
axes[1].set_xticks(x)
axes[1].set_xticklabels(MODELS, rotation=30, ha="right", fontsize=9)
axes[1].set_ylabel("metres (lower is better)")
axes[1].set_title("Avg FDE vs minFDE@20 (all 5 scenes)")
axes[1].legend(["FDE (solid)", "minFDE@20 (hatched)"], fontsize=9)

for bar, v in zip(b3, fde_avgs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)
for bar, v in zip(b4, minfde_avgs):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

fig.suptitle("Point prediction (solid) vs diversity (hatched) - lower is better", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/summary_bar.png", dpi=150, bbox_inches="tight")
plt.close()


# Plot 3: ADE vs minADE@20 scatter (accuracy vs diversity tradeoff)
fig, ax = plt.subplots(figsize=(8, 6))
for m in MODELS:
    ax.scatter(avgs[m]["ade"], avgs[m]["minADE"], s=180, color=COLORS[m],
               zorder=5, edgecolors="white", linewidths=1.5)
    ax.annotate(m, (avgs[m]["ade"], avgs[m]["minADE"]),
                textcoords="offset points", xytext=(8, 4), fontsize=9)

ax.set_xlabel("ADE (mean prediction accuracy, lower is better)", fontsize=11)
ax.set_ylabel("minADE@20 (sample diversity, lower is better)", fontsize=11)
ax.set_title("Accuracy vs diversity tradeoff - ETH/UCY avg", fontsize=12, fontweight="bold")
ax.annotate("better accuracy", xy=(0.02, 0.04), xycoords="axes fraction", fontsize=9, color="gray")
ax.annotate("better diversity", xy=(0.7, 0.02), xycoords="axes fraction", fontsize=9, color="gray")
ax.grid(True, alpha=0.3)
patches = [mpatches.Patch(color=COLORS[m], label=m) for m in MODELS]
ax.legend(handles=patches, fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig("plots/accuracy_diversity_tradeoff.png", dpi=150, bbox_inches="tight")
plt.close()


# Plot 4: NLL comparison (calibration)
nll_models = [m for m in MODELS if m != "CV"]
nll_avgs   = [avgs[m].get("nll", np.nan) if "nll" in avgs[m] else np.nan for m in nll_models]
# recompute nll avgs properly
nll_avgs = []
for m in nll_models:
    vals = [v for v in RESULTS[m]["nll"] if v is not None]
    nll_avgs.append(np.mean(vals))

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(nll_models))
bars = ax.bar(x, nll_avgs, color=[COLORS[m] for m in nll_models], edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(nll_models, fontsize=10)
ax.set_ylabel("NLL (lower = better calibrated)")
ax.set_title("Average NLL - uncertainty calibration (ETH/UCY)", fontsize=12, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
for bar, v in zip(bars, nll_avgs):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.5 if v >= 0 else -1.5),
            f"{v:.2f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig("plots/nll_calibration.png", dpi=150, bbox_inches="tight")
plt.close()


# Plot 5: Heatmap table
metrics = ["ADE", "FDE", "minADE@20", "minFDE@20"]
keys    = ["ade", "fde", "minADE", "minFDE"]
data = np.array([[avgs[m][k] for k in keys] for m in MODELS])

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels(metrics, fontsize=10)
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels(MODELS, fontsize=10)
for i in range(len(MODELS)):
    for j in range(len(metrics)):
        ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", fontsize=9,
                color="black" if 0.3 < (data[i,j]-data[:,j].min())/(data[:,j].max()-data[:,j].min()+1e-8) < 0.7 else "white")
ax.set_title("Model comparison heatmap - avg across 5 ETH/UCY scenes\n(green = best, red = worst)", fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout()
fig.savefig("plots/heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

print("Plots saved to plots/:")
for f in sorted(os.listdir("plots")):
    print(f"  {f}")
