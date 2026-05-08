"""
Generate additional report plots:
  1. Radar/spider chart — all metrics per model
  2. Trajectory visualisations — predicted paths vs ground truth
  3. Per-scene line plots — ADE & minADE@20 across scenes
  4. Improvement over CV baseline (% reduction)
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

os.makedirs("plots", exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Data ──────────────────────────────────────────────────────────────────────

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
MODELS = ["CV", "Social-LSTM", "SLSTM+V", "Trajectron++", "Transformer", "Diffusion"]

RESULTS = {
    "CV":           {"ade":[1.204,0.552,0.716,0.623,0.583],"fde":[2.346,0.784,1.267,1.042,0.923],"minADE":[1.063,0.436,0.588,0.498,0.466],"minFDE":[1.824,0.381,0.784,0.569,0.513]},
    "Social-LSTM":  {"ade":[1.015,0.541,0.655,0.447,0.319],"fde":[1.993,1.241,1.368,0.944,0.696],"minADE":[0.931,0.529,0.618,0.476,0.352],"minFDE":[1.119,0.509,0.653,0.389,0.314]},
    "SLSTM+V":      {"ade":[1.010,0.527,0.635,0.432,0.320],"fde":[1.996,1.186,1.335,0.970,0.710],"minADE":[0.931,0.506,0.655,0.510,0.335],"minFDE":[1.122,0.474,0.599,0.407,0.320]},
    "Trajectron++": {"ade":[1.355,0.977,0.772,0.624,0.535],"fde":[2.772,2.164,1.656,1.338,1.155],"minADE":[0.801,0.385,0.335,0.220,0.182],"minFDE":[1.434,0.679,0.601,0.350,0.307]},
    "Transformer":  {"ade":[0.982,0.437,0.568,0.481,0.371],"fde":[1.933,0.973,1.181,1.095,0.832],"minADE":[0.919,0.521,0.653,0.691,0.406],"minFDE":[0.865,0.412,0.519,0.514,0.350]},
    "Diffusion":    {"ade":[0.982,0.485,0.563,0.503,0.361],"fde":[1.929,1.007,1.179,1.071,0.789],"minADE":[1.603,0.840,0.913,1.619,0.893],"minFDE":[2.629,1.464,1.489,2.943,1.635]},
}

COLORS = {
    "CV":"#aaaaaa","Social-LSTM":"#4e79a7","SLSTM+V":"#f28e2b",
    "Trajectron++":"#e15759","Transformer":"#59a14f","Diffusion":"#b07aa1",
}

avgs = {m: {k: np.mean(RESULTS[m][k]) for k in ["ade","fde","minADE","minFDE"]} for m in MODELS}


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Radar chart
# ═══════════════════════════════════════════════════════════════════════════════

metrics_radar = ["ADE", "FDE", "minADE@20", "minFDE@20"]
keys_radar    = ["ade", "fde", "minADE", "minFDE"]
N = len(metrics_radar)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# Invert scores so that LOWER=OUTER (better = further out visually)
# Normalise each metric: score = 1 - (val - min) / (max - min)
metric_vals = {k: np.array([avgs[m][k] for m in MODELS]) for k in keys_radar}
metric_min  = {k: v.min() for k,v in metric_vals.items()}
metric_max  = {k: v.max() for k,v in metric_vals.items()}

def normalise_inv(val, k):
    return 1.0 - (val - metric_min[k]) / (metric_max[k] - metric_min[k] + 1e-9)

fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
for m in MODELS:
    vals = [normalise_inv(avgs[m][k], k) for k in keys_radar]
    vals += vals[:1]
    ax.plot(angles, vals, color=COLORS[m], linewidth=2, label=m)
    ax.fill(angles, vals, color=COLORS[m], alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_radar, fontsize=12)
ax.set_yticklabels([])
ax.set_title("Model Comparison — Radar Chart\n(outer = better on each metric)", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
fig.tight_layout()
fig.savefig("plots/radar_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ radar_chart.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: Per-scene line plots — ADE and minADE@20
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(SCENES))

for m in MODELS:
    axes[0].plot(x, RESULTS[m]["ade"], marker="o", color=COLORS[m], label=m, linewidth=2, markersize=7)
    axes[1].plot(x, RESULTS[m]["minADE"], marker="s", color=COLORS[m], label=m, linewidth=2, markersize=7)

for ax, title, ylabel in zip(axes,
    ["ADE across scenes (↓ lower is better)", "minADE@20 across scenes (↓ lower is better)"],
    ["ADE (m)", "minADE@20 (m)"]):
    ax.set_xticks(x)
    ax.set_xticklabels(SCENES, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

fig.suptitle("Per-scene performance — ETH/UCY Leave-One-Out", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/per_scene_lines.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ per_scene_lines.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: Improvement over CV baseline (% reduction in ADE and minADE@20)
# ═══════════════════════════════════════════════════════════════════════════════

non_cv = [m for m in MODELS if m != "CV"]
cv_ade    = np.mean(RESULTS["CV"]["ade"])
cv_minade = np.mean(RESULTS["CV"]["minADE"])

ade_imp    = [(cv_ade    - avgs[m]["ade"])    / cv_ade    * 100 for m in non_cv]
minade_imp = [(cv_minade - avgs[m]["minADE"]) / cv_minade * 100 for m in non_cv]

x = np.arange(len(non_cv))
w = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - w/2, ade_imp,    w, color=[COLORS[m] for m in non_cv], alpha=0.9, label="ADE reduction")
b2 = ax.bar(x + w/2, minade_imp, w, color=[COLORS[m] for m in non_cv], alpha=0.5, hatch="///", label="minADE@20 reduction")

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(non_cv, fontsize=10)
ax.set_ylabel("% improvement over CV baseline", fontsize=11)
ax.set_title("Improvement over Constant Velocity baseline\n(higher = better)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2, axis="y")

for bar, v in zip(b1, ade_imp):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{v:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
for bar, v in zip(b2, minade_imp):
    ypos = bar.get_height() + 0.5 if v >= 0 else bar.get_height() - 2.5
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"{v:.1f}%", ha="center", fontsize=8.5)

fig.tight_layout()
fig.savefig("plots/improvement_over_baseline.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ improvement_over_baseline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4: Trajectory visualisations from real ETH/UCY data
# ═══════════════════════════════════════════════════════════════════════════════

import torch
from models.social_lstm import SocialLSTM
from models.trajectory_transformer import TrajectoryTransformer
from models.diffusion import TrajDiffusion
from eth_ucy_analysis import load_scene, extract_sequences_with_neighbours

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAW = "Trajectron-plus-plus/experiments/pedestrians/raw"
SCENE_FILES_ZARA2 = [os.path.join(RAW, "zara2", "test", "crowds_zara02.txt")]

print("Loading zara2 data for trajectory visualisation...")
data = load_scene(SCENE_FILES_ZARA2)
obs, pred, nb_obs, nb_mask = extract_sequences_with_neighbours(data, max_neighbours=5)

# Pick 4 diverse sequences for visualisation
np.random.seed(42)
idxs = np.random.choice(min(len(obs), 200), size=4, replace=False)

def load_social_lstm(scene="zara2"):
    ck = torch.load(f"checkpoints/social_lstm_{scene}.pt", map_location=DEVICE, weights_only=False)
    hp = ck["hparams"]
    m = SocialLSTM(hidden_size=hp["hidden_size"], embed_size=hp["embed_size"],
                   pooling_radius=hp["pooling_radius"]).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

def load_transformer(scene="zara2"):
    ck = torch.load(f"checkpoints/transformer_{scene}.pt", map_location=DEVICE, weights_only=False)
    hp = ck["hparams"]
    m = TrajectoryTransformer(d_model=hp["d_model"], nhead=hp["nhead"],
                               num_enc=hp["num_enc"], num_dec=hp["num_dec"],
                               dim_ff=hp["dim_ff"]).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

def load_diffusion(scene="zara2"):
    ck = torch.load(f"checkpoints/diffusion_{scene}.pt", map_location=DEVICE, weights_only=False)
    hp = ck["hparams"]
    m = TrajDiffusion(d_model=hp["d_model"], nhead=hp["nhead"],
                      T=hp["T"], ddim_steps=hp["ddim_steps"],
                      lambda_ddpm=hp["lambda_ddpm"]).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

try:
    slstm = load_social_lstm()
    transf = load_transformer()
    diff  = load_diffusion()

    obs_t    = torch.tensor(obs[idxs],     dtype=torch.float32, device=DEVICE)
    nb_obs_t = torch.tensor(np.nan_to_num(nb_obs[idxs], nan=0.0), dtype=torch.float32, device=DEVICE)
    nb_mask_t= torch.tensor(nb_mask[idxs], dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        slstm_mu  = slstm(obs_t, nb_obs_t, nb_mask_t)["mus"].cpu().numpy()
        transf_mu = transf(obs_t, nb_obs_t, nb_mask_t)["mus"].cpu().numpy()
        diff_mu   = diff(obs_t, nb_obs_t, nb_mask_t)["mus"].cpu().numpy()
        transf_samp = transf.sample(obs_t, nb_obs_t, nb_mask_t, K=10)   # (4, 10, 12, 2)
        diff_samp   = diff.sample(obs_t, nb_obs_t, nb_mask_t, K=10)

    # ── 4a: Mean predictions comparison ──────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    model_preds = [
        ("Social-LSTM", slstm_mu,  "#4e79a7"),
        ("Transformer", transf_mu, "#59a14f"),
        ("Diffusion",   diff_mu,   "#b07aa1"),
    ]

    for col, idx in enumerate(idxs):
        ax = axes[0, col]
        ax2 = axes[1, col]

        obs_xy  = obs[idx]
        pred_xy = pred[idx]

        for ax_cur in [ax, ax2]:
            ax_cur.plot(obs_xy[:,0], obs_xy[:,1], "k-o", ms=4, lw=2, label="Observed" if col==0 else "")
            ax_cur.plot(pred_xy[:,0], pred_xy[:,1], "k--", ms=3, lw=1.5, alpha=0.7, label="Ground truth" if col==0 else "")
            ax_cur.plot(pred_xy[0,0], pred_xy[0,1], "k^", ms=6)
            ax_cur.plot(pred_xy[-1,0], pred_xy[-1,1], "k*", ms=8)

        # Top row: all 3 mean predictions
        for name, mu, color in model_preds:
            ax.plot(mu[col,:,0], mu[col,:,1], "-", color=color, lw=2, label=name if col==0 else "")
            ax.plot(mu[col,-1,0], mu[col,-1,1], "o", color=color, ms=7)
        ax.set_title(f"Sequence {col+1}", fontsize=11)
        ax.set_aspect("equal"); ax.grid(True, alpha=0.2)

        # Bottom row: Transformer samples (diversity)
        for k in range(10):
            ax2.plot(transf_samp[col,k,:,0], transf_samp[col,k,:,1],
                     "-", color="#59a14f", alpha=0.25, lw=1)
        ax2.plot(transf_mu[col,:,0], transf_mu[col,:,1], "-", color="#59a14f", lw=2.5)
        ax2.set_title(f"Transformer K=10 samples", fontsize=10)
        ax2.set_aspect("equal"); ax2.grid(True, alpha=0.2)

    # Legends
    handles = [mpatches.Patch(color=c, label=n) for n,_,c in model_preds]
    handles += [plt.Line2D([0],[0],color="k",lw=2,label="Observed"),
                plt.Line2D([0],[0],color="k",lw=1.5,ls="--",label="Ground truth")]
    axes[0,3].legend(handles=handles, fontsize=9, loc="center")
    axes[0,3].axis("off")
    axes[1,3].axis("off")

    fig.suptitle("Trajectory Predictions — zara2 test set (mean predictions + sample diversity)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("plots/trajectory_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ trajectory_comparison.png")

    # ── 4b: Diffusion samples vs Transformer samples ─────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    sample_data = [
        ("Transformer K=10", transf_samp, transf_mu, "#59a14f"),
        ("Diffusion K=10",   diff_samp,   diff_mu,   "#b07aa1"),
    ]

    for row, (name, samps, mu, color) in enumerate(sample_data):
        for col, idx_pos in enumerate(range(4)):
            if idx_pos >= len(idxs): break
            ax = axes[row, col]
            obs_xy  = obs[idxs[idx_pos]]
            pred_xy = pred[idxs[idx_pos]]
            ax.plot(obs_xy[:,0], obs_xy[:,1], "k-o", ms=4, lw=2)
            ax.plot(pred_xy[:,0], pred_xy[:,1], "k--", lw=1.5, alpha=0.7)
            for k in range(10):
                ax.plot(samps[idx_pos,k,:,0], samps[idx_pos,k,:,1],
                        "-", color=color, alpha=0.3, lw=1)
            ax.plot(mu[idx_pos,:,0], mu[idx_pos,:,1], "-", color=color, lw=2.5)
            ax.plot(mu[idx_pos,-1,0], mu[idx_pos,-1,1], "o", color=color, ms=7)
            ax.set_title(f"{'Obs' if row==0 else ''}", fontsize=9)
            ax.set_aspect("equal"); ax.grid(True, alpha=0.2)
            if col == 0: ax.set_ylabel(name, fontsize=10, fontweight="bold")

    fig.suptitle("Sample Diversity: Transformer vs Diffusion (K=10 samples per model)\nBlack dashed = ground truth",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig("plots/sample_diversity_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ sample_diversity_comparison.png")

except Exception as e:
    print(f"  [skip trajectory plots] {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5: FDE per scene (mirrors ADE plot)
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
x = np.arange(len(MODELS))
for i, scene in enumerate(SCENES):
    vals = [RESULTS[m]["fde"][i] for m in MODELS]
    bars = axes[i].bar(x, vals, color=[COLORS[m] for m in MODELS], edgecolor="white")
    axes[i].set_title(scene, fontsize=13, fontweight="bold")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels([m.replace("Social-","S-") for m in MODELS], rotation=45, ha="right", fontsize=8)
    axes[i].set_ylabel("FDE (m)" if i == 0 else "")
    best_idx = np.argmin(vals)
    bars[best_idx].set_edgecolor("black"); bars[best_idx].set_linewidth(2)

fig.suptitle("FDE per scene (↓ lower is better) — ETH/UCY Leave-One-Out", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/fde_per_scene.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ fde_per_scene.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 6: minADE@20 per scene
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=False)
for i, scene in enumerate(SCENES):
    vals = [RESULTS[m]["minADE"][i] for m in MODELS]
    bars = axes[i].bar(x, vals, color=[COLORS[m] for m in MODELS], edgecolor="white")
    axes[i].set_title(scene, fontsize=13, fontweight="bold")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels([m.replace("Social-","S-") for m in MODELS], rotation=45, ha="right", fontsize=8)
    axes[i].set_ylabel("minADE@20 (m)" if i == 0 else "")
    best_idx = np.argmin(vals)
    bars[best_idx].set_edgecolor("black"); bars[best_idx].set_linewidth(2)

fig.suptitle("minADE@20 per scene (↓ lower is better) — ETH/UCY Leave-One-Out", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/minade_per_scene.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ minade_per_scene.png")

print("\nAll plots saved to plots/")
