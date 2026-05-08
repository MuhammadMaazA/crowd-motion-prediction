"""
Publication-quality plots matching the style of CVPR/ECCV trajectory prediction papers
(Social-GAN, Trajectron++, Social-STGCNN style).
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import torch

os.makedirs("plots/paper", exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Style: match published papers ────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "figure.dpi":        150,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAW = "Trajectron-plus-plus/experiments/pedestrians/raw"

from eth_ucy_analysis import load_scene, extract_sequences_with_neighbours
from models.social_lstm import SocialLSTM
from models.trajectory_transformer import TrajectoryTransformer
from models.diffusion import TrajDiffusion

# ── Model colours (Tableau palette as used in most papers) ───────────────────
MODEL_COLORS = {
    "Observed":     "#2d2d2d",
    "Ground Truth": "#555555",
    "CV":           "#aaaaaa",
    "Social-LSTM":  "#4878CF",
    "SLSTM+V":      "#6ACC65",
    "Transformer":  "#D65F5F",
    "Diffusion":    "#B47CC7",
    "Trajectron++": "#C4AD66",
}

# ── Load data & models ────────────────────────────────────────────────────────
def load_slstm(scene, v=False):
    s = "v" if v else ""
    ck = torch.load(f"checkpoints/social_lstm{s}_{scene}.pt", map_location=DEVICE, weights_only=False)
    hp = ck["hparams"]
    m  = SocialLSTM(hidden_size=hp["hidden_size"], embed_size=hp["embed_size"],
                    pooling_radius=hp["pooling_radius"],
                    use_velocity=hp.get("use_velocity",False)).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

def load_transf(scene):
    ck = torch.load(f"checkpoints/transformer_{scene}.pt", map_location=DEVICE, weights_only=False)
    hp = ck["hparams"]
    m  = TrajectoryTransformer(d_model=hp["d_model"],nhead=hp["nhead"],
                                num_enc=hp["num_enc"],num_dec=hp["num_dec"],
                                dim_ff=hp["dim_ff"]).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

def load_diff(scene):
    ck = torch.load(f"checkpoints/diffusion_{scene}.pt", map_location=DEVICE, weights_only=False)
    hp = ck["hparams"]
    m  = TrajDiffusion(d_model=hp["d_model"],nhead=hp["nhead"],
                       T=hp["T"],ddim_steps=hp["ddim_steps"],
                       lambda_ddpm=hp["lambda_ddpm"]).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

print("Loading data...")
scenes_data = {}
for scene, fname in [
    ("zara1", "crowds_zara01.txt"),
    ("hotel", "biwi_hotel.txt"),
    ("univ",  "students001.txt"),
]:
    data = load_scene([os.path.join(RAW, scene, "test", fname)])
    obs, pred, nb_obs, nb_mask = extract_sequences_with_neighbours(data, max_neighbours=5)
    scenes_data[scene] = (obs, pred, nb_obs, nb_mask)

# Use zara1 for main plots (clean, well-spaced trajectories)
np.random.seed(7)
obs_np, pred_np, nb_obs_np, nb_mask_np = scenes_data["zara1"]
# Pick sequences with decent movement (filter static ones)
movement = np.linalg.norm(pred_np[:,-1] - obs_np[:,0], axis=-1)
good = np.where(movement > 1.0)[0]
idxs = np.random.choice(good, size=min(6, len(good)), replace=False)

obs_t     = torch.tensor(obs_np[idxs],    dtype=torch.float32, device=DEVICE)
nb_obs_t  = torch.tensor(np.nan_to_num(nb_obs_np[idxs], nan=0.0), dtype=torch.float32, device=DEVICE)
nb_mask_t = torch.tensor(nb_mask_np[idxs], dtype=torch.bool, device=DEVICE)

print("Running inference...")
slstm = load_slstm("zara1")
transf = load_transf("zara1")
diff   = load_diff("zara1")

with torch.no_grad():
    slstm_mu    = slstm(obs_t, nb_obs_t, nb_mask_t)["mus"].cpu().numpy()
    transf_mu   = transf(obs_t, nb_obs_t, nb_mask_t)["mus"].cpu().numpy()
    diff_mu     = diff(obs_t, nb_obs_t, nb_mask_t)["mus"].cpu().numpy()
    transf_samp = transf.sample(obs_t, nb_obs_t, nb_mask_t, K=20)
    diff_samp   = diff.sample(obs_t, nb_obs_t, nb_mask_t, K=20)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Qualitative comparison grid — paper style
# Social-GAN / Trajectron++ style: rows=models, cols=scenarios
# ═══════════════════════════════════════════════════════════════════════════════

n_col = 4
fig_rows = [
    ("Social-LSTM",  slstm_mu,  None,         MODEL_COLORS["Social-LSTM"]),
    ("Transformer",  transf_mu, transf_samp,  MODEL_COLORS["Transformer"]),
    ("Diffusion",    diff_mu,   diff_samp,     MODEL_COLORS["Diffusion"]),
]

fig, axes = plt.subplots(len(fig_rows), n_col,
                         figsize=(3.2 * n_col, 2.8 * len(fig_rows)))

for row, (model_name, mu, samps, color) in enumerate(fig_rows):
    for col in range(n_col):
        ax = axes[row, col]
        si = col  # sequence index

        obs_xy  = obs_np[idxs[si]]
        pred_xy = pred_np[idxs[si]]

        # Observed path
        ax.plot(obs_xy[:,0], obs_xy[:,1],
                color=MODEL_COLORS["Observed"], lw=2.5, solid_capstyle="round", zorder=4)
        ax.plot(obs_xy[-1,0], obs_xy[-1,1], "o",
                color=MODEL_COLORS["Observed"], ms=7, zorder=5)

        # Ground truth future
        ax.plot(pred_xy[:,0], pred_xy[:,1],
                color=MODEL_COLORS["Ground Truth"], lw=2, ls="--",
                dashes=(5, 3), zorder=3, alpha=0.8)
        ax.plot(pred_xy[-1,0], pred_xy[-1,1], "*",
                color=MODEL_COLORS["Ground Truth"], ms=9, zorder=5)

        # K samples (faded)
        if samps is not None:
            for k in range(min(20, samps.shape[1])):
                ax.plot(samps[si,k,:,0], samps[si,k,:,1],
                        color=color, lw=0.8, alpha=0.15, zorder=2)

        # Mean prediction
        ax.plot(mu[si,:,0], mu[si,:,1],
                color=color, lw=2.5, solid_capstyle="round", zorder=4)
        ax.plot(mu[si,-1,0], mu[si,-1,1], "o",
                color=color, ms=7, zorder=5)

        # Clean axes
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)

        if row == 0:
            ax.set_title(f"Scenario {col+1}", fontsize=11, fontweight="bold", pad=6)
        if col == 0:
            ax.set_ylabel(model_name, fontsize=11, fontweight="bold",
                          color=color, labelpad=8)

# Legend
legend_elements = [
    Line2D([0],[0], color=MODEL_COLORS["Observed"], lw=2.5, label="Observed"),
    Line2D([0],[0], color=MODEL_COLORS["Ground Truth"], lw=2, ls="--", label="Ground Truth"),
    Line2D([0],[0], color=MODEL_COLORS["Social-LSTM"], lw=2.5, label="Social-LSTM"),
    Line2D([0],[0], color=MODEL_COLORS["Transformer"], lw=2.5, label="Transformer (ours)"),
    Line2D([0],[0], color=MODEL_COLORS["Diffusion"],   lw=2.5, label="Diffusion (ours)"),
    Line2D([0],[0], color=MODEL_COLORS["Transformer"], lw=1.5, alpha=0.4, label="Samples (K=20)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3,
           fontsize=10, bbox_to_anchor=(0.5, -0.04),
           frameon=True, edgecolor="#cccccc")

fig.suptitle("Qualitative Trajectory Prediction — zara1 Test Set",
             fontsize=13, fontweight="bold", y=1.01)
plt.subplots_adjust(hspace=0.12, wspace=0.08)
fig.savefig("plots/paper/qualitative_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ qualitative_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Clean ADE/FDE bar chart — CVPR table-companion style
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_AVG = {
    "CV":           {"ADE": 0.736, "FDE": 1.272, "minADE": 0.610, "minFDE": 0.815},
    "Social-LSTM":  {"ADE": 0.596, "FDE": 1.248, "minADE": 0.581, "minFDE": 0.597},
    "SLSTM+V":      {"ADE": 0.585, "FDE": 1.240, "minADE": 0.587, "minFDE": 0.585},
    "Trajectron++": {"ADE": 0.853, "FDE": 1.817, "minADE": 0.385, "minFDE": 0.674},
    "Transformer":  {"ADE": 0.568, "FDE": 1.203, "minADE": 0.638, "minFDE": 0.532},
    "Diffusion":    {"ADE": 0.579, "FDE": 1.195, "minADE": 1.174, "minFDE": 2.032},
}
BAR_COLORS = {m: MODEL_COLORS.get(m, "#888888") for m in RESULTS_AVG}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, (metric, title) in zip(axes, [
    ("ADE",    "Average Displacement Error (ADE ↓)"),
    ("minADE", "Best-of-20 ADE (minADE@20 ↓)"),
]):
    models = list(RESULTS_AVG.keys())
    vals   = [RESULTS_AVG[m][metric] for m in models]
    colors = [BAR_COLORS[m] for m in models]
    x = np.arange(len(models))

    bars = ax.bar(x, vals, color=colors, width=0.6, edgecolor="white",
                  linewidth=0.5, zorder=3)

    # Highlight best
    best = np.argmin(vals)
    bars[best].set_edgecolor("#1a1a1a"); bars[best].set_linewidth(2)

    # Value labels
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("metres", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

# Mark ours
for ax in axes:
    ax.get_xticklabels()[4].set_color(MODEL_COLORS["Transformer"])
    ax.get_xticklabels()[4].set_fontweight("bold")
    ax.get_xticklabels()[5].set_color(MODEL_COLORS["Diffusion"])
    ax.get_xticklabels()[5].set_fontweight("bold")

fig.suptitle("ETH/UCY Leave-One-Out — Average across 5 scenes",
             fontsize=12, fontweight="bold")
plt.subplots_adjust(wspace=0.3)
fig.savefig("plots/paper/ade_fde_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ ade_fde_bar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Per-scene table-style bar chart — like Table 1 companion figure
# ═══════════════════════════════════════════════════════════════════════════════

SCENES = ["ETH", "Hotel", "Univ", "Zara1", "Zara2"]
SCENE_ADE = {
    "Social-LSTM":  [1.015, 0.541, 0.655, 0.447, 0.319],
    "SLSTM+V":      [1.010, 0.527, 0.635, 0.432, 0.320],
    "Trajectron++": [1.355, 0.977, 0.772, 0.624, 0.535],
    "Transformer":  [0.982, 0.437, 0.568, 0.481, 0.371],
    "Diffusion":    [0.982, 0.485, 0.563, 0.503, 0.361],
}

plot_models = list(SCENE_ADE.keys())
x = np.arange(len(SCENES))
width = 0.15
offsets = np.linspace(-(len(plot_models)-1)/2, (len(plot_models)-1)/2, len(plot_models)) * width

fig, ax = plt.subplots(figsize=(12, 4.5))

for i, (m, offset) in enumerate(zip(plot_models, offsets)):
    vals = SCENE_ADE[m]
    bars = ax.bar(x + offset, vals, width * 0.9,
                  color=MODEL_COLORS.get(m, "#888"), label=m,
                  edgecolor="white", linewidth=0.3, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(SCENES, fontsize=12)
ax.set_ylabel("ADE (m)", fontsize=12)
ax.set_title("Per-scene ADE — ETH/UCY Leave-One-Out Benchmark",
             fontsize=12, fontweight="bold")
ax.yaxis.grid(True, alpha=0.3, zorder=0); ax.set_axisbelow(True)
ax.legend(fontsize=10, loc="upper right",
          frameon=True, edgecolor="#cccccc",
          ncol=2)
ax.set_ylim(0, 1.6)

# Bold "ours" labels
for label in ax.get_legend().get_texts():
    if label.get_text() in ("Transformer", "Diffusion", "SLSTM+V"):
        label.set_fontweight("bold")

fig.savefig("plots/paper/per_scene_grouped.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ per_scene_grouped.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Multi-modal sample visualisation — Trajectron++ paper style
# Shows K=20 samples fanning out from observation
# ═══════════════════════════════════════════════════════════════════════════════

# Pick 3 scenarios with clear multi-modal future
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

for col, si in enumerate([0, 2, 4]):
    ax = axes[col]
    obs_xy  = obs_np[idxs[si]]
    pred_xy = pred_np[idxs[si]]

    # Background grey samples from Diffusion
    for k in range(20):
        ax.plot(diff_samp[si,k,:,0], diff_samp[si,k,:,1],
                color=MODEL_COLORS["Diffusion"], lw=1, alpha=0.2, zorder=1)

    # Transformer samples
    for k in range(20):
        ax.plot(transf_samp[si,k,:,0], transf_samp[si,k,:,1],
                color=MODEL_COLORS["Transformer"], lw=1, alpha=0.2, zorder=1)

    # Observed (thick)
    ax.plot(obs_xy[:,0], obs_xy[:,1],
            color="black", lw=3, solid_capstyle="round", zorder=5)
    ax.plot(obs_xy[:,0], obs_xy[:,1], "o",
            color="black", ms=5, zorder=6)

    # GT future
    ax.plot(pred_xy[:,0], pred_xy[:,1],
            color="#555555", lw=2.5, ls=(0,(4,2)), zorder=4)
    ax.plot(pred_xy[-1,0], pred_xy[-1,1], "*",
            color="#555555", ms=10, zorder=6)

    # Mean predictions
    ax.plot(transf_mu[si,:,0], transf_mu[si,:,1],
            color=MODEL_COLORS["Transformer"], lw=2.5, zorder=5)
    ax.plot(diff_mu[si,:,0], diff_mu[si,:,1],
            color=MODEL_COLORS["Diffusion"], lw=2.5, zorder=5)

    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_title(f"Scenario {col+1}", fontsize=11, fontweight="bold")

legend_elems = [
    Line2D([0],[0], color="black", lw=3, label="Observed"),
    Line2D([0],[0], color="#555555", lw=2, ls=(0,(4,2)), label="Ground Truth"),
    Line2D([0],[0], color=MODEL_COLORS["Transformer"], lw=2.5, label="Transformer mean"),
    Line2D([0],[0], color=MODEL_COLORS["Diffusion"],   lw=2.5, label="Diffusion mean"),
    Line2D([0],[0], color=MODEL_COLORS["Transformer"], lw=2, alpha=0.4, label="K=20 samples"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=5,
           fontsize=9.5, bbox_to_anchor=(0.5, -0.1),
           frameon=True, edgecolor="#cccccc")

fig.suptitle("Multi-Modal Sample Diversity: Transformer vs Diffusion — zara1",
             fontsize=12, fontweight="bold", y=1.02)
plt.subplots_adjust(wspace=0.06)
fig.savefig("plots/paper/multimodal_samples.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ multimodal_samples.png")

print("\nPaper-style plots saved to plots/paper/:")
for f in sorted(os.listdir("plots/paper")):
    print(f"  {f}")
