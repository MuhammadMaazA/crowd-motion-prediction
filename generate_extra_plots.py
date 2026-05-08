"""
Extra report plots:
  1. Prediction horizon degradation (ADE at each future timestep 1..12)
  2. Uncertainty ellipse visualisation
  3. Error distribution violin plots
  4. Model size vs performance bubble chart
  5. Training curves (from checkpoint data)
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import torch

os.makedirs("plots", exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eth_ucy_analysis import load_scene, extract_sequences_with_neighbours
from models.social_lstm import SocialLSTM
from models.trajectory_transformer import TrajectoryTransformer
from models.diffusion import TrajDiffusion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAW = "Trajectron-plus-plus/experiments/pedestrians/raw"

COLORS = {
    "CV":"#aaaaaa","Social-LSTM":"#4e79a7","SLSTM+V":"#f28e2b",
    "Trajectron++":"#e15759","Transformer":"#59a14f","Diffusion":"#b07aa1",
}


# ── Load models & data ────────────────────────────────────────────────────────

def load_slstm(scene, velocity=False):
    suffix = "v" if velocity else ""
    ck = torch.load(f"checkpoints/social_lstm{suffix}_{scene}.pt", map_location=DEVICE, weights_only=False)
    hp = ck["hparams"]
    m = SocialLSTM(hidden_size=hp["hidden_size"], embed_size=hp["embed_size"],
                   pooling_radius=hp["pooling_radius"],
                   use_velocity=hp.get("use_velocity", False)).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

def load_transformer(scene):
    ck = torch.load(f"checkpoints/transformer_{scene}.pt", map_location=DEVICE, weights_only=False)
    hp = ck["hparams"]
    m = TrajectoryTransformer(d_model=hp["d_model"], nhead=hp["nhead"],
                               num_enc=hp["num_enc"], num_dec=hp["num_dec"],
                               dim_ff=hp["dim_ff"]).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

def load_diffusion(scene):
    ck = torch.load(f"checkpoints/diffusion_{scene}.pt", map_location=DEVICE, weights_only=False)
    hp = ck["hparams"]
    m = TrajDiffusion(d_model=hp["d_model"], nhead=hp["nhead"],
                      T=hp["T"], ddim_steps=hp["ddim_steps"],
                      lambda_ddpm=hp["lambda_ddpm"]).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

print("Loading zara2 data...")
data = load_scene([os.path.join(RAW, "zara2", "test", "crowds_zara02.txt")])
obs_np, pred_np, nb_obs_np, nb_mask_np = extract_sequences_with_neighbours(data, max_neighbours=5)

np.random.seed(0)
N_eval = min(200, len(obs_np))
idxs = np.random.choice(len(obs_np), N_eval, replace=False)
obs_np  = obs_np[idxs];  pred_np  = pred_np[idxs]
nb_obs_np = nb_obs_np[idxs]; nb_mask_np = nb_mask_np[idxs]

obs_t    = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
nb_obs_t = torch.tensor(np.nan_to_num(nb_obs_np, nan=0.0), dtype=torch.float32, device=DEVICE)
nb_mask_t= torch.tensor(nb_mask_np, dtype=torch.bool, device=DEVICE)

print("Running model inference...")
models_dict = {}
try: models_dict["Social-LSTM"]  = load_slstm("zara2")
except: pass
try: models_dict["SLSTM+V"]     = load_slstm("zara2", velocity=True)
except: pass
try: models_dict["Transformer"] = load_transformer("zara2")
except: pass
try: models_dict["Diffusion"]   = load_diffusion("zara2")
except: pass

preds_dict = {}
with torch.no_grad():
    for name, model in models_dict.items():
        preds_dict[name] = model(obs_t, nb_obs_t, nb_mask_t)
        print(f"  {name}: done")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Prediction horizon degradation
# ═══════════════════════════════════════════════════════════════════════════════

timestep_labels = [f"{(t+1)*0.4:.1f}s" for t in range(12)]
x = np.arange(12)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ADE at each timestep (using mean prediction)
for name, preds in preds_dict.items():
    mus = preds["mus"].cpu().numpy()                         # (N, 12, 2)
    step_ade = np.linalg.norm(mus - pred_np, axis=-1).mean(axis=0)  # (12,)
    axes[0].plot(x, step_ade, marker="o", color=COLORS[name], label=name, lw=2, ms=5)

# CV baseline
cv_vel = obs_np[:, -1] - obs_np[:, -2]
cv_pred = obs_np[:, -1:] + cv_vel[:, None] * (np.arange(1, 13)[None, :, None])
cv_ade_step = np.linalg.norm(cv_pred - pred_np, axis=-1).mean(axis=0)
axes[0].plot(x, cv_ade_step, marker="o", color=COLORS["CV"], label="CV", lw=2, ms=5, linestyle="--")

axes[0].set_xticks(x)
axes[0].set_xticklabels(timestep_labels, rotation=45)
axes[0].set_xlabel("Prediction horizon", fontsize=11)
axes[0].set_ylabel("ADE (m) at timestep t", fontsize=11)
axes[0].set_title("Prediction error vs horizon\n(how fast accuracy degrades)", fontsize=12, fontweight="bold")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

# Final displacement growth (FDE at each step = distance from start)
for name, preds in preds_dict.items():
    mus = preds["mus"].cpu().numpy()
    step_fde = np.linalg.norm(mus - pred_np, axis=-1).mean(axis=0)
    axes[1].plot(x, np.cumsum(step_fde)/np.arange(1,13), marker="s",
                 color=COLORS[name], label=name, lw=2, ms=5)
axes[1].plot(x, np.cumsum(cv_ade_step)/np.arange(1,13), marker="s",
             color=COLORS["CV"], label="CV", lw=2, ms=5, linestyle="--")
axes[1].set_xticks(x)
axes[1].set_xticklabels(timestep_labels, rotation=45)
axes[1].set_xlabel("Prediction horizon", fontsize=11)
axes[1].set_ylabel("Cumulative avg ADE (m)", fontsize=11)
axes[1].set_title("Cumulative prediction error vs horizon", fontsize=12, fontweight="bold")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle("Prediction Horizon Analysis — zara2 (4.8s total, 0.4s/step)", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/horizon_degradation.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ horizon_degradation.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: Uncertainty ellipse visualisation
# ═══════════════════════════════════════════════════════════════════════════════

# Pick 2 sequences and show predicted uncertainty ellipses at each timestep
seq_idxs = [0, 3]

fig, axes = plt.subplots(len(seq_idxs), len(preds_dict), figsize=(4*len(preds_dict), 4*len(seq_idxs)))
if len(seq_idxs) == 1: axes = axes[np.newaxis, :]
if len(preds_dict) == 1: axes = axes[:, np.newaxis]

for row, si in enumerate(seq_idxs):
    for col, (name, preds) in enumerate(preds_dict.items()):
        ax = axes[row, col]
        mus    = preds["mus"].cpu().numpy()[si]      # (12, 2)
        sigmas = preds["sigmas"].cpu().numpy()[si]   # (12, 2)
        rhos   = preds["rhos"].cpu().numpy()[si, :, 0]  # (12,)

        obs_xy  = obs_np[si]
        pred_xy = pred_np[si]

        # Observed trajectory
        ax.plot(obs_xy[:,0], obs_xy[:,1], "k-o", ms=5, lw=2, zorder=5)
        # Ground truth future
        ax.plot(pred_xy[:,0], pred_xy[:,1], "k--", ms=3, lw=1.5, alpha=0.6, zorder=5)
        # Predicted mean
        ax.plot(mus[:,0], mus[:,1], "-", color=COLORS[name], lw=2, zorder=4)

        # Uncertainty ellipses at each timestep
        alpha_vals = np.linspace(0.6, 0.15, 12)
        for t in range(12):
            sx, sy = sigmas[t, 0], sigmas[t, 1]
            rho    = rhos[t]
            # 2-sigma ellipse
            angle = np.degrees(0.5 * np.arctan2(2*rho*sx*sy, sx**2 - sy**2))
            w_ = 2 * 2 * sx
            h_ = 2 * 2 * sy
            ell = Ellipse((mus[t,0], mus[t,1]), width=w_, height=h_, angle=angle,
                          color=COLORS[name], alpha=alpha_vals[t], zorder=3)
            ax.add_patch(ell)
            ax.plot(mus[t,0], mus[t,1], ".", color=COLORS[name], ms=5, zorder=6)

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        if row == 0: ax.set_title(name, fontsize=11, fontweight="bold", color=COLORS[name])
        if col == 0: ax.set_ylabel(f"Sequence {row+1}", fontsize=10)

fig.suptitle("Predicted Uncertainty Ellipses (2σ) — Growing uncertainty over time\nBlack dashed = ground truth",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/uncertainty_ellipses.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ uncertainty_ellipses.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3: Error distribution violin plots
# ═══════════════════════════════════════════════════════════════════════════════

all_ades = {}
for name, preds in preds_dict.items():
    mus = preds["mus"].cpu().numpy()
    per_seq_ade = np.linalg.norm(mus - pred_np, axis=-1).mean(axis=1)  # (N,)
    all_ades[name] = per_seq_ade

# CV
cv_pred_full = obs_np[:, -1:] + (obs_np[:, -1] - obs_np[:, -2])[:, None] * np.arange(1,13)[None,:,None]
cv_ade_seq   = np.linalg.norm(cv_pred_full - pred_np, axis=-1).mean(axis=1)
all_ades["CV"] = cv_ade_seq

plot_order  = ["CV", "Social-LSTM", "SLSTM+V", "Transformer", "Diffusion"]
plot_order  = [m for m in plot_order if m in all_ades]
violin_data = [all_ades[m] for m in plot_order]

fig, ax = plt.subplots(figsize=(10, 5))
parts = ax.violinplot(violin_data, positions=range(len(plot_order)),
                      showmeans=True, showmedians=True, showextrema=False)

for i, (pc, m) in enumerate(zip(parts["bodies"], plot_order)):
    pc.set_facecolor(COLORS[m])
    pc.set_alpha(0.7)
parts["cmeans"].set_color("black"); parts["cmeans"].set_linewidth(2)
parts["cmedians"].set_color("red"); parts["cmedians"].set_linewidth(1.5)

ax.set_xticks(range(len(plot_order)))
ax.set_xticklabels(plot_order, fontsize=10)
ax.set_ylabel("Per-sequence ADE (m)", fontsize=11)
ax.set_title("Error Distribution per Model — zara2\n(black line = mean, red = median)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.2, axis="y")
ax.set_ylim(0, None)

# Annotate means
for i, m in enumerate(plot_order):
    mean_val = np.mean(all_ades[m])
    ax.text(i, mean_val + 0.02, f"{mean_val:.3f}", ha="center", fontsize=8.5, fontweight="bold")

fig.tight_layout()
fig.savefig("plots/error_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ error_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 4: Model size vs performance bubble chart
# ═══════════════════════════════════════════════════════════════════════════════

avg_ade_all = {
    "CV":           0.736,
    "Social-LSTM":  0.596,
    "SLSTM+V":      0.585,
    "Trajectron++": 0.853,
    "Transformer":  0.568,
    "Diffusion":    0.579,
}
avg_minade_all = {
    "CV":           0.610,
    "Social-LSTM":  0.581,
    "SLSTM+V":      0.587,
    "Trajectron++": 0.385,
    "Transformer":  0.638,
    "Diffusion":    1.174,
}
params = {
    "CV":           0,
    "Social-LSTM":  455,
    "SLSTM+V":      455,
    "Trajectron++": 3000,
    "Transformer":  537,
    "Diffusion":    956,
}  # in thousands

fig, ax = plt.subplots(figsize=(9, 6))
for m in avg_ade_all:
    size = max(params[m], 50)  # min bubble size
    ax.scatter(avg_ade_all[m], avg_minade_all[m],
               s=size/5 + 100, color=COLORS[m], alpha=0.8,
               edgecolors="white", linewidths=1.5, zorder=5)
    offset = (0.01, 0.01)
    if m == "Trajectron++": offset = (0.01, -0.04)
    if m == "CV": offset = (0.01, 0.02)
    ax.annotate(f"{m}\n({params[m]}K params)",
                (avg_ade_all[m], avg_minade_all[m]),
                textcoords="offset points", xytext=(8, 4), fontsize=8.5)

ax.set_xlabel("Average ADE (m) ↓ — point prediction accuracy", fontsize=11)
ax.set_ylabel("Average minADE@20 (m) ↓ — sample diversity", fontsize=11)
ax.set_title("Model Size vs Performance — ETH/UCY avg\n(bubble size ∝ parameter count)", fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3)

# Add ideal region annotation
ax.annotate("ideal", xy=(0.5, 0.35), fontsize=10, color="green", alpha=0.5,
            fontweight="bold")
ax.annotate("", xy=(0.52, 0.37), xytext=(0.62, 0.45),
            arrowprops=dict(arrowstyle="->", color="green", alpha=0.4))

fig.tight_layout()
fig.savefig("plots/model_size_vs_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ model_size_vs_performance.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 5: Sigma (uncertainty) growth over prediction horizon
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
timestep_labels = [f"{(t+1)*0.4:.1f}s" for t in range(12)]
x = np.arange(12)

for name, preds in preds_dict.items():
    sigmas = preds["sigmas"].cpu().numpy()               # (N, 12, 2)
    mean_sigma_x = sigmas[:, :, 0].mean(axis=0)         # (12,)
    mean_sigma_y = sigmas[:, :, 1].mean(axis=0)
    mean_sigma   = (mean_sigma_x + mean_sigma_y) / 2

    axes[0].plot(x, mean_sigma_x, color=COLORS[name], lw=2, label=name)
    axes[1].plot(x, mean_sigma,   color=COLORS[name], lw=2, label=name)

for ax, ylabel, title in zip(axes,
    ["σx (m)", "Mean σ = (σx+σy)/2 (m)"],
    ["Predicted uncertainty σx over horizon", "Mean predicted uncertainty over horizon"]):
    ax.set_xticks(x)
    ax.set_xticklabels(timestep_labels, rotation=45)
    ax.set_xlabel("Prediction horizon", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

fig.suptitle("Predicted uncertainty growth — well-calibrated models should grow with horizon",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig("plots/uncertainty_growth.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ uncertainty_growth.png")

print("\nAll extra plots saved to plots/")
