"""
gen_qualitative_fig.py  —  2-model qualitative figure (no Diffusion)
Social-LSTM vs Transformer on 4 zara1 test scenarios.
Run from year-long/:
    source crowdnav-env/bin/activate && python report/gen_qualitative_fig.py
Output → report/figures/fig03_qualitative.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(__file__), "figures")

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.labelsize":    9,
    "axes.titlesize":    9.5,
    "legend.fontsize":   8.5,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.alpha":        0.22,
    "grid.linestyle":    "--",
    "grid.color":        "#cccccc",
})

C_OBS  = "#1a2f6e"
C_GT   = "#cc2222"
C_SLSTM = "#4878CF"
C_TRSF  = "#2ca02c"

# ── Data ──────────────────────────────────────────────────────────────────────
from eth_ucy_analysis import extract_sequences_with_neighbours, SCENES, load_scene
from models.social_lstm import SocialLSTM
from models.trajectory_transformer import TrajectoryTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCENE  = "zara1"

data = load_scene(SCENES[SCENE])
obs_np, pred_np, nb_np, nm_np = extract_sequences_with_neighbours(
    data, obs_len=8, pred_len=12, max_neighbours=5)

n = len(obs_np)
split = int(0.8 * n)
obs_np  = obs_np[split:]
pred_np = pred_np[split:]
nb_np   = nb_np[split:]
nm_np   = nm_np[split:]

N = len(obs_np)
idxs = [int(N * f) for f in [0.05, 0.28, 0.55, 0.80]]

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading Social-LSTM …")
slstm = SocialLSTM(obs_len=8, pred_len=12, hidden_size=128,
                   embed_size=64, pooling_radius=2.0)
ck = torch.load(f"checkpoints/social_lstm_{SCENE}.pt", map_location=DEVICE)
slstm.load_state_dict(ck.get("model_state", ck.get("model_state_dict", ck)))
slstm = slstm.to(DEVICE).eval()

print("Loading Transformer …")
transf = TrajectoryTransformer(obs_len=8, pred_len=12, d_model=128,
                               nhead=4, num_enc=2, num_dec=2, max_nb=5)
ck = torch.load(f"checkpoints/transformer_{SCENE}.pt", map_location=DEVICE)
transf.load_state_dict(ck.get("model_state_dict", ck.get("model_state", ck)))
transf = transf.to(DEVICE).eval()

def predict(model, i):
    obs = torch.tensor(obs_np[[i]], dtype=torch.float32, device=DEVICE)
    nb  = torch.tensor(np.nan_to_num(nb_np[[i]], nan=0.0), dtype=torch.float32, device=DEVICE)
    nm  = torch.tensor(nm_np[[i]], dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        return model(obs, nb, nm)["mus"].cpu().numpy()[0]  # (T, 2)

def centre(traj, o):
    return traj - o

def clean_ax(ax):
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel(r"$x$ position (m)", fontsize=8, color="#555")
    ax.set_ylabel(r"$y$ position (m)", fontsize=8, color="#555")

# ── Figure: 2 rows × 4 cols ───────────────────────────────────────────────────
ROWS = [("Social-LSTM", slstm, C_SLSTM), ("Transformer ★", transf, C_TRSF)]
fig, axes = plt.subplots(2, 4, figsize=(12, 5.5), constrained_layout=True)

for row, (lbl, model, color) in enumerate(ROWS):
    for col, idx in enumerate(idxs):
        ax  = axes[row, col]
        o   = obs_np[idx, -1]
        oc  = centre(obs_np[idx], o)
        pc  = centre(pred_np[idx], o)
        mu  = centre(predict(model, idx), o)

        ade = np.linalg.norm(predict(model, idx) - pred_np[idx], axis=-1).mean()

        # Ground truth
        ax.plot(pc[:, 0], pc[:, 1], color=C_GT, lw=1.6, ls="--", alpha=0.8, zorder=3)
        ax.plot(pc[-1, 0], pc[-1, 1], "*", color=C_GT, ms=9, zorder=5)

        # Model mean
        ax.plot(mu[:, 0], mu[:, 1], color=color, lw=2.2, zorder=4)
        ax.plot(mu[-1, 0], mu[-1, 1], "o", color=color, ms=6,
                markeredgecolor="white", markeredgewidth=0.9, zorder=6)

        # Observed
        ax.plot(oc[:, 0], oc[:, 1], color=C_OBS, lw=2.2, zorder=4)
        ax.plot(oc[-1, 0], oc[-1, 1], "o", color=C_OBS, ms=6,
                markeredgecolor="white", markeredgewidth=0.9, zorder=6)

        ax.text(0.97, 0.04, f"ADE = {ade:.2f} m",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="#333",
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec="#cccccc", alpha=0.9))

        clean_ax(ax)
        if row == 0:
            ax.set_title(f"Scenario {col + 1}", fontsize=9.5, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"{lbl}\n" + r"$y$ position (m)",
                          fontsize=9, fontweight="bold", color=color)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_elements = [
    Line2D([0], [0], color=C_OBS,   lw=2.2, label="Observed (8 steps)"),
    Line2D([0], [0], color=C_GT,    lw=1.6, ls="--", label="Ground truth (12 steps)"),
    Line2D([0], [0], color=C_SLSTM, lw=2.2, label="Social-LSTM mean"),
    Line2D([0], [0], color=C_TRSF,  lw=2.2, label="Transformer mean ★"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           fontsize=9, frameon=True, edgecolor="#cccccc",
           bbox_to_anchor=(0.5, -0.05))

out = os.path.join(OUT, "fig03_qualitative.png")
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.05)
print(f"Saved → {out}")
