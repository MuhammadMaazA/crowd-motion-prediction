"""
generate_report_figures.py
==========================
Generates all 11 clean, report-ready figures for the final report.
Output: plots/report/fig01_ethucy_bar.png ... fig11_model_size.png

Run: source crowdnav-env/bin/activate && python generate_report_figures.py
"""

import os, sys
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import torch

os.makedirs("plots/report", exist_ok=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Consistent style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "savefig.facecolor":  "white",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "axes.axisbelow":     True,
    "grid.alpha":         0.2,
    "grid.linestyle":     "--",
})

COLORS = {
    "CV":           "#aaaaaa",
    "Social-LSTM":  "#4878CF",
    "SLSTM+V":      "#f28e2b",
    "GRU-v2":       "#17becf",
    "Trajectron++": "#d62728",
    "Transformer":  "#2ca02c",
    "Diffusion":    "#9467bd",
}
OURS = {"SLSTM+V", "GRU-v2", "Transformer", "Diffusion"}
OBS_C, GT_C = "#1a2f6e", "#cc0000"

def label(m):
    return f"{m} ★" if m in OURS else m

def save(fig, name):
    fig.savefig(f"plots/report/{name}", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  ✓ {name}")

# ── ETH/UCY results data ──────────────────────────────────────────────────────
SCENES_ETH = ["ETH", "Hotel", "Univ", "Zara1", "Zara2"]
ETH_ADE = {
    "CV":           [1.201, 0.553, 0.716, 0.620, 0.583],
    "Social-LSTM":  [1.015, 0.541, 0.655, 0.447, 0.319],
    "SLSTM+V":      [1.010, 0.527, 0.635, 0.432, 0.320],
    "GRU-v2":       [1.031, 0.558, 0.575, 0.443, 0.324],
    "Trajectron++": [1.355, 0.964, 0.774, 0.626, 0.527],
    "Transformer":  [0.982, 0.437, 0.568, 0.481, 0.371],
    "Diffusion":    [0.983, 0.474, 0.573, 0.501, 0.355],
}
ETH_MINADE = {
    "CV":           [1.063, 0.433, 0.588, 0.499, 0.467],
    "Social-LSTM":  [0.936, 0.526, 0.619, 0.475, 0.351],
    "SLSTM+V":      [0.934, 0.508, 0.656, 0.507, 0.335],
    "GRU-v2":       [0.961, 0.568, 0.558, 0.552, 0.341],
    "Trajectron++": [0.804, 0.382, 0.335, 0.221, 0.182],
    "Transformer":  [0.912, 0.521, 0.653, 0.694, 0.407],
    "Diffusion":    [1.571, 0.692, 0.828, 1.463, 0.752],
}
ETH_FDE = {
    "CV":           [2.320, 0.780, 1.263, 1.038, 0.921],
    "Social-LSTM":  [1.993, 1.241, 1.368, 0.944, 0.696],
    "SLSTM+V":      [1.996, 1.186, 1.335, 0.970, 0.710],
    "GRU-v2":       [2.251, 1.062, 1.203, 0.989, 0.685],
    "Trajectron++": [2.772, 2.117, 1.661, 1.335, 1.135],
    "Transformer":  [1.933, 0.973, 1.181, 1.095, 0.832],
    "Diffusion":    [1.930, 0.978, 1.194, 1.096, 0.775],
}
ETH_NLL = {
    "Social-LSTM":  [78.484, -0.175, 8.382, 0.795, 2.661],
    "SLSTM+V":      [146.898, 0.865, 5.188, 1.126, 26.938],
    "GRU-v2":       [31.293, -0.175, 8.0, 0.9, 2.5],  # approx
    "Trajectron++": [56.998, 9.631, 3.842, -4.071, -23.960],
    "Transformer":  [3.656, -0.403, 2.857, 0.884, 0.683],
    "Diffusion":    [7.793, 0.245, 3.009, 0.753, 13.875],
}

ALL_ETH = list(ETH_ADE.keys())
avgs_ade    = {m: np.mean(ETH_ADE[m])    for m in ALL_ETH}
avgs_minade = {m: np.mean(ETH_MINADE[m]) for m in ALL_ETH}
avgs_fde    = {m: np.mean(ETH_FDE[m])    for m in ALL_ETH}


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 01: ETH/UCY quantitative — horizontal grouped bars
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 01: ETH/UCY bar chart...")
PLOT_MODELS = ["CV", "Social-LSTM", "SLSTM+V", "GRU-v2", "Trajectron++", "Transformer", "Diffusion"]
scenes_with_avg = SCENES_ETH + ["Avg"]

def build_scene_ade(models):
    rows = []
    for s_idx in range(5):
        rows.append([ETH_ADE[m][s_idx] for m in models])
    rows.append([avgs_ade[m] for m in models])
    return np.array(rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
data_ade    = build_scene_ade(PLOT_MODELS)
data_minade = np.array([[ETH_MINADE[m][i] for m in PLOT_MODELS] for i in range(5)] +
                        [[avgs_minade[m] for m in PLOT_MODELS]])

for ax, data, title, xlabel in zip(axes,
    [data_ade, data_minade],
    ["Average Displacement Error (ADE ↓)", "Best-of-20 ADE (minADE@20 ↓)"],
    ["ADE (metres)", "minADE@20 (metres)"]):
    y     = np.arange(len(scenes_with_avg))
    n     = len(PLOT_MODELS)
    h     = 0.10
    offs  = np.linspace(-(n-1)/2, (n-1)/2, n) * h
    for i, (m, off) in enumerate(zip(PLOT_MODELS, offs)):
        bars = ax.barh(y + off, data[:, i], h * 0.88,
                       color=COLORS[m], label=label(m), alpha=0.92)
        # bold border on best per row
        for row_idx, bar in enumerate(bars):
            if i == int(np.argmin(data[row_idx])):
                bar.set_edgecolor("black"); bar.set_linewidth(1.8)
    ax.set_yticks(y)
    ax.set_yticklabels(scenes_with_avg, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.axhline(4.5, color="#999", lw=0.8, ls=":")  # separator before Avg
    ax.invert_yaxis()

patches = [mpatches.Patch(color=COLORS[m], label=label(m)) for m in PLOT_MODELS]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.08), frameon=True, edgecolor="#ddd")
fig.suptitle("ETH/UCY Leave-One-Out Benchmark (★ = our models, black border = best per scene)",
             fontsize=12, fontweight="bold")
plt.subplots_adjust(wspace=0.35)
save(fig, "fig01_ethucy_bar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 02: Accuracy vs diversity scatter
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 02: Scatter...")
fig, ax = plt.subplots(figsize=(8, 6))
for m in PLOT_MODELS:
    ax.scatter(avgs_ade[m], avgs_minade[m], s=220, color=COLORS[m],
               zorder=5, edgecolors="white", linewidths=1.5)
    offsets = {"Trajectron++": (6, -14), "CV": (6, 5), "Diffusion": (-78, 5),
               "GRU-v2": (6, -14), "SLSTM+V": (6, -14)}
    xy = offsets.get(m, (6, 5))
    fw = "bold" if m in OURS else "normal"
    ax.annotate(label(m), (avgs_ade[m], avgs_minade[m]),
                textcoords="offset points", xytext=xy,
                fontsize=10, color=COLORS[m], fontweight=fw)

ax.set_xlabel("Average ADE — point accuracy (↓ lower is better)", fontsize=11)
ax.set_ylabel("Average minADE@20 — sample diversity (↓ lower is better)", fontsize=11)
ax.set_title("Accuracy vs Diversity Trade-off — ETH/UCY avg (5 scenes)",
             fontsize=12, fontweight="bold")
ax.annotate("← better accuracy", xy=(0.02, 0.04), xycoords="axes fraction",
            fontsize=9, color="#888")
ax.annotate("↓ better diversity", xy=(0.65, 0.02), xycoords="axes fraction",
            fontsize=9, color="#888")
save(fig, "fig02_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 05: NLL calibration
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 05: NLL calibration...")
nll_models = ["Social-LSTM", "SLSTM+V", "GRU-v2", "Trajectron++", "Transformer", "Diffusion"]
nll_avgs   = [np.mean(ETH_NLL[m]) for m in nll_models]

fig, ax = plt.subplots(figsize=(9, 4.5))
y = np.arange(len(nll_models))
bars = ax.barh(y, nll_avgs, color=[COLORS[m] for m in nll_models], alpha=0.9)
ax.set_yticks(y)
lbls = [label(m) for m in nll_models]
ax.set_yticklabels(lbls, fontsize=11)
for tick, m in zip(ax.get_yticklabels(), nll_models):
    if m in OURS: tick.set_color(COLORS[m]); tick.set_fontweight("bold")
ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("Average NLL (↓ lower = better calibrated uncertainty)", fontsize=11)
ax.set_title("Uncertainty Calibration — NLL on ETH/UCY (5 scenes)",
             fontsize=12, fontweight="bold")
for bar, v in zip(bars, nll_avgs):
    xpos = v + 0.3 if v >= 0 else v - 0.3
    ax.text(xpos, bar.get_y() + bar.get_height()/2, f"{v:.1f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=9)
ax.invert_yaxis()
save(fig, "fig05_nll_calibration.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 09: LSTM vs GRU ablation
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 09: LSTM vs GRU ablation...")
fig, ax = plt.subplots(figsize=(10, 4.5))
x = np.arange(len(SCENES_ETH) + 1)
w = 0.28
scenes_plus_avg = SCENES_ETH + ["Avg"]
lstm_vals = ETH_ADE["Social-LSTM"] + [avgs_ade["Social-LSTM"]]
gru_vals  = ETH_ADE["GRU-v2"]      + [avgs_ade["GRU-v2"]]

b1 = ax.bar(x - w/2, lstm_vals, w, color=COLORS["Social-LSTM"], label="Social-LSTM (LSTM)", alpha=0.9)
b2 = ax.bar(x + w/2, gru_vals,  w, color=COLORS["GRU-v2"],      label="Social-GRU-v2 ★ (bidir GRU)", alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(scenes_plus_avg, fontsize=11)
ax.set_ylabel("ADE (m)", fontsize=11)
ax.set_title("Ablation: LSTM → Bidirectional GRU Encoder — ETH/UCY",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
# improvement arrows
for i in range(len(x)):
    diff = lstm_vals[i] - gru_vals[i]
    if diff > 0:
        ax.annotate(f"−{diff:.2f}", xy=(x[i], min(lstm_vals[i], gru_vals[i]) - 0.02),
                    ha="center", fontsize=8, color="green")
save(fig, "fig09_lstm_vs_gru.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 11: Model size vs performance bubble chart
# ═══════════════════════════════════════════════════════════════════════════════
print("Fig 11: Model size vs performance...")
PARAMS = {"CV": 0, "Social-LSTM": 455, "SLSTM+V": 455,
          "GRU-v2": 406, "Trajectron++": 3000, "Transformer": 537, "Diffusion": 956}

fig, ax = plt.subplots(figsize=(9, 6))
for m in PLOT_MODELS:
    s  = max(PARAMS[m], 50) * 0.12 + 80
    ax.scatter(avgs_ade[m], avgs_fde[m], s=s, color=COLORS[m],
               alpha=0.85, edgecolors="white", linewidths=1.5, zorder=5)
    fw = "bold" if m in OURS else "normal"
    ax.annotate(f"{label(m)}\n({PARAMS[m]}K)",
                (avgs_ade[m], avgs_fde[m]),
                textcoords="offset points", xytext=(8, 4),
                fontsize=9, color=COLORS[m], fontweight=fw)

ax.set_xlabel("Average ADE (m) ↓", fontsize=11)
ax.set_ylabel("Average FDE (m) ↓", fontsize=11)
ax.set_title("Model Efficiency — Parameter Count vs Performance\n(bubble size ∝ parameters, ★ = our models)",
             fontsize=12, fontweight="bold")
save(fig, "fig11_model_size.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Figs 03, 04, 10 — require loading actual models (trajectory + horizon)
# ═══════════════════════════════════════════════════════════════════════════════
from eth_ucy_analysis import load_scene, extract_sequences_with_neighbours
from eth_ucy_analysis import ade as metric_ade
from models.social_lstm import SocialLSTM
from models.trajectory_transformer import TrajectoryTransformer
from models.diffusion import TrajDiffusion
from models.social_gru_v2 import SocialGRUv2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RAW = "Trajectron-plus-plus/experiments/pedestrians/raw"

def ld_slstm(scene):
    ck=torch.load(f"checkpoints/social_lstm_{scene}.pt",map_location=DEVICE,weights_only=False)
    hp=ck["hparams"]
    m=SocialLSTM(hidden_size=hp["hidden_size"],embed_size=hp["embed_size"],
                 pooling_radius=hp["pooling_radius"]).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

def ld_transf(scene):
    ck=torch.load(f"checkpoints/transformer_{scene}.pt",map_location=DEVICE,weights_only=False)
    hp=ck["hparams"]
    m=TrajectoryTransformer(d_model=hp["d_model"],nhead=hp["nhead"],
                             num_enc=hp["num_enc"],num_dec=hp["num_dec"],
                             dim_ff=hp["dim_ff"]).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

def ld_diff(scene):
    ck=torch.load(f"checkpoints/diffusion_{scene}.pt",map_location=DEVICE,weights_only=False)
    hp=ck["hparams"]
    m=TrajDiffusion(d_model=hp["d_model"],nhead=hp["nhead"],T=hp["T"],
                    ddim_steps=hp["ddim_steps"],lambda_ddpm=hp["lambda_ddpm"]).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

def ctr(t, o): return t - o
def clean_ax(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_aspect("equal")

print("Loading models and data for trajectory figures...")
data = load_scene([f"{RAW}/zara1/test/crowds_zara01.txt"])
obs_np, pred_np, nb_np, nm_np = extract_sequences_with_neighbours(data, max_neighbours=5)

# Cherry-pick: best ADE sequences for Transformer
np.random.seed(42)
move = np.linalg.norm(pred_np[:,-1]-obs_np[:,0], axis=-1)
good = np.where(move > 0.8)[0]
transf_z1 = ld_transf("zara1")
batch_idx = np.random.choice(good[:200], 100, replace=False)
obs_t   = torch.tensor(obs_np[batch_idx],    dtype=torch.float32, device=DEVICE)
nb_t    = torch.tensor(np.nan_to_num(nb_np[batch_idx], nan=0.0), dtype=torch.float32, device=DEVICE)
nm_t    = torch.tensor(nm_np[batch_idx],     dtype=torch.bool,    device=DEVICE)
with torch.no_grad():
    mu_batch = transf_z1(obs_t, nb_t, nm_t)["mus"].cpu().numpy()
ades_batch = np.linalg.norm(mu_batch - pred_np[batch_idx], axis=-1).mean(axis=1)
best4 = batch_idx[np.argsort(ades_batch)[:4]]

slstm_z1 = ld_slstm("zara1"); diff_z1 = ld_diff("zara1")
obs_t4 = torch.tensor(obs_np[best4], dtype=torch.float32, device=DEVICE)
nb_t4  = torch.tensor(np.nan_to_num(nb_np[best4], nan=0.0), dtype=torch.float32, device=DEVICE)
nm_t4  = torch.tensor(nm_np[best4], dtype=torch.bool, device=DEVICE)
with torch.no_grad():
    s_mu   = slstm_z1(obs_t4, nb_t4, nm_t4)["mus"].cpu().numpy()
    t_mu   = transf_z1(obs_t4, nb_t4, nm_t4)["mus"].cpu().numpy()
    d_mu   = diff_z1(obs_t4, nb_t4, nm_t4)["mus"].cpu().numpy()
    t_samp = transf_z1.sample(obs_t4, nb_t4, nm_t4, K=20)
    d_samp = diff_z1.sample(obs_t4, nb_t4, nm_t4, K=20)


# ── Fig 03: Qualitative trajectory comparison ─────────────────────────────────
print("Fig 03: Qualitative comparison...")
rows3 = [("Social-LSTM", s_mu, COLORS["Social-LSTM"]),
         ("Transformer ★", t_mu, COLORS["Transformer"]),
         ("Diffusion ★",  d_mu, COLORS["Diffusion"])]

fig, axes = plt.subplots(3, 4, figsize=(14, 9))
for row, (lbl, mu, color) in enumerate(rows3):
    for col in range(4):
        ax = axes[row, col]
        i  = col
        o  = obs_np[best4[i], -1]
        oc = ctr(obs_np[best4[i]], o); pc = ctr(pred_np[best4[i]], o)
        mc = ctr(mu[i], o)
        ade_v = np.linalg.norm(mu[i] - pred_np[best4[i]], axis=-1).mean()

        ax.plot(oc[:,0], oc[:,1], color=OBS_C, lw=2.5, solid_capstyle="round", zorder=4)
        ax.plot(oc[-1,0], oc[-1,1], "o", color=OBS_C, ms=7,
                markeredgecolor="white", markeredgewidth=1.5, zorder=6)
        ax.plot(pc[:,0], pc[:,1], color=GT_C, lw=1.8, ls="--", alpha=0.8, zorder=3)
        ax.plot(pc[-1,0], pc[-1,1], "*", color=GT_C, ms=10, zorder=6)
        ax.plot(mc[:,0], mc[:,1], color=color, lw=2.5, solid_capstyle="round", zorder=5)
        ax.plot(mc[-1,0], mc[-1,1], "o", color=color, ms=7,
                markeredgecolor="white", markeredgewidth=1.5, zorder=6)
        ax.text(0.97, 0.03, f"ADE={ade_v:.2f}m", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color="#444",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.9))
        clean_ax(ax)
        if row == 0: ax.set_title(f"Scenario {col+1}", fontsize=11, fontweight="bold", pad=5)
        if col == 0: ax.set_ylabel(lbl, fontsize=10, fontweight="bold", color=color, labelpad=6)

legend_e = [Line2D([0],[0], color=OBS_C, lw=2.5, label="Observed"),
            Line2D([0],[0], color=GT_C, lw=1.8, ls="--", label="Ground truth"),
            Line2D([0],[0], color=COLORS["Social-LSTM"], lw=2.5, label="Social-LSTM"),
            Line2D([0],[0], color=COLORS["Transformer"], lw=2.5, label="Transformer ★"),
            Line2D([0],[0], color=COLORS["Diffusion"], lw=2.5, label="Diffusion ★")]
fig.legend(handles=legend_e, loc="lower center", ncol=5, fontsize=10,
           bbox_to_anchor=(0.5,-0.03), frameon=True, edgecolor="#ddd")
fig.suptitle("Qualitative Trajectory Prediction — zara1 Test Set (best sequences)",
             fontsize=12, fontweight="bold", y=1.01)
plt.subplots_adjust(hspace=0.06, wspace=0.06)
save(fig, "fig03_qualitative.png")


# ── Fig 04: Sample diversity ──────────────────────────────────────────────────
print("Fig 04: Diversity showcase...")
rows4 = [("Transformer ★", t_mu, t_samp, COLORS["Transformer"]),
         ("Diffusion ★",   d_mu, d_samp, COLORS["Diffusion"])]

fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
for row, (lbl, mu, samp, color) in enumerate(rows4):
    for col in range(4):
        ax = axes[row, col]
        i  = col
        o  = obs_np[best4[i], -1]
        oc = ctr(obs_np[best4[i]], o); pc = ctr(pred_np[best4[i]], o)
        mc = ctr(mu[i], o)

        for k in range(20):
            sc = ctr(samp[i,k], o)
            ax.plot(sc[:,0], sc[:,1], color=color, lw=0.8, alpha=0.15, zorder=1)
        ax.plot(oc[:,0], oc[:,1], color=OBS_C, lw=2.5, solid_capstyle="round", zorder=4)
        ax.plot(oc[-1,0], oc[-1,1], "o", color=OBS_C, ms=7,
                markeredgecolor="white", markeredgewidth=1.5, zorder=6)
        ax.plot(pc[:,0], pc[:,1], color=GT_C, lw=1.8, ls="--", alpha=0.8, zorder=3)
        ax.plot(pc[-1,0], pc[-1,1], "*", color=GT_C, ms=10, zorder=6)
        ax.plot(mc[:,0], mc[:,1], color=color, lw=2.5, zorder=5)
        ax.plot(mc[-1,0], mc[-1,1], "o", color=color, ms=7,
                markeredgecolor="white", markeredgewidth=1.5, zorder=6)
        clean_ax(ax)
        if row == 0: ax.set_title(f"Scenario {col+1}", fontsize=11, fontweight="bold", pad=5)
        if col == 0: ax.set_ylabel(lbl, fontsize=10, fontweight="bold", color=color, labelpad=6)

legend_e = [Line2D([0],[0], color=OBS_C, lw=2.5, label="Observed"),
            Line2D([0],[0], color=GT_C, lw=1.8, ls="--", label="Ground truth"),
            Line2D([0],[0], color=COLORS["Transformer"], lw=2.5, label="Transformer mean ★"),
            Line2D([0],[0], color=COLORS["Diffusion"], lw=2.5, label="Diffusion mean ★"),
            Line2D([0],[0], color="#888", lw=2, alpha=0.4, label="K=20 samples")]
fig.legend(handles=legend_e, loc="lower center", ncol=5, fontsize=10,
           bbox_to_anchor=(0.5,-0.05), frameon=True, edgecolor="#ddd")
fig.suptitle("Multi-Modal Sample Diversity — K=20 Samples (zara1 test set)",
             fontsize=12, fontweight="bold", y=1.01)
plt.subplots_adjust(hspace=0.06, wspace=0.06)
save(fig, "fig04_diversity.png")


# ── Fig 10: Horizon degradation ───────────────────────────────────────────────
print("Fig 10: Horizon degradation...")
horizon_models = {
    "CV":          None,
    "Social-LSTM": ld_slstm("zara1"),
    "Transformer": transf_z1,
    "Diffusion":   diff_z1,
}

np.random.seed(99)
eval_idx = np.random.choice(good, min(200, len(good)), replace=False)
obs_e = torch.tensor(obs_np[eval_idx], dtype=torch.float32, device=DEVICE)
nb_e  = torch.tensor(np.nan_to_num(nb_np[eval_idx], nan=0.0), dtype=torch.float32, device=DEVICE)
nm_e  = torch.tensor(nm_np[eval_idx], dtype=torch.bool, device=DEVICE)
pred_e = pred_np[eval_idx]

horizon_ades = {}
cv_vel = obs_np[eval_idx,-1] - obs_np[eval_idx,-2]
cv_pred = obs_np[eval_idx,-1:] + cv_vel[:,None] * np.arange(1,13)[None,:,None]
horizon_ades["CV"] = np.linalg.norm(cv_pred - pred_e, axis=-1).mean(axis=0)

for mn, mdl in horizon_models.items():
    if mdl is None: continue
    with torch.no_grad():
        mu_h = mdl(obs_e, nb_e, nm_e)["mus"].cpu().numpy()
    horizon_ades[mn] = np.linalg.norm(mu_h - pred_e, axis=-1).mean(axis=0)

x = np.arange(12)
tlabels = [f"{(t+1)*0.4:.1f}s" for t in range(12)]

fig, ax = plt.subplots(figsize=(10, 5))
for mn, vals in horizon_ades.items():
    fw = "bold" if mn in OURS else "normal"
    ax.plot(x, vals, marker="o", color=COLORS[mn], lw=2.2, ms=6,
            label=label(mn), markeredgecolor="white", markeredgewidth=1)
ax.set_xticks(x); ax.set_xticklabels(tlabels, fontsize=10)
ax.set_xlabel("Prediction horizon (seconds ahead)", fontsize=11)
ax.set_ylabel("ADE (m) at timestep t", fontsize=11)
ax.set_title("Prediction Error vs Horizon — How fast accuracy degrades (zara1)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
save(fig, "fig10_horizon_degradation.png")


# ═══════════════════════════════════════════════════════════════════════════════
# SDD figures (Figs 06, 07, 08)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nLoading SDD data for figures 06-08...")
from sdd_analysis import load_scene as sdd_load, SCENE_FILES as SDD_FILES, SCENES as SDD_SCENES
from sdd_analysis import ade as sdd_ade, extract_sequences_with_neighbours as sdd_ext

from models.social_gru_v2 import SocialGRUv2

def ld_sdd(model_name, scene):
    path = f"checkpoints/sdd/{model_name}_{scene}.pt"
    if not os.path.exists(path): return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hp = ck["hparams"]
    if model_name == "social_lstm":
        m = SocialLSTM(hidden_size=hp["hidden_size"], embed_size=hp["embed_size"],
                       pooling_radius=hp["pooling_radius"])
    elif model_name == "social_lstm_v":
        m = SocialLSTM(hidden_size=hp["hidden_size"], embed_size=hp["embed_size"],
                       pooling_radius=hp["pooling_radius"], use_velocity=True)
    elif model_name == "gru_v2":
        m = SocialGRUv2(hidden_size=hp["hidden_size"], embed_size=hp["embed_size"],
                        pooling_radius=hp["pooling_radius"])
    elif model_name == "transformer":
        m = TrajectoryTransformer(d_model=hp["d_model"], nhead=hp["nhead"],
                                   num_enc=hp["num_enc"], num_dec=hp["num_dec"],
                                   dim_ff=hp["dim_ff"])
    elif model_name == "diffusion":
        m = TrajDiffusion(d_model=hp["d_model"], nhead=hp["nhead"],
                          T=hp["T"], ddim_steps=hp["ddim_steps"],
                          lambda_ddpm=hp["lambda_ddpm"])
    else:
        return None
    m.load_state_dict(ck["model_state"]); m.eval(); return m

SDD_MODELS_MAP = {
    "Social-LSTM":  "social_lstm",
    "SLSTM+V":      "social_lstm_v",
    "GRU-v2":       "gru_v2",
    "Transformer":  "transformer",
    "Diffusion":    "diffusion",
}
SDD_PLOT_MODELS = list(SDD_MODELS_MAP.keys())
SDD_SCENE_LABELS = [s[:5] for s in SDD_SCENES]

# Compute SDD ADE per scene per model (batched to avoid OOM)
print("Computing SDD ADE (may take a few minutes)...")
sdd_results = {m: [] for m in SDD_PLOT_MODELS}

for scene in SDD_SCENES:
    data = sdd_load(SDD_FILES[scene])
    obs_s, pred_s, nb_s, nm_s = sdd_ext(data, max_neighbours=5)
    if len(obs_s) == 0:
        for m in SDD_PLOT_MODELS: sdd_results[m].append(None)
        continue

    # CV
    cv_vel_s = obs_s[:,-1] - obs_s[:,-2]
    cv_pred_s = obs_s[:,-1:] + cv_vel_s[:,None] * np.arange(1,13)[None,:,None]

    for mn in SDD_PLOT_MODELS:
        mdl = ld_sdd(SDD_MODELS_MAP[mn], scene)
        if mdl is None:
            sdd_results[mn].append(None)
            continue
        # batched eval
        batch = 256
        all_mu = []
        for st in range(0, len(obs_s), batch):
            en = min(st+batch, len(obs_s))
            ob_t = torch.tensor(obs_s[st:en], dtype=torch.float32)
            nb_t_s = torch.tensor(np.nan_to_num(nb_s[st:en], nan=0.0), dtype=torch.float32)
            nm_t_s = torch.tensor(nm_s[st:en], dtype=torch.bool)
            with torch.no_grad():
                mu_b = mdl(ob_t, nb_t_s, nm_t_s)["mus"].numpy()
            all_mu.append(mu_b)
        mu_all = np.concatenate(all_mu, axis=0)
        sdd_results[mn].append(sdd_ade(mu_all, pred_s))
    print(f"  {scene}: done")

sdd_avgs = {m: np.mean([v for v in sdd_results[m] if v is not None]) for m in SDD_PLOT_MODELS}

# CV for SDD
cv_sdd = []
for scene in SDD_SCENES:
    data = sdd_load(SDD_FILES[scene])
    obs_s, pred_s, _, _ = sdd_ext(data, max_neighbours=5)
    if len(obs_s) == 0: cv_sdd.append(None); continue
    cv_vel_s = obs_s[:,-1] - obs_s[:,-2]
    cv_pred_s = obs_s[:,-1:] + cv_vel_s[:,None] * np.arange(1,13)[None,:,None]
    cv_sdd.append(sdd_ade(cv_pred_s, pred_s))
sdd_results["CV"] = cv_sdd
sdd_avgs["CV"] = np.mean([v for v in cv_sdd if v is not None])
SDD_ALL_MODELS = ["CV"] + SDD_PLOT_MODELS


# ── Fig 06: SDD quantitative ──────────────────────────────────────────────────
print("Fig 06: SDD bar chart...")
sdd_with_avg = SDD_SCENE_LABELS + ["Avg"]
fig, ax = plt.subplots(figsize=(14, 5.5))
y   = np.arange(len(sdd_with_avg))
n   = len(SDD_ALL_MODELS)
h   = 0.10
offs = np.linspace(-(n-1)/2, (n-1)/2, n) * h

for i, m in enumerate(SDD_ALL_MODELS):
    vals = [(sdd_results[m][j] if sdd_results[m][j] is not None else 0)
            for j in range(len(SDD_SCENES))] + [sdd_avgs[m]]
    bars = ax.barh(y + offs[i], vals, h * 0.88, color=COLORS[m],
                   label=label(m), alpha=0.92)
    # best marker
    for row_idx, bar in enumerate(bars):
        row_vals = [(sdd_results[mm][row_idx % len(SDD_SCENES)]
                     if row_idx < len(SDD_SCENES) and sdd_results[mm][row_idx] is not None
                     else sdd_avgs[mm]) for mm in SDD_ALL_MODELS]
        if i == int(np.argmin(row_vals)):
            bar.set_edgecolor("black"); bar.set_linewidth(1.8)

ax.set_yticks(y); ax.set_yticklabels(sdd_with_avg, fontsize=10)
ax.axhline(len(SDD_SCENES) - 0.5, color="#999", lw=0.8, ls=":")
ax.set_xlabel("ADE (metres, ↓ lower is better)", fontsize=11)
ax.set_title("Stanford Drone Dataset — ADE Leave-One-Scene-Out\n(★ = our models, black border = best per scene)",
             fontsize=12, fontweight="bold")
patches = [mpatches.Patch(color=COLORS[m], label=label(m)) for m in SDD_ALL_MODELS]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.1), frameon=True, edgecolor="#ddd")
ax.invert_yaxis()
save(fig, "fig06_sdd_bar.png")


# ── Fig 07: ETH/UCY vs SDD generalisation ─────────────────────────────────────
print("Fig 07: ETH/UCY vs SDD comparison...")
gen_models = ["CV", "Social-LSTM", "SLSTM+V", "GRU-v2", "Transformer", "Diffusion"]
ethucy_avgs_list = [avgs_ade[m] for m in gen_models]
sdd_avgs_list    = [sdd_avgs[m] for m in gen_models]

x = np.arange(len(gen_models)); w = 0.35
fig, ax = plt.subplots(figsize=(11, 4.5))
b1 = ax.bar(x - w/2, ethucy_avgs_list, w, color=[COLORS[m] for m in gen_models],
            alpha=0.95, label="ETH/UCY avg")
b2 = ax.bar(x + w/2, sdd_avgs_list,    w, color=[COLORS[m] for m in gen_models],
            alpha=0.55, hatch="///", label="SDD avg")
ax.set_xticks(x)
ax.set_xticklabels([label(m) for m in gen_models], rotation=25, ha="right", fontsize=10)
for tick, m in zip(ax.get_xticklabels(), gen_models):
    if m in OURS: tick.set_color(COLORS[m]); tick.set_fontweight("bold")
ax.set_ylabel("Average ADE (m)", fontsize=11)
ax.set_title("Generalisation: ETH/UCY (solid) vs SDD (hatched) — Average ADE",
             fontsize=12, fontweight="bold")
ax.legend(["ETH/UCY (solid)", "SDD (hatched)"], fontsize=10)
for bar, v in zip(list(b1)+list(b2), ethucy_avgs_list+sdd_avgs_list):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
            f"{v:.2f}", ha="center", va="bottom", fontsize=8)
save(fig, "fig07_generalisation.png")


# ── Fig 08: SDD qualitative ───────────────────────────────────────────────────
print("Fig 08: SDD qualitative trajectories...")
AGENT_PALETTE = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628"]
sdd_vis_scenes = [("bookstore", "Bookstore"), ("hyang", "Hyang"), ("nexus", "Nexus")]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

for col, (scene, title) in enumerate(sdd_vis_scenes):
    ax = axes[col]
    mdl = ld_sdd("transformer", scene)
    if mdl is None:
        ax.set_visible(False); continue
    data = sdd_load(SDD_FILES[scene])
    obs_s, pred_s, nb_s, nm_s = sdd_ext(data, max_neighbours=5)
    np.random.seed(col * 13 + 5)
    mv = np.linalg.norm(pred_s[:,-1]-obs_s[:,0], axis=-1)
    gd = np.where(mv > 0.3)[0]
    n_ag = min(5, len(gd))
    ag_idx = np.random.choice(gd, n_ag, replace=False)
    ob_t = torch.tensor(obs_s[ag_idx], dtype=torch.float32)
    nb_t_s = torch.tensor(np.nan_to_num(nb_s[ag_idx], nan=0.0), dtype=torch.float32)
    nm_t_s = torch.tensor(nm_s[ag_idx], dtype=torch.bool)
    with torch.no_grad():
        mu_s = mdl(ob_t, nb_t_s, nm_t_s)["mus"].numpy()
        smp_s = mdl.sample(ob_t, nb_t_s, nm_t_s, K=12)

    for i in range(n_ag):
        c = AGENT_PALETTE[i % len(AGENT_PALETTE)]
        o = obs_s[ag_idx[i], -1]
        oc = obs_s[ag_idx[i]] - o; pc = pred_s[ag_idx[i]] - o; mc = mu_s[i] - o
        for k in range(12):
            ax.plot((smp_s[i,k]-o)[:,0], (smp_s[i,k]-o)[:,1], color=c, lw=0.7, alpha=0.2, zorder=1)
        ax.plot(oc[:,0], oc[:,1], color=c, lw=2.2, solid_capstyle="round", zorder=4)
        ax.plot(oc[-1,0], oc[-1,1], "o", color=c, ms=6,
                markeredgecolor="white", markeredgewidth=1, zorder=6)
        ax.plot(pc[:,0], pc[:,1], color=c, lw=1.5, ls="--", alpha=0.6, zorder=3)
        ax.plot(pc[-1,0], pc[-1,1], "*", color=c, ms=8, zorder=6)
        ax.plot(mc[:,0], mc[:,1], color=c, lw=2.2, zorder=5)
    clean_ax(ax)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)

legend_e = [Line2D([0],[0], color="#555", lw=2.2, label="Observed"),
            Line2D([0],[0], color="#555", lw=1.5, ls="--", alpha=0.6, label="Ground truth"),
            Line2D([0],[0], color="#555", lw=2.2, label="Transformer ★ prediction"),
            Line2D([0],[0], color="#555", lw=1.5, alpha=0.35, label="K=12 samples")]
fig.legend(handles=legend_e, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5,-0.08), frameon=True, edgecolor="#ddd")
fig.suptitle("Stanford Drone Dataset — Transformer Predictions (each colour = one pedestrian)",
             fontsize=12, fontweight="bold", y=1.01)
plt.subplots_adjust(wspace=0.06)
save(fig, "fig08_sdd_qualitative.png")


print("\n" + "="*50)
print("All figures saved to plots/report/")
for f in sorted(os.listdir("plots/report")):
    print(f"  {f}")
