"""
SDD Trajectory Prediction Plots — clean publication style
==========================================================
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import torch

WORK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK)
os.makedirs("plots/sdd", exist_ok=True)

# ── Global style (clean, publication-ready) ───────────────────────────────────
plt.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           11,
    "axes.titlesize":      12,
    "axes.labelsize":      11,
    "xtick.labelsize":     10,
    "ytick.labelsize":     10,
    "legend.fontsize":     10,
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.edgecolor":      "#cccccc",
    "axes.linewidth":      0.8,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "grid.color":          "#eeeeee",
    "grid.linewidth":      0.6,
    "axes.grid":           True,
    "axes.grid.axis":      "y",
    "xtick.bottom":        False,
    "ytick.left":          True,
    "xtick.major.size":    0,
    "savefig.facecolor":   "white",
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.08,
})

OBS_COLOR  = "#000000"
GT_COLOR   = "#1a44cc"
CV_COLOR   = "#2ca02c"
MODEL_COLORS = {
    "Social-LSTM":  "#4878CF",
    "SLSTM+V":      "#6ACC65",
    "GRU-v2":       "#e67e22",
    "Transformer":  "#D65F5F",
    "Diffusion":    "#9b59b6",
}
SAMPLE_ALPHA = 0.25
K = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = os.path.join(WORK, "checkpoints", "sdd")

# ── Load models ───────────────────────────────────────────────────────────────
from sdd_analysis import load_scene, extract_sequences_with_neighbours, SCENE_FILES
from models.social_lstm import SocialLSTM
from models.social_gru_v2 import SocialGRUv2
from models.trajectory_transformer import TrajectoryTransformer
from models.diffusion import TrajDiffusion
from models.cv_baseline import ConstantVelocityPredictor

def _load_slstm(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    hp = ck["hparams"]
    m  = SocialLSTM(hidden_size=hp["hidden_size"], embed_size=hp["embed_size"],
                    pooling_radius=hp["pooling_radius"],
                    use_velocity=hp.get("use_velocity", False)).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

def _load_gru_v2(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    hp = ck["hparams"]
    m  = SocialGRUv2(hidden_size=hp.get("hidden_size", 128),
                     embed_size=hp.get("embed_size", 64),
                     pooling_radius=hp.get("pooling_radius", 2.0)).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

def _load_transf(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    hp = ck["hparams"]
    m  = TrajectoryTransformer(d_model=hp.get("d_model",128), nhead=hp.get("nhead",4),
                                num_enc=hp.get("num_enc",2),  num_dec=hp.get("num_dec",2),
                                dim_ff=hp.get("dim_ff",128)).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

def _load_diff(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    hp = ck["hparams"]
    m  = TrajDiffusion(d_model=hp.get("d_model",128), nhead=hp.get("nhead",4),
                       T=hp.get("T",100), ddim_steps=hp.get("ddim_steps",20),
                       lambda_ddpm=hp.get("lambda_ddpm",0.1)).to(DEVICE).eval()
    m.load_state_dict(ck["model_state"]); return m

def load_models(scene):
    models = {}
    p = lambda name: os.path.join(CKPT, f"{name}_{scene}.pt")
    if os.path.exists(p("social_lstm")):   models["Social-LSTM"]  = _load_slstm(p("social_lstm"))
    if os.path.exists(p("social_lstm_v")): models["SLSTM+V"]      = _load_slstm(p("social_lstm_v"))
    if os.path.exists(p("gru_v2")):        models["GRU-v2"]       = _load_gru_v2(p("gru_v2"))
    if os.path.exists(p("transformer")):   models["Transformer"]  = _load_transf(p("transformer"))
    if os.path.exists(p("diffusion")):     models["Diffusion"]    = _load_diff(p("diffusion"))
    return models

def get_samples(model, obs_t, nb_obs_t, nb_mask_t):
    with torch.no_grad():
        samps = model.sample(obs_t, nb_obs_t, nb_mask_t, K=K)  # (1, K, T, 2)
    return samps[0]  # (K, T, 2)

# ── Qualitative plot: one sequence, all models ────────────────────────────────

def plot_qualitative(scene, n_sequences=4, seed=42):
    """
    Clean qualitative panel — reference image style:
    • Black observed (dots + line)
    • Blue GT dashed (squares)
    • Thin coral sample trajectories with x endpoints
    • Mean prediction in model colour (solid)
    • Light grid, no spines, equal aspect
    """
    print(f"  Loading data for {scene}...")
    data = load_scene(SCENE_FILES[scene])
    obs_np, pred_np, nb_obs_np, nb_mask_np = extract_sequences_with_neighbours(
        data, obs_len=8, pred_len=12, max_neighbours=5)
    if len(obs_np) == 0:
        print(f"  No sequences for {scene}, skipping"); return

    np.random.seed(seed)
    move = np.linalg.norm(pred_np[:, -1] - obs_np[:, 0], axis=-1)
    good = np.where(move > 0.5)[0]
    if len(good) < n_sequences:
        good = np.arange(len(obs_np))
    idxs = np.random.choice(good, size=min(n_sequences, len(good)), replace=False)

    print(f"  Loading models for {scene}...")
    models = load_models(scene)
    cv     = ConstantVelocityPredictor(noise_std=0.25)

    model_names = ["CV"] + list(models.keys())
    n_cols = len(model_names)
    n_rows = len(idxs)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.8 * n_cols, 2.8 * n_rows),
                             facecolor="white")
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_color = {"CV": CV_COLOR}
    col_color.update(MODEL_COLORS)
    for col, mname in enumerate(model_names):
        c = col_color.get(mname, "#333333")
        axes[0, col].set_title(mname, fontsize=11, fontweight="bold", color=c, pad=6)

    for row, idx in enumerate(idxs):
        obs    = obs_np[idx]
        pred   = pred_np[idx]
        origin = obs[-1]
        obs_c  = obs  - origin
        pred_c = pred - origin

        obs_t     = torch.tensor(obs_np[[idx]], dtype=torch.float32, device=DEVICE)
        nb_obs_t  = torch.tensor(np.nan_to_num(nb_obs_np[[idx]], nan=0.0),
                                 dtype=torch.float32, device=DEVICE)
        nb_mask_t = torch.tensor(nb_mask_np[[idx]], dtype=torch.bool, device=DEVICE)

        cv_samps = cv.predict_samples(obs_np[[idx]], K=K, pred_len=12)[0] - origin
        all_samps = {"CV": cv_samps}
        for mname, model in models.items():
            s = get_samples(model, obs_t, nb_obs_t, nb_mask_t)
            s_np = s.cpu().numpy() if hasattr(s, "cpu") else np.array(s)
            all_samps[mname] = s_np - origin

        for col, mname in enumerate(model_names):
            ax     = axes[row, col]
            samps  = all_samps[mname]
            mcolor = CV_COLOR if mname == "CV" else MODEL_COLORS.get(mname, "#888888")

            # samples — thin coral
            for k in range(K):
                ax.plot(samps[k, :, 0], samps[k, :, 1],
                        color="#f08080", alpha=0.18, lw=0.7, zorder=2)
                ax.plot(samps[k, -1, 0], samps[k, -1, 1],
                        "x", color="#f08080", alpha=0.40, ms=3, zorder=2)
            # mean prediction
            ax.plot(samps[0, :, 0], samps[0, :, 1],
                    color=mcolor, lw=2.0, alpha=0.95, zorder=3)
            # observed
            ax.plot(obs_c[:, 0], obs_c[:, 1],
                    color=OBS_COLOR, lw=2.2, marker="o", ms=4, zorder=5,
                    solid_capstyle="round")
            # GT future
            ax.plot(pred_c[:, 0], pred_c[:, 1],
                    color=GT_COLOR, lw=2.0, linestyle="--",
                    marker="s", ms=4, zorder=5)

            ax.grid(True, color="#eeeeee", linewidth=0.5, zorder=0)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.set_aspect("equal")
            ax.set_facecolor("white")

    legend_els = [
        Line2D([0], [0], color=OBS_COLOR, lw=2, marker="o", ms=5, label="Observed"),
        Line2D([0], [0], color=GT_COLOR,  lw=2, linestyle="--", marker="s", ms=5, label="Ground truth"),
        Line2D([0], [0], color="#f08080", lw=1.5, alpha=0.7, label="Samples (K=20)"),
    ] + [mpatches.Patch(color=CV_COLOR if m == "CV" else MODEL_COLORS.get(m, "#888"), label=m)
         for m in model_names]

    fig.legend(handles=legend_els, loc="lower center", ncol=len(legend_els),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"SDD — {scene}   (8 obs → 12 pred,  K=20 samples)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = f"plots/sdd/qualitative_{scene}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Bar charts: ADE/FDE per scene ─────────────────────────────────────────────

RESULTS = {
    # From evaluate_sdd.py output
    "ADE": {
        "CV":          [0.759, 0.718, 1.298, 1.068, 0.874, 1.236, 0.879, 0.576],
        "Social-LSTM": [0.510, 0.465, 0.926, 0.831, 0.596, 0.958, 0.583, 0.507],
        "SLSTM+V":     [0.508, 0.451, 0.920, 0.832, 0.594, 0.948, 0.586, 0.405],
        "GRU-v2":      [0.505, 0.463, 0.933, 0.823, 0.591, 0.965, 0.588, 0.468],
        "Transformer": [0.504, 0.454, 0.939, 0.826, 0.604, 0.968, 0.591, 0.366],
        "Diffusion":   [0.513, 0.464, 0.933, 0.837, 0.596, 0.976, 0.589, 0.476],
    },
    "FDE": {
        "CV":          [1.273, 1.170, 2.378, 1.987, 1.499, 2.288, 1.496, 0.865],
        "Social-LSTM": [1.028, 0.967, 1.829, 1.732, 1.190, 1.918, 1.138, 1.065],
        "SLSTM+V":     [1.036, 0.933, 1.819, 1.742, 1.190, 1.944, 1.158, 0.897],
        "GRU-v2":      [1.021, 1.011, 1.846, 1.724, 1.197, 2.052, 1.156, 1.023],
        "Transformer": [1.008, 0.922, 1.840, 1.715, 1.206, 1.916, 1.179, 0.820],
        "Diffusion":   [1.023, 0.924, 1.831, 1.723, 1.172, 1.930, 1.154, 0.920],
    },
    "minADE@20": {
        "CV":          [0.635, 0.597, 1.164, 0.935, 0.744, 1.098, 0.747, 0.457],
        "Social-LSTM": [0.628, 0.589, 0.980, 0.904, 0.673, 1.122, 0.688, 0.680],
        "SLSTM+V":     [0.611, 0.542, 0.976, 0.911, 0.705, 1.050, 0.706, 0.538],
        "GRU-v2":      [0.611, 0.582, 0.970, 0.943, 0.738, 1.069, 0.759, 0.675],
        "Transformer": [0.612, 0.581, 1.011, 0.953, 0.727, 1.069, 0.768, 0.740],
        "Diffusion":   [1.196, 1.306, 1.685, 1.896, 1.568, 2.959, 1.866, 1.205],
    },
    "minFDE@20": {
        "CV":          [0.826, 0.750, 1.893, 1.500, 1.027, 1.771, 1.011, 0.449],
        "Social-LSTM": [0.475, 0.468, 0.850, 0.778, 0.535, 0.863, 0.528, 0.470],
        "SLSTM+V":     [0.466, 0.451, 0.830, 0.756, 0.545, 0.840, 0.528, 0.415],
        "GRU-v2":      [0.466, 0.464, 0.858, 0.762, 0.562, 0.858, 0.556, 0.471],
        "Transformer": [0.475, 0.455, 0.870, 0.786, 0.574, 0.847, 0.565, 0.495],
        "Diffusion":   [2.200, 2.478, 3.045, 3.591, 2.858, 5.410, 3.454, 2.240],
    },
}
SCENES    = ["bookstore", "coupa", "deathCircle", "gates", "hyang", "little", "nexus", "quad"]
BAR_COLORS = {
    "CV":          "#aaaaaa",
    "Social-LSTM": "#4878CF",
    "SLSTM+V":     "#6ACC65",
    "GRU-v2":      "#e67e22",
    "Transformer": "#D65F5F",
    "Diffusion":   "#9b59b6",
}


def plot_per_scene_bars():
    """Clean per-scene ADE and FDE grouped bar charts."""
    models = list(BAR_COLORS.keys())
    x      = np.arange(len(SCENES))
    n      = len(models)
    width  = 0.70 / n  # total bar group width = 0.70
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

    for metric, ylabel in [("ADE", "ADE (m)"), ("FDE", "FDE (m)")]:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        for i, m in enumerate(models):
            ax.bar(x + offsets[i], RESULTS[metric][m], width * 0.92,
                   label=m, color=BAR_COLORS[m], zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(SCENES, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_title(f"SDD — {metric} per scene  (leave-one-out, 8 scenes)",
                     fontweight="bold")
        ax.legend(frameon=False, ncol=len(models), loc="upper center",
                  bbox_to_anchor=(0.5, 1.13), fontsize=9.5)
        ax.set_xlim(-0.55, len(SCENES) - 0.45)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        out = f"plots/sdd/{metric.lower()}_per_scene.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


def plot_avg_summary():
    """Clean horizontal summary: ADE / FDE / minADE@20 / minFDE@20 avg."""
    models   = list(BAR_COLORS.keys())
    metrics  = ["ADE", "FDE", "minADE@20", "minFDE@20"]
    labels   = ["ADE", "FDE", "minADE@20", "minFDE@20"]
    avgs     = {met: [np.mean(RESULTS[met][m]) for m in models] for met in metrics}

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=False)
    fig.suptitle("SDD — Average across 8 scenes", fontsize=13, fontweight="bold", y=1.02)

    for ax, met, lab in zip(axes, metrics, labels):
        vals   = avgs[met]
        colors = [BAR_COLORS[m] for m in models]
        bars   = ax.bar(range(len(models)), vals, color=colors, zorder=3, width=0.6)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        ax.set_title(lab, fontweight="bold", fontsize=11)
        ax.set_ylabel("metres", fontsize=10)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        # value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.008 * max(vals),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        # highlight best (lowest) bar
        best = np.argmin(vals)
        bars[best].set_edgecolor("#111111")
        bars[best].set_linewidth(1.8)

    plt.tight_layout()
    out = "plots/sdd/summary_bars.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_diversity_vs_accuracy():
    """Scatter: ADE vs minADE@20 — clean, no clutter."""
    models = list(BAR_COLORS.keys())
    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    for m in models:
        x = np.mean(RESULTS["ADE"][m])
        y = np.mean(RESULTS["minADE@20"][m])
        ax.scatter(x, y, color=BAR_COLORS[m], s=140, zorder=5,
                   edgecolors="white", linewidths=1.2)
        ax.annotate(m, (x, y), textcoords="offset points",
                    xytext=(7, 3), fontsize=9.5)

    ax.set_xlabel("ADE ↓  (m)")
    ax.set_ylabel("minADE@20 ↓  (m)")
    ax.set_title("Accuracy vs Diversity tradeoff — SDD",
                 fontweight="bold")
    out = "plots/sdd/accuracy_vs_diversity.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_radar():
    """Clean radar chart — 4 learned models only."""
    models   = ["Social-LSTM", "SLSTM+V", "GRU-v2", "Transformer"]
    metrics  = ["ADE", "FDE", "minADE@20", "minFDE@20"]
    n        = len(metrics)
    angles   = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles  += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.set_facecolor("white")

    for m in models:
        vals = [np.mean(RESULTS[met][m]) for met in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, lw=2, label=m, color=BAR_COLORS[m])
        ax.fill(angles, vals, alpha=0.06, color=BAR_COLORS[m])

    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=10)
    ax.set_title("SDD — model comparison\n(avg over 8 scenes)",
                 fontsize=11, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9,
              frameon=False)
    ax.grid(color="#dddddd", linewidth=0.6)
    out = "plots/sdd/radar.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Generating SDD bar/summary plots ===")
    plot_per_scene_bars()
    plot_avg_summary()
    plot_diversity_vs_accuracy()
    plot_radar()

    print("\n=== Generating qualitative trajectory plots ===")
    # Use a few well-behaved scenes for qualitative
    for scene in ["coupa", "bookstore", "quad"]:
        print(f"\n--- {scene} ---")
        plot_qualitative(scene, n_sequences=4, seed=7)

    print("\nAll SDD plots saved to plots/sdd/")
