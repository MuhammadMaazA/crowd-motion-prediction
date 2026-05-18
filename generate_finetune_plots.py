"""
generate_finetune_plots.py
==========================
Evaluates from-scratch vs SDD-pretrained+fine-tuned models on ETH/UCY,
then generates publication-quality comparison plots.

Outputs: plots/finetune/fig_ft_*.png

Run:
    source crowdnav-env/bin/activate
    python generate_finetune_plots.py [--no-eval] [--save-json results_ft.json]
"""

import os, sys, json, argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")

WORK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from eth_ucy_analysis import (
    load_scene, extract_sequences, extract_sequences_with_neighbours,
    ade, fde, best_of_k_ade, best_of_k_fde
)
from models.social_lstm import SocialLSTM, bivariate_gaussian_nll
from models.trajectory_transformer import TrajectoryTransformer
from models.diffusion import TrajDiffusion
from models.social_gru_v2 import SocialGRUv2

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
RAW = os.path.join(WORK, "Trajectron-plus-plus/experiments/pedestrians/raw")
SCENE_FILES = {
    "eth":   [os.path.join(RAW, "eth",   "test", "biwi_eth.txt")],
    "hotel": [os.path.join(RAW, "hotel", "test", "biwi_hotel.txt")],
    "univ":  [os.path.join(RAW, "univ",  "test", "students001.txt"),
              os.path.join(RAW, "univ",  "test", "students003.txt")],
    "zara1": [os.path.join(RAW, "zara1", "test", "crowds_zara01.txt")],
    "zara2": [os.path.join(RAW, "zara2", "test", "crowds_zara02.txt")],
}
SCENES   = ["eth", "hotel", "univ", "zara1", "zara2"]
SCENE_LABELS = ["ETH", "Hotel", "Univ", "Zara1", "Zara2"]
OBS_LEN, PRED_LEN, K = 8, 12, 20
K_DIFFUSION = 10   # fewer samples for Diffusion (DDIM is expensive)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = ["Social-LSTM", "SLSTM+V", "GRU-v2", "Transformer", "Diffusion"]

COLORS_SCRATCH = {
    "Social-LSTM": "#4878CF",
    "SLSTM+V":     "#f28e2b",
    "GRU-v2":      "#17becf",
    "Transformer": "#2ca02c",
    "Diffusion":   "#9467bd",
}
COLORS_FT = {k: c for k, c in COLORS_SCRATCH.items()}  # same hue, hatched

os.makedirs("plots/finetune", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint paths
# ─────────────────────────────────────────────────────────────────────────────
CKPT_SCRATCH = {
    "Social-LSTM": lambda s: f"checkpoints/social_lstm_{s}.pt",
    "SLSTM+V":     lambda s: f"checkpoints/social_lstmv_{s}.pt",
    "GRU-v2":      lambda s: f"checkpoints/social_gruv2_{s}.pt",
    "Transformer": lambda s: f"checkpoints/transformer_{s}.pt",
    "Diffusion":   lambda s: f"checkpoints/diffusion_{s}.pt",
}
CKPT_FT = {
    "Social-LSTM": lambda s: f"checkpoints/ft_social_lstm_{s}.pt",
    "SLSTM+V":     lambda s: f"checkpoints/ft_social_lstmv_{s}.pt",
    "GRU-v2":      lambda s: f"checkpoints/ft_social_gruv2_{s}.pt",
    "Transformer": lambda s: f"checkpoints/ft_transformer_{s}.pt",
    "Diffusion":   lambda s: f"checkpoints/ft_diffusion_{s}.pt",
}

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_name, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    hp   = ckpt["hparams"]
    if model_name in ("Social-LSTM", "SLSTM+V"):
        m = SocialLSTM(
            obs_len=OBS_LEN, pred_len=PRED_LEN,
            hidden_size=hp["hidden_size"], embed_size=hp["embed_size"],
            pooling_radius=hp["pooling_radius"],
            use_velocity=hp.get("use_velocity", False),
        ).to(DEVICE)
    elif model_name == "GRU-v2":
        m = SocialGRUv2(
            obs_len=OBS_LEN, pred_len=PRED_LEN,
            hidden_size=hp["hidden_size"], embed_size=hp["embed_size"],
            pooling_radius=hp["pooling_radius"],
            use_velocity=hp.get("use_velocity", False),
        ).to(DEVICE)
    elif model_name == "Transformer":
        m = TrajectoryTransformer(
            obs_len=OBS_LEN, pred_len=PRED_LEN,
            d_model=hp.get("d_model", 128), nhead=hp.get("nhead", 4),
            num_enc=hp.get("num_enc", 2),   num_dec=hp.get("num_dec", 2),
            dim_ff=hp.get("dim_ff", 128),
        ).to(DEVICE)
    elif model_name == "Diffusion":
        m = TrajDiffusion(
            obs_len=OBS_LEN, pred_len=PRED_LEN,
            d_model=hp.get("d_model", 128), nhead=hp.get("nhead", 4),
            T=hp.get("T", 100), ddim_steps=hp.get("ddim_steps", 20),
            lambda_ddpm=hp.get("lambda_ddpm", 0.1),
        ).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    return m


EVAL_BATCH = 256   # process at most this many trajectories at once

MAX_SEQ_PER_SCENE = 2000   # cap to keep evaluation fast; 2k is statistically robust

# Scene data cache: load each scene once and reuse across all models
_SCENE_CACHE = {}

def get_scene_data(scene):
    if scene not in _SCENE_CACHE:
        print(f"  [cache] Loading {scene} test sequences...", flush=True)
        data = load_scene(SCENE_FILES[scene])
        obs, pred, nb_obs, nb_mask = extract_sequences_with_neighbours(
            data, obs_len=OBS_LEN, pred_len=PRED_LEN, max_neighbours=5
        )
        N = len(obs)
        if N > MAX_SEQ_PER_SCENE:
            rng = np.random.default_rng(42)
            idx = rng.choice(N, MAX_SEQ_PER_SCENE, replace=False)
            idx.sort()
            obs, pred, nb_obs, nb_mask = obs[idx], pred[idx], nb_obs[idx], nb_mask[idx]
            print(f"  [cache] {scene}: {N} → subsampled {MAX_SEQ_PER_SCENE}", flush=True)
        else:
            print(f"  [cache] {scene}: {N} sequences", flush=True)
        _SCENE_CACHE[scene] = (obs, pred, nb_obs, nb_mask)
    return _SCENE_CACHE[scene]


def eval_model(model, scene, model_name=""):
    k = K_DIFFUSION if model_name == "Diffusion" else K
    obs, pred, nb_obs, nb_mask = get_scene_data(scene)
    if len(obs) == 0:
        return None

    all_mus, all_samples, all_nll = [], [], []
    N = len(obs)
    for i in range(0, N, EVAL_BATCH):
        obs_b     = torch.tensor(obs[i:i+EVAL_BATCH],  dtype=torch.float32).to(DEVICE)
        pred_b    = torch.tensor(pred[i:i+EVAL_BATCH], dtype=torch.float32).to(DEVICE)
        nb_obs_b  = torch.tensor(np.nan_to_num(nb_obs[i:i+EVAL_BATCH],  nan=0.0),
                                 dtype=torch.float32).to(DEVICE)
        nb_mask_b = torch.tensor(nb_mask[i:i+EVAL_BATCH], dtype=torch.bool).to(DEVICE)
        with torch.no_grad():
            out      = model(obs_b, nb_obs_b, nb_mask_b)
            mus_b    = out["mus"].cpu().numpy()
            samp_b   = model.sample(obs_b, nb_obs_b, nb_mask_b, K=k)  # (B,K,T,2)
            nll_b    = bivariate_gaussian_nll(out, pred_b).item()
        all_mus.append(mus_b)
        all_samples.append(samp_b)
        all_nll.append(nll_b)

    mus_np    = np.concatenate(all_mus,     axis=0)  # (N, T, 2)
    samples   = np.concatenate(all_samples, axis=0)  # (N, K, T, 2)
    nll_val   = float(np.mean(all_nll))

    return {
        "ade":       float(ade(mus_np, pred)),
        "fde":       float(fde(mus_np, pred)),
        "minADE_20": float(best_of_k_ade(samples, pred)),
        "minFDE_20": float(best_of_k_fde(samples, pred)),
        "nll":       float(nll_val),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded from-scratch ETH/UCY results (validated in generate_report_figures.py)
# ─────────────────────────────────────────────────────────────────────────────
_SCRATCH_ADE = {
    "Social-LSTM": [1.015, 0.541, 0.655, 0.447, 0.319],
    "SLSTM+V":     [1.010, 0.527, 0.635, 0.432, 0.320],
    "GRU-v2":      [1.031, 0.558, 0.575, 0.443, 0.324],
    "Transformer": [0.982, 0.437, 0.568, 0.481, 0.371],
    "Diffusion":   [0.983, 0.474, 0.573, 0.501, 0.355],
}
_SCRATCH_FDE = {
    "Social-LSTM": [1.993, 1.241, 1.368, 0.944, 0.696],
    "SLSTM+V":     [1.996, 1.186, 1.335, 0.970, 0.710],
    "GRU-v2":      [2.251, 1.062, 1.203, 0.989, 0.685],
    "Transformer": [1.933, 0.973, 1.181, 1.095, 0.832],
    "Diffusion":   [1.930, 0.978, 1.194, 1.096, 0.775],
}
_SCRATCH_MINADE = {
    "Social-LSTM": [0.936, 0.526, 0.619, 0.475, 0.351],
    "SLSTM+V":     [0.934, 0.508, 0.656, 0.507, 0.335],
    "GRU-v2":      [0.961, 0.568, 0.558, 0.552, 0.341],
    "Transformer": [0.912, 0.521, 0.653, 0.694, 0.407],
    "Diffusion":   [1.571, 0.692, 0.828, 1.463, 0.752],
}
_SCRATCH_MINFDE = {
    "Social-LSTM": [0.870, 1.020, 1.135, 0.920, 0.640],
    "SLSTM+V":     [0.870, 0.940, 1.150, 0.870, 0.635],
    "GRU-v2":      [1.020, 0.950, 1.020, 0.990, 0.640],
    "Transformer": [0.775, 0.665, 1.060, 0.981, 0.680],
    "Diffusion":   [2.120, 1.090, 1.440, 2.140, 1.020],
}


def build_scratch_results():
    """Build scratch results dict from hardcoded validated numbers."""
    results_scratch = {m: {} for m in MODELS}
    for m in MODELS:
        for si, s in enumerate(SCENES):
            results_scratch[m][s] = {
                "ade":       _SCRATCH_ADE[m][si],
                "fde":       _SCRATCH_FDE[m][si],
                "minADE_20": _SCRATCH_MINADE[m][si],
                "minFDE_20": _SCRATCH_MINFDE[m][si],
                "nll":       None,
            }
    return results_scratch


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(ft_only=False):
    results = {"scratch": {m: {} for m in MODELS},
               "ft":      {m: {} for m in MODELS}}

    if ft_only:
        print("Using hardcoded from-scratch results (validated).")
        results["scratch"] = build_scratch_results()
        variants = [("ft", CKPT_FT)]
    else:
        variants = [("scratch", CKPT_SCRATCH), ("ft", CKPT_FT)]

    # Pre-load all scene data once (avoids repeated O(N²) neighbour computation)
    print("\nPre-loading scene test data (this may take a minute for 'univ')...")
    for scene in SCENES:
        get_scene_data(scene)
    print("Scene data loaded.\n")

    for variant, ckpt_fns in variants:
        print(f"\n=== Evaluating {variant} models ===")
        for model_name in MODELS:
            print(f"  {model_name}:")
            for scene in SCENES:
                ckpt_path = os.path.join(WORK, ckpt_fns[model_name](scene))
                if not os.path.exists(ckpt_path):
                    print(f"    [{scene}] MISSING: {ckpt_path}")
                    results[variant][model_name][scene] = None
                    continue
                try:
                    model = load_model(model_name, ckpt_path)
                    r = eval_model(model, scene, model_name=model_name)
                    results[variant][model_name][scene] = r
                    print(f"    [{scene}]  ADE={r['ade']:.3f}  FDE={r['fde']:.3f}  "
                          f"minADE@20={r['minADE_20']:.3f}")
                    del model
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"    [{scene}] ERROR: {e}")
                    results[variant][model_name][scene] = None
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.facecolor": "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "axes.axisbelow":    True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
})

def save(fig, name):
    fig.savefig(f"plots/finetune/{name}", dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  saved: plots/finetune/{name}")


def get_metric(results, variant, model, scene, metric):
    r = results[variant][model].get(scene)
    return r[metric] if r else None


def scene_avg(results, variant, model, metric):
    vals = [get_metric(results, variant, model, s, metric) for s in SCENES]
    vals = [v for v in vals if v is not None]
    return np.mean(vals) if vals else None


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Rendered results table  — scratch vs fine-tuned (all scenes + avg)
# ─────────────────────────────────────────────────────────────────────────────
def plot_overall_comparison(results):
    """Clean table: one row per model, columns = scenes, colour-coded ft rows."""
    metrics   = [("ade", "ADE"), ("fde", "FDE"), ("minADE_20", "minADE@20")]
    scenes_wa = SCENES + ["avg"]
    scene_la  = SCENE_LABELS + ["Avg"]

    MODEL_COLORS = ["#ddeeff", "#fff3cc", "#ddffee", "#f3ddff", "#ffe0e0"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("white")

    for ax, (metric, mlabel) in zip(axes, metrics):
        ax.set_title(f"{mlabel}  (↓ lower is better)", fontsize=13, fontweight="bold", pad=14)
        ax.axis("off")

        def _val(variant, m, s):
            if s == "avg":
                v = scene_avg(results, variant, m, metric)
            else:
                v = get_metric(results, variant, m, s, metric)
            return v if v is not None else np.nan

        sc_vals = np.array([[_val("scratch", m, s) for s in scenes_wa] for m in MODELS])
        ft_vals = np.array([[_val("ft",      m, s) for s in scenes_wa] for m in MODELS])

        # Build cell_text and cell_colors
        # Columns: [Model, ETH, Hotel, Univ, Zara1, Zara2, Avg]
        col_labels_full = ["Model"] + scene_la
        n_cols = len(col_labels_full)

        cell_text   = [col_labels_full]
        cell_colors = [["#d0d0d0"] * n_cols]

        for mi, model in enumerate(MODELS):
            mc = MODEL_COLORS[mi]
            # scratch row
            sc_row = [f"{model}\n(scratch)"]
            sc_bg  = [mc]
            for ci, s in enumerate(scenes_wa):
                sv = sc_vals[mi, ci]
                sc_row.append(f"{sv:.3f}" if not np.isnan(sv) else "—")
                sc_bg.append(mc)
            cell_text.append(sc_row)
            cell_colors.append(sc_bg)

            # ft row
            ft_row = ["  ↳ fine-tuned"]
            ft_bg  = ["white"]
            for ci, s in enumerate(scenes_wa):
                sv = sc_vals[mi, ci]
                fv = ft_vals[mi, ci]
                ft_row.append(f"{fv:.3f}" if not np.isnan(fv) else "—")
                if not np.isnan(sv) and not np.isnan(fv):
                    delta = (fv - sv) / sv
                    if delta < -0.02:
                        ft_bg.append("#aaeaaa")   # green
                    elif delta > 0.02:
                        ft_bg.append("#f8c8c8")   # red
                    else:
                        ft_bg.append("#fffbe6")   # neutral
                else:
                    ft_bg.append("white")
            cell_text.append(ft_row)
            cell_colors.append(ft_bg)

        tbl = ax.table(
            cellText=cell_text,
            cellLoc="center",
            loc="center",
            cellColours=cell_colors,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.0, 2.0)

        # Bold header row and Avg column
        for col in range(n_cols):
            tbl[0, col].set_text_props(fontweight="bold")
        for row in range(len(cell_text)):
            tbl[row, n_cols - 1].set_text_props(fontweight="bold")
            tbl[row, 0].set_text_props(fontweight="bold", ha="left")

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#aaeaaa", edgecolor="#aaa", label="Fine-tuned improved ≥2%"),
        Patch(facecolor="#fffbe6", edgecolor="#aaa", label="Fine-tuned similar (<2%)"),
        Patch(facecolor="#f8c8c8", edgecolor="#aaa", label="Fine-tuned worse ≥2%"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, -0.01), frameon=True, edgecolor="#ccc")
    fig.suptitle("SDD Pre-training → ETH/UCY Fine-tuning  vs  From-Scratch Training",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "fig_ft_01_overall.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Δ% improvement heatmap — clean, one value per cell
# ─────────────────────────────────────────────────────────────────────────────
def plot_improvement_heatmap(results):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("white")

    for ax, metric, title in zip(
        axes,
        ["ade",    "fde"],
        ["ADE",    "FDE"],
    ):
        mat = np.full((len(MODELS), len(SCENES) + 1), np.nan)  # +1 for avg col
        scenes_wa = SCENES + ["avg"]
        for i, m in enumerate(MODELS):
            for j, s in enumerate(SCENES):
                sc = get_metric(results, "scratch", m, s, metric)
                ft = get_metric(results, "ft",      m, s, metric)
                if sc and ft:
                    mat[i, j] = (sc - ft) / sc * 100
            # avg column
            sc_a = scene_avg(results, "scratch", m, metric)
            ft_a = scene_avg(results, "ft",      m, metric)
            if sc_a and ft_a:
                mat[i, -1] = (sc_a - ft_a) / sc_a * 100

        vmax = max(10.0, np.nanmax(np.abs(mat)))
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

        xlabels = SCENE_LABELS + ["Avg"]
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, fontsize=11)
        ax.set_yticks(range(len(MODELS)))
        ax.set_yticklabels(MODELS, fontsize=11)
        ax.set_title(f"{title}  Δ%  (green = improvement)", fontsize=13,
                     fontweight="bold", pad=10)
        ax.tick_params(length=0)
        ax.grid(False)

        # Vertical separator before Avg column
        ax.axvline(len(SCENES) - 0.5, color="#999", lw=1.5, ls="--")

        # Annotate each cell
        for i in range(len(MODELS)):
            for j in range(len(xlabels)):
                v = mat[i, j]
                if not np.isnan(v):
                    color = "white" if abs(v) > vmax * 0.6 else "#222"
                    weight = "bold" if j == len(SCENES) else "normal"
                    ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                            fontsize=10, color=color, fontweight=weight)

        plt.colorbar(im, ax=ax, shrink=0.8, label="% reduction (positive = better)")

    fig.suptitle("Fine-tuning % Improvement over From-Scratch",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig_ft_02_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Dot/line plot — scratch vs ft per scene per model
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_scene(results):
    """Connected dot plot: scratch (filled circle) → ft (open diamond) per scene."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.patch.set_facecolor("white")

    x = np.arange(len(SCENES))
    offsets = np.linspace(-0.3, 0.3, len(MODELS))

    for ax, metric, ylabel in zip(axes, ["ade", "fde"], ["ADE (m)", "FDE (m)"]):
        for mi, model in enumerate(MODELS):
            sc_vals = [get_metric(results, "scratch", model, s, metric) for s in SCENES]
            ft_vals = [get_metric(results, "ft",      model, s, metric) for s in SCENES]
            xpos = x + offsets[mi]
            c = COLORS_SCRATCH[model]

            # Draw connecting line scratch → ft
            for xi, (sv, fv) in enumerate(zip(sc_vals, ft_vals)):
                if sv is not None and fv is not None:
                    ax.plot([xpos[xi], xpos[xi]], [sv, fv],
                            color=c, lw=1.5, alpha=0.5, zorder=1)

            # Scratch: filled circle
            ax.scatter(xpos, sc_vals, marker="o", s=55, color=c,
                       zorder=3, label=f"{model} scratch")
            # Fine-tuned: open diamond
            ax.scatter(xpos, ft_vals, marker="D", s=40, color=c,
                       facecolors="white", edgecolors=c, linewidths=1.8,
                       zorder=4, label=f"{model} ft")

        ax.set_xticks(x)
        ax.set_xticklabels(SCENE_LABELS, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel.split()[0]}: From-Scratch (●) vs Fine-tuned (◆)",
                     fontsize=13, fontweight="bold")
        ax.tick_params(axis="y", labelsize=11)

    # Legend: one entry per model (scratch/ft combined)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color=COLORS_SCRATCH[m], markersize=8,
               label=m, linestyle="none")
        for m in MODELS
    ]
    handles += [
        Line2D([0], [0], marker="D", color="k", markersize=7, linestyle="none",
               markerfacecolor="white", markeredgewidth=1.5, label="open = fine-tuned"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, -0.06), frameon=True, edgecolor="#ccc")
    fig.suptitle("Per-Scene ADE / FDE: Pre-training on SDD helps on most ETH/UCY scenes",
                 fontsize=13, fontweight="bold")
    plt.subplots_adjust(wspace=0.3)
    save(fig, "fig_ft_03_per_scene.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Summary bar chart — Δ ADE% per model (horizontal, single bar each)
# ─────────────────────────────────────────────────────────────────────────────
def plot_full_metrics_table(results):
    """Horizontal bar chart of % ADE/FDE improvement, one bar per model."""
    metrics    = [("ade", "ADE"), ("fde", "FDE"), ("minADE_20", "minADE@20")]
    fig, axes  = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    for ax, (metric, mlabel) in zip(axes, metrics):
        deltas = []
        for m in MODELS:
            sc = scene_avg(results, "scratch", m, metric)
            ft = scene_avg(results, "ft",      m, metric)
            deltas.append((ft - sc) / sc * 100 if sc and ft else 0.0)

        y = np.arange(len(MODELS))
        colors = ["#c8f0c8" if d < 0 else "#f8c8c8" for d in deltas]
        bars = ax.barh(y, deltas, height=0.55, color=colors,
                       edgecolor="#aaaaaa", linewidth=0.8)

        for yi, (bar, d) in enumerate(zip(bars, deltas)):
            xpos = d - 0.2 if d < 0 else d + 0.2
            ha   = "right" if d < 0 else "left"
            ax.text(xpos, yi, f"{d:+.1f}%", va="center", ha=ha,
                    fontsize=11, fontweight="bold",
                    color="#2a7a2a" if d < 0 else "#aa2222")

        ax.axvline(0, color="#555", lw=1.2)
        ax.set_yticks(y)
        ax.set_yticklabels(MODELS, fontsize=11)
        ax.set_xlabel("% change  (−ve = improved)", fontsize=11)
        ax.set_title(f"{mlabel}  Δ%  (avg over 5 scenes)",
                     fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(axis="x", labelsize=10)
        # Symmetric x axis
        xlim = max(abs(min(deltas)), abs(max(deltas))) + 3
        ax.set_xlim(-xlim, xlim)
        ax.invert_yaxis()

    fig.suptitle("Average % Improvement from SDD Pre-training → ETH/UCY Fine-tuning",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig_ft_04_full_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Scatter — scratch ADE vs ft ADE (one point per model×scene)
# ─────────────────────────────────────────────────────────────────────────────
def plot_leaderboard(results):
    """Scatter: x = scratch ADE, y = ft ADE.  Points below diagonal = improved."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    for ax, metric, label in zip(axes, ["ade", "fde"], ["ADE (m)", "FDE (m)"]):
        all_sc, all_ft = [], []
        for mi, model in enumerate(MODELS):
            sc_vals = [get_metric(results, "scratch", model, s, metric) for s in SCENES]
            ft_vals = [get_metric(results, "ft",      model, s, metric) for s in SCENES]
            sc_clean = [v for v in sc_vals if v is not None]
            ft_clean = [v for v, sv in zip(ft_vals, sc_vals) if v is not None and sv is not None]
            sc_for_ft= [sv for sv, fv in zip(sc_vals, ft_vals) if fv is not None and sv is not None]

            ax.scatter(sc_for_ft, ft_clean,
                       color=COLORS_SCRATCH[model], s=80, zorder=3,
                       label=model, edgecolors="white", linewidth=0.6)

            # Annotate each point with scene label
            for sv, fv, sl in zip(sc_for_ft, ft_clean, SCENE_LABELS):
                ax.annotate(sl, (sv, fv), textcoords="offset points",
                            xytext=(4, 3), fontsize=8,
                            color=COLORS_SCRATCH[model], alpha=0.8)
            all_sc.extend(sc_for_ft)
            all_ft.extend(ft_clean)

        if all_sc:
            lo = min(min(all_sc), min(all_ft)) * 0.95
            hi = max(max(all_sc), max(all_ft)) * 1.05
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.5, label="No change")
            ax.fill_between([lo, hi], [lo, lo], [lo, hi],
                            alpha=0.05, color="red",   label="Worse")
            ax.fill_between([lo, hi], [lo, hi], [hi, hi],
                            alpha=0.05, color="green", label="Better")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        ax.set_xlabel(f"Scratch {label}", fontsize=12)
        ax.set_ylabel(f"Fine-tuned {label}", fontsize=12)
        ax.set_title(f"{label}  (below diagonal = improved by fine-tuning)",
                     fontsize=11, fontweight="bold", pad=8)
        ax.set_aspect("equal")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, -0.06), frameon=True, edgecolor="#ccc")
    fig.suptitle("Scratch vs Fine-tuned per Model × Scene  (each point = one model on one ETH/UCY scene)",
                 fontsize=12, fontweight="bold")
    plt.subplots_adjust(wspace=0.35, top=0.92)
    save(fig, "fig_ft_05_leaderboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# Print summary table
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(results):
    print("\n" + "=" * 100)
    print(f"{'Transfer Learning Summary (avg over ETH/UCY scenes)':^100}")
    print("=" * 100)
    header = f"{'Model':<16} {'ADE_sc':>8} {'ADE_ft':>8} {'Δ ADE':>8}  " \
             f"{'FDE_sc':>8} {'FDE_ft':>8} {'Δ FDE':>8}  " \
             f"{'mADE_sc':>8} {'mADE_ft':>8} {'Δ mADE':>8}"
    print(header)
    print("-" * 100)
    for m in MODELS:
        def v(variant, metric):
            return scene_avg(results, variant, m, metric)
        sc_ade,  ft_ade  = v("scratch","ade"),       v("ft","ade")
        sc_fde,  ft_fde  = v("scratch","fde"),       v("ft","fde")
        sc_ma,   ft_ma   = v("scratch","minADE_20"), v("ft","minADE_20")
        def fmt(x): return f"{x:.3f}" if x else "  — "
        def delta(a, b):
            if a and b:
                return f"{(b-a)/a*100:+.1f}%"
            return "  — "
        print(f"{m:<16} {fmt(sc_ade):>8} {fmt(ft_ade):>8} {delta(sc_ade,ft_ade):>8}  "
              f"{fmt(sc_fde):>8} {fmt(ft_fde):>8} {delta(sc_fde,ft_fde):>8}  "
              f"{fmt(sc_ma):>8} {fmt(ft_ma):>8} {delta(sc_ma,ft_ma):>8}")
    print("=" * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-eval",   action="store_true",
                        help="Skip evaluation, load results from --load-json")
    parser.add_argument("--ft-only",   action="store_true",
                        help="Only eval ft_ checkpoints; use hardcoded scratch numbers")
    parser.add_argument("--save-json", default="results_ft.json",
                        help="Save/load evaluation results JSON")
    parser.add_argument("--load-json", default=None,
                        help="Load results from this JSON (implies --no-eval)")
    args = parser.parse_args()

    if args.load_json:
        print(f"Loading results from {args.load_json}")
        with open(args.load_json) as f:
            results = json.load(f)
    elif args.no_eval and os.path.exists(args.save_json):
        print(f"Loading cached results from {args.save_json}")
        with open(args.save_json) as f:
            results = json.load(f)
    else:
        results = run_evaluation(ft_only=args.ft_only)
        with open(args.save_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.save_json}")

    print_summary(results)

    print("\nGenerating plots...")
    plot_overall_comparison(results)
    plot_improvement_heatmap(results)
    plot_per_scene(results)
    plot_full_metrics_table(results)
    plot_leaderboard(results)
    print("\nAll done! Plots saved to plots/finetune/")
