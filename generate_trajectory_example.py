"""
Generate a qualitative trajectory prediction example for the report.

The figures use a real ETH/UCY held-out sequence and overlay:
- observed trajectory
- ground-truth future
- CV baseline
- Social-LSTM
- Social-LSTM+V
- Transformer mean prediction
- Diffusion mean prediction
- sampled futures from each native model

Run:
    python generate_trajectory_example.py
"""

import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK)

from eth_ucy_analysis import load_scene, extract_sequences_with_neighbours
from models.cv_baseline import ConstantVelocityPredictor
from models.social_lstm import SocialLSTM
from models.trajectory_transformer import TrajectoryTransformer
from models.diffusion import TrajDiffusion


RAW = os.path.join(WORK, "Trajectron-plus-plus/experiments/pedestrians/raw")
SCENE_FILES = {
    "eth": [os.path.join(RAW, "eth", "test", "biwi_eth.txt")],
    "hotel": [os.path.join(RAW, "hotel", "test", "biwi_hotel.txt")],
    "univ": [
        os.path.join(RAW, "univ", "test", "students001.txt"),
        os.path.join(RAW, "univ", "test", "students003.txt"),
    ],
    "zara1": [os.path.join(RAW, "zara1", "test", "crowds_zara01.txt")],
    "zara2": [os.path.join(RAW, "zara2", "test", "crowds_zara02.txt")],
}


def load_transformer(scene, device):
    ckpt = torch.load(
        os.path.join(WORK, "checkpoints", f"transformer_{scene}.pt"),
        map_location=device,
        weights_only=False,
    )
    model = TrajectoryTransformer(**ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_social_lstm(scene, variant, device):
    ckpt_name = "social_lstmv" if variant == "social_lstm_v" else "social_lstm"
    ckpt = torch.load(
        os.path.join(WORK, "checkpoints", f"{ckpt_name}_{scene}.pt"),
        map_location=device,
        weights_only=False,
    )
    model = SocialLSTM(**ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_diffusion(scene, device):
    ckpt = torch.load(
        os.path.join(WORK, "checkpoints", f"diffusion_{scene}.pt"),
        map_location=device,
        weights_only=False,
    )
    model = TrajDiffusion(**ckpt["hparams"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def pick_example(obs, pred):
    """Pick a readable trajectory with clear forward motion in local coordinates."""
    candidates = []
    for i in range(len(obs)):
        to_local = make_local_transform(obs[i])
        local_future = to_local(pred[i])
        final_x, final_y = local_future[-1]
        max_lateral = np.max(np.abs(local_future[:, 1]))
        obs_disp = np.linalg.norm(obs[i, -1] - obs[i, 0])
        if 2.2 < final_x < 4.6 and abs(final_y) < 0.20 and max_lateral < 0.28 and obs_disp > 0.8:
            score = final_x - 1.5 * max_lateral
            candidates.append((score, i))
    if candidates:
        candidates.sort()
        return int(candidates[int(0.55 * len(candidates))][1])

    future_disp = np.linalg.norm(pred[:, -1] - obs[:, -1], axis=1)
    obs_disp = np.linalg.norm(obs[:, -1] - obs[:, 0], axis=1)
    score = future_disp + 0.5 * obs_disp
    return int(np.argsort(score)[int(0.72 * len(score))])


def plot_path(ax, pts, label, color, marker=None, linestyle="-", alpha=1.0, lw=2.4):
    ax.plot(
        pts[:, 0],
        pts[:, 1],
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        alpha=alpha,
        linewidth=lw,
        markersize=4,
    )


def make_local_transform(obs_path):
    """Return a function that centers paths at prediction start and aligns heading."""
    origin = obs_path[-1].copy()
    heading = obs_path[-1] - obs_path[-2]
    theta = np.arctan2(heading[1], heading[0]) if np.linalg.norm(heading) > 1e-6 else 0.0
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, s], [-s, c]], dtype=np.float32)

    def transform(path):
        arr = np.asarray(path, dtype=np.float32)
        flat = arr.reshape(-1, 2)
        out = (flat - origin) @ rot.T
        return out.reshape(arr.shape)

    return transform


def run_model_predictions(scene, obs_t, nb_obs_t, nb_mask_t, obs_np, device):
    models = {
        "CV": {
            "color": "#9e9e9e",
            "marker": "D",
            "linestyle": "--",
            "samples": ConstantVelocityPredictor(noise_std=0.30).predict_samples(obs_np, K=12)[0],
        },
        "Social-LSTM": {
            "color": "#4e79a7",
            "marker": "v",
            "linestyle": "--",
            "model": load_social_lstm(scene, "social_lstm", device),
        },
        "Social-LSTM+V": {
            "color": "#f28e2b",
            "marker": "P",
            "linestyle": "--",
            "model": load_social_lstm(scene, "social_lstm_v", device),
        },
        "Transformer": {
            "color": "#59a14f",
            "marker": "s",
            "linestyle": "--",
            "model": load_transformer(scene, device),
        },
        "Diffusion": {
            "color": "#b07aa1",
            "marker": "^",
            "linestyle": "--",
            "model": load_diffusion(scene, device),
        },
    }

    with torch.no_grad():
        for name, info in models.items():
            if name == "CV":
                info["mean"] = ConstantVelocityPredictor(noise_std=0.0).predict_distribution(obs_np)["mus"][0]
                continue
            model = info["model"]
            info["mean"] = model(obs_t, nb_obs_t, nb_mask_t)["mus"].cpu().numpy()[0]
            info["samples"] = model.sample(obs_t, nb_obs_t, nb_mask_t, K=12)[0]
    return models


def setup_axes(ax, title):
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def apply_clean_limits(ax, x_min, x_max, y_min, y_max):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)


def compute_limits(paths, pad=0.35):
    stacked = np.concatenate([np.asarray(p).reshape(-1, 2) for p in paths], axis=0)
    mins = stacked.min(axis=0) - pad
    maxs = stacked.max(axis=0) + pad
    return mins[0], maxs[0], mins[1], maxs[1]


def draw_reference(ax, obs_local, gt_local, show_labels=True):
    obs_label = "Observed" if show_labels else None
    gt_label = "Ground truth" if show_labels else None
    start_label = "Prediction start" if show_labels else None
    plot_path(ax, obs_local, obs_label, "#203864", marker="o", lw=3.0)
    plot_path(ax, gt_local, gt_label, "#111111", marker="o", lw=3.0)
    ax.scatter(0, 0, color="#f28e2b", s=80, zorder=6, label=start_label)


def save_mean_panel_plot(local_models, obs_local, gt_local, out_path):
    all_paths = [obs_local, gt_local] + [info["mean"] for info in local_models.values()]
    x_min, x_max, y_min, y_max = compute_limits(all_paths, pad=0.35)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    axes = axes.ravel()
    for ax, (name, info) in zip(axes, local_models.items()):
        draw_reference(ax, obs_local, gt_local, show_labels=False)
        plot_path(
            ax,
            info["mean"],
            f"{name} prediction",
            info["color"],
            marker=info["marker"],
            linestyle="-",
            lw=2.8,
        )
        apply_clean_limits(ax, x_min, x_max, y_min, y_max)
        ax.set_title(name, fontsize=15, fontweight="bold")
        ax.set_xlabel("forward displacement (m)")
        ax.set_ylabel("lateral displacement (m)")

    axes[-1].axis("off")
    axes[-1].text(
        0.02,
        0.85,
        "Mean prediction panels\n\n"
        "Blue: observed history\n"
        "Black: true future\n"
        "Orange: prediction start\n"
        "Coloured: model mean\n\n"
        "Local coordinates make the\n"
        "last observed position the origin.",
        transform=axes[-1].transAxes,
        fontsize=13,
        va="top",
    )
    fig.suptitle("Qualitative trajectory predictions on ETH/UCY zara2", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def save_endpoint_uncertainty_plot(local_models, obs_local, gt_local, out_path):
    all_paths = [obs_local, gt_local]
    for info in local_models.values():
        all_paths.append(info["mean"])
        all_paths.append(info["samples"][:, -1])
    x_min, x_max, y_min, y_max = compute_limits(all_paths, pad=0.45)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    axes = axes.ravel()
    for ax, (name, info) in zip(axes, local_models.items()):
        draw_reference(ax, obs_local, gt_local, show_labels=False)
        endpoints = info["samples"][:, -1]
        ax.scatter(
            endpoints[:, 0],
            endpoints[:, 1],
            color=info["color"],
            alpha=0.35,
            s=42,
            edgecolors="none",
            label="sampled final positions",
        )
        ax.scatter(
            info["mean"][-1, 0],
            info["mean"][-1, 1],
            color=info["color"],
            marker=info["marker"],
            s=85,
            edgecolors="black",
            linewidths=0.5,
            zorder=6,
            label="mean final position",
        )
        ax.scatter(
            gt_local[-1, 0],
            gt_local[-1, 1],
            color="#111111",
            s=85,
            zorder=7,
            label="true final position",
        )
        apply_clean_limits(ax, x_min, x_max, y_min, y_max)
        ax.set_title(name, fontsize=15, fontweight="bold")
        ax.set_xlabel("forward displacement (m)")
        ax.set_ylabel("lateral displacement (m)")

    axes[-1].axis("off")
    axes[-1].text(
        0.02,
        0.85,
        "Endpoint uncertainty\n\n"
        "Dots show sampled final\n"
        "positions only, avoiding\n"
        "overlapping trajectory paths.\n\n"
        "Tighter clusters indicate more\n"
        "confident final-position belief.",
        transform=axes[-1].transAxes,
        fontsize=13,
        va="top",
    )
    fig.suptitle("Sampled final-position uncertainty by model", fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def save_clean_final_position_plot(local_models, obs_local, gt_local, out_path):
    """Single report-ready plot: path context plus model final-position markers."""
    paths = [obs_local, gt_local]
    final_points = []
    for name, info in local_models.items():
        final_points.append(info["mean"][-1])
    x_min, x_max, y_min, y_max = compute_limits(paths + [np.array(final_points)], pad=0.35)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    draw_reference(ax, obs_local, gt_local, show_labels=True)

    for name, info in local_models.items():
        final = info["mean"][-1]
        ax.scatter(
            final[0],
            final[1],
            color=info["color"],
            marker=info["marker"],
            s=100,
            edgecolors="white",
            linewidths=0.8,
            zorder=8,
            label=name,
        )
        ax.plot(
            [0, final[0]],
            [0, final[1]],
            color=info["color"],
            alpha=0.35,
            linewidth=1.6,
            linestyle="--",
        )

    ax.scatter(gt_local[-1, 0], gt_local[-1, 1], color="#111111", s=105, zorder=9)
    setup_axes(ax, "Predicted final positions from the same observed trajectory")
    ax.set_xlabel("forward displacement from last observation (m)")
    ax.set_ylabel("lateral displacement (m)")
    apply_clean_limits(ax, x_min, x_max, y_min, y_max)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(
        dedup.values(),
        dedup.keys(),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def save_clean_small_multiples(local_models, obs_local, gt_local, out_path):
    """Minimal mean-trajectory panels with no samples and no explanatory panel."""
    paths = [obs_local, gt_local] + [info["mean"] for info in local_models.values()]
    x_min, x_max, y_min, y_max = compute_limits(paths, pad=0.28)

    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharex=True, sharey=True)
    for ax, (name, info) in zip(axes, local_models.items()):
        plot_path(ax, obs_local, None, "#203864", marker="o", lw=2.6)
        plot_path(ax, gt_local, None, "#111111", marker="o", lw=2.6)
        plot_path(ax, info["mean"], None, info["color"], marker=info["marker"], linestyle="-", lw=2.4)
        ax.scatter(0, 0, color="#f28e2b", s=58, zorder=6)
        apply_clean_limits(ax, x_min, x_max, y_min, y_max)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.20)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("forward (m)", fontsize=9)
    axes[0].set_ylabel("lateral (m)", fontsize=9)
    fig.suptitle("Mean future trajectory by model", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def save_minimal_endpoint_plot(local_models, obs_local, gt_local, out_path):
    """Minimal report figure: observed path, true future, final predictions only."""
    final_points = np.array([info["mean"][-1] for info in local_models.values()])
    x_min, x_max, y_min, y_max = compute_limits([obs_local, gt_local, final_points], pad=0.30)

    fig, ax = plt.subplots(figsize=(8.8, 4.4))

    ax.plot(obs_local[:, 0], obs_local[:, 1], color="#203864", linewidth=3.0, label="Observed")
    ax.plot(gt_local[:, 0], gt_local[:, 1], color="#111111", linewidth=3.0, label="Ground truth")
    ax.scatter(0, 0, color="#f28e2b", s=90, zorder=6, label="Prediction start")
    ax.scatter(gt_local[-1, 0], gt_local[-1, 1], color="#111111", s=90, zorder=7)

    label_offsets = {
        "CV": (0.07, 0.08),
        "Social-LSTM": (0.06, -0.13),
        "Social-LSTM+V": (0.06, 0.12),
        "Transformer": (0.07, -0.12),
        "Diffusion": (0.07, 0.08),
    }
    short_names = {
        "CV": "CV",
        "Social-LSTM": "SLSTM",
        "Social-LSTM+V": "SLSTM+V",
        "Transformer": "Trans.",
        "Diffusion": "Diff.",
    }

    for name, info in local_models.items():
        final = info["mean"][-1]
        ax.scatter(
            final[0],
            final[1],
            color=info["color"],
            marker=info["marker"],
            s=115,
            edgecolors="white",
            linewidths=0.8,
            zorder=8,
        )
        dx, dy = label_offsets.get(name, (0.06, 0.06))
        ax.text(
            final[0] + dx,
            final[1] + dy,
            short_names.get(name, name),
            color=info["color"],
            fontsize=10,
            fontweight="bold",
            va="center",
        )

    ax.set_title("Predicted final positions for one ETH/UCY trajectory", fontsize=14, fontweight="bold")
    ax.set_xlabel("forward displacement from last observation (m)")
    ax.set_ylabel("lateral displacement (m)")
    ax.grid(True, alpha=0.18)
    ax.axhline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#dddddd", linewidth=0.8, zorder=0)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("auto")
    ax.legend(loc="upper left", fontsize=9, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def save_final_error_bar(local_models, gt_local, out_path):
    names = list(local_models.keys())
    errors = [float(np.linalg.norm(info["mean"][-1] - gt_local[-1])) for info in local_models.values()]
    colors = [info["color"] for info in local_models.values()]

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(names))
    bars = ax.bar(x, errors, color=colors, width=0.62, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["CV", "SLSTM", "SLSTM+V", "Trans.", "Diff."])
    ax.set_ylabel("final position error (m)")
    ax.set_title("Final-position error for the same trajectory", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.20)
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, err + 0.03, f"{err:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    np.random.seed(7)
    torch.manual_seed(7)

    scene = "zara2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_scene(SCENE_FILES[scene])
    obs, pred, nb_obs, nb_mask = extract_sequences_with_neighbours(
        data, obs_len=8, pred_len=12, max_neighbours=5
    )
    idx = pick_example(obs, pred)

    obs_i = obs[idx : idx + 1].astype(np.float32)
    pred_i = pred[idx : idx + 1].astype(np.float32)
    nb_obs_i = np.nan_to_num(nb_obs[idx : idx + 1], nan=0.0).astype(np.float32)
    nb_mask_i = nb_mask[idx : idx + 1]

    obs_t = torch.tensor(obs_i, dtype=torch.float32, device=device)
    nb_obs_t = torch.tensor(nb_obs_i, dtype=torch.float32, device=device)
    nb_mask_t = torch.tensor(nb_mask_i, dtype=torch.bool, device=device)

    models = run_model_predictions(scene, obs_t, nb_obs_t, nb_mask_t, obs_i, device)

    obs_path = obs_i[0]
    gt_path = pred_i[0]
    to_local = make_local_transform(obs_path)
    obs_local = to_local(obs_path)
    gt_local = to_local(gt_path)
    local_models = {}
    for name, info in models.items():
        local_models[name] = {
            **info,
            "mean": to_local(info["mean"]),
            "samples": to_local(info["samples"]),
        }

    os.makedirs(os.path.join(WORK, "plots"), exist_ok=True)

    save_mean_panel_plot(
        local_models,
        obs_local,
        gt_local,
        os.path.join(WORK, "plots", "trajectory_mean_panels_zara2.png"),
    )
    save_endpoint_uncertainty_plot(
        local_models,
        obs_local,
        gt_local,
        os.path.join(WORK, "plots", "trajectory_endpoint_uncertainty_zara2.png"),
    )
    save_clean_final_position_plot(
        local_models,
        obs_local,
        gt_local,
        os.path.join(WORK, "plots", "trajectory_final_positions_clean_zara2.png"),
    )
    save_clean_small_multiples(
        local_models,
        obs_local,
        gt_local,
        os.path.join(WORK, "plots", "trajectory_mean_small_multiples_clean_zara2.png"),
    )
    save_minimal_endpoint_plot(
        local_models,
        obs_local,
        gt_local,
        os.path.join(WORK, "plots", "trajectory_minimal_final_positions_zara2.png"),
    )
    save_final_error_bar(
        local_models,
        gt_local,
        os.path.join(WORK, "plots", "trajectory_final_error_bar_zara2.png"),
    )

    # Figure 0: report-ready local-frame mean predictions.
    limit_paths = [obs_local, gt_local] + [info["mean"] for info in local_models.values()]
    x_min, x_max, y_min, y_max = compute_limits(limit_paths, pad=0.25)
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    plot_path(ax, obs_local, "Observed", "#203864", marker="o", lw=3.0)
    plot_path(ax, gt_local, "Ground truth", "#111111", marker="o", lw=3.0)
    for name, info in local_models.items():
        plot_path(
            ax,
            info["mean"],
            name,
            info["color"],
            marker=info["marker"],
            linestyle=info["linestyle"],
            lw=2.0,
        )
    ax.scatter(0, 0, color="#f28e2b", s=95, zorder=6, label="Prediction start")
    setup_axes(ax, "Local-frame trajectory prediction example")
    ax.set_xlabel("forward displacement from last observation (m)")
    ax.set_ylabel("lateral displacement (m)")
    apply_clean_limits(ax, x_min, x_max, y_min, y_max)
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(
        dedup.values(),
        dedup.keys(),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=True,
    )
    out = os.path.join(WORK, "plots", "trajectory_mean_relative_zara2.png")
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Figure 1: clean comparison of model mean predictions.
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_path(ax, obs_path, "Observed trajectory", "#2f4b7c", marker="o", lw=3.0)
    plot_path(ax, gt_path, "Ground truth future", "#111111", marker="o", lw=3.0)
    for name, info in models.items():
        plot_path(
            ax,
            info["mean"],
            f"{name} mean",
            info["color"],
            marker=info["marker"],
            linestyle=info["linestyle"],
            lw=2.2,
        )

    ax.scatter(obs_path[0, 0], obs_path[0, 1], color="#2f4b7c", s=80, zorder=5)
    ax.scatter(obs_path[-1, 0], obs_path[-1, 1], color="#f28e2b", s=90, zorder=5, label="Prediction start")
    ax.scatter(gt_path[-1, 0], gt_path[-1, 1], color="#111111", s=90, zorder=5, label="True final position")
    setup_axes(ax, "Mean trajectory predictions on ETH/UCY zara2")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(
        dedup.values(),
        dedup.keys(),
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=True,
    )
    out = os.path.join(WORK, "plots", "trajectory_all_models_zara2.png")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Backwards-compatible name used by the earlier progress summary.
    old_out = os.path.join(WORK, "plots", "trajectory_example_zara2.png")
    fig, ax = plt.subplots(figsize=(8, 7))
    plot_path(ax, obs_path, "Observed trajectory", "#2f4b7c", marker="o", lw=3.0)
    plot_path(ax, gt_path, "Ground truth future", "#111111", marker="o", lw=3.0)
    plot_path(ax, models["Transformer"]["mean"], "Transformer prediction", models["Transformer"]["color"], marker="s", linestyle="--")
    plot_path(ax, models["Diffusion"]["mean"], "Diffusion mean prediction", models["Diffusion"]["color"], marker="^", linestyle="--")
    for sample in models["Diffusion"]["samples"]:
        ax.plot(sample[:, 0], sample[:, 1], color=models["Diffusion"]["color"], alpha=0.18, linewidth=1.4)
    ax.scatter(obs_path[-1, 0], obs_path[-1, 1], color="#f28e2b", s=90, zorder=5, label="Prediction start")
    ax.scatter(gt_path[-1, 0], gt_path[-1, 1], color="#111111", s=90, zorder=5, label="True final position")
    setup_axes(ax, "Example trajectory prediction on ETH/UCY zara2")
    handles, labels = ax.get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    ax.legend(dedup.values(), dedup.keys(), loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(old_out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {old_out}")

    # Figure 2: report-ready local-frame sample futures from all native models.
    all_points = [obs_local, gt_local]
    for info in local_models.values():
        all_points.append(info["mean"])
        all_points.extend(list(info["samples"]))
    x_min, x_max, y_min, y_max = compute_limits(all_points, pad=0.35)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))
    axes = axes.ravel()
    for ax, (name, info) in zip(axes, local_models.items()):
        plot_path(ax, obs_local, "Observed", "#203864", marker="o", lw=2.4)
        plot_path(ax, gt_local, "Ground truth", "#111111", marker="o", lw=2.4)
        for sample in info["samples"]:
            ax.plot(sample[:, 0], sample[:, 1], color=info["color"], alpha=0.22, linewidth=1.4)
        plot_path(ax, info["mean"], f"{name} mean", info["color"], marker=info["marker"], linestyle="--", lw=2.2)
        ax.scatter(0, 0, color="#f28e2b", s=55, zorder=5)
        setup_axes(ax, name)
        ax.set_xlabel("forward displacement (m)")
        ax.set_ylabel("lateral displacement (m)")
        apply_clean_limits(ax, x_min, x_max, y_min, y_max)

    axes[-1].axis("off")
    axes[-1].text(
        0.03,
        0.88,
        "Each panel shows 12 sampled futures\nfrom the same observed trajectory.\n\n"
        "Lower spread means more deterministic\nprediction; wider spread shows\nmore sampled uncertainty.",
        transform=axes[-1].transAxes,
        fontsize=12,
        va="top",
    )

    fig.suptitle("Sampled future trajectories in local coordinates", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(WORK, "plots", "trajectory_samples_relative_zara2.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Backwards-compatible sample grid filename.
    old_samples = os.path.join(WORK, "plots", "trajectory_samples_all_models_zara2.png")
    import shutil
    shutil.copyfile(out, old_samples)
    print(f"Saved {old_samples}")


if __name__ == "__main__":
    main()
