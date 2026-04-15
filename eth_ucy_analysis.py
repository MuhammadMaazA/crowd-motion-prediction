"""
ETH/UCY Pedestrian Trajectory Dataset Analysis
COMP0225 – UCL Integrated Crowd Navigation System
D1: Multi-Modal Prediction Module

Covers:
  1. Per-scene statistics (crowd density, speed, NN distance, valid sequences)
  2. Visualisations: trajectory overlays, density heatmaps, crowd-density over
     time, speed distribution, obs/pred examples
  3. Interaction-aware dataloader (obs/pred + neighbour context)
  4. Evaluation metrics: ADE, FDE, Best-of-K ADE/FDE, collision rate
  5. Model recommendations tailored to this project's three-layer architecture
  6. Gap-analysis: which scenes best illustrate G2/G4/G6
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE = "Trajectron-plus-plus/experiments/pedestrians/raw"

SCENES = {
    "eth":   [os.path.join(BASE, "eth",   "test", "biwi_eth.txt")],
    "hotel": [os.path.join(BASE, "hotel", "test", "biwi_hotel.txt")],
    "univ":  [os.path.join(BASE, "univ",  "test", "students001.txt"),
              os.path.join(BASE, "univ",  "test", "students003.txt")],
    "zara1": [os.path.join(BASE, "zara1", "test", "crowds_zara01.txt")],
    "zara2": [os.path.join(BASE, "zara2", "test", "crowds_zara02.txt")],
}

OUTPUT_DIR = "eth_ucy_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OBS_LEN      = 8
PRED_LEN     = 12
SEQ_LEN      = OBS_LEN + PRED_LEN   # 20 frames
SAFETY_RADIUS = 0.2                  # metres


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_scene(paths):
    """
    Load one or more tab-separated ETH/UCY files into a single DataFrame.

    Columns: frame (float), ped (float), x (float), y (float).
    When multiple files are combined (e.g. univ students001 + students003),
    pedestrian IDs are offset per file to avoid collisions.
    """
    frames = []
    ped_offset = 0
    for p in paths:
        df = pd.read_csv(p, sep="\t", header=None,
                         names=["frame", "ped", "x", "y"])
        df["ped"] = df["ped"] + ped_offset
        ped_offset += int(df["ped"].max()) + 1
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values(["frame", "ped"]).reset_index(drop=True)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 2. Per-scene statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(data, scene_name):
    n_peds   = data["ped"].nunique()
    n_frames = data["frame"].nunique()

    peds_per_frame = data.groupby("frame")["ped"].count()
    avg_crowd = peds_per_frame.mean()
    max_crowd = peds_per_frame.max()

    # Speed: displacement / Δt for consecutive frames of each pedestrian
    # Only use steps equal to the minimum inter-frame gap for that pedestrian.
    speeds = []
    for ped_id, traj in data.groupby("ped"):
        traj = traj.sort_values("frame")
        if len(traj) < 2:
            continue
        dx         = np.diff(traj["x"].values)
        dy         = np.diff(traj["y"].values)
        dt_frames  = np.diff(traj["frame"].values)
        min_step   = dt_frames.min()
        if min_step <= 0:
            continue
        mask  = dt_frames == min_step
        dists = np.sqrt(dx[mask]**2 + dy[mask]**2)
        spd   = dists / 0.4   # 0.4 s per frame step
        speeds.extend(spd.tolist())

    mean_speed = np.mean(speeds) if speeds else float("nan")
    std_speed  = np.std(speeds)  if speeds else float("nan")

    # Average nearest-neighbour distance across frames that have ≥2 pedestrians
    nn_dists = []
    for _, group in data.groupby("frame"):
        if len(group) < 2:
            continue
        pts = group[["x", "y"]].values
        D   = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        nn_dists.extend(D.min(axis=1).tolist())
    mean_nn = np.mean(nn_dists) if nn_dists else float("nan")

    x_range = data["x"].max() - data["x"].min()
    y_range = data["y"].max() - data["y"].min()

    print(f"\n{'='*55}")
    print(f"  Scene: {scene_name.upper()}")
    print(f"{'='*55}")
    print(f"  Unique pedestrians  : {n_peds}")
    print(f"  Unique frames       : {n_frames}")
    print(f"  Avg peds/frame      : {avg_crowd:.2f}")
    print(f"  Max peds/frame      : {int(max_crowd)}")
    print(f"  Mean speed (m/s)    : {mean_speed:.3f}")
    print(f"  Std  speed (m/s)    : {std_speed:.3f}")
    print(f"  Mean NN distance (m): {mean_nn:.3f}")
    print(f"  Spatial extent      : x={x_range:.1f} m, y={y_range:.1f} m")

    return {
        "n_peds": n_peds, "n_frames": n_frames,
        "avg_crowd": avg_crowd, "max_crowd": max_crowd,
        "mean_speed": mean_speed, "std_speed": std_speed,
        "mean_nn": mean_nn,
        "x_range": x_range, "y_range": y_range,
        "speeds": speeds,
        "peds_per_frame": peds_per_frame,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Reusable DataLoader
# ─────────────────────────────────────────────────────────────────────────────

def extract_sequences(data, obs_len=OBS_LEN, pred_len=PRED_LEN):
    """
    Extract (obs, pred) sequence pairs using the standard ETH/UCY protocol.

    A valid sequence requires a pedestrian to be present for obs_len + pred_len
    consecutive frames at the scene's native stride.

    Parameters
    ----------
    data     : DataFrame [frame, ped, x, y]
    obs_len  : observation length in frames (default 8)
    pred_len : prediction length in frames  (default 12)

    Returns
    -------
    obs  : np.ndarray  shape (N, obs_len,  2)  – absolute (x, y) in metres
    pred : np.ndarray  shape (N, pred_len, 2)
    """
    seq_len    = obs_len + pred_len
    all_frames = np.sort(data["frame"].unique())
    if len(all_frames) < 2:
        return np.empty((0, obs_len, 2)), np.empty((0, pred_len, 2))
    stride = int(np.min(np.diff(all_frames)))

    obs_list, pred_list = [], []

    for ped_id, traj in data.groupby("ped"):
        traj         = traj.sort_values("frame")
        frames       = traj["frame"].values
        xy           = traj[["x", "y"]].values
        frame_to_xy  = {f: xy[i] for i, f in enumerate(frames)}

        first_frame = frames[0]
        last_frame  = frames[-1]
        start       = first_frame

        while start + stride * (seq_len - 1) <= last_frame:
            window = [start + stride * k for k in range(seq_len)]
            if all(f in frame_to_xy for f in window):
                coords = np.array([frame_to_xy[f] for f in window])
                obs_list.append(coords[:obs_len])
                pred_list.append(coords[obs_len:])
            start += stride

    if not obs_list:
        return np.empty((0, obs_len, 2)), np.empty((0, pred_len, 2))
    return np.array(obs_list), np.array(pred_list)


def extract_sequences_with_neighbours(data, obs_len=OBS_LEN, pred_len=PRED_LEN,
                                      max_neighbours=10):
    """
    Extract sequences WITH neighbour context for interaction-aware models.

    For each valid (obs, pred) sequence of the *focal* pedestrian, also extract
    the positions of up to max_neighbours other pedestrians present during the
    observation window (sorted by mean distance to focal agent, closest first).

    Parameters
    ----------
    data           : DataFrame [frame, ped, x, y]
    obs_len        : observation steps (default 8)
    pred_len       : prediction steps  (default 12)
    max_neighbours : max number of neighbours to include (default 10)

    Returns
    -------
    obs       : (N, obs_len, 2)
    pred      : (N, pred_len, 2)
    nb_obs    : (N, max_neighbours, obs_len, 2) – padded with NaN for absent agents
    nb_masks  : (N, max_neighbours) bool – True where neighbour data is valid
    """
    seq_len    = obs_len + pred_len
    all_frames = np.sort(data["frame"].unique())
    if len(all_frames) < 2:
        empty = np.empty((0, obs_len, 2))
        return (empty, empty,
                np.empty((0, max_neighbours, obs_len, 2)),
                np.empty((0, max_neighbours), dtype=bool))
    stride = int(np.min(np.diff(all_frames)))

    # Pre-build lookup: frame → dict{ped_id → (x, y)}
    frame_to_agents = {}
    for _, row in data.iterrows():
        f = row["frame"]
        if f not in frame_to_agents:
            frame_to_agents[f] = {}
        frame_to_agents[f][row["ped"]] = np.array([row["x"], row["y"]])

    obs_list, pred_list, nb_obs_list, nb_mask_list = [], [], [], []

    for ped_id, traj in data.groupby("ped"):
        traj         = traj.sort_values("frame")
        frames       = traj["frame"].values
        xy           = traj[["x", "y"]].values
        frame_to_xy  = {f: xy[i] for i, f in enumerate(frames)}

        first_frame = frames[0]
        last_frame  = frames[-1]
        start       = first_frame

        while start + stride * (seq_len - 1) <= last_frame:
            window = [start + stride * k for k in range(seq_len)]
            if not all(f in frame_to_xy for f in window):
                start += stride
                continue

            coords = np.array([frame_to_xy[f] for f in window])
            obs_window = window[:obs_len]

            # Collect all other pedestrians present in the obs window
            neighbour_ids = set()
            for f in obs_window:
                if f in frame_to_agents:
                    neighbour_ids.update(frame_to_agents[f].keys())
            neighbour_ids.discard(ped_id)

            # Rank by mean distance to focal agent during obs window
            focal_centre = coords[:obs_len].mean(axis=0)
            ranked = []
            for nid in neighbour_ids:
                nb_positions = []
                for f in obs_window:
                    if f in frame_to_agents and nid in frame_to_agents[f]:
                        nb_positions.append(frame_to_agents[f][nid])
                if nb_positions:
                    mean_pos  = np.mean(nb_positions, axis=0)
                    mean_dist = np.linalg.norm(mean_pos - focal_centre)
                    ranked.append((mean_dist, nid, nb_positions))
            ranked.sort(key=lambda x: x[0])

            # Build padded neighbour array
            nb_arr  = np.full((max_neighbours, obs_len, 2), np.nan)
            nb_mask = np.zeros(max_neighbours, dtype=bool)
            for slot, (_, nid, _) in enumerate(ranked[:max_neighbours]):
                for t_idx, f in enumerate(obs_window):
                    if f in frame_to_agents and nid in frame_to_agents[f]:
                        nb_arr[slot, t_idx] = frame_to_agents[f][nid]
                # Mark as valid if at least one timestep is present
                nb_mask[slot] = not np.all(np.isnan(nb_arr[slot]))

            obs_list.append(coords[:obs_len])
            pred_list.append(coords[obs_len:])
            nb_obs_list.append(nb_arr)
            nb_mask_list.append(nb_mask)
            start += stride

    if not obs_list:
        empty = np.empty((0, obs_len, 2))
        return (empty, empty,
                np.empty((0, max_neighbours, obs_len, 2)),
                np.empty((0, max_neighbours), dtype=bool))

    return (np.array(obs_list),
            np.array(pred_list),
            np.array(nb_obs_list),
            np.array(nb_mask_list))


def count_valid_sequences(data, obs_len=OBS_LEN, pred_len=PRED_LEN):
    obs, pred = extract_sequences(data, obs_len, pred_len)
    n = len(obs)
    print(f"  Valid sequences     : {n}  (obs={obs_len}, pred={pred_len})")
    return n, obs, pred


# ─────────────────────────────────────────────────────────────────────────────
# 4. Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def ade(pred_traj, gt_traj):
    """
    Average Displacement Error.

    Parameters
    ----------
    pred_traj : (N, pred_len, 2)  – predicted trajectories
    gt_traj   : (N, pred_len, 2)  – ground-truth trajectories

    Returns
    -------
    float – mean L2 distance over all agents and timesteps
    """
    assert pred_traj.shape == gt_traj.shape
    return float(np.mean(np.linalg.norm(pred_traj - gt_traj, axis=-1)))


def fde(pred_traj, gt_traj):
    """
    Final Displacement Error.

    Parameters
    ----------
    pred_traj : (N, pred_len, 2)
    gt_traj   : (N, pred_len, 2)

    Returns
    -------
    float – mean L2 distance at the last predicted timestep
    """
    return float(np.mean(np.linalg.norm(pred_traj[:, -1] - gt_traj[:, -1],
                                        axis=-1)))


def best_of_k_ade(samples, gt_traj):
    """
    Best-of-K ADE: for each sequence, pick the sample closest to ground truth.

    Parameters
    ----------
    samples  : (N, K, pred_len, 2)  – K sampled trajectories per sequence
    gt_traj  : (N,    pred_len, 2)

    Returns
    -------
    float – mean ADE over sequences, using each sequence's best sample
    """
    N, K, pred_len, _ = samples.shape
    # Per-sample ADE for each sequence: (N, K)
    errors = np.linalg.norm(
        samples - gt_traj[:, None, :, :], axis=-1).mean(axis=-1)
    best_idx = errors.argmin(axis=1)           # (N,)
    best     = samples[np.arange(N), best_idx] # (N, pred_len, 2)
    return ade(best, gt_traj)


def best_of_k_fde(samples, gt_traj):
    """
    Best-of-K FDE: for each sequence, pick the sample closest in final position.

    Parameters
    ----------
    samples  : (N, K, pred_len, 2)
    gt_traj  : (N,    pred_len, 2)

    Returns
    -------
    float
    """
    N, K, pred_len, _ = samples.shape
    final_errors = np.linalg.norm(
        samples[:, :, -1, :] - gt_traj[:, None, -1, :], axis=-1)  # (N, K)
    best_idx = final_errors.argmin(axis=1)
    best     = samples[np.arange(N), best_idx]
    return fde(best, gt_traj)


def collision_rate(pred_traj, all_ped_gt, radius=SAFETY_RADIUS):
    """
    Collision rate: fraction of predicted positions within `radius` metres of
    any other pedestrian's ground-truth position at the same timestep.

    Parameters
    ----------
    pred_traj   : (N, pred_len, 2) – predicted trajectory for the focal agent
    all_ped_gt  : (M, pred_len, 2) – ground-truth trajectories of ALL other peds
                  (do NOT include the focal agent)
    radius      : collision threshold in metres (default 0.2 m)

    Returns
    -------
    float – collision rate in [0, 1]
    """
    N, pred_len, _ = pred_traj.shape
    M              = all_ped_gt.shape[0]
    if M == 0:
        return 0.0

    collisions = 0
    total      = 0

    for t in range(pred_len):
        # pairwise distances: (N, M)
        focal_t = pred_traj[:, t, :]      # (N, 2)
        others  = all_ped_gt[:, t, :]     # (M, 2)
        dists   = cdist(focal_t, others)  # (N, M)
        collisions += int(np.any(dists < radius, axis=1).sum())
        total      += N

    return collisions / total if total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_trajectories(data, scene_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    peds    = data["ped"].unique()
    colours = cm.tab20(np.linspace(0, 1, min(len(peds), 20)))
    for i, ped_id in enumerate(peds):
        traj = data[data["ped"] == ped_id].sort_values("frame")
        ax.plot(traj["x"], traj["y"],
                color=colours[i % 20], alpha=0.55, linewidth=0.8)
    ax.set_title(f"{scene_name} – All Trajectories ({len(peds)} peds)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{scene_name}_trajectories.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_heatmap(data, scene_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    h = ax.hist2d(data["x"], data["y"], bins=60, cmap="hot_r", density=True)
    fig.colorbar(h[3], ax=ax, label="density")
    ax.set_title(f"{scene_name} – Pedestrian Density Heatmap")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{scene_name}_heatmap.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_crowd_density_over_time(peds_per_frame, scene_name):
    """Line plot: number of pedestrians visible at each frame."""
    fig, ax = plt.subplots(figsize=(9, 3))
    frames = np.sort(peds_per_frame.index)
    counts = peds_per_frame[frames].values
    ax.plot(frames, counts, linewidth=1.2, color="steelblue")
    ax.fill_between(frames, counts, alpha=0.25, color="steelblue")
    ax.axhline(counts.mean(), color="crimson", linestyle="--", linewidth=1,
               label=f"Mean={counts.mean():.1f}")
    ax.set_title(f"{scene_name} – Crowd Density Over Time")
    ax.set_xlabel("Frame"); ax.set_ylabel("Pedestrians in scene")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{scene_name}_density_over_time.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_speed_distribution(speeds, scene_name):
    """Histogram of instantaneous pedestrian speeds."""
    if not speeds:
        print(f"  [skip] No speed data for {scene_name}")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(speeds, bins=50, color="darkorange", edgecolor="white",
            linewidth=0.4, density=True)
    ax.axvline(np.mean(speeds), color="navy", linestyle="--",
               label=f"Mean={np.mean(speeds):.2f} m/s")
    ax.axvline(np.mean(speeds) + np.std(speeds), color="navy", linestyle=":",
               label=f"±1σ ({np.std(speeds):.2f} m/s)")
    ax.axvline(max(0, np.mean(speeds) - np.std(speeds)), color="navy",
               linestyle=":")
    ax.set_title(f"{scene_name} – Speed Distribution")
    ax.set_xlabel("Speed (m/s)"); ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{scene_name}_speed_dist.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


def plot_obs_pred_examples(obs, pred, scene_name, n_examples=5):
    rng     = np.random.default_rng(42)
    indices = rng.choice(len(obs), size=min(n_examples, len(obs)), replace=False)
    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4),
                             squeeze=False)
    for col, idx in enumerate(indices):
        ax = axes[0][col]
        o = obs[idx]; p = pred[idx]
        ax.plot(o[:, 0], o[:, 1], "bo-",  lw=1.5, ms=4,  label="Obs (8 steps)")
        ax.plot(p[:, 0], p[:, 1], "r^--", lw=1.5, ms=4,  label="GT pred (12 steps)")
        ax.plot(o[0,  0], o[0,  1], "gs", ms=7,   label="Start")
        ax.plot(p[-1, 0], p[-1, 1], "kx", ms=9, mew=2, label="End")
        ax.set_title(f"Seq {idx}"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect("equal"); ax.legend(fontsize=6)
    fig.suptitle(f"{scene_name} – Obs/Pred Examples", fontsize=12)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{scene_name}_obs_pred_examples.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Model recommendations
# ─────────────────────────────────────────────────────────────────────────────

MODEL_RECS = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║          MODEL RECOMMENDATIONS  –  COMP0225 Three-Person Team                  ║
║          (D1: Prediction · D2: Planning · D3: Conformal Safety)                ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ── PERSON 1 (Baseline) ──────────────────────────────────────────────────────  ║
║  Model   : Constant Velocity + Gaussian noise  (deterministic CV + k samples)  ║
║  Input   : last 2 obs positions → velocity vector                               ║
║  Output  : k=20 samples by adding isotropic Gaussian noise  σ~0.3 m            ║
║  Multi-modal? Weakly – only because of noise injection                          ║
║  PyTorch : ~40 lines, no repo needed                                            ║
║  Training: none – analytical model                                               ║
║  → Conformal safety: treat the k samples as an empirical distribution;          ║
║    compute nonconformity scores on val set; conformal region = k-sample hull    ║
║  Why essential: sets the performance floor; exposes how much naive uncertainty  ║
║    quantification inflates or deflates conformal regions (G4 illustration)      ║
║                                                                                  ║
║  ── PERSON 2 (Medium) ─────────────────────────────────────────────────────────  ║
║  Model   : Social-LSTM  (Alahi et al., CVPR 2016)                               ║
║  Input   : (x,y) sequences; social pooling grid of neighbours                   ║
║  Output  : bivariate Gaussian at each step  (μx, μy, σx, σy, ρ)                ║
║  Multi-modal? No – unimodal Gaussian per step                                    ║
║  PyTorch : github.com/quancore/social-lstm                                       ║
║  Training: ~1–2 h on a single GPU                                                ║
║  → Conformal safety: use predicted Gaussian as base distribution; compute       ║
║    nonconformity scores r_i = ||ŷ_i - y_i||₂ on val; coverage at 1-α quantile  ║
║    of {r_i} directly inflates the ellipse — interpretable, clean connection     ║
║  Why medium: introduces interaction awareness; unimodal output makes conformal  ║
║    set construction straightforward (G2: avg prediction ≠ safe planning)        ║
║                                                                                  ║
║  ── PERSON 3 (State-of-the-Art) ───────────────────────────────────────────────  ║
║  Model   : Trajectron++  (Salzmann et al., ECCV 2020)                            ║
║  Input   : (x,y) sequences + semantic map (optional); agent type labels          ║
║  Output  : full GMM distribution P(y|x) via CVAE; supports ego-conditional pred  ║
║  Multi-modal? Yes – full distribution, sample k=20 or compute GMM parameters    ║
║  PyTorch : github.com/StanfordASL/Trajectron-plus-plus  (already cloned!)       ║
║  Training: ~4–8 h on a GPU; config files already in the cloned repo              ║
║  → Conformal safety: use GMM NLL as nonconformity score; build conformal         ║
║    prediction sets as level sets of the distribution at the 1-α quantile        ║
║    (Lindemann et al., RA-L 2023 framework applies directly)                      ║
║  Why SoTA: best ADE/FDE on ETH/UCY without scene images; calibrated uncertainty ║
║    allows NLL-based evaluation; ego-conditional output supports closed-loop test ║
║                                                                                  ║
║  ── CONFORMAL SAFETY (D3) CONNECTION ─────────────────────────────────────────  ║
║  The three models produce outputs of increasing richness:                        ║
║    CV+noise → sample set → empirical quantile of nonconformity scores           ║
║    Social-LSTM → Gaussian → analytical ellipse inflation                         ║
║    Trajectron++ → GMM → level-set conformal regions                             ║
║  This lets the team directly compare how output quality affects conformal        ║
║  region size (illustrates G3: coverage ≠ composes; G4: conservatism-freezing)   ║
║                                                                                  ║
║  ── METRICS TO REPORT ─────────────────────────────────────────────────────────  ║
║    • ADE, FDE (standard closed-loop surrogates)                                  ║
║    • minADE@20 / minFDE@20 (best-of-20-samples; penalises mode collapse)        ║
║    • NLL (only Trajectron++ and Social-LSTM natively; CV requires a fit)         ║
║    • Calibration ECE (expected calibration error on marginal step distributions) ║
║    • Collision rate of predicted trajectories (see collision_rate() above)       ║
║    • Conformal region area / volume at 90% and 95% coverage                      ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# 7. Gap analysis
# ─────────────────────────────────────────────────────────────────────────────

GAP_ANALYSIS = """
╔══════════════════════════════════════════════════════════════════════════════════╗
║          GAP ANALYSIS  –  Best scenes for each identified gap                   ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  G2  "Open-loop prediction improvements don't translate to closed-loop safety"  ║
║  Best scene : ETH                                                                ║
║  Why        : ETH is the *hardest* scene (high speed 2.3 m/s, 6.3 avg peds,    ║
║               22 m corridor).  A model can achieve low ADE on ETH in open loop  ║
║               (extrapolating straight walks) yet generate unsafe trajectories   ║
║               when those predictions feed a planner – pedestrians arrive faster ║
║               than expected.  Compare open-loop ADE vs planner collision rate   ║
║               on ETH to make G2 concrete.                                        ║
║  Secondary  : zara1 (dense but more predictable – contrast with ETH)            ║
║                                                                                  ║
║  G4  "Conservatism-freezing: safer systems eventually prevent navigation"       ║
║  Best scene : univ                                                               ║
║  Why        : univ has by far the densest crowds (73.5 avg peds/frame, NN       ║
║               distance only 0.72 m).  As conformal coverage α increases, the   ║
║               prediction regions of nearby pedestrians overlap, leaving no      ║
║               feasible path for the robot.  This is the most direct empirical  ║
║               evidence for G4.  Plot conformal region area vs crowd density     ║
║               using univ sequences to produce the conservatism-coverage curve.  ║
║  Secondary  : zara2 (dense, moderate speed – shows G4 at intermediate density) ║
║                                                                                  ║
║  G6  "Evaluation over-measures surrogates, under-measures social success"       ║
║  Best scene : hotel                                                              ║
║  Why        : hotel is a narrow corridor (7.6 m wide) with very regular,        ║
║               slow-moving crowd (1.04 m/s).  ADE/FDE are easy to minimise here ║
║               (straight-line extrapolation works well) yet the corridor setting ║
║               is exactly where socially-acceptable navigation – giving way,     ║
║               queuing, avoiding side-by-side walkers – matters most.            ║
║               Use hotel to show that a model ranking first on ADE may rank last ║
║               on social comfort metrics, motivating G6.                          ║
║  Secondary  : zara1 (shoppers with frequent stops; ADE misleading for turns)   ║
║                                                                                  ║
║  SUMMARY TABLE                                                                   ║
║  Gap   Primary scene   Key statistic to highlight                                ║
║  G2    eth             speed=2.3 m/s; low ADE ≠ safe planner                   ║
║  G4    univ            crowd=73.5 peds/frame, NN=0.72 m; regions inflate fast   ║
║  G6    hotel           corridor 7.6 m; ADE easy but social fidelity is hard     ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    all_stats = {}

    for scene_name, paths in SCENES.items():
        print(f"\nLoading scene: {scene_name}")
        data = load_scene(paths)

        stats                = compute_stats(data, scene_name)
        n_seq, obs, pred     = count_valid_sequences(data)
        stats["n_sequences"] = n_seq
        all_stats[scene_name] = stats

        # ── Plots ────────────────────────────────────────────────────────────
        plot_trajectories(data, scene_name)
        plot_heatmap(data, scene_name)
        plot_crowd_density_over_time(stats["peds_per_frame"], scene_name)
        plot_speed_distribution(stats["speeds"], scene_name)
        if n_seq > 0:
            plot_obs_pred_examples(obs, pred, scene_name)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*80)
    print("  SUMMARY TABLE")
    print("="*80)
    hdr = (f"{'Scene':<8} {'Peds':>5} {'Frames':>7} {'AvgCrowd':>9} "
           f"{'MaxCrowd':>9} {'Speed m/s':>10} {'NN_dist':>8} {'Seqs':>7}")
    print(hdr)
    print("-"*80)
    for sn, s in all_stats.items():
        print(f"{sn:<8} {s['n_peds']:>5} {s['n_frames']:>7} "
              f"{s['avg_crowd']:>9.2f} {s['max_crowd']:>9.0f} "
              f"{s['mean_speed']:>10.3f} {s['mean_nn']:>8.3f} "
              f"{s['n_sequences']:>7}")

    # ── Recommendations & gap analysis ───────────────────────────────────────
    print(MODEL_RECS)
    print(GAP_ANALYSIS)

    # ── Demo: evaluation metric shapes ───────────────────────────────────────
    print("="*55)
    print("  METRIC FUNCTION SIGNATURES (demo with random data)")
    print("="*55)
    N, K, T = 50, 20, 12
    gt      = np.random.randn(N, T, 2)
    pred_s  = np.random.randn(N, T, 2)
    samples = np.random.randn(N, K, T, 2)
    print(f"  ADE               : {ade(pred_s, gt):.4f}")
    print(f"  FDE               : {fde(pred_s, gt):.4f}")
    print(f"  Best-of-{K} ADE   : {best_of_k_ade(samples, gt):.4f}")
    print(f"  Best-of-{K} FDE   : {best_of_k_fde(samples, gt):.4f}")
    others = np.random.randn(30, T, 2)
    print(f"  Collision rate    : {collision_rate(pred_s, others):.4f}")

    print(f"\nAll plots saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("Script complete.")


if __name__ == "__main__":
    main()
