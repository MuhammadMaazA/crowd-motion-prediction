"""
Stanford Drone Dataset (SDD) — Data Loading and Sequence Extraction
====================================================================
Mirrors the interface of eth_ucy_analysis.py so all existing models
(Social-LSTM, Transformer, Diffusion) train on SDD without modification.

SDD annotation format (raw)
---------------------------
  track_id xmin ymin xmax ymax frame_id lost occluded generated label
  e.g.: 0 211 1038 239 1072 10000 1 0 0 "Biker"

We:
  - Filter label == "Pedestrian"
  - Filter lost == 0 (tracking intact)
  - Centroid: x = (xmin+xmax)/2, y = (ymin+ymax)/2 (pixels)
  - Subsample to 2.5 FPS (every 12th frame from 30 FPS)
  - Normalise pixel coords to metres using scale=0.0417 (1 pixel ≈ 0.04m,
    a standard approximation matching SDD papers reporting ~0.5m ADE)

Scenes (8 total, leave-one-out evaluation)
------------------------------------------
  bookstore, coupa, deathCircle, gates, hyang, little, nexus, quad

Each scene has multiple video files which are merged with ID offsetting
to avoid pedestrian ID collisions (same approach as ETH/UCY univ scene).
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple

# ── Constants ──────────────────────────────────────────────────────────────────

SDD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sdd")

# Pixel → metre scale factor (matches SDD evaluation in published papers)
PIXEL_TO_METRE = 0.0417

# Subsample: SDD is ~30 FPS, we want 2.5 FPS → every 12th frame
FRAME_STRIDE = 12

SCENE_FILES = {
    "bookstore":    [os.path.join(SDD_DIR, f"bookstore_video{i}.txt")   for i in range(7)],
    "coupa":        [os.path.join(SDD_DIR, f"coupa_video{i}.txt")       for i in range(4)],
    "deathCircle":  [os.path.join(SDD_DIR, f"deathCircle_video{i}.txt") for i in range(5)],
    "gates":        [os.path.join(SDD_DIR, f"gates_video{i}.txt")       for i in range(9)],
    "hyang":        [os.path.join(SDD_DIR, f"hyang_video{i}.txt")       for i in range(15)],
    "little":       [os.path.join(SDD_DIR, f"little_video{i}.txt")      for i in range(4)],
    "nexus":        [os.path.join(SDD_DIR, f"nexus_video{i}.txt")       for i in range(12)],
    "quad":         [os.path.join(SDD_DIR, f"quad_video{i}.txt")        for i in range(4)],
}

SCENES = list(SCENE_FILES.keys())


# ── Data loading ───────────────────────────────────────────────────────────────

def load_sdd_file(filepath: str, id_offset: int = 0) -> pd.DataFrame:
    """
    Load one SDD annotation file → DataFrame with columns [frame, ped, x, y].

    Filters pedestrians only, drops lost tracks, converts to metres,
    subsamples to 2.5 FPS.
    """
    rows = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            track_id = int(parts[0])
            xmin, ymin, xmax, ymax = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            frame_id = int(parts[5])
            lost     = int(parts[6])
            label    = parts[9].strip('"')

            if label != "Pedestrian" or lost == 1:
                continue

            cx = (xmin + xmax) / 2.0 * PIXEL_TO_METRE
            cy = (ymin + ymax) / 2.0 * PIXEL_TO_METRE
            rows.append((frame_id, track_id + id_offset, cx, cy))

    if not rows:
        return pd.DataFrame(columns=["frame", "ped", "x", "y"])

    df = pd.DataFrame(rows, columns=["frame", "ped", "x", "y"])

    # Subsample to 2.5 FPS: keep only frames divisible by FRAME_STRIDE
    min_frame = df["frame"].min()
    df = df[(df["frame"] - min_frame) % FRAME_STRIDE == 0].copy()
    # Renumber frames to consecutive integers
    df["frame"] = (df["frame"] - min_frame) // FRAME_STRIDE
    return df.sort_values(["frame", "ped"]).reset_index(drop=True)


def load_scene(files: List[str]) -> pd.DataFrame:
    """
    Load and merge multiple SDD annotation files for one scene.
    Offsets pedestrian IDs to avoid collisions across files.

    Parameters
    ----------
    files : list of file paths (all videos for one scene)

    Returns
    -------
    DataFrame with columns [frame, ped, x, y]
    """
    dfs = []
    id_offset = 0
    for fpath in files:
        if not os.path.exists(fpath):
            continue
        df = load_sdd_file(fpath, id_offset=id_offset)
        if len(df) == 0:
            continue
        dfs.append(df)
        id_offset += int(df["ped"].max()) + 1

    if not dfs:
        return pd.DataFrame(columns=["frame", "ped", "x", "y"])

    combined = pd.concat(dfs, ignore_index=True)
    return combined.sort_values(["frame", "ped"]).reset_index(drop=True)


# ── Sequence extraction — vectorised per-pedestrian approach ───────────────────

def _build_ped_arrays(data: pd.DataFrame, seq_len: int
                      ) -> dict:
    """
    For each pedestrian, build a dict of frame→(x,y) and find all valid
    start frames that yield a consecutive seq_len window.
    Returns {ped_id: (sorted_frames_array, xy_array)} for peds with enough data.
    """
    result = {}
    for ped, grp in data.groupby("ped"):
        grp = grp.sort_values("frame")
        frames = grp["frame"].values
        xy     = grp[["x", "y"]].values
        if len(frames) < seq_len:
            continue
        result[ped] = (frames, xy)
    return result


def extract_sequences(data: pd.DataFrame,
                      obs_len: int = 8,
                      pred_len: int = 12
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (obs, pred) sequence pairs — vectorised per-pedestrian.

    Returns
    -------
    obs  : (N, obs_len, 2)
    pred : (N, pred_len, 2)
    """
    seq_len = obs_len + pred_len
    all_obs, all_pred = [], []

    ped_data = _build_ped_arrays(data, seq_len)

    for ped, (frames, xy) in ped_data.items():
        for i in range(len(frames) - seq_len + 1):
            win = frames[i: i + seq_len]
            # Require strictly consecutive frames
            if win[-1] - win[0] != seq_len - 1:
                continue
            all_obs.append(xy[i: i + obs_len])
            all_pred.append(xy[i + obs_len: i + seq_len])

    if not all_obs:
        return np.zeros((0, obs_len, 2), dtype=np.float32), \
               np.zeros((0, pred_len, 2), dtype=np.float32)
    return np.array(all_obs, dtype=np.float32), np.array(all_pred, dtype=np.float32)


def extract_sequences_with_neighbours(
        data: pd.DataFrame,
        obs_len: int = 8,
        pred_len: int = 12,
        max_neighbours: int = 5
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract sequences with neighbour context — vectorised per-pedestrian.

    Returns
    -------
    obs      : (N, obs_len, 2)
    pred     : (N, pred_len, 2)
    nb_obs   : (N, max_neighbours, obs_len, 2)  NaN where absent
    nb_mask  : (N, max_neighbours) bool
    """
    seq_len = obs_len + pred_len
    _empty  = (np.zeros((0, obs_len, 2), dtype=np.float32),
               np.zeros((0, pred_len, 2), dtype=np.float32),
               np.full((0, max_neighbours, obs_len, 2), np.nan, dtype=np.float32),
               np.zeros((0, max_neighbours), dtype=bool))

    ped_data = _build_ped_arrays(data, obs_len)  # obs only needed for neighbours

    # Build a frame→{ped: xy} lookup for fast neighbour queries
    frame_to_peds: dict = {}
    for ped, (frames, xy) in ped_data.items():
        for j, f in enumerate(frames):
            if f not in frame_to_peds:
                frame_to_peds[f] = {}
            frame_to_peds[f][ped] = xy[j]

    all_obs, all_pred, all_nb_obs, all_nb_mask = [], [], [], []

    ped_data_full = _build_ped_arrays(data, seq_len)

    for ped, (frames, xy) in ped_data_full.items():
        for i in range(len(frames) - seq_len + 1):
            win = frames[i: i + seq_len]
            if win[-1] - win[0] != seq_len - 1:
                continue

            obs_xy  = xy[i: i + obs_len]
            pred_xy = xy[i + obs_len: i + seq_len]
            mean_pos = obs_xy.mean(axis=0)

            # Gather neighbours present in all obs frames
            obs_frames_set = set(win[:obs_len])
            candidate_peds = set(frame_to_peds.get(win[0], {}).keys())
            for f in win[1:obs_len]:
                candidate_peds &= set(frame_to_peds.get(f, {}).keys())
            candidate_peds.discard(ped)

            # Build neighbour trajectories and sort by distance
            nb_list = []
            for nb_ped in candidate_peds:
                nb_traj = np.array([frame_to_peds[f][nb_ped] for f in win[:obs_len]])
                dist    = np.linalg.norm(nb_traj.mean(axis=0) - mean_pos)
                nb_list.append((dist, nb_traj))
            nb_list.sort(key=lambda x: x[0])

            nb_obs_arr  = np.full((max_neighbours, obs_len, 2), np.nan, dtype=np.float32)
            nb_mask_arr = np.zeros(max_neighbours, dtype=bool)
            for k, (_, nb_traj) in enumerate(nb_list[:max_neighbours]):
                nb_obs_arr[k]  = nb_traj
                nb_mask_arr[k] = True

            all_obs.append(obs_xy)
            all_pred.append(pred_xy)
            all_nb_obs.append(nb_obs_arr)
            all_nb_mask.append(nb_mask_arr)

    if not all_obs:
        return _empty

    return (np.array(all_obs,     dtype=np.float32),
            np.array(all_pred,    dtype=np.float32),
            np.array(all_nb_obs,  dtype=np.float32),
            np.array(all_nb_mask, dtype=bool))


# ── Metrics (identical to eth_ucy_analysis.py) ────────────────────────────────

def ade(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred_traj - gt_traj, axis=-1)))

def fde(pred_traj: np.ndarray, gt_traj: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(pred_traj[:, -1] - gt_traj[:, -1], axis=-1)))

def best_of_k_ade(samples: np.ndarray, gt_traj: np.ndarray) -> float:
    errors = np.linalg.norm(samples - gt_traj[:, None], axis=-1).mean(axis=-1)  # (N, K)
    return float(errors.min(axis=1).mean())

def best_of_k_fde(samples: np.ndarray, gt_traj: np.ndarray) -> float:
    errors = np.linalg.norm(samples[:, :, -1] - gt_traj[:, None, -1], axis=-1)  # (N, K)
    return float(errors.min(axis=1).mean())


# ── Quick dataset stats ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SDD Dataset Statistics")
    print("=" * 50)
    total_seq = 0
    for scene, files in SCENE_FILES.items():
        data = load_scene(files)
        if len(data) == 0:
            print(f"  {scene:15s}: no data")
            continue
        obs, pred = extract_sequences(data)
        n = len(obs)
        total_seq += n
        n_peds = data["ped"].nunique()
        print(f"  {scene:15s}: {n_peds:4d} pedestrians, {n:5d} sequences")
    print(f"{'TOTAL':15s}: {total_seq:5d} sequences across 8 scenes")
