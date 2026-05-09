"""
Preprocess Stanford Drone Dataset for Trajectron++ training.
Converts SDD bounding-box annotations → Trajectron++ pkl Environment objects.

For each of the 8 SDD scenes as holdout, creates:
  experiments/processed/sdd_{holdout}_train.pkl
  experiments/processed/sdd_{holdout}_test.pkl

Usage
-----
source crowdnav-env/bin/activate
python process_sdd_trajectron.py
"""

import sys, os
import numpy as np
import pandas as pd
import dill

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "Trajectron-plus-plus", "trajectron"))
from environment import Environment, Scene, Node
from environment import derivative_of

SDD_DIR     = os.path.join(os.path.dirname(__file__), "data", "sdd")
PROC_DIR    = os.path.join(os.path.dirname(__file__),
                           "Trajectron-plus-plus", "experiments", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

PIXEL_TO_M  = 0.0417
FRAME_STRIDE= 12       # subsample 30fps → 2.5fps
DT          = 0.4      # seconds per timestep

SCENES = ["bookstore", "coupa", "deathCircle", "gates",
          "hyang", "little", "nexus", "quad"]
VIDEOS = {
    "bookstore": 7, "coupa": 4, "deathCircle": 5, "gates": 9,
    "hyang": 15,    "little": 4, "nexus": 12,     "quad": 4,
}

standardization = {
    'PEDESTRIAN': {
        'position':     {'x': {'mean': 0, 'std': 1},  'y': {'mean': 0, 'std': 1}},
        'velocity':     {'x': {'mean': 0, 'std': 2},  'y': {'mean': 0, 'std': 2}},
        'acceleration': {'x': {'mean': 0, 'std': 1},  'y': {'mean': 0, 'std': 1}},
    }
}
data_columns = pd.MultiIndex.from_product(
    [['position', 'velocity', 'acceleration'], ['x', 'y']])


def read_sdd_video(fpath, id_offset=0):
    """Load one SDD video annotation → DataFrame (frame, ped, x_m, y_m)."""
    rows = []
    with open(fpath) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 10: continue
            track_id = int(p[0])
            xmin,ymin,xmax,ymax = float(p[1]),float(p[2]),float(p[3]),float(p[4])
            frame_id = int(p[5])
            lost     = int(p[6])
            label    = p[9].strip('"')
            if label != "Pedestrian" or lost == 1: continue
            cx = (xmin + xmax) / 2 * PIXEL_TO_M
            cy = (ymin + ymax) / 2 * PIXEL_TO_M
            rows.append((frame_id, track_id + id_offset, cx, cy))
    if not rows:
        return pd.DataFrame(columns=["frame","ped","x","y"])
    df = pd.DataFrame(rows, columns=["frame","ped","x","y"])
    min_f = df["frame"].min()
    df = df[(df["frame"] - min_f) % FRAME_STRIDE == 0].copy()
    df["frame"] = (df["frame"] - min_f) // FRAME_STRIDE
    return df.sort_values(["frame","ped"]).reset_index(drop=True)


def load_scene_df(scene_name):
    """Merge all videos for a scene into one DataFrame with unique ped IDs."""
    dfs, id_off = [], 0
    for i in range(VIDEOS[scene_name]):
        fpath = os.path.join(SDD_DIR, f"{scene_name}_video{i}.txt")
        if not os.path.exists(fpath): continue
        df = read_sdd_video(fpath, id_offset=id_off)
        if len(df): dfs.append(df); id_off += int(df["ped"].max()) + 1
    if not dfs: return pd.DataFrame(columns=["frame","ped","x","y"])
    return pd.concat(dfs, ignore_index=True).sort_values(["frame","ped"]).reset_index(drop=True)


def make_env(scenes_list=None):
    env = Environment(node_type_list=["PEDESTRIAN"],
                      standardization=standardization)
    env.attention_radius = {
        (env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 3.0
    }
    if scenes_list is not None:
        env.scenes = [s for s in scenes_list if s is not None]
    return env


# Build a shared env to get the proper NodeType enum
_env_ref = make_env()
PEDESTRIAN_TYPE = _env_ref.NodeType.PEDESTRIAN


def df_to_scene(df, scene_name, split, augment_train=True):
    """Convert a trajectory DataFrame to a Trajectron++ Scene object."""
    if len(df) == 0:
        return None

    df = df.copy()
    df["x"] -= df["x"].mean()
    df["y"] -= df["y"].mean()

    max_ts = int(df["frame"].max())
    scene  = Scene(timesteps=max_ts + 1, dt=DT,
                   name=f"{scene_name}_{split}")

    for ped_id, grp in df.groupby("ped"):
        grp = grp.sort_values("frame")
        frames = grp["frame"].values
        if len(frames) < 2: continue
        x  = grp["x"].values
        y  = grp["y"].values
        vx = derivative_of(x, DT)
        vy = derivative_of(y, DT)
        ax = derivative_of(vx, DT)
        ay = derivative_of(vy, DT)
        data_dict = {('position','x'):x, ('position','y'):y,
                     ('velocity','x'):vx,('velocity','y'):vy,
                     ('acceleration','x'):ax,('acceleration','y'):ay}
        node_data = pd.DataFrame(data_dict, columns=data_columns)
        # Use enum NodeType (not string) — required by scene_graph.py
        node = Node(node_type=PEDESTRIAN_TYPE, node_id=str(ped_id),
                    data=node_data)
        node.first_timestep = int(frames[0])
        scene.nodes.append(node)

    return scene if scene.nodes else None
    env.scenes = [s for s in scenes_list if s is not None]
    return env


def process_holdout(holdout_scene):
    """Create train + test pkl for one leave-one-out split."""
    train_scenes, test_scenes = [], []

    for scene_name in SCENES:
        print(f"  Loading {scene_name}...", end="", flush=True)
        df = load_scene_df(scene_name)
        if len(df) == 0:
            print(" (empty)")
            continue

        if scene_name == holdout_scene:
            sc = df_to_scene(df, scene_name, "test", augment_train=False)
            if sc: test_scenes.append(sc)
            print(f" test ({len(sc.nodes) if sc else 0} agents)")
        else:
            sc = df_to_scene(df, scene_name, "train", augment_train=True)
            if sc: train_scenes.append(sc)
            print(f" train ({len(sc.nodes) if sc else 0} agents)")

    train_env = make_env(train_scenes)
    test_env  = make_env(test_scenes)

    train_path = os.path.join(PROC_DIR, f"sdd_{holdout_scene}_train.pkl")
    test_path  = os.path.join(PROC_DIR, f"sdd_{holdout_scene}_test.pkl")

    with open(train_path, "wb") as f: dill.dump(train_env, f, protocol=dill.HIGHEST_PROTOCOL)
    with open(test_path,  "wb") as f: dill.dump(test_env,  f, protocol=dill.HIGHEST_PROTOCOL)

    print(f"  Saved: {os.path.basename(train_path)} ({len(train_env.scenes)} scenes)")
    print(f"  Saved: {os.path.basename(test_path)}  ({len(test_env.scenes)} scenes)")


if __name__ == "__main__":
    for holdout in SCENES:
        print(f"\n=== Processing holdout: {holdout} ===")
        process_holdout(holdout)
    print("\nAll SDD pkl files created.")
