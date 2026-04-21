"""
Unified Evaluation Script — D1 Comparison Table
================================================
Evaluates CV baseline, Social-LSTM, and Trajectron++ on all 5 ETH/UCY scenes
and prints a clean comparison table.

Usage
-----
source crowdnav-env/bin/activate
python evaluate_all.py

Requirements
------------
- Social-LSTM checkpoints in checkpoints/social_lstm_{scene}.pt
- Trajectron++ checkpoints in Trajectron-plus-plus/experiments/logs/
- ETH/UCY data in data/ or Trajectron-plus-plus/experiments/pedestrians/raw/
"""

import os
import sys
import numpy as np
import torch

WORK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK)

from eth_ucy_analysis import (
    load_scene, extract_sequences, extract_sequences_with_neighbours,
    ade, fde, best_of_k_ade, best_of_k_fde
)
from models.cv_baseline import ConstantVelocityPredictor
from models.social_lstm import SocialLSTM, bivariate_gaussian_nll

# ── Scene file paths ───────────────────────────────────────────────────────────
RAW = os.path.join(WORK, "Trajectron-plus-plus/experiments/pedestrians/raw")
SCENE_FILES = {
    "eth":   [os.path.join(RAW, "eth",   "test", "biwi_eth.txt")],
    "hotel": [os.path.join(RAW, "hotel", "test", "biwi_hotel.txt")],
    "univ":  [os.path.join(RAW, "univ",  "test", "students001.txt"),
              os.path.join(RAW, "univ",  "test", "students003.txt")],
    "zara1": [os.path.join(RAW, "zara1", "test", "crowds_zara01.txt")],
    "zara2": [os.path.join(RAW, "zara2", "test", "crowds_zara02.txt")],
}

SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]
OBS_LEN, PRED_LEN, K = 8, 12, 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── CV Baseline ────────────────────────────────────────────────────────────────

def eval_cv(scene):
    data = load_scene(SCENE_FILES[scene])
    obs, pred = extract_sequences(data, obs_len=OBS_LEN, pred_len=PRED_LEN)
    if len(obs) == 0:
        return None
    cv = ConstantVelocityPredictor(noise_std=0.30)
    samples = cv.predict_samples(obs, K=K, pred_len=PRED_LEN)   # (N, K, 12, 2)
    mean_pred = samples[:, 0]                                    # deterministic
    return {
        "ade":       ade(mean_pred, pred),
        "fde":       fde(mean_pred, pred),
        "minADE_20": best_of_k_ade(samples, pred),
        "minFDE_20": best_of_k_fde(samples, pred),
    }


# ── Social-LSTM ────────────────────────────────────────────────────────────────

def _load_social_lstm(ckpt_path):
    """Load a Social-LSTM checkpoint and return (model, data_loader_fn)."""
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    hp   = ckpt["hparams"]
    model = SocialLSTM(
        obs_len=OBS_LEN, pred_len=PRED_LEN,
        hidden_size=hp["hidden_size"],
        embed_size=hp["embed_size"],
        pooling_radius=hp["pooling_radius"],
        use_velocity=hp.get("use_velocity", False),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _eval_social_lstm_model(model, scene):
    """Shared eval logic for any Social-LSTM variant."""
    data = load_scene(SCENE_FILES[scene])
    obs, pred, nb_obs, nb_mask = extract_sequences_with_neighbours(
        data, obs_len=OBS_LEN, pred_len=PRED_LEN, max_neighbours=5
    )
    if len(obs) == 0:
        return None

    obs_t     = torch.tensor(obs,     dtype=torch.float32).to(DEVICE)
    pred_t    = torch.tensor(pred,    dtype=torch.float32).to(DEVICE)
    nb_obs_t  = torch.tensor(np.nan_to_num(nb_obs, nan=0.0), dtype=torch.float32).to(DEVICE)
    nb_mask_t = torch.tensor(nb_mask, dtype=torch.bool).to(DEVICE)

    with torch.no_grad():
        preds   = model(obs_t, nb_obs_t, nb_mask_t)
        mus_np  = preds["mus"].cpu().numpy()
        samples = model.sample(obs_t, nb_obs_t, nb_mask_t, K=K)  # (N, K, T, 2)
        nll_val = bivariate_gaussian_nll(preds, pred_t).item()

    return {
        "ade":       ade(mus_np, pred),
        "fde":       fde(mus_np, pred),
        "minADE_20": best_of_k_ade(samples, pred),
        "minFDE_20": best_of_k_fde(samples, pred),
        "nll":       nll_val,
    }


def eval_social_lstm(scene):
    ckpt_path = os.path.join(WORK, "checkpoints", f"social_lstm_{scene}.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [skip] No checkpoint for {scene}")
        return None
    return _eval_social_lstm_model(_load_social_lstm(ckpt_path), scene)


def eval_social_lstm_v(scene):
    """Velocity-augmented Social-LSTM (our contribution)."""
    ckpt_path = os.path.join(WORK, "checkpoints", f"social_lstmv_{scene}.pt")
    if not os.path.exists(ckpt_path):
        return None
    return _eval_social_lstm_model(_load_social_lstm(ckpt_path), scene)


# ── Trajectron++ ───────────────────────────────────────────────────────────────

def eval_trajectronpp(scene):
    """
    Load the best Trajectron++ checkpoint for a scene and evaluate on the test set.
    Looks for checkpoints saved by train.py in experiments/logs/.
    """
    import dill
    sys.path.insert(0, os.path.join(WORK, "Trajectron-plus-plus", "trajectron"))

    try:
        from model.model_registrar import ModelRegistrar
        from model.trajectron import Trajectron
        from argument_parser import args as tpp_args
    except ImportError as e:
        print(f"  [skip] Trajectron++ import failed: {e}")
        return None

    # Find the checkpoint directory for this scene that has the highest completed epoch
    log_base = os.path.join(WORK, "Trajectron-plus-plus", "experiments", "logs")
    scene_dirs = [d for d in os.listdir(log_base) if scene in d] if os.path.exists(log_base) else []
    if not scene_dirs:
        print(f"  [skip] No Trajectron++ checkpoint found for {scene}")
        return None

    best_dir, best_epoch = None, -1
    for d in scene_dirs:
        path = os.path.join(log_base, d)
        pts = [f for f in os.listdir(path) if f.startswith("model_registrar") and f.endswith(".pt")]
        if pts:
            epoch = max(int(f.replace("model_registrar-", "").replace(".pt", "")) for f in pts)
            if epoch > best_epoch:
                best_epoch, best_dir = epoch, path

    if best_dir is None:
        print(f"  [skip] No .pt files found for {scene}")
        return None

    ckpt_dir  = best_dir
    epoch_num = best_epoch

    # Load test data
    test_pkl = os.path.join(WORK, "Trajectron-plus-plus", "experiments", "processed",
                            f"{scene}_test.pkl")
    with open(test_pkl, "rb") as f:
        test_env = dill.load(f, encoding="latin1")

    # Load model
    model_reg = ModelRegistrar(ckpt_dir, "cpu")
    model_reg.load_models(epoch_num)

    conf_path = os.path.join(WORK, "Trajectron-plus-plus", "experiments",
                             "pedestrians", "models",
                             f"{scene}_attention_radius_3", "config.json")
    import json
    with open(conf_path) as f:
        hyperparams = json.load(f)

    hyperparams["maximum_history_length"] = 7
    hyperparams["prediction_horizon"]     = PRED_LEN

    trajectron = Trajectron(model_reg, hyperparams, None, "cpu")
    trajectron.set_environment(test_env)
    trajectron.set_annealing_params()

    from utils.trajectory_utils import prediction_output_to_trajectories

    # Collect predictions and ground truth
    all_preds, all_gt = [], []

    for scene_obj in test_env.scenes:
        timesteps = np.arange(0, scene_obj.timesteps)

        with torch.no_grad():
            predictions = trajectron.predict(
                scene_obj, timesteps,
                ph=PRED_LEN,
                num_samples=K,
                min_future_timesteps=PRED_LEN,
                full_dist=False
            )

        if not predictions:
            continue

        # Use Trajectron++'s own utility to extract (predictions, futures)
        pred_dict, _, futures_dict = prediction_output_to_trajectories(
            predictions,
            dt=scene_obj.dt,
            max_h=hyperparams["maximum_history_length"],
            ph=PRED_LEN,
        )

        for ts in pred_dict:
            for node in pred_dict[ts]:
                preds_arr = pred_dict[ts][node]   # (1, K, T, 2) or (K, T, 2)
                if preds_arr.ndim == 4:
                    preds_arr = preds_arr[0]       # squeeze to (K, T, 2)
                gt_xy = futures_dict[ts][node]     # (T, 2)
                if gt_xy.shape[0] < PRED_LEN:
                    continue
                all_preds.append(preds_arr[:, :PRED_LEN])
                all_gt.append(gt_xy[:PRED_LEN])

    if not all_preds:
        print(f"  [skip] No valid predictions for {scene}")
        return None

    samples_np = np.stack(all_preds)    # (N, K, T, 2)
    gt_np      = np.stack(all_gt)       # (N, T, 2)
    mean_pred  = samples_np[:, 0]       # deterministic (first sample)

    # NLL via model's eval_loss on the test EnvironmentDataset
    nll_val = None
    try:
        from model.dataset import EnvironmentDataset, collate
        from torch.utils.data import DataLoader

        nll_dataset = EnvironmentDataset(
            test_env,
            hyperparams['state'],
            hyperparams['pred_state'],
            scene_freq_mult=False,
            node_freq_mult=False,
            hyperparams=hyperparams,
            min_history_timesteps=hyperparams['minimum_history_length'],
            min_future_timesteps=hyperparams['prediction_horizon'],
            return_robot=True,
        )
        nll_values = []
        for node_type_ds in nll_dataset:
            if len(node_type_ds) == 0:
                continue
            node_type = node_type_ds.node_type
            if node_type not in trajectron.node_models_dict:
                continue
            loader = DataLoader(node_type_ds, collate_fn=collate,
                                batch_size=256, shuffle=False, num_workers=0)
            for batch in loader:
                with torch.no_grad():
                    nll_batch = trajectron.eval_loss(batch, node_type)
                    nll_values.extend(nll_batch.tolist())
        if nll_values:
            nll_val = float(np.mean(nll_values))
    except Exception as e:
        print(f"  [warn] Trajectron++ NLL failed: {e}")

    return {
        "ade":       ade(mean_pred, gt_np),
        "fde":       fde(mean_pred, gt_np),
        "minADE_20": best_of_k_ade(samples_np, gt_np),
        "minFDE_20": best_of_k_fde(samples_np, gt_np),
        "nll":       nll_val,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def fmt(v):
    return f"{v:.3f}" if v is not None else "  —  "


def print_table(results):
    all_models = ["CV", "Social-LSTM", "Social-LSTM+V", "Trajectron++"]
    # Only show Social-LSTM+V column if any results exist for it
    has_v = any(results["Social-LSTM+V"].get(s) for s in SCENES)
    models = all_models if has_v else [m for m in all_models if m != "Social-LSTM+V"]

    metrics = ["ade", "fde", "minADE_20", "minFDE_20", "nll"]
    metric_labels = ["ADE", "FDE", "minADE@20", "minFDE@20", "NLL (Social-LSTM only)"]

    width = 10 + 14 * len(models)
    print("\n" + "=" * width)
    print(f"{'D1: Prediction Model Comparison — ETH/UCY Leave-One-Out':^{width}}")
    print("=" * width)

    for metric, label in zip(metrics, metric_labels):
        print(f"\n{label}")
        header = f"{'Scene':<10}" + "".join(f"{m:>14}" for m in models)
        print(header)
        print("-" * len(header))
        avgs = {m: [] for m in models}
        for scene in SCENES:
            row = f"{scene:<10}"
            for model_name in models:
                v = results[model_name].get(scene, {})
                val = v.get(metric) if v else None
                row += f"{fmt(val):>14}"
                if val is not None:
                    avgs[model_name].append(val)
            print(row)
        # Average row
        row = f"{'avg':<10}"
        for model_name in models:
            vals = avgs[model_name]
            row += f"{fmt(np.mean(vals) if vals else None):>14}"
        print("-" * len(header))
        print(row)

    print("=" * width + "\n")


if __name__ == "__main__":
    print("Evaluating all models on ETH/UCY test sets...")

    results = {"CV": {}, "Social-LSTM": {}, "Social-LSTM+V": {}, "Trajectron++": {}}

    for scene in SCENES:
        print(f"\n--- {scene} ---")

        print("  CV baseline...")
        results["CV"][scene] = eval_cv(scene)

        print("  Social-LSTM...")
        results["Social-LSTM"][scene] = eval_social_lstm(scene)

        print("  Social-LSTM+V...")
        results["Social-LSTM+V"][scene] = eval_social_lstm_v(scene)

        print("  Trajectron++...")
        results["Trajectron++"][scene] = eval_trajectronpp(scene)

    print_table(results)
