"""
SDD Evaluation Script
=====================
Evaluates CV, Social-LSTM, Social-LSTM+V, GRU-v2, Transformer, Diffusion
on all 8 SDD scenes (leave-one-out) and prints a comparison table.

Usage
-----
source crowdnav-env/bin/activate
python evaluate_sdd.py
"""

import os
import sys
import numpy as np
import torch

WORK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK)

import sdd_analysis as sdd
from sdd_analysis import (
    load_scene, extract_sequences, extract_sequences_with_neighbours,
    SCENE_FILES,
)
from eth_ucy_analysis import ade, fde, best_of_k_ade, best_of_k_fde
from models.cv_baseline import ConstantVelocityPredictor
from models.social_lstm import SocialLSTM, bivariate_gaussian_nll
from models.social_gru_v2 import SocialGRUv2
from models.trajectory_transformer import TrajectoryTransformer
from models.diffusion import TrajDiffusion

SCENES = ["bookstore", "coupa", "deathCircle", "gates", "hyang", "little", "nexus", "quad"]
OBS_LEN, PRED_LEN, K = 8, 12, 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = os.path.join(WORK, "checkpoints", "sdd")


# ── CV Baseline ────────────────────────────────────────────────────────────────

def eval_cv(scene):
    data = load_scene(SCENE_FILES[scene])
    obs, pred = extract_sequences(data, obs_len=OBS_LEN, pred_len=PRED_LEN)
    if len(obs) == 0:
        return None
    cv = ConstantVelocityPredictor(noise_std=0.30)
    samples = cv.predict_samples(obs, K=K, pred_len=PRED_LEN)
    mean_pred = samples[:, 0]
    return {
        "ade":       ade(mean_pred, pred),
        "fde":       fde(mean_pred, pred),
        "minADE_20": best_of_k_ade(samples, pred),
        "minFDE_20": best_of_k_fde(samples, pred),
    }


# ── Shared social model eval ───────────────────────────────────────────────────

EVAL_BATCH = 512   # process sequences in chunks to avoid OOM on large SDD scenes

def _eval_social_model(model, scene):
    data = load_scene(SCENE_FILES[scene])
    obs, pred, nb_obs, nb_mask = extract_sequences_with_neighbours(
        data, obs_len=OBS_LEN, pred_len=PRED_LEN, max_neighbours=5
    )
    if len(obs) == 0:
        return None

    N = len(obs)
    all_mus, all_samples, all_nll = [], [], []

    for start in range(0, N, EVAL_BATCH):
        end = min(start + EVAL_BATCH, N)
        obs_t     = torch.tensor(obs[start:end],                             dtype=torch.float32).to(DEVICE)
        pred_t    = torch.tensor(pred[start:end],                            dtype=torch.float32).to(DEVICE)
        nb_obs_t  = torch.tensor(np.nan_to_num(nb_obs[start:end], nan=0.0), dtype=torch.float32).to(DEVICE)
        nb_mask_t = torch.tensor(nb_mask[start:end],                        dtype=torch.bool).to(DEVICE)

        with torch.no_grad():
            preds   = model(obs_t, nb_obs_t, nb_mask_t)
            mus_np  = preds["mus"].cpu().numpy()
            samps   = model.sample(obs_t, nb_obs_t, nb_mask_t, K=K)   # (B, K, T, 2)
            nll_b   = bivariate_gaussian_nll(preds, pred_t).item()

        all_mus.append(mus_np)
        all_samples.append(samps)
        all_nll.append(nll_b * (end - start))

    mus_np  = np.concatenate(all_mus,     axis=0)   # (N, T, 2)
    samples = np.concatenate(all_samples, axis=0)   # (N, K, T, 2)
    nll_val = sum(all_nll) / N

    return {
        "ade":       ade(mus_np, pred),
        "fde":       fde(mus_np, pred),
        "minADE_20": best_of_k_ade(samples, pred),
        "minFDE_20": best_of_k_fde(samples, pred),
        "nll":       nll_val,
    }


def _load_social_lstm(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
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


def eval_social_lstm(scene):
    ckpt = os.path.join(CKPT_DIR, f"social_lstm_{scene}.pt")
    if not os.path.exists(ckpt):
        return None
    return _eval_social_model(_load_social_lstm(ckpt), scene)


def eval_social_lstm_v(scene):
    ckpt = os.path.join(CKPT_DIR, f"social_lstm_v_{scene}.pt")
    if not os.path.exists(ckpt):
        return None
    return _eval_social_model(_load_social_lstm(ckpt), scene)


# ── GRU-v2 ────────────────────────────────────────────────────────────────────

def _load_gru_v2(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    hp   = ckpt["hparams"]
    model = SocialGRUv2(
        obs_len=OBS_LEN, pred_len=PRED_LEN,
        hidden_size=hp.get("hidden_size", 128),
        embed_size=hp.get("embed_size", 64),
        pooling_radius=hp.get("pooling_radius", 2.0),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def eval_gru_v2(scene):
    ckpt = os.path.join(CKPT_DIR, f"gru_v2_{scene}.pt")
    if not os.path.exists(ckpt):
        return None
    return _eval_social_model(_load_gru_v2(ckpt), scene)


# ── Transformer ───────────────────────────────────────────────────────────────

def _load_transformer(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    hp   = ckpt["hparams"]
    model = TrajectoryTransformer(
        obs_len=OBS_LEN, pred_len=PRED_LEN,
        d_model=hp.get("d_model", 128), nhead=hp.get("nhead", 4),
        num_enc=hp.get("num_enc", 2),   num_dec=hp.get("num_dec", 2),
        dim_ff=hp.get("dim_ff", 128),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def eval_transformer(scene):
    ckpt = os.path.join(CKPT_DIR, f"transformer_{scene}.pt")
    if not os.path.exists(ckpt):
        return None
    return _eval_social_model(_load_transformer(ckpt), scene)


# ── Diffusion ─────────────────────────────────────────────────────────────────

def _load_diffusion(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    hp   = ckpt["hparams"]
    model = TrajDiffusion(
        obs_len=OBS_LEN, pred_len=PRED_LEN,
        d_model=hp.get("d_model", 128), nhead=hp.get("nhead", 4),
        T=hp.get("T", 100), ddim_steps=hp.get("ddim_steps", 20),
        lambda_ddpm=hp.get("lambda_ddpm", 0.1),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def eval_diffusion(scene):
    ckpt = os.path.join(CKPT_DIR, f"diffusion_{scene}.pt")
    if not os.path.exists(ckpt):
        return None
    return _eval_social_model(_load_diffusion(ckpt), scene)


# ── Table printing ─────────────────────────────────────────────────────────────

def fmt(v):
    return f"{v:.3f}" if v is not None else "  —  "


def print_table(results):
    all_models = ["CV", "Social-LSTM", "Social-LSTM+V", "GRU-v2", "Transformer", "Diffusion"]
    models = [m for m in all_models if any(results.get(m, {}).get(s) for s in SCENES)]

    metrics       = ["ade", "fde", "minADE_20", "minFDE_20", "nll"]
    metric_labels = ["ADE", "FDE", "minADE@20", "minFDE@20", "NLL"]

    width = 13 + 14 * len(models)
    print("\n" + "=" * width)
    print(f"{'SDD Prediction Model Comparison — Leave-One-Out':^{width}}")
    print("=" * width)

    for metric, label in zip(metrics, metric_labels):
        print(f"\n{label}")
        header = f"{'Scene':<13}" + "".join(f"{m:>14}" for m in models)
        print(header)
        print("-" * len(header))
        avgs = {m: [] for m in models}
        for scene in SCENES:
            row = f"{scene:<13}"
            for mname in models:
                v   = results[mname].get(scene, {})
                val = v.get(metric) if v else None
                row += f"{fmt(val):>14}"
                if val is not None:
                    avgs[mname].append(val)
            print(row)
        row = f"{'avg':<13}"
        for mname in models:
            vals = avgs[mname]
            row += f"{fmt(np.mean(vals) if vals else None):>14}"
        print("-" * len(header))
        print(row)

    print("=" * width + "\n")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Evaluating SDD models on {DEVICE}...")

    results = {
        "CV": {}, "Social-LSTM": {}, "Social-LSTM+V": {},
        "GRU-v2": {}, "Transformer": {}, "Diffusion": {},
    }

    for scene in SCENES:
        print(f"\n--- {scene} ---")

        print("  CV baseline...")
        results["CV"][scene] = eval_cv(scene)

        print("  Social-LSTM...")
        results["Social-LSTM"][scene] = eval_social_lstm(scene)

        print("  Social-LSTM+V...")
        results["Social-LSTM+V"][scene] = eval_social_lstm_v(scene)

        print("  GRU-v2...")
        results["GRU-v2"][scene] = eval_gru_v2(scene)

        print("  Transformer...")
        results["Transformer"][scene] = eval_transformer(scene)

        print("  Diffusion...")
        results["Diffusion"][scene] = eval_diffusion(scene)

    print_table(results)
