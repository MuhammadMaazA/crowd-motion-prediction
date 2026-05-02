"""
Training script for TrajDiffusion on ETH/UCY
=============================================
Leave-one-out cross-validation: train on 4 scenes, evaluate on the held-out one.

Usage
-----
source ../crowdnav-env/bin/activate
python train_diffusion.py --holdout eth --epochs 200
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORK)

from eth_ucy_analysis import (
    load_scene, extract_sequences_with_neighbours,
    ade, fde, best_of_k_ade, best_of_k_fde
)
from models.diffusion import TrajDiffusion

RAW = os.path.join(WORK, "Trajectron-plus-plus/experiments/pedestrians/raw")
SCENE_FILES = {
    "eth":   [os.path.join(RAW, "eth",   "test", "biwi_eth.txt")],
    "hotel": [os.path.join(RAW, "hotel", "test", "biwi_hotel.txt")],
    "univ":  [os.path.join(RAW, "univ",  "test", "students001.txt"),
              os.path.join(RAW, "univ",  "test", "students003.txt")],
    "zara1": [os.path.join(RAW, "zara1", "test", "crowds_zara01.txt")],
    "zara2": [os.path.join(RAW, "zara2", "test", "crowds_zara02.txt")],
}


class PedestrianDataset(Dataset):
    def __init__(self, scene_names, obs_len=8, pred_len=12, max_neighbours=5):
        all_obs, all_pred, all_nb_obs, all_nb_mask = [], [], [], []
        for sname in scene_names:
            data = load_scene(SCENE_FILES[sname])
            obs, pred, nb_obs, nb_mask = extract_sequences_with_neighbours(
                data, obs_len=obs_len, pred_len=pred_len,
                max_neighbours=max_neighbours
            )
            if len(obs) == 0:
                continue
            all_obs.append(obs);     all_pred.append(pred)
            all_nb_obs.append(nb_obs); all_nb_mask.append(nb_mask)

        self.obs     = np.concatenate(all_obs,     axis=0).astype(np.float32)
        self.pred    = np.concatenate(all_pred,    axis=0).astype(np.float32)
        self.nb_obs  = np.nan_to_num(np.concatenate(all_nb_obs, axis=0), nan=0.0).astype(np.float32)
        self.nb_mask = np.concatenate(all_nb_mask, axis=0)
        print(f"  Dataset: {len(self.obs)} sequences from {scene_names}")

    def __len__(self):  return len(self.obs)
    def __getitem__(self, idx):
        return (self.obs[idx], self.pred[idx],
                self.nb_obs[idx], self.nb_mask[idx].astype(np.float32))


def collate_fn(batch):
    obs, pred, nb_obs, nb_mask = zip(*batch)
    return (torch.tensor(np.stack(obs),     dtype=torch.float32),
            torch.tensor(np.stack(pred),    dtype=torch.float32),
            torch.tensor(np.stack(nb_obs),  dtype=torch.float32),
            torch.tensor(np.stack(nb_mask), dtype=torch.bool))


def evaluate(model, val_loader, device, K=20):
    model.eval()
    all_obs, all_pred, all_nb_obs, all_nb_mask = [], [], [], []
    with torch.no_grad():
        for obs, pred, nb_obs, nb_mask in val_loader:
            all_obs.append(obs);       all_pred.append(pred)
            all_nb_obs.append(nb_obs); all_nb_mask.append(nb_mask)

    obs_t     = torch.cat(all_obs).to(device)
    pred_np   = torch.cat(all_pred).numpy()
    nb_obs_t  = torch.cat(all_nb_obs).to(device)
    nb_mask_t = torch.cat(all_nb_mask).to(device)

    with torch.no_grad():
        preds   = model(obs_t, nb_obs_t, nb_mask_t)
        mus_np  = preds["mus"].cpu().numpy()
        samples = model.sample(obs_t, nb_obs_t, nb_mask_t, K=K)

    return {
        "ade":       ade(mus_np, pred_np),
        "fde":       fde(mus_np, pred_np),
        "minADE_20": best_of_k_ade(samples, pred_np),
        "minFDE_20": best_of_k_fde(samples, pred_np),
    }


def train(holdout, epochs=200, batch_size=64, lr=1e-3, d_model=128, nhead=4,
          T=100, ddim_steps=20, lambda_ddpm=0.1, max_neighbours=5,
          eval_every=10, K_eval=20, device_str="cuda"):

    device   = torch.device(device_str if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(WORK, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    train_scenes = [s for s in SCENE_FILES if s != holdout]
    print(f"\nTraining on: {train_scenes}  |  Val on: [{holdout}]")

    train_ds = PedestrianDataset(train_scenes, max_neighbours=max_neighbours)
    val_ds   = PedestrianDataset([holdout],    max_neighbours=max_neighbours)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    model = TrajDiffusion(
        obs_len=8, pred_len=12,
        d_model=d_model, nhead=nhead,
        max_nb=max_neighbours, T=T,
        ddim_steps=ddim_steps, lambda_ddpm=lambda_ddpm,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device:     {device}  |  T={T}  ddim_steps={ddim_steps}  λ={lambda_ddpm}")

    train_losses, val_ades = [], []
    best_ade, patience_counter = float("inf"), 0
    early_stop_patience = 20

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        for obs, pred, nb_obs, nb_mask in train_loader:
            obs = obs.to(device); pred = pred.to(device)
            nb_obs = nb_obs.to(device); nb_mask = nb_mask.to(device)

            optimizer.zero_grad()
            loss = model.nll_loss(obs, nb_obs, nb_mask, pred)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item(); n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        if epoch % eval_every == 0 or epoch == epochs:
            metrics = evaluate(model, val_loader, device, K=K_eval)
            val_ades.append(metrics["ade"])
            scheduler.step(metrics["ade"])

            print(f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}  "
                  f"ADE={metrics['ade']:.3f}  FDE={metrics['fde']:.3f}  "
                  f"minADE@{K_eval}={metrics['minADE_20']:.3f}  "
                  f"minFDE@{K_eval}={metrics['minFDE_20']:.3f}")

            if metrics["ade"] < best_ade:
                best_ade = metrics["ade"]; patience_counter = 0
                ckpt_path = os.path.join(ckpt_dir, f"diffusion_{holdout}.pt")
                torch.save({
                    "epoch": epoch, "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ade": best_ade, "holdout": holdout,
                    "hparams": {"d_model": d_model, "nhead": nhead,
                                "T": T, "ddim_steps": ddim_steps,
                                "lambda_ddpm": lambda_ddpm},
                }, ckpt_path)
                print(f"  → Saved best checkpoint (ADE={best_ade:.3f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        else:
            if epoch % 5 == 0:
                print(f"Epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}")

    # Training curve
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(train_losses)
    axes[0].set_title(f"Diffusion Loss ({holdout} held out)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("NLL + λ·MSE")
    val_epochs = list(range(eval_every, epochs + 1, eval_every))
    if epochs not in val_epochs: val_epochs.append(epochs)
    axes[1].plot(val_epochs[:len(val_ades)], val_ades, marker="o")
    axes[1].set_title(f"Val ADE on {holdout}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("ADE (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(ckpt_dir, f"diffusion_{holdout}_curve.png"), dpi=150)
    plt.close(fig)
    print(f"Best ADE on {holdout}: {best_ade:.3f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout",     type=str,   default="eth", choices=list(SCENE_FILES.keys()))
    parser.add_argument("--epochs",      type=int,   default=200)
    parser.add_argument("--batch_size",  type=int,   default=64)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--d_model",     type=int,   default=128)
    parser.add_argument("--nhead",       type=int,   default=4)
    parser.add_argument("--T",           type=int,   default=100)
    parser.add_argument("--ddim_steps",  type=int,   default=20)
    parser.add_argument("--lambda_ddpm", type=float, default=0.1)
    parser.add_argument("--max_nb",      type=int,   default=5)
    parser.add_argument("--eval_every",  type=int,   default=10)
    parser.add_argument("--K",           type=int,   default=20)
    parser.add_argument("--device",      type=str,   default="cuda")
    args = parser.parse_args()

    train(holdout=args.holdout, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, d_model=args.d_model, nhead=args.nhead,
          T=args.T, ddim_steps=args.ddim_steps, lambda_ddpm=args.lambda_ddpm,
          max_neighbours=args.max_nb, eval_every=args.eval_every,
          K_eval=args.K, device_str=args.device)
