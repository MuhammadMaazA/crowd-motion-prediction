"""
Unified training script for SDD (Stanford Drone Dataset)
=========================================================
Leave-one-scene-out cross-validation over 8 scenes.
Supports all models: Social-LSTM, Social-LSTM+V, Transformer, Diffusion.

Usage
-----
source ../crowdnav-env/bin/activate
python train_sdd.py --model social_lstm --holdout bookstore --epochs 50
python train_sdd.py --model transformer  --holdout bookstore --epochs 50
python train_sdd.py --model diffusion    --holdout bookstore --epochs 50

SDD is much larger than ETH/UCY (~250K sequences vs ~40K), so we train for
fewer epochs. Early stopping (patience=10 eval cycles) handles convergence.
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

from sdd_analysis import (
    load_scene, extract_sequences_with_neighbours,
    ade, fde, best_of_k_ade, best_of_k_fde,
    SCENE_FILES, SCENES,
)
from models.social_lstm import SocialLSTM
from models.trajectory_transformer import TrajectoryTransformer
from models.diffusion import TrajDiffusion
from models.social_gru_v2 import SocialGRUv2

CKPT_DIR = os.path.join(WORK, "checkpoints", "sdd")


CACHE_DIR = os.path.join(WORK, "data", "sdd_cache")


class SDDDataset(Dataset):
    def __init__(self, scene_names, obs_len=8, pred_len=12, max_neighbours=5):
        os.makedirs(CACHE_DIR, exist_ok=True)
        all_obs, all_pred, all_nb_obs, all_nb_mask = [], [], [], []
        for sname in scene_names:
            cache_path = os.path.join(CACHE_DIR, f"{sname}_nb{max_neighbours}.npz")
            if os.path.exists(cache_path):
                c = np.load(cache_path)
                obs, pred, nb_obs, nb_mask = c["obs"], c["pred"], c["nb_obs"], c["nb_mask"]
                print(f"    {sname}: {len(obs)} sequences (cached)")
            else:
                data = load_scene(SCENE_FILES[sname])
                obs, pred, nb_obs, nb_mask = extract_sequences_with_neighbours(
                    data, obs_len=obs_len, pred_len=pred_len,
                    max_neighbours=max_neighbours,
                )
                np.savez_compressed(cache_path, obs=obs, pred=pred,
                                    nb_obs=nb_obs, nb_mask=nb_mask)
                print(f"    {sname}: {len(obs)} sequences (cached to disk)")
            if len(obs) == 0:
                continue
            all_obs.append(obs);       all_pred.append(pred)
            all_nb_obs.append(nb_obs); all_nb_mask.append(nb_mask)

        self.obs     = np.concatenate(all_obs,     axis=0).astype(np.float32)
        self.pred    = np.concatenate(all_pred,    axis=0).astype(np.float32)
        self.nb_obs  = np.nan_to_num(np.concatenate(all_nb_obs, axis=0), nan=0.0).astype(np.float32)
        self.nb_mask = np.concatenate(all_nb_mask, axis=0)
        print(f"  Total: {len(self.obs)} sequences")

    def __len__(self):
        return len(self.obs)

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
            all_obs.append(obs); all_pred.append(pred)
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


def build_model(args):
    if args.model == "social_lstm":
        return SocialLSTM(obs_len=8, pred_len=12, hidden_size=128,
                          embed_size=64, pooling_radius=2.0,
                          use_velocity=False)
    elif args.model == "social_lstm_v":
        return SocialLSTM(obs_len=8, pred_len=12, hidden_size=128,
                          embed_size=64, pooling_radius=2.0,
                          use_velocity=True)
    elif args.model == "gru_v2":
        return SocialGRUv2(obs_len=8, pred_len=12, hidden_size=128,
                           embed_size=64, pooling_radius=2.0)
    elif args.model == "transformer":
        return TrajectoryTransformer(obs_len=8, pred_len=12, d_model=128,
                                     nhead=4, num_enc=2, num_dec=2, dim_ff=128)
    elif args.model == "diffusion":
        return TrajDiffusion(obs_len=8, pred_len=12, d_model=128, nhead=4,
                             T=100, ddim_steps=50, lambda_ddpm=0.3)
    else:
        raise ValueError(f"Unknown model: {args.model}")


def ckpt_hparams(args):
    if args.model in ("social_lstm", "social_lstm_v", "gru_v2"):
        return {"hidden_size": 128, "embed_size": 64, "pooling_radius": 2.0,
                "use_velocity": args.model == "social_lstm_v"}
    elif args.model == "transformer":
        return {"d_model": 128, "nhead": 4, "num_enc": 2, "num_dec": 2, "dim_ff": 128}
    elif args.model == "diffusion":
        return {"d_model": 128, "nhead": 4, "T": 100, "ddim_steps": 50, "lambda_ddpm": 0.3}


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_scenes = [s for s in SCENES if s != args.holdout]
    print(f"\nModel: {args.model}  |  Holdout: {args.holdout}")
    print(f"Train: {train_scenes}")

    print("Loading training data...")
    train_ds = SDDDataset(train_scenes)
    print("Loading validation data...")
    val_ds   = SDDDataset([args.holdout])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=512, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    model = build_model(args).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}  Device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5)

    best_ade, patience_counter = float("inf"), 0
    train_losses, val_ades = [], []

    for epoch in range(1, args.epochs + 1):
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

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate(model, val_loader, device, K=args.K)
            val_ades.append(metrics["ade"])
            scheduler.step(metrics["ade"])
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}  "
                  f"ADE={metrics['ade']:.3f}  FDE={metrics['fde']:.3f}  "
                  f"minADE@{args.K}={metrics['minADE_20']:.3f}  "
                  f"minFDE@{args.K}={metrics['minFDE_20']:.3f}")
            if metrics["ade"] < best_ade:
                best_ade = metrics["ade"]; patience_counter = 0
                ckpt_path = os.path.join(CKPT_DIR, f"{args.model}_{args.holdout}.pt")
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "ade": best_ade, "holdout": args.holdout,
                            "hparams": ckpt_hparams(args)}, ckpt_path)
                print(f"  → Saved best (ADE={best_ade:.3f})")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch}"); break
        else:
            if epoch % 5 == 0:
                print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}")

    # Training curve
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(train_losses)
    axes[0].set_title(f"{args.model} SDD Loss ({args.holdout} held out)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    val_epochs = list(range(args.eval_every, args.epochs + 1, args.eval_every))
    axes[1].plot(val_epochs[:len(val_ades)], val_ades, marker="o")
    axes[1].set_title(f"Val ADE on {args.holdout}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("ADE (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(CKPT_DIR, f"{args.model}_{args.holdout}_curve.png"), dpi=150)
    plt.close(fig)
    print(f"Best ADE on {args.holdout}: {best_ade:.3f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, default="transformer",
                        choices=["social_lstm", "social_lstm_v", "gru_v2", "transformer", "diffusion"])
    parser.add_argument("--holdout",    type=str, default="bookstore", choices=SCENES)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--K",          type=int, default=20)
    parser.add_argument("--patience",   type=int, default=10)
    parser.add_argument("--device",     type=str, default="cuda")
    args = parser.parse_args()
    train(args)
