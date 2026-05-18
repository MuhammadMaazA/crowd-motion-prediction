"""
Fine-tuning: SDD pretrained → ETH/UCY
======================================
Loads an SDD pretrained checkpoint, then fine-tunes with leave-one-out
cross-validation on ETH/UCY at a reduced learning rate.

Usage
-----
source ../crowdnav-env/bin/activate

# Benchmark: fine-tune Social-LSTM+V from SDD pretrain
python models/finetune_eth.py --model social_lstm_v --holdout eth \
    --pretrain checkpoints/sdd/social_lstm_v_bookstore.pt

# Fine-tune all models for one scene
python models/finetune_eth.py --model transformer --holdout hotel \
    --pretrain checkpoints/sdd/transformer_bookstore.pt

Supported models: social_lstm, social_lstm_v, gru_v2, transformer, diffusion

Outputs
-------
  checkpoints/ft_social_lstm_v_eth.pt
  checkpoints/ft_transformer_hotel.pt
  ...  (pattern: checkpoints/ft_{model}_{holdout}.pt)
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
    ade, fde, best_of_k_ade, best_of_k_fde,
)
from models.social_lstm import SocialLSTM
from models.trajectory_transformer import TrajectoryTransformer
from models.diffusion import TrajDiffusion
from models.social_gru_v2 import SocialGRUv2

# ── Scene paths ────────────────────────────────────────────────────────────
RAW = os.path.join(WORK, "Trajectron-plus-plus/experiments/pedestrians/raw")
SCENE_FILES = {
    "eth":   [os.path.join(RAW, "eth",   "test", "biwi_eth.txt")],
    "hotel": [os.path.join(RAW, "hotel", "test", "biwi_hotel.txt")],
    "univ":  [os.path.join(RAW, "univ",  "test", "students001.txt"),
              os.path.join(RAW, "univ",  "test", "students003.txt")],
    "zara1": [os.path.join(RAW, "zara1", "test", "crowds_zara01.txt")],
    "zara2": [os.path.join(RAW, "zara2", "test", "crowds_zara02.txt")],
}

CKPT_DIR = os.path.join(WORK, "checkpoints")


# ── Dataset ────────────────────────────────────────────────────────────────

class PedestrianDataset(Dataset):
    def __init__(self, scene_names, obs_len=8, pred_len=12, max_neighbours=5):
        all_obs, all_pred, all_nb_obs, all_nb_mask = [], [], [], []
        for sname in scene_names:
            data = load_scene(SCENE_FILES[sname])
            obs, pred, nb_obs, nb_mask = extract_sequences_with_neighbours(
                data, obs_len=obs_len, pred_len=pred_len,
                max_neighbours=max_neighbours,
            )
            if len(obs) == 0:
                continue
            all_obs.append(obs);       all_pred.append(pred)
            all_nb_obs.append(nb_obs); all_nb_mask.append(nb_mask)

        self.obs     = np.concatenate(all_obs,     axis=0).astype(np.float32)
        self.pred    = np.concatenate(all_pred,    axis=0).astype(np.float32)
        self.nb_obs  = np.nan_to_num(np.concatenate(all_nb_obs, axis=0), nan=0.0).astype(np.float32)
        self.nb_mask = np.concatenate(all_nb_mask, axis=0)
        print(f"  Dataset: {len(self.obs)} sequences from {scene_names}")

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


# ── Evaluation ─────────────────────────────────────────────────────────────

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


# ── Model builder ──────────────────────────────────────────────────────────

def build_model(model_name, max_neighbours=5):
    """Construct model matching the architecture used in train_sdd.py."""
    if model_name == "social_lstm":
        return SocialLSTM(obs_len=8, pred_len=12, hidden_size=128,
                          embed_size=64, pooling_radius=2.0, use_velocity=False)
    elif model_name == "social_lstm_v":
        return SocialLSTM(obs_len=8, pred_len=12, hidden_size=128,
                          embed_size=64, pooling_radius=2.0, use_velocity=True)
    elif model_name == "gru_v2":
        return SocialGRUv2(obs_len=8, pred_len=12, hidden_size=128,
                           embed_size=64, pooling_radius=2.0)
    elif model_name == "transformer":
        return TrajectoryTransformer(obs_len=8, pred_len=12, d_model=128,
                                     nhead=4, num_enc=2, num_dec=2, dim_ff=128,
                                     max_nb=max_neighbours)
    elif model_name == "diffusion":
        return TrajDiffusion(obs_len=8, pred_len=12, d_model=128, nhead=4,
                             max_nb=max_neighbours, T=100,
                             ddim_steps=50, lambda_ddpm=0.3)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ── Fine-tune loop ─────────────────────────────────────────────────────────

def finetune(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(CKPT_DIR, exist_ok=True)

    train_scenes = [s for s in SCENE_FILES if s != args.holdout]
    print(f"\nModel:      {args.model}")
    print(f"Pretrain:   {args.pretrain}")
    print(f"Train on:   {train_scenes}  |  Val/Test on: [{args.holdout}]")
    print(f"LR:         {args.lr}  |  Epochs: {args.epochs}")

    train_ds = PedestrianDataset(train_scenes)
    val_ds   = PedestrianDataset([args.holdout])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    model = build_model(args.model).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}  Device: {device}")

    # ── Load pretrained SDD weights ──────────────────────────────────────
    pretrain_path = os.path.join(WORK, args.pretrain) if not os.path.isabs(args.pretrain) else args.pretrain
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"Pretrain checkpoint not found: {pretrain_path}")

    ckpt = torch.load(pretrain_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=True)
    print(f"Loaded pretrain weights from: {pretrain_path}")
    print(f"  (pretrained on SDD scene: {ckpt.get('holdout', '?')}, "
          f"SDD ADE: {ckpt.get('ade', float('nan')):.3f})")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    train_losses, val_ades = [], []
    best_ade, patience_counter = float("inf"), 0
    early_stop_patience = 15  # slightly more lenient for fine-tuning

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
                ckpt_path = os.path.join(CKPT_DIR, f"ft_{args.model}_{args.holdout}.pt")
                torch.save({
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "ade":          best_ade,
                    "holdout":      args.holdout,
                    "pretrain_src": args.pretrain,
                    "model_name":   args.model,
                }, ckpt_path)
                print(f"  → Saved best checkpoint (ADE={best_ade:.3f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
        else:
            if epoch % 5 == 0:
                print(f"Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.4f}")

    # ── Training curve ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(train_losses)
    axes[0].set_title(f"ft_{args.model} Loss ({args.holdout} held out)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("NLL")
    val_epochs = list(range(args.eval_every, args.epochs + 1, args.eval_every))
    if args.epochs not in val_epochs:
        val_epochs.append(args.epochs)
    axes[1].plot(val_epochs[:len(val_ades)], val_ades, marker="o")
    axes[1].set_title(f"Val ADE on {args.holdout}")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("ADE (m)")
    fig.tight_layout()
    curve_path = os.path.join(CKPT_DIR, f"ft_{args.model}_{args.holdout}_curve.png")
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f"\nBest ADE on {args.holdout}: {best_ade:.3f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune SDD-pretrained model on ETH/UCY")
    parser.add_argument("--model",      type=str, required=True,
                        choices=["social_lstm", "social_lstm_v", "gru_v2",
                                 "transformer", "diffusion"],
                        help="Model architecture")
    parser.add_argument("--holdout",    type=str, required=True,
                        choices=list(SCENE_FILES.keys()),
                        help="ETH/UCY scene to hold out for evaluation")
    parser.add_argument("--pretrain",   type=str, required=True,
                        help="Path to SDD pretrained checkpoint "
                             "(absolute or relative to project root)")
    parser.add_argument("--epochs",     type=int,   default=50,
                        help="Max fine-tuning epochs (default: 50)")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-4,
                        help="Fine-tuning LR (default: 1e-4, 10× lower than scratch)")
    parser.add_argument("--eval_every", type=int,   default=5)
    parser.add_argument("--K",          type=int,   default=20)
    parser.add_argument("--device",     type=str,   default="cuda")
    args = parser.parse_args()
    finetune(args)
