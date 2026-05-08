"""
Social-GRU v2 — Bidirectional Encoder
=======================================
Improvement over Social-GRU v1: the observation encoder is bidirectional,
processing the trajectory both forward and backward before pooling.
Bidirectional encoding gives each timestep access to the full observation
window context, not just the past — better at capturing deceleration,
turning intent, and interaction patterns.

Architecture change vs v1:
  - encoder: nn.GRU(bidirectional=True) instead of GRUCell loop
  - forward + backward hidden states are concatenated → projected to hidden_size
  - everything else identical
"""

import math
import numpy as np
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.social_lstm import (
    SocialPooling, bivariate_gaussian_nll, sample_bivariate_gaussian
)


class SocialGRUv2(nn.Module):
    """
    Social-GRU with bidirectional encoder.

    Parameters
    ----------
    obs_len         : observation window
    pred_len        : prediction horizon
    hidden_size     : GRU hidden dimension (per direction)
    embed_size      : position embedding size
    pooling_radius  : social pooling radius (metres)
    dropout         : dropout in decoder
    use_velocity    : augment encoder with (vx, vy)
    """

    def __init__(self,
                 obs_len:        int   = 8,
                 pred_len:       int   = 12,
                 hidden_size:    int   = 128,
                 embed_size:     int   = 64,
                 pooling_radius: float = 2.0,
                 dropout:        float = 0.1,
                 use_velocity:   bool  = False):
        super().__init__()
        self.obs_len      = obs_len
        self.pred_len     = pred_len
        self.hidden_size  = hidden_size
        self.use_velocity = use_velocity

        enc_input_dim = 4 if use_velocity else 2

        # ── Focal encoder: bidirectional GRU ──────────────────────────────
        self.pos_embed_enc = nn.Sequential(
            nn.Linear(enc_input_dim, embed_size), nn.ReLU()
        )
        # Outputs 2*hidden_size (forward + backward), projected back to hidden_size
        self.encoder_gru = nn.GRU(
            input_size=embed_size, hidden_size=hidden_size,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        self.bidir_proj = nn.Linear(2 * hidden_size, hidden_size)

        # ── Social pooling ─────────────────────────────────────────────────
        self.social_pool = SocialPooling(hidden_size, embed_size, pooling_radius)

        # ── Neighbour encoder: unidirectional GRUCell ─────────────────────
        # (neighbours only observed sequentially, bidirectional not necessary)
        self.pos_embed_nb = nn.Sequential(
            nn.Linear(2, embed_size), nn.ReLU()
        )
        self.nb_encoder = nn.GRUCell(embed_size, hidden_size)

        # ── Decoder: unidirectional GRUCell ───────────────────────────────
        self.pos_embed_dec = nn.Sequential(
            nn.Linear(2, embed_size), nn.ReLU()
        )
        self.decoder     = nn.GRUCell(embed_size + hidden_size, hidden_size)
        self.dropout     = nn.Dropout(p=dropout)
        self.output_head = nn.Linear(hidden_size, 5)

    def _encode(self, obs, nb_obs, nb_mask):
        """
        Returns
        -------
        h          : (N, hidden)        focal encoder output
        nb_h_final : (N, M, hidden)     neighbour encoder final states
        """
        N, T, _    = obs.shape
        device     = obs.device
        _, M, _, _ = nb_obs.shape

        # ── Bidirectional focal encoder ───────────────────────────────────
        # Build velocity if needed
        if self.use_velocity:
            vel       = torch.zeros_like(obs)
            vel[:, 1:] = obs[:, 1:] - obs[:, :-1]
            enc_input = torch.cat([obs, vel], dim=-1)   # (N, T, 4)
        else:
            enc_input = obs                              # (N, T, 2)

        emb_seq      = self.pos_embed_enc(enc_input)    # (N, T, embed)
        out, _       = self.encoder_gru(emb_seq)        # (N, T, 2*hidden)
        # Use last timestep of each direction: out[:, -1, :hidden] (fwd) + out[:, 0, hidden:] (bwd)
        h_fwd        = out[:, -1, :self.hidden_size]
        h_bwd        = out[:, 0,  self.hidden_size:]
        h            = self.bidir_proj(torch.cat([h_fwd, h_bwd], dim=-1))  # (N, hidden)

        # ── Social context from last obs timestep ─────────────────────────
        # Run neighbour GRUCells sequentially
        nb_h_flat    = torch.zeros(N * M, self.hidden_size, device=device)
        nb_mask_flat = nb_mask.reshape(N * M).float().unsqueeze(-1)

        for t in range(T):
            nb_pos_flat = nb_obs[:, :, t, :].reshape(N * M, 2)
            nb_emb_flat = self.pos_embed_nb(nb_pos_flat)
            nb_h_flat   = self.nb_encoder(nb_emb_flat, nb_h_flat)
            nb_h_flat   = nb_h_flat * nb_mask_flat

        # Apply social context to h (additive, using last frame)
        focal_last  = obs[:, -1, :]
        nb_pos_last = nb_obs[:, :, -1, :]
        nb_h_t      = nb_h_flat.reshape(N, M, self.hidden_size)
        social_ctx  = self.social_pool(focal_last, h, nb_pos_last, nb_h_t, nb_mask)
        h           = h + 0.5 * social_ctx  # fuse

        return h, nb_h_flat.reshape(N, M, self.hidden_size)

    def forward(self, obs, nb_obs, nb_mask) -> dict:
        N      = obs.shape[0]
        device = obs.device

        origin      = obs[:, -1:, :]
        obs_rel     = obs - origin
        nb_obs_safe = torch.nan_to_num(nb_obs, nan=0.0)
        nb_origin   = nb_obs_safe[:, :, -1:, :]
        nb_obs_rel  = nb_obs_safe - nb_origin

        h, nb_h_final = self._encode(obs_rel, nb_obs_rel, nb_mask)

        mus_list, sigmas_list, rhos_list = [], [], []
        cur_pos     = torch.zeros(N, 2, device=device)
        nb_pos_last = torch.zeros(N, nb_obs_safe.shape[1], 2, device=device)
        nb_h_last   = nb_h_final

        for _ in range(self.pred_len):
            social_ctx = self.social_pool(cur_pos, h, nb_pos_last, nb_h_last, nb_mask)
            emb        = self.pos_embed_dec(cur_pos)
            h          = self.decoder(torch.cat([emb, social_ctx], dim=-1), h)
            h_out      = self.dropout(h)

            raw    = self.output_head(h_out)
            mu     = raw[:, :2]
            sigma  = torch.exp(torch.clamp(raw[:, 2:4], -4, 4))
            rho    = torch.tanh(raw[:, 4:5])
            cur_pos = mu

            mus_list.append(mu); sigmas_list.append(sigma); rhos_list.append(rho)

        mus_rel = torch.stack(mus_list, dim=1)
        return {"mus": mus_rel + origin,
                "sigmas": torch.stack(sigmas_list, dim=1),
                "rhos":   torch.stack(rhos_list,   dim=1)}

    def nll_loss(self, obs, nb_obs, nb_mask, targets):
        return bivariate_gaussian_nll(self.forward(obs, nb_obs, nb_mask), targets)

    @torch.no_grad()
    def sample(self, obs, nb_obs, nb_mask, K=20) -> np.ndarray:
        preds = self.forward(obs, nb_obs, nb_mask)
        return sample_bivariate_gaussian(
            preds["mus"], preds["sigmas"], preds["rhos"], K
        ).cpu().numpy()

    def predict_samples(self, obs, nb_obs, nb_mask, K=20, device="cuda") -> np.ndarray:
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(dev).eval()
        return self.sample(
            torch.tensor(obs, dtype=torch.float32, device=dev),
            torch.tensor(np.nan_to_num(nb_obs, nan=0.0), dtype=torch.float32, device=dev),
            torch.tensor(nb_mask, dtype=torch.bool, device=dev), K=K)


if __name__ == "__main__":
    print("SocialGRUv2 smoke test...")
    model = SocialGRUv2()
    dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    obs     = torch.randn(16, 8, 2, device=dev)
    nb_obs  = torch.randn(16, 5, 8, 2, device=dev)
    nb_mask = torch.ones(16, 5, dtype=torch.bool, device=dev)
    targets = torch.randn(16, 12, 2, device=dev)
    preds = model(obs, nb_obs, nb_mask)
    print(f"  mus: {preds['mus'].shape}")
    loss = model.nll_loss(obs, nb_obs, nb_mask, targets)
    print(f"  NLL: {loss.item():.4f}")
    samps = model.sample(obs, nb_obs, nb_mask, K=20)
    print(f"  samples: {samps.shape}")
    print("PASSED.")
