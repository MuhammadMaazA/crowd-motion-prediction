"""
Social-GRU for Pedestrian Trajectory Prediction
================================================
Drop-in replacement for Social-LSTM using GRUCell instead of LSTMCell.

Key differences from Social-LSTM:
- GRUCell has 3 gates vs LSTM's 4 → ~25% fewer recurrent parameters
- No cell state (c) — GRU only maintains hidden state (h)
- Everything else identical: social pooling, per-neighbour encoders,
  relative coord normalisation, bivariate Gaussian output

Interface identical to SocialLSTM — works with train_social_lstm.py
and evaluate_all.py without modification.
"""

import math
import numpy as np
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.social_lstm import (
    SocialPooling,
    bivariate_gaussian_nll,
    sample_bivariate_gaussian,
)


class SocialGRU(nn.Module):
    """
    Social-GRU pedestrian trajectory predictor.

    Parameters
    ----------
    obs_len         : observation window (timesteps)
    pred_len        : prediction horizon (timesteps)
    hidden_size     : GRU hidden dimension
    embed_size      : position embedding size
    pooling_radius  : social pooling radius (metres)
    dropout         : dropout probability in decoder
    use_velocity    : if True, augment encoder input with (vx, vy)
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

        # ── Encoder ────────────────────────────────────────────────────────
        enc_input_dim = 4 if use_velocity else 2
        self.pos_embed_enc = nn.Sequential(
            nn.Linear(enc_input_dim, embed_size), nn.ReLU()
        )
        self.encoder = nn.GRUCell(embed_size + hidden_size, hidden_size)

        # ── Social pooling ─────────────────────────────────────────────────
        self.social_pool = SocialPooling(hidden_size, embed_size, pooling_radius)

        # ── Neighbour encoder ─────────────────────────────────────────────
        self.pos_embed_nb = nn.Sequential(
            nn.Linear(2, embed_size), nn.ReLU()
        )
        self.nb_encoder = nn.GRUCell(embed_size, hidden_size)

        # ── Decoder ────────────────────────────────────────────────────────
        self.pos_embed_dec = nn.Sequential(
            nn.Linear(2, embed_size), nn.ReLU()
        )
        self.decoder     = nn.GRUCell(embed_size + hidden_size, hidden_size)
        self.dropout     = nn.Dropout(p=dropout)
        self.output_head = nn.Linear(hidden_size, 5)

    # ------------------------------------------------------------------

    def _init_hidden(self, N: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(N, self.hidden_size, device=device)

    def _encode(self, obs, nb_obs, nb_mask):
        """
        Returns
        -------
        h           : (N, hidden)       final focal-agent GRU state
        nb_h_final  : (N, M, hidden)    final per-neighbour GRU states
        """
        N, T, _    = obs.shape
        device     = obs.device
        h          = self._init_hidden(N, device)
        _, M, _, _ = nb_obs.shape

        nb_h_flat    = torch.zeros(N * M, self.hidden_size, device=device)
        nb_mask_flat = nb_mask.reshape(N * M).float().unsqueeze(-1)

        prev_pos = None
        for t in range(T):
            focal_pos   = obs[:, t, :]
            nb_pos_t    = nb_obs[:, :, t, :]

            # Neighbour GRU step
            nb_pos_flat = nb_pos_t.reshape(N * M, 2)
            nb_emb_flat = self.pos_embed_nb(nb_pos_flat)
            nb_h_flat   = self.nb_encoder(nb_emb_flat, nb_h_flat)
            nb_h_flat   = nb_h_flat * nb_mask_flat

            nb_h_t      = nb_h_flat.reshape(N, M, self.hidden_size)
            social_ctx  = self.social_pool(focal_pos, h, nb_pos_t, nb_h_t, nb_mask)

            # Focal encoder step
            if self.use_velocity:
                vel     = focal_pos - prev_pos if prev_pos is not None else torch.zeros_like(focal_pos)
                enc_inp = torch.cat([focal_pos, vel], dim=-1)
            else:
                enc_inp = focal_pos
            prev_pos = focal_pos

            emb     = self.pos_embed_enc(enc_inp)
            h       = self.encoder(torch.cat([emb, social_ctx], dim=-1), h)

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
        cur_pos      = torch.zeros(N, 2, device=device)
        nb_pos_last  = torch.zeros(N, nb_obs_safe.shape[1], 2, device=device)
        nb_h_last    = nb_h_final

        for _ in range(self.pred_len):
            social_ctx = self.social_pool(cur_pos, h, nb_pos_last, nb_h_last, nb_mask)
            emb        = self.pos_embed_dec(cur_pos)
            h          = self.decoder(torch.cat([emb, social_ctx], dim=-1), h)
            h_out      = self.dropout(h)

            raw     = self.output_head(h_out)
            mu      = raw[:, :2]
            sigma   = torch.exp(torch.clamp(raw[:, 2:4], -4, 4))
            rho     = torch.tanh(raw[:, 4:5])
            cur_pos = mu

            mus_list.append(mu)
            sigmas_list.append(sigma)
            rhos_list.append(rho)

        mus_rel = torch.stack(mus_list, dim=1)
        return {
            "mus":    mus_rel + origin,
            "sigmas": torch.stack(sigmas_list, dim=1),
            "rhos":   torch.stack(rhos_list,   dim=1),
        }

    def nll_loss(self, obs, nb_obs, nb_mask, targets):
        return bivariate_gaussian_nll(self.forward(obs, nb_obs, nb_mask), targets)

    @torch.no_grad()
    def sample(self, obs, nb_obs, nb_mask, K: int = 20) -> np.ndarray:
        preds = self.forward(obs, nb_obs, nb_mask)
        return sample_bivariate_gaussian(
            preds["mus"], preds["sigmas"], preds["rhos"], K
        ).cpu().numpy()

    def predict_samples(self, obs, nb_obs, nb_mask, K=20, device="cuda") -> np.ndarray:
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(dev).eval()
        obs_t     = torch.tensor(obs,                             dtype=torch.float32, device=dev)
        nb_obs_t  = torch.tensor(np.nan_to_num(nb_obs, nan=0.0), dtype=torch.float32, device=dev)
        nb_mask_t = torch.tensor(nb_mask,                        dtype=torch.bool,    device=dev)
        return self.sample(obs_t, nb_obs_t, nb_mask_t, K=K)


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Running SocialGRU smoke test ...")
    N, obs_len, pred_len, max_nb = 16, 8, 12, 5
    model  = SocialGRU(obs_len=obs_len, pred_len=pred_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    obs     = torch.randn(N, obs_len, 2,          device=device)
    nb_obs  = torch.randn(N, max_nb, obs_len, 2,  device=device)
    nb_mask = torch.ones(N, max_nb, dtype=torch.bool, device=device)
    targets = torch.randn(N, pred_len, 2,          device=device)

    preds = model(obs, nb_obs, nb_mask)
    print(f"  mus:    {preds['mus'].shape}")
    print(f"  sigmas: {preds['sigmas'].shape}")
    print(f"  rhos:   {preds['rhos'].shape}")
    loss = model.nll_loss(obs, nb_obs, nb_mask, targets)
    print(f"  NLL:    {loss.item():.4f}")
    samps = model.sample(obs, nb_obs, nb_mask, K=20)
    print(f"  samples:{samps.shape}")
    print("Smoke test PASSED.")
