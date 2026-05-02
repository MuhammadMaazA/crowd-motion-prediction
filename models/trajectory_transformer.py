"""
Trajectory Transformer for Pedestrian Trajectory Prediction
============================================================
Transformer encoder-decoder architecture for ETH/UCY prediction.

Architecture
------------
  Encoder : linear projection + learned positional encoding →
            2-layer Pre-LN TransformerEncoder (self-attention over obs window)
  Memory  : focal encoder output + flattened neighbour tokens (48 tokens total)
  Decoder : 12 learned query tokens → 2-layer Pre-LN TransformerDecoder
            (cross-attention to memory)
  Head    : Linear(128→5) → bivariate Gaussian params

Data Interface (from eth_ucy_analysis.py)
-----------------------------------------
  obs     : (N, obs_len, 2)               focal agent positions
  nb_obs  : (N, max_nb, obs_len, 2)       neighbour positions (NaN = absent)
  nb_mask : (N, max_nb) bool              True where neighbour slot is valid

Outputs (identical to SocialLSTM interface)
-------------------------------------------
  forward()       → {"mus": (N,T,2), "sigmas": (N,T,2), "rhos": (N,T,1)}
  sample()        → (N, K, pred_len, 2)  numpy array
  nll_loss()      → scalar tensor
  predict_samples() → numpy wrapper
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse bivariate Gaussian helpers from Social-LSTM
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.social_lstm import bivariate_gaussian_nll, sample_bivariate_gaussian


class TrajectoryTransformer(nn.Module):
    """
    Transformer-based pedestrian trajectory predictor.

    Parameters
    ----------
    obs_len         : observation window length (timesteps)
    pred_len        : prediction horizon (timesteps)
    d_model         : transformer dimension
    nhead           : number of attention heads
    num_enc         : number of encoder layers
    num_dec         : number of decoder layers
    dim_ff          : feedforward dimension inside transformer layers
    max_nb          : maximum number of neighbours
    dropout         : dropout probability
    use_velocity    : unused (kept for checkpoint compatibility with training script)
    """

    def __init__(self,
                 obs_len:   int   = 8,
                 pred_len:  int   = 12,
                 d_model:   int   = 128,
                 nhead:     int   = 4,
                 num_enc:   int   = 2,
                 num_dec:   int   = 2,
                 dim_ff:    int   = 128,
                 max_nb:    int   = 5,
                 dropout:   float = 0.1,
                 use_velocity: bool = False):
        super().__init__()
        self.obs_len  = obs_len
        self.pred_len = pred_len
        self.d_model  = d_model
        self.max_nb   = max_nb

        # ── Input projection (shared for focal and neighbour tokens) ─────────
        self.input_proj = nn.Linear(2, d_model)

        # ── Positional encodings (learned) ───────────────────────────────────
        self.obs_pos_enc   = nn.Embedding(obs_len,  d_model)
        self.nb_pos_enc    = nn.Embedding(obs_len,  d_model)
        self.query_pos_enc = nn.Embedding(pred_len, d_model)

        # ── Learned decoder query tokens ─────────────────────────────────────
        self.pred_queries = nn.Parameter(torch.zeros(1, pred_len, d_model))
        nn.init.normal_(self.pred_queries, std=0.02)

        # ── Transformer encoder (Pre-LN for stable training on small datasets)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc,
                                             enable_nested_tensor=False)

        # ── Transformer decoder ───────────────────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec)

        self.dropout = nn.Dropout(dropout)

        # ── Output head: d_model → 5 bivariate Gaussian params ───────────────
        self.output_head = nn.Linear(d_model, 5)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self,
                obs:     torch.Tensor,
                nb_obs:  torch.Tensor,
                nb_mask: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        obs     : (N, obs_len, 2)
        nb_obs  : (N, max_nb, obs_len, 2)  NaN where absent
        nb_mask : (N, max_nb) bool

        Returns
        -------
        dict with mus (N,pred_len,2), sigmas (N,pred_len,2), rhos (N,pred_len,1)
        """
        N      = obs.shape[0]
        device = obs.device

        # ── Relative coordinate normalisation ─────────────────────────────────
        origin    = obs[:, -1:, :]                           # (N, 1, 2)
        obs_rel   = obs - origin                             # (N, obs_len, 2)

        nb_safe   = torch.nan_to_num(nb_obs, nan=0.0)
        nb_origin = nb_safe[:, :, -1:, :]                   # (N, max_nb, 1, 2)
        nb_rel    = nb_safe - nb_origin                      # (N, max_nb, obs_len, 2)

        # ── Token projection ──────────────────────────────────────────────────
        t_idx     = torch.arange(self.obs_len, device=device)

        focal_tok = self.input_proj(obs_rel) + self.obs_pos_enc(t_idx)   # (N, T, d)

        # Neighbours: (N, max_nb, obs_len, d) → flatten to (N, max_nb*obs_len, d)
        nb_tok = self.input_proj(nb_rel) + self.nb_pos_enc(t_idx)        # (N, M, T, d)
        nb_flat = nb_tok.reshape(N, self.max_nb * self.obs_len, self.d_model)

        # ── Focal encoder ─────────────────────────────────────────────────────
        focal_enc = self.encoder(focal_tok)                              # (N, T, d)

        # ── Build memory: focal + neighbour tokens ────────────────────────────
        memory = torch.cat([focal_enc, nb_flat], dim=1)                  # (N, 8+M*T, d)

        # Padding mask: True = ignore.  First 8 tokens always valid.
        # Remaining max_nb*obs_len tokens: mask out absent neighbour slots.
        nb_tok_mask = (~nb_mask).unsqueeze(-1).expand(
            N, self.max_nb, self.obs_len
        ).reshape(N, self.max_nb * self.obs_len)                         # (N, M*T)

        focal_valid = torch.zeros(N, self.obs_len, dtype=torch.bool, device=device)
        mem_mask    = torch.cat([focal_valid, nb_tok_mask], dim=1)       # (N, 8+M*T)

        # ── Decoder with learned queries ──────────────────────────────────────
        q_idx   = torch.arange(self.pred_len, device=device)
        queries = self.pred_queries.expand(N, -1, -1) + self.query_pos_enc(q_idx)

        dec_out = self.decoder(
            queries, memory,
            memory_key_padding_mask=mem_mask,
        )                                                                # (N, pred_len, d)
        dec_out = self.dropout(dec_out)

        # ── Output head ───────────────────────────────────────────────────────
        raw     = self.output_head(dec_out)                              # (N, pred_len, 5)

        mus_rel = raw[:, :, :2]                                          # (N, T, 2)
        log_s   = raw[:, :, 2:4]
        atanh_r = raw[:, :, 4:5]

        sigmas  = torch.exp(torch.clamp(log_s, -4, 4))                  # positive
        rhos    = torch.tanh(atanh_r)                                    # (-1, 1)

        return {
            "mus":    mus_rel + origin,     # back to absolute coords
            "sigmas": sigmas,
            "rhos":   rhos,
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def nll_loss(self,
                 obs:     torch.Tensor,
                 nb_obs:  torch.Tensor,
                 nb_mask: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        preds = self.forward(obs, nb_obs, nb_mask)
        return bivariate_gaussian_nll(preds, targets)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self,
               obs:     torch.Tensor,
               nb_obs:  torch.Tensor,
               nb_mask: torch.Tensor,
               K: int = 20) -> np.ndarray:
        """Returns (N, K, pred_len, 2)."""
        preds = self.forward(obs, nb_obs, nb_mask)
        samps = sample_bivariate_gaussian(
            preds["mus"], preds["sigmas"], preds["rhos"], K
        )
        return samps.cpu().numpy()

    def predict_samples(self,
                        obs:     np.ndarray,
                        nb_obs:  np.ndarray,
                        nb_mask: np.ndarray,
                        K:       int = 20,
                        device:  str = "cuda") -> np.ndarray:
        """Numpy in → numpy out. Shape (N, K, pred_len, 2)."""
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(dev).eval()
        obs_t     = torch.tensor(obs,                              dtype=torch.float32, device=dev)
        nb_obs_t  = torch.tensor(np.nan_to_num(nb_obs, nan=0.0),  dtype=torch.float32, device=dev)
        nb_mask_t = torch.tensor(nb_mask,                         dtype=torch.bool,    device=dev)
        return self.sample(obs_t, nb_obs_t, nb_mask_t, K=K)


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Running TrajectoryTransformer smoke test ...")
    N, obs_len, pred_len, max_nb = 16, 8, 12, 5

    model  = TrajectoryTransformer(obs_len=obs_len, pred_len=pred_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    obs     = torch.randn(N, obs_len,  2,           device=device)
    nb_obs  = torch.randn(N, max_nb,   obs_len, 2,  device=device)
    nb_mask = torch.ones(N, max_nb,  dtype=torch.bool, device=device)
    targets = torch.randn(N, pred_len, 2,           device=device)

    preds = model(obs, nb_obs, nb_mask)
    print(f"  mus:    {preds['mus'].shape}")
    print(f"  sigmas: {preds['sigmas'].shape}")
    print(f"  rhos:   {preds['rhos'].shape}")

    loss = model.nll_loss(obs, nb_obs, nb_mask, targets)
    print(f"  NLL loss: {loss.item():.4f}")

    samps = model.sample(obs, nb_obs, nb_mask, K=20)
    print(f"  samples: {samps.shape}")

    print("Smoke test PASSED.")
