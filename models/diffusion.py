"""
Diffusion Model for Pedestrian Trajectory Prediction
=====================================================
DDPM-style generative model conditioned on observed trajectories and neighbours.

Architecture
------------
  Conditioning encoder : Transformer with CLS token aggregates (obs + neighbours)
  Gaussian head        : Linear head on CLS token → bivariate Gaussian params
                         (used for forward() and NLL training branch)
  Denoiser             : Small Transformer takes (noisy_traj, time_embed, context)
                         → predicted noise (used for DDPM training branch and sampling)
  Sampling             : DDIM with 20 steps for fast inference

Training loss
-------------
  L = NLL(Gaussian_head) + 0.1 * MSE(denoiser_noise)

The Gaussian head gives a differentiable, fast forward() output compatible with
bivariate_gaussian_nll. The denoiser adds sample diversity via the sample() method.

Data Interface (from eth_ucy_analysis.py) — identical to SocialLSTM
---------------------------------------------------------------------
  obs     : (N, obs_len, 2)
  nb_obs  : (N, max_nb, obs_len, 2)  NaN = absent
  nb_mask : (N, max_nb) bool
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.social_lstm import bivariate_gaussian_nll


def sinusoidal_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    """Standard DDPM sinusoidal time embedding. t: (N,) int → (N, dim)."""
    half   = dim // 2
    freqs  = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args   = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (N, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (N, dim)


class TrajDiffusion(nn.Module):
    """
    DDPM-style diffusion model for pedestrian trajectory prediction.

    Parameters
    ----------
    obs_len      : observation window (timesteps)
    pred_len     : prediction horizon (timesteps)
    d_model      : transformer dimension
    nhead        : attention heads
    max_nb       : maximum neighbours
    T            : diffusion steps (forward process)
    ddim_steps   : DDIM inference steps (< T, speeds up sampling)
    lambda_ddpm  : weight on DDPM MSE auxiliary loss
    dropout      : dropout probability
    use_velocity : unused, kept for training script compatibility
    """

    def __init__(self,
                 obs_len:      int   = 8,
                 pred_len:     int   = 12,
                 d_model:      int   = 128,
                 nhead:        int   = 4,
                 max_nb:       int   = 5,
                 T:            int   = 100,
                 ddim_steps:   int   = 20,
                 lambda_ddpm:  float = 0.1,
                 dropout:      float = 0.1,
                 use_velocity: bool  = False):
        super().__init__()
        self.obs_len     = obs_len
        self.pred_len    = pred_len
        self.d_model     = d_model
        self.max_nb      = max_nb
        self.T           = T
        self.ddim_steps  = ddim_steps
        self.lambda_ddpm = lambda_ddpm

        # ── Shared input projection (focal + neighbours) ─────────────────────
        self.input_proj  = nn.Linear(2, d_model)
        self.obs_pos_enc = nn.Embedding(obs_len,  d_model)
        self.nb_pos_enc  = nn.Embedding(obs_len,  d_model)

        # ── CLS token (BERT-style scene aggregator) ──────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # ── Conditioning encoder (2 Pre-LN layers) ───────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.cond_encoder = nn.TransformerEncoder(enc_layer, num_layers=2,
                                                  enable_nested_tensor=False)

        # ── Gaussian head: CLS → bivariate Gaussian params for forward() ─────
        self.gaussian_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, pred_len * 5),
        )

        # ── Time embedding MLP ───────────────────────────────────────────────
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, d_model),
        )

        # ── Trajectory tokens for denoiser ───────────────────────────────────
        self.traj_proj    = nn.Linear(2, d_model)
        self.traj_pos_enc = nn.Embedding(pred_len, d_model)

        # ── Denoiser (2 Pre-LN layers) ───────────────────────────────────────
        den_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.denoiser  = nn.TransformerEncoder(den_layer, num_layers=2,
                                               enable_nested_tensor=False)
        self.noise_proj = nn.Linear(d_model, 2)

        # ── DDPM noise schedule (buffers, not parameters) ────────────────────
        betas        = torch.linspace(1e-4, 0.02, T)
        alphas       = 1.0 - betas
        acp          = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas",       betas)
        self.register_buffer("alphas",      alphas)
        self.register_buffer("acp",         acp)
        self.register_buffer("sqrt_acp",    acp.sqrt())
        self.register_buffer("sqrt_1m_acp", (1.0 - acp).sqrt())

        # Precompute DDIM step indices [T-1, T-1-stride, ..., 0]
        stride    = max(1, T // ddim_steps)
        ddim_idx  = list(range(T - 1, -1, -stride))
        if ddim_idx[-1] != 0:
            ddim_idx.append(0)
        self.register_buffer("ddim_idx", torch.tensor(ddim_idx, dtype=torch.long))

    # ------------------------------------------------------------------
    # Context encoder helper
    # ------------------------------------------------------------------

    def _encode_context(self,
                        obs:     torch.Tensor,
                        nb_obs:  torch.Tensor,
                        nb_mask: torch.Tensor):
        """
        Returns
        -------
        context : (N, d_model)  — CLS token output from conditioning encoder
        origin  : (N, 1, 2)    — last observed position (for coord normalisation)
        """
        N      = obs.shape[0]
        device = obs.device

        # Relative normalisation
        origin    = obs[:, -1:, :]
        obs_rel   = obs - origin

        nb_safe   = torch.nan_to_num(nb_obs, nan=0.0)
        nb_origin = nb_safe[:, :, -1:, :]
        nb_rel    = nb_safe - nb_origin

        # Token projection
        t_idx     = torch.arange(self.obs_len, device=device)
        f_tok     = self.input_proj(obs_rel) + self.obs_pos_enc(t_idx)  # (N, T, d)
        nb_tok    = self.input_proj(nb_rel)  + self.nb_pos_enc(t_idx)   # (N, M, T, d)
        nb_flat   = nb_tok.reshape(N, self.max_nb * self.obs_len, self.d_model)

        # CLS prepend → [cls | focal_8 | nb_40]
        cls_exp   = self.cls_token.expand(N, 1, self.d_model)
        seq       = torch.cat([cls_exp, f_tok, nb_flat], dim=1)         # (N, 49, d)

        # Padding mask: True = ignore absent neighbour slots
        nb_tok_mask = (~nb_mask).unsqueeze(-1).expand(
            N, self.max_nb, self.obs_len
        ).reshape(N, self.max_nb * self.obs_len)
        valid_prefix = torch.zeros(N, 1 + self.obs_len, dtype=torch.bool, device=device)
        src_mask    = torch.cat([valid_prefix, nb_tok_mask], dim=1)     # (N, 49)

        enc_out   = self.cond_encoder(seq, src_key_padding_mask=src_mask)
        context   = enc_out[:, 0, :]                                    # (N, d) — CLS

        return context, origin

    # ------------------------------------------------------------------
    # Denoiser helper
    # ------------------------------------------------------------------

    def _denoise(self,
                 x_t:     torch.Tensor,   # (B, pred_len, 2)  noisy trajectory
                 t:       torch.Tensor,   # (B,)              diffusion timestep
                 context: torch.Tensor    # (B, d_model)
                 ) -> torch.Tensor:       # (B, pred_len, 2)  predicted noise
        B      = x_t.shape[0]
        device = x_t.device

        t_emb  = self.time_mlp(sinusoidal_embedding(t, self.d_model))   # (B, d)

        p_idx  = torch.arange(self.pred_len, device=device)
        x_tok  = self.traj_proj(x_t) + self.traj_pos_enc(p_idx)         # (B, P, d)

        # Sequence: [time_token | context_token | traj_tokens]
        seq    = torch.cat([
            t_emb.unsqueeze(1),       # (B, 1, d)
            context.unsqueeze(1),     # (B, 1, d)
            x_tok,                    # (B, P, d)
        ], dim=1)                                                        # (B, P+2, d)

        out    = self.denoiser(seq)                                      # (B, P+2, d)
        return self.noise_proj(out[:, 2:, :])                            # (B, P, 2)

    # ------------------------------------------------------------------
    # Forward — uses Gaussian head (not denoiser)
    # ------------------------------------------------------------------

    def forward(self,
                obs:     torch.Tensor,
                nb_obs:  torch.Tensor,
                nb_mask: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        obs     : (N, obs_len, 2)
        nb_obs  : (N, max_nb, obs_len, 2)
        nb_mask : (N, max_nb) bool

        Returns
        -------
        {"mus": (N,pred_len,2), "sigmas": (N,pred_len,2), "rhos": (N,pred_len,1)}
        """
        context, origin = self._encode_context(obs, nb_obs, nb_mask)

        raw    = self.gaussian_head(context)                             # (N, pred_len*5)
        raw    = raw.reshape(obs.shape[0], self.pred_len, 5)            # (N, P, 5)

        mus_rel = raw[:, :, :2]
        log_s   = raw[:, :, 2:4]
        atanh_r = raw[:, :, 4:5]

        sigmas  = torch.exp(torch.clamp(log_s, -4, 4))
        rhos    = torch.tanh(atanh_r)

        return {
            "mus":    mus_rel + origin,
            "sigmas": sigmas,
            "rhos":   rhos,
        }

    # ------------------------------------------------------------------
    # Loss — NLL + DDPM MSE
    # ------------------------------------------------------------------

    def nll_loss(self,
                 obs:     torch.Tensor,
                 nb_obs:  torch.Tensor,
                 nb_mask: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        N      = obs.shape[0]
        device = obs.device

        # Branch 1: NLL on Gaussian head
        preds   = self.forward(obs, nb_obs, nb_mask)
        nll     = bivariate_gaussian_nll(preds, targets)

        # Branch 2: DDPM MSE (trains denoiser, reinforces encoder)
        context, origin = self._encode_context(obs, nb_obs, nb_mask)
        tgt_rel = targets - origin                                       # relative coords

        t       = torch.randint(1, self.T + 1, (N,), device=device)
        eps     = torch.randn_like(tgt_rel)

        acp_t   = self.acp[t - 1].view(N, 1, 1)
        x_t     = self.sqrt_acp[t - 1].view(N, 1, 1) * tgt_rel \
                + self.sqrt_1m_acp[t - 1].view(N, 1, 1) * eps

        eps_pred = self._denoise(x_t, t, context)
        ddpm_mse = F.mse_loss(eps_pred, eps)

        return nll + self.lambda_ddpm * ddpm_mse

    # ------------------------------------------------------------------
    # Sampling — DDIM
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self,
               obs:     torch.Tensor,
               nb_obs:  torch.Tensor,
               nb_mask: torch.Tensor,
               K: int = 20) -> np.ndarray:
        """
        Draw K diverse trajectories via DDIM (ddim_steps inference steps).

        Returns
        -------
        np.ndarray  shape (N, K, pred_len, 2)
        """
        N      = obs.shape[0]
        device = obs.device

        context, origin = self._encode_context(obs, nb_obs, nb_mask)    # (N, d), (N,1,2)

        # Broadcast context for K samples: (N*K, d)
        ctx_K   = context.unsqueeze(1).expand(N, K, self.d_model).reshape(N * K, self.d_model)

        # Start from pure Gaussian noise
        x = torch.randn(N * K, self.pred_len, 2, device=device)

        ddim_ts = self.ddim_idx.tolist()
        for i in range(len(ddim_ts) - 1):
            t_curr = ddim_ts[i]
            t_prev = ddim_ts[i + 1]

            t_batch  = torch.full((N * K,), t_curr, dtype=torch.long, device=device)
            eps_pred = self._denoise(x, t_batch, ctx_K)                 # (N*K, P, 2)

            acp_curr = self.acp[t_curr]
            x0_pred  = (x - self.sqrt_1m_acp[t_curr] * eps_pred) \
                     / (self.sqrt_acp[t_curr] + 1e-8)
            x0_pred  = x0_pred.clamp(-10.0, 10.0)

            acp_prev = self.acp[t_prev]
            x = acp_prev.sqrt() * x0_pred + (1 - acp_prev).sqrt() * eps_pred

        # Final step: return clean estimate
        t_batch  = torch.full((N * K,), ddim_ts[-1], dtype=torch.long, device=device)
        eps_pred = self._denoise(x, t_batch, ctx_K)
        acp_last = self.acp[ddim_ts[-1]]
        x0_final = (x - (1 - acp_last).sqrt() * eps_pred) / (acp_last.sqrt() + 1e-8)
        x0_final = x0_final.clamp(-10.0, 10.0)

        samples  = x0_final.reshape(N, K, self.pred_len, 2)
        samples  = samples + origin.unsqueeze(1)                        # back to absolute
        return samples.cpu().numpy()

    def predict_samples(self,
                        obs:     np.ndarray,
                        nb_obs:  np.ndarray,
                        nb_mask: np.ndarray,
                        K:       int = 20,
                        device:  str = "cuda") -> np.ndarray:
        """Numpy in → numpy out. Shape (N, K, pred_len, 2)."""
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
    print("Running TrajDiffusion smoke test ...")
    N, obs_len, pred_len, max_nb = 16, 8, 12, 5

    model  = TrajDiffusion(obs_len=obs_len, pred_len=pred_len)
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
    print(f"  NLL+DDPM loss: {loss.item():.4f}")

    samps = model.sample(obs, nb_obs, nb_mask, K=20)
    print(f"  samples: {samps.shape}")

    print("Smoke test PASSED.")
