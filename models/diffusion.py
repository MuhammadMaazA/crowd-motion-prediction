"""
Diffusion Model for Pedestrian Trajectory Prediction
=====================================================
DDPM-style generative model conditioned on observed trajectories and neighbours.

Architecture
------------
  Conditioning encoder : Transformer with CLS token — encodes (obs + neighbours)
                         Returns full sequence (49 tokens) for denoiser cross-attention
  Gaussian head        : MLP on CLS token → bivariate Gaussian params
                         Trained on detached context so encoder optimises for denoiser
  Denoiser             : TransformerDecoder — queries are (time + noisy_traj tokens),
                         memory is full encoder output via cross-attention (richer than
                         single CLS token used previously)
  Sampling             : DDIM with 50 steps, chunked to avoid OOM

Training loss
-------------
  L = NLL(Gaussian_head, detached_context) + lambda_ddpm * MSE(denoiser_noise)
  lambda_ddpm = 1.0  (equal weighting; encoder gradient comes only from DDPM branch)

Data Interface — identical to SocialLSTM
-----------------------------------------
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
    half  = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / (half - 1)
    )
    args  = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


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
    T            : diffusion steps
    ddim_steps   : DDIM inference steps
    lambda_ddpm  : weight on DDPM MSE loss (1.0 = equal to NLL)
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
                 ddim_steps:   int   = 50,
                 lambda_ddpm:  float = 0.3,
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
        # total encoder sequence length: 1 CLS + obs_len focal + max_nb*obs_len nb
        self.enc_seq_len = 1 + obs_len + max_nb * obs_len

        # ── Shared input projection ──────────────────────────────────────────
        self.input_proj  = nn.Linear(2, d_model)
        self.obs_pos_enc = nn.Embedding(obs_len,  d_model)
        self.nb_pos_enc  = nn.Embedding(obs_len,  d_model)

        # ── CLS token ────────────────────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # ── Conditioning encoder (2 Pre-LN layers) ───────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.cond_encoder = nn.TransformerEncoder(enc_layer, num_layers=2,
                                                  enable_nested_tensor=False)

        # ── Gaussian head: CLS → bivariate Gaussian (uses detached context) ──
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

        # ── Denoiser: TransformerDecoder cross-attends to full encoder output ─
        # Queries: [time_token | traj_tokens] (1 + pred_len tokens)
        # Memory:  full encoder output (enc_seq_len tokens)  ← richer than CLS only
        den_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.denoiser   = nn.TransformerDecoder(den_layer, num_layers=3)
        self.noise_proj = nn.Linear(d_model, 2)

        # ── DDPM noise schedule ──────────────────────────────────────────────
        betas  = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        acp    = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas",       betas)
        self.register_buffer("alphas",      alphas)
        self.register_buffer("acp",         acp)
        self.register_buffer("sqrt_acp",    acp.sqrt())
        self.register_buffer("sqrt_1m_acp", (1.0 - acp).sqrt())

        # DDIM step indices
        stride   = max(1, T // ddim_steps)
        ddim_idx = list(range(T - 1, -1, -stride))
        if ddim_idx[-1] != 0:
            ddim_idx.append(0)
        self.register_buffer("ddim_idx", torch.tensor(ddim_idx, dtype=torch.long))

    # ------------------------------------------------------------------
    # Context encoder — returns full sequence for denoiser cross-attention
    # ------------------------------------------------------------------

    def _encode_context(self, obs, nb_obs, nb_mask):
        """
        Returns
        -------
        enc_out  : (N, enc_seq_len, d_model)  full encoder output
        context  : (N, d_model)               CLS token (for Gaussian head)
        src_mask : (N, enc_seq_len) bool       True = ignore (absent nb slots)
        origin   : (N, 1, 2)
        """
        N      = obs.shape[0]
        device = obs.device

        origin    = obs[:, -1:, :]
        obs_rel   = obs - origin
        nb_safe   = torch.nan_to_num(nb_obs, nan=0.0)
        nb_origin = nb_safe[:, :, -1:, :]
        nb_rel    = nb_safe - nb_origin

        t_idx   = torch.arange(self.obs_len, device=device)
        f_tok   = self.input_proj(obs_rel) + self.obs_pos_enc(t_idx)      # (N, T, d)
        nb_tok  = self.input_proj(nb_rel)  + self.nb_pos_enc(t_idx)       # (N, M, T, d)
        nb_flat = nb_tok.reshape(N, self.max_nb * self.obs_len, self.d_model)

        cls_exp  = self.cls_token.expand(N, 1, self.d_model)
        seq      = torch.cat([cls_exp, f_tok, nb_flat], dim=1)            # (N, 49, d)

        nb_tok_mask  = (~nb_mask).unsqueeze(-1).expand(
            N, self.max_nb, self.obs_len
        ).reshape(N, self.max_nb * self.obs_len)
        valid_prefix = torch.zeros(N, 1 + self.obs_len, dtype=torch.bool, device=device)
        src_mask     = torch.cat([valid_prefix, nb_tok_mask], dim=1)      # (N, 49)

        enc_out  = self.cond_encoder(seq, src_key_padding_mask=src_mask)  # (N, 49, d)
        context  = enc_out[:, 0, :]                                       # (N, d)

        return enc_out, context, src_mask, origin

    # ------------------------------------------------------------------
    # Denoiser — cross-attends to full encoder output
    # ------------------------------------------------------------------

    def _denoise(self,
                 x_t:      torch.Tensor,   # (B, pred_len, 2)
                 t:        torch.Tensor,   # (B,)
                 enc_out:  torch.Tensor,   # (B, enc_seq_len, d)
                 mem_mask: torch.Tensor    # (B, enc_seq_len) bool
                 ) -> torch.Tensor:        # (B, pred_len, 2)
        device = x_t.device
        t_emb  = self.time_mlp(sinusoidal_embedding(t, self.d_model))    # (B, d)
        p_idx  = torch.arange(self.pred_len, device=device)
        x_tok  = self.traj_proj(x_t) + self.traj_pos_enc(p_idx)          # (B, P, d)

        # Queries: time token + noisy trajectory tokens
        queries = torch.cat([t_emb.unsqueeze(1), x_tok], dim=1)          # (B, P+1, d)

        # Cross-attend to full encoder output (richer than single CLS token)
        out = self.denoiser(queries, enc_out,
                            memory_key_padding_mask=mem_mask)             # (B, P+1, d)
        return self.noise_proj(out[:, 1:, :])                             # (B, P, 2)

    # ------------------------------------------------------------------
    # Forward — Gaussian head on detached CLS context
    # ------------------------------------------------------------------

    def forward(self, obs, nb_obs, nb_mask) -> dict:
        _, context, _, origin = self._encode_context(obs, nb_obs, nb_mask)

        raw = self.gaussian_head(context).reshape(
            obs.shape[0], self.pred_len, 5
        )
        return {
            "mus":    raw[:, :, :2] + origin,
            "sigmas": torch.exp(torch.clamp(raw[:, :, 2:4], -4, 4)),
            "rhos":   torch.tanh(raw[:, :, 4:5]),
        }

    # ------------------------------------------------------------------
    # Loss — NLL (Gaussian head) + DDPM MSE (denoiser)
    # ------------------------------------------------------------------

    def nll_loss(self, obs, nb_obs, nb_mask, targets) -> torch.Tensor:
        N      = obs.shape[0]
        device = obs.device

        enc_out, context, src_mask, origin = self._encode_context(obs, nb_obs, nb_mask)
        tgt_rel = targets - origin

        # Branch 1: NLL — Gaussian head (shared encoder, full gradient)
        raw  = self.gaussian_head(context).reshape(N, self.pred_len, 5)
        preds = {
            "mus":    raw[:, :, :2] + origin,
            "sigmas": torch.exp(torch.clamp(raw[:, :, 2:4], -4, 4)),
            "rhos":   torch.tanh(raw[:, :, 4:5]),
        }
        nll = bivariate_gaussian_nll(preds, targets)

        # Branch 2: DDPM MSE — denoiser with full encoder context
        t        = torch.randint(1, self.T + 1, (N,), device=device)
        eps      = torch.randn_like(tgt_rel)
        x_t      = self.sqrt_acp[t-1].view(N,1,1) * tgt_rel \
                 + self.sqrt_1m_acp[t-1].view(N,1,1) * eps
        eps_pred = self._denoise(x_t, t, enc_out, src_mask)
        ddpm_mse = F.mse_loss(eps_pred, eps)

        return nll + self.lambda_ddpm * ddpm_mse

    # ------------------------------------------------------------------
    # DDIM sampling — chunked to avoid OOM
    # ------------------------------------------------------------------

    def _sample_chunk(self, x, enc_out_K, mem_mask_K, ddim_ts, device, eta: float = 1.0):
        """
        Generalised sampler: eta=0 → deterministic DDIM, eta=1 → stochastic DDPM.
        Higher eta = more diversity in samples (better minADE@20).
        """
        NK = x.shape[0]
        for i in range(len(ddim_ts) - 1):
            tc, tp   = ddim_ts[i], ddim_ts[i + 1]
            t_b      = torch.full((NK,), tc, dtype=torch.long, device=device)
            eps_pred = self._denoise(x, t_b, enc_out_K, mem_mask_K)

            acp_c = self.acp[tc]
            acp_p = self.acp[tp]

            x0    = (x - self.sqrt_1m_acp[tc] * eps_pred) / (acp_c.sqrt() + 1e-8)
            x0    = x0.clamp(-15.0, 15.0)

            # Stochasticity controlled by eta
            # sigma^2 = eta^2 * (1-acp_p)/(1-acp_c) * (1 - acp_c/acp_p)
            sigma2    = (eta ** 2) * (1 - acp_p) / (1 - acp_c + 1e-8) * (1 - acp_c / (acp_p + 1e-8))
            sigma2    = sigma2.clamp(min=0)
            direction = ((1 - acp_p - sigma2).clamp(min=0).sqrt()) * eps_pred

            x = acp_p.sqrt() * x0 + direction
            if eta > 0:
                x = x + sigma2.sqrt() * torch.randn_like(x)

        # Final clean step (no noise)
        tc   = ddim_ts[-1]
        t_b  = torch.full((NK,), tc, dtype=torch.long, device=device)
        eps_pred = self._denoise(x, t_b, enc_out_K, mem_mask_K)
        x0   = (x - self.sqrt_1m_acp[tc] * eps_pred) / (self.sqrt_acp[tc] + 1e-8)
        return x0.clamp(-15.0, 15.0)

    @torch.no_grad()
    def sample(self, obs, nb_obs, nb_mask, K: int = 20, chunk: int = 32,
               eta: float = 1.0) -> np.ndarray:
        """Returns (N, K, pred_len, 2)."""
        N      = obs.shape[0]
        device = obs.device
        ddim_ts = self.ddim_idx.tolist()
        all_samples = []

        for start in range(0, N, chunk):
            end     = min(start + chunk, N)
            c       = end - start

            enc_c, _, mask_c, orig_c = self._encode_context(
                obs[start:end], nb_obs[start:end], nb_mask[start:end]
            )
            S = enc_c.shape[1]  # enc_seq_len

            # Expand encoder output and mask for K samples
            enc_K  = enc_c.unsqueeze(1).expand(c, K, S, self.d_model).reshape(c*K, S, self.d_model)
            mask_K = mask_c.unsqueeze(1).expand(c, K, S).reshape(c*K, S)

            x   = torch.randn(c * K, self.pred_len, 2, device=device)
            x0  = self._sample_chunk(x, enc_K, mask_K, ddim_ts, device, eta=eta)

            samps = x0.reshape(c, K, self.pred_len, 2) + orig_c.unsqueeze(1)
            all_samples.append(samps.cpu())

        return torch.cat(all_samples, dim=0).numpy()

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
    print("Running TrajDiffusion smoke test ...")
    N, obs_len, pred_len, max_nb = 16, 8, 12, 5

    model  = TrajDiffusion(obs_len=obs_len, pred_len=pred_len)
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
    print(f"  NLL+DDPM loss: {loss.item():.4f}")

    samps = model.sample(obs, nb_obs, nb_mask, K=20)
    print(f"  samples: {samps.shape}")
    print("Smoke test PASSED.")
