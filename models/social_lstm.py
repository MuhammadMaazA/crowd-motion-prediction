"""
Social-LSTM for Pedestrian Trajectory Prediction
=================================================
Clean PyTorch 2.x implementation of:
    Alahi et al., "Social LSTM: Human Trajectory Prediction in Crowded Spaces",
    CVPR 2016.

Architecture
------------
  Encoder: position embedding → LSTM (hidden=128)
  Social pooling: aggregate nearest-neighbour hidden states (≤ pooling_radius m)
  Decoder: LSTM → 5 output params (µx, µy, log_σx, log_σy, atanh_ρ)
         one set of params per prediction timestep

Data Interface (from eth_ucy_analysis.py)
-----------------------------------------
  obs    : (N, obs_len, 2)               focal agent positions
  nb_obs : (N, max_nb, obs_len, 2)       neighbour positions (NaN = absent)
  nb_mask: (N, max_nb)  bool             True where neighbour slot is valid

Outputs
-------
  forward()         → dict with mus, sigmas, rhos per timestep
  sample()          → (N, K, pred_len, 2)  numpy array
  nll_loss()        → scalar tensor (bivariate Gaussian NLL)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


# ─────────────────────────────────────────────────────────────────────────────
# Bivariate Gaussian helpers
# ─────────────────────────────────────────────────────────────────────────────

def bivariate_gaussian_nll(pred_params: dict,
                            targets: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood of a bivariate Gaussian.

    Parameters
    ----------
    pred_params : dict with:
        mus    : (N, T, 2)
        sigmas : (N, T, 2)   positive standard deviations
        rhos   : (N, T, 1)   correlation in (-1, 1)
    targets : (N, T, 2)

    Returns
    -------
    Scalar tensor — mean NLL over (N, T).
    """
    mu_x   = pred_params["mus"][..., 0]       # (N, T)
    mu_y   = pred_params["mus"][..., 1]
    sig_x  = pred_params["sigmas"][..., 0]    # positive
    sig_y  = pred_params["sigmas"][..., 1]
    rho    = pred_params["rhos"][..., 0]      # in (-1, 1)

    tx = targets[..., 0]
    ty = targets[..., 1]

    zx   = (tx - mu_x) / (sig_x + 1e-6)
    zy   = (ty - mu_y) / (sig_y + 1e-6)
    rho2 = rho ** 2

    z = (zx**2 + zy**2 - 2 * rho * zx * zy) / (1 - rho2 + 1e-6)
    log_norm = (torch.log(sig_x + 1e-6)
                + torch.log(sig_y + 1e-6)
                + 0.5 * torch.log(1 - rho2 + 1e-6)
                + math.log(2 * math.pi))

    nll = 0.5 * z + log_norm
    return nll.mean()


def sample_bivariate_gaussian(mus: torch.Tensor,
                               sigmas: torch.Tensor,
                               rhos: torch.Tensor,
                               K: int) -> torch.Tensor:
    """
    Draw K samples from a bivariate Gaussian at each (N, T) position.

    Returns
    -------
    (N, K, T, 2)
    """
    N, T, _ = mus.shape
    device   = mus.device

    # Build (N, T, 2, 2) covariance matrices
    sig_x  = sigmas[..., 0:1]        # (N, T, 1)
    sig_y  = sigmas[..., 1:2]
    rho    = rhos[..., 0:1]

    cov_xx = sig_x ** 2
    cov_yy = sig_y ** 2
    cov_xy = rho * sig_x * sig_y

    cov = torch.stack([
        torch.stack([cov_xx, cov_xy], dim=-1),
        torch.stack([cov_xy, cov_yy], dim=-1),
    ], dim=-2).squeeze(-3)          # (N, T, 2, 2)

    # Add small diagonal jitter for numerical stability
    jitter = 1e-5 * torch.eye(2, device=device).unsqueeze(0).unsqueeze(0)
    cov    = cov + jitter

    # Cholesky decomposition then sample
    L = torch.linalg.cholesky(cov)  # (N, T, 2, 2)

    eps   = torch.randn(N, K, T, 2, device=device)         # (N, K, T, 2)
    L_exp = L.unsqueeze(1).expand(N, K, T, 2, 2)           # (N, K, T, 2, 2)
    eps_  = eps.unsqueeze(-1)                               # (N, K, T, 2, 1)
    samps = torch.matmul(L_exp, eps_).squeeze(-1)           # (N, K, T, 2)
    samps = samps + mus.unsqueeze(1)                        # (N, K, T, 2)
    return samps


# ─────────────────────────────────────────────────────────────────────────────
# Social Pooling
# ─────────────────────────────────────────────────────────────────────────────

class SocialPooling(nn.Module):
    """
    Max-pool hidden states of neighbours within a fixed spatial radius.

    Parameters
    ----------
    hidden_size     : LSTM hidden dimension
    embed_size      : spatial embedding dimension (for relative position)
    pooling_radius  : only aggregate neighbours within this distance (metres)
    """

    def __init__(self, hidden_size: int = 128,
                 embed_size: int = 64,
                 pooling_radius: float = 2.0):
        super().__init__()
        self.hidden_size    = hidden_size
        self.pooling_radius = pooling_radius

        # Project (relative_x, relative_y) → embed_size
        self.pos_embed = nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )
        # Project (pooled_hidden | pos_embed) → hidden_size
        self.social_embed = nn.Sequential(
            nn.Linear(hidden_size + embed_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self,
                focal_pos: torch.Tensor,       # (N, 2)
                focal_h:   torch.Tensor,       # (N, hidden)
                nb_pos:    torch.Tensor,       # (N, M, 2)  NaN = absent
                nb_h:      torch.Tensor,       # (N, M, hidden)
                nb_mask:   torch.Tensor        # (N, M) bool
                ) -> torch.Tensor:             # (N, hidden)
        """Return a social-context vector for each focal agent."""
        N, M, _ = nb_pos.shape
        device   = focal_pos.device

        # Relative position of each neighbour to focal agent
        rel_pos  = nb_pos - focal_pos.unsqueeze(1)        # (N, M, 2)
        dist     = rel_pos.norm(dim=-1)                   # (N, M)

        # Mask: valid neighbour AND within pooling radius
        radius_mask = (dist <= self.pooling_radius) & nb_mask  # (N, M)

        # Replace NaN positions with 0 for embedding (they'll be masked out)
        rel_pos_safe = torch.nan_to_num(rel_pos, nan=0.0)

        pos_feat  = self.pos_embed(rel_pos_safe)          # (N, M, embed)

        # Concatenate neighbour hidden state with position feature
        combined = torch.cat([nb_h, pos_feat], dim=-1)   # (N, M, hidden+embed)
        pooled_h = self.social_embed(combined)            # (N, M, hidden)

        # Zero out invalid slots then max-pool
        pool_mask = radius_mask.unsqueeze(-1).float()     # (N, M, 1)
        pooled_h  = pooled_h * pool_mask                  # zero invalid

        # If no valid neighbour, return zeros
        any_valid = radius_mask.any(dim=1, keepdim=True)  # (N, 1)
        social    = pooled_h.max(dim=1).values            # (N, hidden)
        social    = social * any_valid.float()

        return social                                      # (N, hidden)


# ─────────────────────────────────────────────────────────────────────────────
# Social-LSTM
# ─────────────────────────────────────────────────────────────────────────────

class SocialLSTM(nn.Module):
    """
    Social-LSTM pedestrian trajectory predictor.

    Parameters
    ----------
    obs_len         : observation window (timesteps)
    pred_len        : prediction horizon (timesteps)
    hidden_size     : LSTM hidden dimension
    embed_size      : position embedding size
    pooling_radius  : social pooling radius (metres)
    dropout         : dropout probability in decoder
    """

    def __init__(self,
                 obs_len: int         = 8,
                 pred_len: int        = 12,
                 hidden_size: int     = 128,
                 embed_size: int      = 64,
                 pooling_radius: float = 2.0,
                 dropout: float       = 0.1):
        super().__init__()
        self.obs_len    = obs_len
        self.pred_len   = pred_len
        self.hidden_size = hidden_size

        # ── Encoder components ──────────────────────────────────────────────
        self.pos_embed_enc = nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )
        self.encoder = nn.LSTMCell(embed_size, hidden_size)

        # ── Social pooling ───────────────────────────────────────────────────
        self.social_pool = SocialPooling(hidden_size, embed_size, pooling_radius)

        # ── Decoder components ───────────────────────────────────────────────
        self.pos_embed_dec = nn.Sequential(
            nn.Linear(2, embed_size),
            nn.ReLU(),
        )
        # Decoder input: position embed + social context
        self.decoder = nn.LSTMCell(embed_size + hidden_size, hidden_size)
        self.dropout  = nn.Dropout(p=dropout)

        # Output head: 5 parameters per step
        #   µx, µy, log_σx, log_σy, atanh(ρ)
        self.output_head = nn.Linear(hidden_size, 5)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_hidden(self, N: int, device: torch.device):
        h = torch.zeros(N, self.hidden_size, device=device)
        c = torch.zeros(N, self.hidden_size, device=device)
        return h, c

    def _encode(self, obs, nb_obs, nb_mask):
        """
        Run encoder LSTM over observation window with social pooling.

        Returns
        -------
        h, c : (N, hidden)  final encoder hidden/cell states
        """
        N, T, _    = obs.shape
        device     = obs.device
        h, c       = self._init_hidden(N, device)
        _, M, _, _ = nb_obs.shape

        for t in range(T):
            # Current positions
            focal_pos = obs[:, t, :]           # (N, 2)
            nb_pos_t  = nb_obs[:, :, t, :]    # (N, M, 2)

            # Social context using previous hidden states of neighbours
            # (we use a single shared LSTM, so we reuse h for neighbours too)
            # For simplicity: tile focal h as proxy neighbour hidden states
            # (a proper multi-agent implementation would maintain separate LSTMs)
            nb_h_t = h.unsqueeze(1).expand(N, M, self.hidden_size)

            social_ctx = self.social_pool(focal_pos, h, nb_pos_t, nb_h_t, nb_mask)

            # Encoder step
            emb      = self.pos_embed_enc(focal_pos)          # (N, embed)
            h, c     = self.encoder(emb, (h, c))

            # Fuse social context: simple additive gate
            h = h + 0.3 * social_ctx

        return h, c

    # ------------------------------------------------------------------
    # Forward pass (returns distribution parameters for decoder rollout)
    # ------------------------------------------------------------------

    def forward(self,
                obs: torch.Tensor,
                nb_obs: torch.Tensor,
                nb_mask: torch.Tensor) -> dict:
        """
        Run full encode → decode, returning bivariate Gaussian parameters.

        Parameters
        ----------
        obs     : (N, obs_len, 2)
        nb_obs  : (N, max_nb, obs_len, 2)  NaN where absent
        nb_mask : (N, max_nb) bool

        Returns
        -------
        dict:
          mus    : (N, pred_len, 2)   predicted means
          sigmas : (N, pred_len, 2)   predicted std devs (> 0)
          rhos   : (N, pred_len, 1)   predicted correlations (in (-1,1))
        """
        N = obs.shape[0]
        device = obs.device

        # ── Normalise to relative coordinates ───────────────────────────────
        # Subtract the last observed position so all coordinates are relative
        # to where the agent is right now.  This makes the model scene-agnostic
        # (no dependence on absolute coordinate origin).
        origin      = obs[:, -1:, :]                       # (N, 1, 2)
        obs_rel     = obs - origin                         # (N, obs_len, 2)

        nb_obs_safe = torch.nan_to_num(nb_obs, nan=0.0)
        # Compute neighbour origins (last valid position per neighbour)
        nb_origin   = nb_obs_safe[:, :, -1:, :]            # (N, M, 1, 2)
        nb_obs_rel  = nb_obs_safe - nb_origin              # (N, M, obs_len, 2)
        # Keep NaN slots zeroed (they were already zeroed by nan_to_num)

        h, c = self._encode(obs_rel, nb_obs_rel, nb_mask)

        # Decoder rollout (all in relative coords; add origin back at the end)
        mus_list    = []
        sigmas_list = []
        rhos_list   = []

        # Start token: last observed position in relative coords = (0, 0)
        cur_pos = torch.zeros(N, 2, device=device)

        # Neighbour positions in decoder: last observed = their relative origin = 0
        nb_pos_last  = torch.zeros(N, nb_obs_safe.shape[1], 2, device=device)
        M            = nb_obs_safe.shape[1]
        nb_h_last    = h.unsqueeze(1).expand(N, M, self.hidden_size)
        nb_mask_last = nb_mask

        for _ in range(self.pred_len):
            social_ctx = self.social_pool(
                cur_pos, h, nb_pos_last, nb_h_last, nb_mask_last
            )

            emb   = self.pos_embed_dec(cur_pos)            # (N, embed)
            inp   = torch.cat([emb, social_ctx], dim=-1)   # (N, embed+hidden)
            h, c  = self.decoder(inp, (h, c))
            h_out = self.dropout(h)

            raw = self.output_head(h_out)                  # (N, 5)

            mu    = raw[:, :2]                             # (N, 2)
            log_s = raw[:, 2:4]
            atanh_r = raw[:, 4:5]

            sigma = torch.exp(torch.clamp(log_s, -4, 4))  # positive
            rho   = torch.tanh(atanh_r)                   # in (-1, 1)

            # Update current position for next step (teacher-force: use µ)
            cur_pos = mu

            mus_list.append(mu)
            sigmas_list.append(sigma)
            rhos_list.append(rho)

        # Convert back to absolute coordinates
        mus_rel = torch.stack(mus_list, dim=1)             # (N, T, 2)
        return {
            "mus":    mus_rel + origin,                    # (N, T, 2) absolute
            "sigmas": torch.stack(sigmas_list, dim=1),     # (N, T, 2)
            "rhos":   torch.stack(rhos_list,   dim=1),     # (N, T, 1)
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def nll_loss(self,
                 obs: torch.Tensor,
                 nb_obs: torch.Tensor,
                 nb_mask: torch.Tensor,
                 targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs, nb_obs, nb_mask : see forward()
        targets : (N, pred_len, 2)  ground-truth future positions

        Returns
        -------
        Scalar tensor.
        """
        preds = self.forward(obs, nb_obs, nb_mask)
        return bivariate_gaussian_nll(preds, targets)

    # ------------------------------------------------------------------
    # Sampling (inference)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self,
               obs: torch.Tensor,
               nb_obs: torch.Tensor,
               nb_mask: torch.Tensor,
               K: int = 20) -> np.ndarray:
        """
        Draw K trajectory samples.

        Returns
        -------
        np.ndarray  shape (N, K, pred_len, 2)
        """
        preds = self.forward(obs, nb_obs, nb_mask)
        samps = sample_bivariate_gaussian(
            preds["mus"], preds["sigmas"], preds["rhos"], K
        )
        return samps.cpu().numpy()

    def predict_samples(self,
                        obs: np.ndarray,
                        nb_obs: np.ndarray,
                        nb_mask: np.ndarray,
                        K: int = 20,
                        device: str = "cuda") -> np.ndarray:
        """
        Convenience wrapper: numpy in → numpy out.

        Parameters
        ----------
        obs     : (N, obs_len, 2)
        nb_obs  : (N, max_nb, obs_len, 2)
        nb_mask : (N, max_nb) bool
        K       : number of samples

        Returns
        -------
        np.ndarray  (N, K, pred_len, 2)
        """
        dev = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(dev)
        self.eval()

        obs_t     = torch.tensor(obs,     dtype=torch.float32, device=dev)
        nb_obs_t  = torch.tensor(np.nan_to_num(nb_obs, nan=0.0),
                                 dtype=torch.float32, device=dev)
        nb_mask_t = torch.tensor(nb_mask, dtype=torch.bool,    device=dev)

        return self.sample(obs_t, nb_obs_t, nb_mask_t, K=K)


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Running Social-LSTM smoke test ...")
    N, obs_len, pred_len = 16, 8, 12
    max_nb = 5

    model = SocialLSTM(obs_len=obs_len, pred_len=pred_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    obs     = torch.randn(N, obs_len,  2,      device=device)
    nb_obs  = torch.randn(N, max_nb,   obs_len, 2, device=device)
    nb_mask = torch.ones(N, max_nb, dtype=torch.bool, device=device)
    targets = torch.randn(N, pred_len, 2, device=device)

    # Forward pass
    preds = model(obs, nb_obs, nb_mask)
    print(f"  mus:    {preds['mus'].shape}")     # (16, 12, 2)
    print(f"  sigmas: {preds['sigmas'].shape}")   # (16, 12, 2)
    print(f"  rhos:   {preds['rhos'].shape}")     # (16, 12, 1)

    # Loss
    loss = model.nll_loss(obs, nb_obs, nb_mask, targets)
    print(f"  NLL loss: {loss.item():.4f}")

    # Sampling
    samps = model.sample(obs, nb_obs, nb_mask, K=20)
    print(f"  samples: {samps.shape}")            # (16, 20, 12, 2)

    print("Smoke test PASSED.")
