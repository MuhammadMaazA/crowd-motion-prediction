"""
Constant Velocity Baseline Predictor
=====================================
Simplest possible multi-step predictor: extrapolate the last observed velocity
forward, optionally adding isotropic Gaussian noise to produce k diverse samples.

Input:   obs  (N, obs_len, 2)  absolute (x, y) positions in metres
Output:  samples (N, K, pred_len, 2)

No training required.  Noise std is the only "hyperparameter" and should be
calibrated on a validation set; 0.30 m works well for ETH/UCY as a first pass.
"""

import numpy as np
import sys
import os

# Allow direct import of eth_ucy_analysis from the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class ConstantVelocityPredictor:
    """
    Constant-velocity predictor with optional additive Gaussian noise.

    Parameters
    ----------
    noise_std : float
        Standard deviation of position noise added to each sample (metres).
        Set to 0.0 for a single deterministic prediction.
    """

    def __init__(self, noise_std: float = 0.30):
        self.noise_std = noise_std

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict_samples(self,
                        obs: np.ndarray,
                        K: int = 20,
                        pred_len: int = 12) -> np.ndarray:
        """
        Parameters
        ----------
        obs      : (N, obs_len, 2)  observed trajectory (absolute positions)
        K        : number of diverse samples to produce
        pred_len : prediction horizon in frames

        Returns
        -------
        np.ndarray  shape (N, K, pred_len, 2)
        """
        obs = np.asarray(obs, dtype=np.float32)
        N = len(obs)

        # Velocity estimate = last displacement
        vel = obs[:, -1] - obs[:, -2]               # (N, 2)

        # Deterministic extrapolation
        steps = np.arange(1, pred_len + 1, dtype=np.float32)  # (pred_len,)
        mean = obs[:, -1:] + vel[:, None] * steps[None, :, None]  # (N, pred_len, 2)

        # Add noise to produce K samples  (broadcast → (N, K, pred_len, 2))
        noise = np.random.randn(N, K, pred_len, 2).astype(np.float32) * self.noise_std
        return mean[:, None] + noise

    def predict_distribution(self, obs: np.ndarray, pred_len: int = 12) -> dict:
        """
        Return the (degenerate) distribution parameters.
        For CV the 'distribution' is just a point estimate with fixed isotropic
        Gaussian noise added; we return the mean and the noise_std.

        Returns
        -------
        dict with keys:
          "mus"    : (N, pred_len, 2)  mean trajectory
          "sigmas" : (N, pred_len, 2)  uniform std (self.noise_std everywhere)
        """
        obs = np.asarray(obs, dtype=np.float32)
        vel = obs[:, -1] - obs[:, -2]
        steps = np.arange(1, pred_len + 1, dtype=np.float32)
        mus = obs[:, -1:] + vel[:, None] * steps[None, :, None]
        N, T, _ = mus.shape
        sigmas = np.full_like(mus, self.noise_std)
        return {"mus": mus, "sigmas": sigmas}


# ------------------------------------------------------------------
# Quick smoke-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from eth_ucy_analysis import load_scene, extract_sequences, ade, fde, best_of_k_ade

    DATA_ROOT = os.path.join(os.path.dirname(__file__), "..",
                             "Trajectron-plus-plus/experiments/pedestrians/raw")

    scene_files = {
        "eth":   [os.path.join(DATA_ROOT, "eth",   "test", "biwi_eth.txt")],
        "hotel": [os.path.join(DATA_ROOT, "hotel", "test", "biwi_hotel.txt")],
        "zara1": [os.path.join(DATA_ROOT, "zara1", "test", "crowds_zara01.txt")],
    }

    model = ConstantVelocityPredictor(noise_std=0.30)

    print(f"\n{'Scene':<8} {'ADE':>8} {'FDE':>8} {'minADE@20':>11}")
    print("-" * 40)
    for scene_name, paths in scene_files.items():
        data = load_scene(paths)
        obs, pred = extract_sequences(data)
        if len(obs) == 0:
            continue

        samples = model.predict_samples(obs, K=20)          # (N, 20, 12, 2)
        mean_pred = samples[:, 0]                           # deterministic mode

        scene_ade  = ade(mean_pred, pred)
        scene_fde  = fde(mean_pred, pred)
        scene_bade = best_of_k_ade(samples, pred)

        print(f"{scene_name:<8} {scene_ade:>8.3f} {scene_fde:>8.3f} {scene_bade:>11.3f}")

    print("\ncv_baseline.py OK")
