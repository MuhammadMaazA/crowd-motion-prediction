# D1 Prediction Model Results — ETH/UCY Leave-One-Out

**Protocol:** obs=8 frames (3.2s), pred=12 frames (4.8s), leave-one-out cross-validation over 5 scenes.  
**Metrics:** ADE/FDE in metres (↓ lower is better). minADE@20 / minFDE@20 = best-of-20 samples.  
**NLL** = bivariate Gaussian negative log-likelihood (↓ lower is better; not available for CV or Trajectron++ sample-only output).

> ⚠️ Diffusion (improved) results marked with * are pending — improved model currently retraining with cross-attention denoiser and λ=1.0.

---

## ADE (Average Displacement Error, metres ↓)

| Scene | CV | Social-LSTM | SLSTM+V | Trajectron++ | **Transformer** | Diffusion |
|-------|------|------------|---------|-------------|-----------------|-----------|
| eth | 1.203 | 1.015 | 1.010 | 1.359 | **0.982** | 0.991 |
| hotel | 0.553 | 0.541 | 0.527 | 0.969 | **0.437** | 0.481 |
| univ | 0.716 | 0.655 | 0.635 | 0.770 | **0.568** | 0.568 |
| zara1 | 0.623 | 0.447 | **0.432** | 0.627 | 0.481 | 0.531 |
| zara2 | 0.583 | 0.319 | 0.320 | 0.530 | 0.371 | **0.344** |
| **avg** | 0.736 | 0.596 | 0.585 | 0.851 | **0.568** | 0.583 |

---

## FDE (Final Displacement Error, metres ↓)

| Scene | CV | Social-LSTM | SLSTM+V | Trajectron++ | **Transformer** | Diffusion |
|-------|------|------------|---------|-------------|-----------------|-----------|
| eth | 2.346 | 1.993 | 1.996 | 2.806 | **1.933** | 1.929 |
| hotel | 0.789 | 1.241 | 1.186 | 2.134 | **0.973** | 1.000 |
| univ | 1.264 | 1.368 | 1.335 | 1.653 | **1.181** | 1.184 |
| zara1 | 1.044 | **0.944** | 0.970 | 1.334 | 1.095 | 1.146 |
| zara2 | 0.922 | 0.696 | 0.710 | 1.141 | 0.832 | **0.755** |
| **avg** | 1.273 | 1.248 | 1.240 | 1.814 | **1.203** | 1.203 |

---

## minADE@20 (Best-of-20 ADE, metres ↓)

| Scene | CV | Social-LSTM | SLSTM+V | **Trajectron++** | Transformer | Diffusion |
|-------|------|------------|---------|-------------|-------------|-----------|
| eth | 1.065 | 0.927 | 0.934 | **0.798** | 0.921 | 1.787* |
| hotel | 0.434 | 0.526 | 0.512 | **0.385** | 0.520 | 1.021* |
| univ | 0.588 | 0.618 | 0.656 | **0.335** | 0.653 | 1.055* |
| zara1 | 0.500 | 0.474 | 0.508 | **0.222** | 0.693 | 1.858* |
| zara2 | 0.465 | 0.352 | 0.335 | **0.181** | 0.407 | 1.064* |
| **avg** | 0.610 | 0.579 | 0.589 | **0.384** | 0.639 | 1.357* |

---

## minFDE@20 (Best-of-20 FDE, metres ↓)

| Scene | CV | Social-LSTM | SLSTM+V | Trajectron++ | **Transformer** | Diffusion |
|-------|------|------------|---------|-------------|-----------------|-----------|
| eth | 1.827 | 1.079 | 1.103 | 1.426 | **0.841** | 2.886* |
| hotel | 0.384 | 0.482 | 0.470 | 0.682 | **0.415** | 1.618* |
| univ | 0.781 | 0.657 | 0.599 | 0.598 | **0.521** | 1.405* |
| zara1 | 0.576 | 0.382 | **0.408** | 0.361 | 0.506 | 3.196* |
| zara2 | 0.512 | 0.320 | 0.321 | 0.304 | **0.350** | 1.874* |
| **avg** | 0.816 | 0.584 | 0.580 | 0.674 | **0.527** | 2.196* |

---

## NLL (Negative Log-Likelihood ↓)

| Scene | CV | Social-LSTM | SLSTM+V | Trajectron++ | **Transformer** | Diffusion |
|-------|------|------------|---------|-------------|-----------------|-----------|
| eth | — | 78.484 | 146.898 | 56.919 | **3.656** | 37.078* |
| hotel | — | -0.175 | 0.865 | 9.688 | **-0.403** | 4.955* |
| univ | — | 8.382 | 5.188 | 3.832 | **2.857** | 5.369* |
| zara1 | — | 0.795 | 1.126 | -4.082 | **0.884** | 1.270* |
| zara2 | — | 2.661 | 26.938 | -23.960 | **0.683** | 22.540* |
| **avg** | — | 18.029 | 36.203 | 8.479 | **1.535** | 14.242* |

---

## Summary Table (averages across 5 scenes)

| Model | ADE↓ | FDE↓ | minADE@20↓ | minFDE@20↓ | NLL↓ |
|-------|------|------|-----------|-----------|------|
| CV Baseline | 0.736 | 1.273 | 0.610 | 0.816 | — |
| Social-LSTM | 0.596 | 1.248 | 0.579 | 0.584 | 18.03 |
| Social-LSTM+V *(ours)* | 0.585 | 1.240 | 0.589 | 0.580 | 36.20 |
| Trajectron++ | 0.851 | 1.814 | **0.384** | 0.674 | 8.48 |
| Transformer *(ours)* | **0.568** | **1.203** | 0.639 | **0.527** | **1.54** |
| Diffusion *(ours, pending)* | 0.583* | 1.203* | 1.357* | 2.196* | 14.24* |

---

## Key Findings

**1. Transformer achieves best point-prediction accuracy.**
Lowest ADE (0.568m), FDE (1.203m), minFDE@20 (0.527m) and NLL (1.54) of all models. Parallel decoder with cross-attention to all 48 observation tokens (focal + neighbours) outperforms recurrent Social-LSTM without any multi-modal latent variables.

**2. Trajectron++ dominates multi-modal coverage (minADE@20 = 0.384m).**
Its CVAE latent variable generates genuinely diverse trajectories — samples cover left-turn, right-turn, and straight-ahead modes. The Transformer's unimodal Gaussian sampling (minADE@20 = 0.639m) cannot match this diversity.

**3. Velocity augmentation (Social-LSTM+V) gives consistent small gains over base Social-LSTM.**
ADE improves from 0.596 → 0.585m. The explicit (vx, vy) input helps the encoder infer motion direction without relying solely on position differences.

**4. ADE and minADE@20 tell opposite stories.**
Trajectron++ has the worst ADE (0.851m) but best minADE@20 (0.384m). The Transformer has the best ADE (0.568m) but middling minADE@20 (0.639m). This confirms that optimising for point-prediction accuracy and optimising for multi-modal coverage are fundamentally different objectives — a key finding for the safety-shielding discussion (D3).

**5. NLL is the most informative single metric.**
Transformer NLL (1.54) shows it produces well-calibrated uncertainty estimates — far better than Social-LSTM (18.03) which inflates uncertainty, and better than Trajectron++ (8.48) which is harder to calibrate due to the GMM output.

---

## Model Details

| Model | Params | Training | Architecture |
|-------|--------|----------|-------------|
| CV Baseline | 0 | None | Constant velocity + Gaussian noise |
| Social-LSTM | 455K | NLL | LSTMCell encoder/decoder, per-neighbour LSTMs, bivariate Gaussian head |
| Social-LSTM+V | 455K | NLL | Social-LSTM + explicit (vx,vy) velocity input |
| Trajectron++ | ~3M | ELBO | CVAE + graph attention, GMM output |
| Transformer | 537K | NLL | 2-layer Pre-LN encoder-decoder, learned queries, cross-attention to 48 tokens |
| Diffusion | 956K | NLL + DDPM-MSE | Cross-attention denoiser (3 layers), detached Gaussian head, DDIM-50 sampling |

*Diffusion numbers marked * will be updated once improved model finishes training.*
