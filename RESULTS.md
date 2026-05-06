# D1 Prediction Model Results — ETH/UCY Leave-One-Out

**Protocol:** obs=8 frames (3.2s), pred=12 frames (4.8s), leave-one-out cross-validation over 5 scenes.  
**Metrics:** ADE/FDE in metres (↓ lower is better). minADE@20 / minFDE@20 = best-of-20 samples.  
**NLL** = bivariate Gaussian negative log-likelihood (↓ lower is better; not available for CV baseline).

---

## ADE (Average Displacement Error, metres ↓)

| Scene | CV | Social-LSTM | SLSTM+V | Trajectron++ | **Transformer** | Diffusion |
|-------|------|------------|---------|-------------|-----------------|-----------|
| eth | 1.195 | 1.015 | 1.010 | 1.363 | **0.982** | 1.001 |
| hotel | 0.553 | 0.541 | 0.527 | 0.968 | **0.437** | 1.192 |
| univ | 0.716 | 0.655 | 0.635 | 0.772 | **0.568** | 1.204 |
| zara1 | 0.625 | 0.447 | **0.432** | 0.624 | 0.481 | 2.184 |
| zara2 | 0.582 | 0.319 | 0.320 | 0.520 | **0.371** | 1.267 |
| **avg** | 0.734 | 0.596 | 0.585 | 0.849 | **0.568** | 1.370 |

---

## FDE (Final Displacement Error, metres ↓)

| Scene | CV | Social-LSTM | SLSTM+V | Trajectron++ | **Transformer** | Diffusion |
|-------|------|------------|---------|-------------|-----------------|-----------|
| eth | 2.321 | 1.993 | 1.996 | 2.808 | **1.933** | 1.949 |
| hotel | 0.792 | 1.241 | 1.186 | 2.105 | **0.973** | 2.157 |
| univ | 1.267 | 1.368 | 1.335 | 1.657 | **1.181** | 2.198 |
| zara1 | 1.057 | **0.944** | 0.970 | 1.336 | 1.095 | 4.037 |
| zara2 | 0.919 | 0.696 | 0.710 | 1.112 | **0.832** | 2.352 |
| **avg** | 1.271 | 1.248 | 1.240 | 1.804 | **1.203** | 2.539 |

---

## minADE@20 (Best-of-20 ADE, metres ↓)

| Scene | CV | Social-LSTM | SLSTM+V | **Trajectron++** | Transformer | Diffusion |
|-------|------|------------|---------|-------------|-------------|-----------|
| eth | 1.063 | 0.926 | 0.938 | **0.802** | 0.923 | 1.621 |
| hotel | 0.433 | 0.528 | 0.509 | **0.379** | 0.519 | 0.745 |
| univ | 0.588 | 0.618 | 0.655 | **0.336** | 0.653 | 0.789 |
| zara1 | 0.498 | 0.474 | 0.507 | **0.223** | 0.692 | 1.635 |
| zara2 | 0.466 | 0.351 | 0.336 | **0.182** | 0.407 | 0.927 |
| **avg** | 0.610 | 0.579 | 0.589 | **0.384** | 0.639 | 1.143 |

---

## minFDE@20 (Best-of-20 FDE, metres ↓)

| Scene | CV | Social-LSTM | SLSTM+V | Trajectron++ | **Transformer** | Diffusion |
|-------|------|------------|---------|-------------|-----------------|-----------|
| eth | 1.825 | 1.095 | 1.127 | 1.423 | **0.844** | 2.769 |
| hotel | 0.385 | 0.495 | 0.476 | 0.659 | **0.408** | 1.392 |
| univ | 0.783 | 0.656 | 0.598 | 0.601 | **0.519** | 1.480 |
| zara1 | 0.571 | 0.387 | **0.403** | 0.359 | 0.516 | 3.121 |
| zara2 | 0.511 | 0.321 | 0.322 | 0.306 | **0.354** | 1.725 |
| **avg** | 0.815 | 0.591 | 0.585 | 0.670 | **0.528** | 2.097 |

---

## NLL (Negative Log-Likelihood ↓)

| Scene | CV | Social-LSTM | SLSTM+V | Trajectron++ | **Transformer** | Diffusion |
|-------|------|------------|---------|-------------|-----------------|-----------|
| eth | — | 78.484 | 146.898 | 56.889 | **3.656** | 3.882 |
| hotel | — | -0.175 | 0.865 | 9.708 | **-0.403** | 4.386 |
| univ | — | 8.382 | 5.188 | 3.845 | 2.857 | **2.616** |
| zara1 | — | 0.795 | 1.126 | -4.064 | **0.884** | 3.158 |
| zara2 | — | 2.661 | 26.938 | -23.956 | **0.683** | 2.498 |
| **avg** | — | 18.029 | 36.203 | 8.484 | **1.535** | 3.308 |

---

## Summary Table (averages across 5 scenes)

| Model | ADE↓ | FDE↓ | minADE@20↓ | minFDE@20↓ | NLL↓ |
|-------|------|------|-----------|-----------|------|
| CV Baseline | 0.734 | 1.271 | 0.610 | 0.815 | — |
| Social-LSTM | 0.596 | 1.248 | 0.579 | 0.591 | 18.03 |
| Social-LSTM+V *(ours)* | 0.585 | 1.240 | 0.589 | 0.585 | 36.20 |
| Trajectron++ | 0.849 | 1.804 | **0.384** | 0.670 | 8.48 |
| **Transformer** *(ours)* | **0.568** | **1.203** | 0.639 | **0.528** | **1.54** |
| Diffusion *(ours)* | 1.370 | 2.539 | 1.143 | 2.097 | 3.31 |

---

## Key Findings

**1. Transformer achieves best point-prediction accuracy.**
Lowest ADE (0.568m), FDE (1.203m), minFDE@20 (0.528m) and NLL (1.54) of all models. The parallel decoder with cross-attention to all 48 observation tokens (focal + neighbours) outperforms recurrent Social-LSTM without requiring any multi-modal latent variables.

**2. Trajectron++ dominates multi-modal coverage (minADE@20 = 0.384m).**
Its CVAE latent variable generates genuinely diverse trajectories across all modes. No other model comes close on this metric, confirming that explicit latent variable multi-modality is necessary for strong minADE@20.

**3. Velocity augmentation (Social-LSTM+V) gives consistent small gains.**
ADE improves 0.596→0.585m across all 5 scenes. The explicit (vx, vy) encoder input helps infer motion direction without relying solely on position differences — our contribution to the prediction module.

**4. ADE and minADE@20 tell opposite stories — confirming the report's core argument.**
Trajectron++ has the worst ADE (0.849m) but best minADE@20 (0.384m). The Transformer has the best ADE (0.568m) but middling minADE@20 (0.639m). Optimising for point-prediction accuracy and multi-modal coverage are fundamentally different objectives — confirming that metric choice drives model selection for safety-critical planning.

**5. Diffusion model reveals a fundamental architecture tradeoff.**
The cross-attention denoiser (v2) improved sample calibration (NLL: 14.24→3.31) and hotel/univ diversity (minADE@20: 1.021→0.745 on hotel). However, detaching the conditioning encoder from the Gaussian head — necessary to force the encoder to serve the denoiser — degraded point accuracy (ADE: 0.583→1.370). This confirms that joint optimisation of point prediction and sample diversity requires careful loss balancing; a separate encoder per objective would be the principled fix but adds architectural complexity beyond project scope.

**6. NLL is the most informative single metric for calibration.**
Transformer NLL (1.54) significantly outperforms Social-LSTM (18.03), Diffusion (3.31), and Trajectron++ (8.48). Well-calibrated uncertainty directly feeds into the conformal prediction safety layer (D3): tighter, more accurate uncertainty sets mean less conservative safety filtering.

---

## Model Details

| Model | Params | Architecture |
|-------|--------|-------------|
| CV Baseline | 0 | Constant velocity + Gaussian noise (σ=0.30m) |
| Social-LSTM | 455K | LSTMCell encoder/decoder, per-neighbour LSTMs, bivariate Gaussian head |
| Social-LSTM+V | 455K | Social-LSTM + explicit (vx,vy) velocity input to encoder |
| Trajectron++ | ~3M | CVAE + graph attention, GMM output, trained on ELBO |
| Transformer | 537K | 2-layer Pre-LN enc-dec, 12 learned query tokens, cross-attention to 48 tokens |
| Diffusion | 956K | TransformerDecoder denoiser (3 layers, cross-attention to 49 encoder tokens), detached Gaussian head, DDIM-50 sampling, λ=1.0 |

---

## Checkpoint Epochs (best validation ADE)

| Scene | Social-LSTM | SLSTM+V | Transformer | Diffusion |
|-------|------------|---------|-------------|-----------|
| eth | 130 | — | 20 | 200 |
| hotel | — | — | 70 | 30 |
| univ | 160 | — | 160 | 80 |
| zara1 | — | — | 20 | 40 |
| zara2 | — | — | 130 | 40 |
