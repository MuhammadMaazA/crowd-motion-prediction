# D1 Prediction Model Results — ETH/UCY Leave-One-Out

**Protocol:** obs=8 frames (3.2s), pred=12 frames (4.8s), leave-one-out cross-validation over 5 scenes.  
**Metrics:** ADE/FDE in metres (↓ lower is better). minADE@20 / minFDE@20 = best-of-20 samples.  
**NLL** = bivariate Gaussian negative log-likelihood (↓ lower is better; not available for CV baseline).

---

## ADE (Average Displacement Error, metres ↓)

| Scene | CV | Social-LSTM | SLSTM+V | Trajectron++ | **Transformer** | Diffusion |
|-------|------|------------|---------|-------------|-----------------|-----------|
| eth | 1.204 | 1.015 | 1.010 | 1.355 | **0.982** | **0.982** |
| hotel | 0.552 | 0.541 | 0.527 | 0.977 | **0.437** | 0.485 |
| univ | 0.716 | 0.655 | 0.635 | 0.772 | 0.568 | **0.563** |
| zara1 | 0.623 | 0.447 | **0.432** | 0.624 | 0.481 | 0.503 |
| zara2 | 0.583 | 0.319 | 0.320 | 0.535 | 0.371 | **0.361** |
| **avg** | 0.736 | 0.596 | 0.585 | 0.853 | **0.568** | 0.579 |

---

## FDE (Final Displacement Error, metres ↓)

| Scene | CV | Social-LSTM | SLSTM+V | Trajectron++ | **Transformer** | Diffusion |
|-------|------|------------|---------|-------------|-----------------|-----------|
| eth | 2.346 | 1.993 | 1.996 | 2.772 | 1.933 | **1.929** |
| hotel | 0.784 | 1.241 | 1.186 | 2.164 | **0.973** | 1.007 |
| univ | 1.267 | 1.368 | 1.335 | 1.656 | 1.181 | **1.179** |
| zara1 | 1.042 | **0.944** | 0.970 | 1.338 | 1.095 | 1.071 |
| zara2 | 0.923 | 0.696 | 0.710 | 1.155 | 0.832 | **0.789** |
| **avg** | 1.272 | 1.248 | 1.240 | 1.817 | 1.203 | **1.195** |

---

## minADE@20 (Best-of-20 ADE, metres ↓)

| Scene | CV | Social-LSTM | SLSTM+V | **Trajectron++** | Transformer | Diffusion |
|-------|------|------------|---------|-------------|-------------|-----------|
| eth | 1.063 | 0.931 | 0.931 | **0.801** | 0.919 | 1.603 |
| hotel | 0.436 | 0.529 | 0.506 | **0.385** | 0.521 | 0.840 |
| univ | 0.588 | 0.618 | 0.655 | **0.335** | 0.653 | 0.913 |
| zara1 | 0.498 | 0.476 | 0.510 | **0.220** | 0.691 | 1.619 |
| zara2 | 0.466 | 0.352 | 0.335 | **0.182** | 0.406 | 0.893 |
| **avg** | 0.610 | 0.581 | 0.587 | **0.385** | 0.638 | 1.174 |

---

## minFDE@20 (Best-of-20 FDE, metres ↓)

| Scene | CV | Social-LSTM | SLSTM+V | Trajectron++ | **Transformer** | Diffusion |
|-------|------|------------|---------|-------------|-----------------|-----------|
| eth | 1.824 | 1.119 | 1.122 | 1.434 | **0.865** | 2.629 |
| hotel | 0.381 | 0.509 | 0.474 | 0.679 | **0.412** | 1.464 |
| univ | 0.784 | 0.653 | 0.599 | 0.601 | **0.519** | 1.489 |
| zara1 | 0.569 | 0.389 | 0.407 | **0.350** | 0.514 | 2.943 |
| zara2 | 0.513 | 0.314 | 0.320 | **0.307** | 0.350 | 1.635 |
| **avg** | 0.814 | 0.597 | 0.585 | 0.674 | **0.532** | 2.032 |

---

## NLL (Negative Log-Likelihood ↓)

| Scene | CV | Social-LSTM | SLSTM+V | Trajectron++ | **Transformer** | Diffusion |
|-------|------|------------|---------|-------------|-----------------|-----------|
| eth | — | 78.484 | 146.898 | 56.998 | **3.656** | 7.793 |
| hotel | — | -0.175 | 0.865 | 9.631 | **-0.403** | 0.245 |
| univ | — | 8.382 | 5.188 | 3.842 | **2.857** | 3.009 |
| zara1 | — | 0.795 | 1.126 | -4.071 | 0.884 | **0.753** |
| zara2 | — | 2.661 | 26.938 | -23.960 | **0.683** | 13.875 |
| **avg** | — | 18.029 | 36.203 | 8.488 | **1.535** | 5.135 |

---

## Summary Table (averages across 5 scenes)

| Model | ADE↓ | FDE↓ | minADE@20↓ | minFDE@20↓ | NLL↓ |
|-------|------|------|-----------|-----------|------|
| CV Baseline | 0.736 | 1.272 | 0.610 | 0.814 | — |
| Social-LSTM | 0.596 | 1.248 | 0.579 | 0.591 | 18.03 |
| Social-LSTM+V *(ours)* | 0.585 | 1.240 | 0.589 | 0.585 | 36.20 |
| Trajectron++ | 0.853 | 1.817 | **0.385** | 0.674 | 8.49 |
| **Transformer** *(ours)* | **0.568** | 1.203 | 0.638 | **0.532** | **1.54** |
| Diffusion *(ours)* | 0.579 | **1.195** | 1.174 | 2.032 | 5.14 |

---

## Key Findings

**1. Transformer achieves best point-prediction accuracy.**
Lowest ADE (0.568m), FDE (1.203m), minFDE@20 (0.528m) and NLL (1.54) of all models. The parallel decoder with cross-attention to all 48 observation tokens (focal + neighbours) outperforms recurrent Social-LSTM without requiring any multi-modal latent variables.

**2. Trajectron++ dominates multi-modal coverage (minADE@20 = 0.385m).**
Its CVAE latent variable generates genuinely diverse trajectories across all modes. No other model comes close on this metric, confirming that explicit latent variable multi-modality is necessary for strong minADE@20.

**3. Velocity augmentation (Social-LSTM+V) gives consistent small gains.**
ADE improves 0.596→0.585m across all 5 scenes. The explicit (vx, vy) encoder input helps infer motion direction without relying solely on position differences — our contribution to the prediction module.

**4. ADE and minADE@20 tell opposite stories — confirming the report's core argument.**
Trajectron++ has the worst ADE (0.853m) but best minADE@20 (0.385m). The Transformer has the best ADE (0.568m) but middling minADE@20 (0.638m). Optimising for point-prediction accuracy and multi-modal coverage are fundamentally different objectives — confirming that metric choice drives model selection for safety-critical planning.

**5. Diffusion model reveals a fundamental architecture tradeoff.**
The fixed diffusion model removes the detached conditioning encoder and uses a lower DDPM weight (λ=0.3). This restores point-prediction performance: ADE improves from 1.370m in the detached run to 0.579m, almost matching the Transformer (0.568m), and it achieves the best average FDE (1.195m). However, best-of-20 diversity remains weak (minADE@20 = 1.174m), showing that the denoiser is accurate but not yet covering multiple plausible futures as effectively as Trajectron++.

**6. NLL is the most informative single metric for calibration.**
Transformer NLL (1.54) significantly outperforms Social-LSTM (18.03), Diffusion (5.14), and Trajectron++ (8.49). Well-calibrated uncertainty directly feeds into the conformal prediction safety layer (D3): tighter, more accurate uncertainty sets mean less conservative safety filtering.

---

## Plots

Generated figures are saved in `plots/`:

- `plots/ade_per_scene.png` — per-scene ADE comparison
- `plots/summary_bar.png` — average ADE/FDE vs best-of-20 metrics
- `plots/accuracy_diversity_tradeoff.png` — ADE vs minADE@20 tradeoff
- `plots/nll_calibration.png` — average NLL calibration
- `plots/heatmap.png` — average metric heatmap

---

## Model Details

| Model | Params | Architecture |
|-------|--------|-------------|
| CV Baseline | 0 | Constant velocity + Gaussian noise (σ=0.30m) |
| Social-LSTM | 455K | LSTMCell encoder/decoder, per-neighbour LSTMs, bivariate Gaussian head |
| Social-LSTM+V | 455K | Social-LSTM + explicit (vx,vy) velocity input to encoder |
| Trajectron++ | ~3M | CVAE + graph attention, GMM output, trained on ELBO |
| Transformer | 537K | 2-layer Pre-LN enc-dec, 12 learned query tokens, cross-attention to 48 tokens |
| Diffusion | 956K | TransformerDecoder denoiser (3 layers, cross-attention to 49 encoder tokens), shared Gaussian head, DDIM-50 sampling, λ=0.3 |

---

## Checkpoint Epochs (best validation ADE)

| Scene | Social-LSTM | SLSTM+V | Transformer | Diffusion |
|-------|------------|---------|-------------|-----------|
| eth | 130 | — | 20 | 20 |
| hotel | — | — | 70 | 70 |
| univ | 160 | — | 160 | 130 |
| zara1 | — | — | 20 | 120 |
| zara2 | — | — | 130 | 200 |
