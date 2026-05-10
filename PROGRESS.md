# Project Progress Summary

Last updated: 2026-05-08 14:43 BST

## Current Status

D1 trajectory prediction is substantially complete on ETH/UCY. We have trained and evaluated:

- CV baseline
- Social-LSTM
- Social-LSTM+V, our velocity-augmented Social-LSTM
- Trajectron++
- Trajectory Transformer, our main attention-based model
- Diffusion predictor, now fixed with shared conditioning encoder and `lambda_ddpm=0.3`

The ETH/UCY results, comparison plots, and model discussion are now written in `RESULTS.md`. Generated figures are in `plots/`.

Stanford Drone Dataset (SDD) support has also been added and training is currently running.

## ETH/UCY Final Results

Protocol:

- Dataset: ETH/UCY
- Split: leave-one-scene-out over `eth`, `hotel`, `univ`, `zara1`, `zara2`
- Observation horizon: 8 frames
- Prediction horizon: 12 frames
- Metrics: ADE, FDE, minADE@20, minFDE@20, NLL

Average results across all 5 scenes:

| Model | ADE ↓ | FDE ↓ | minADE@20 ↓ | minFDE@20 ↓ | NLL ↓ |
|---|---:|---:|---:|---:|---:|
| CV Baseline | 0.736 | 1.272 | 0.610 | 0.814 | - |
| Social-LSTM | 0.596 | 1.248 | 0.581 | 0.597 | 18.03 |
| Social-LSTM+V | 0.585 | 1.240 | 0.587 | 0.585 | 36.20 |
| Trajectron++ | 0.853 | 1.817 | **0.385** | 0.674 | 8.49 |
| Transformer | **0.568** | 1.203 | 0.638 | **0.532** | **1.54** |
| Diffusion | 0.579 | **1.195** | 1.174 | 2.032 | 5.14 |

Main takeaways:

- Transformer is the best point predictor by ADE.
- Diffusion is now competitive after the fix: ADE `0.579`, close to Transformer `0.568`, and best FDE `1.195`.
- Trajectron++ is still clearly best for multi-modal coverage: minADE@20 `0.385`.
- Social-LSTM+V improves over Social-LSTM on ADE: `0.596 -> 0.585`.
- The key research finding is the tradeoff between point accuracy and diversity: models that predict one accurate future do not necessarily cover many plausible futures.

## Diffusion Fix

The first improved diffusion attempt used a detached context encoder and `lambda_ddpm=1.0`. That improved some denoising behaviour but badly hurt point prediction.

The current fixed version:

- removes `context.detach()`
- shares the encoder between Gaussian head and denoiser
- uses `lambda_ddpm=0.3`
- trains with DDIM-50 sampling

This changed diffusion from a weak result to a usable model:

| Diffusion Version | ADE ↓ | FDE ↓ | minADE@20 ↓ | NLL ↓ |
|---|---:|---:|---:|---:|
| Detached encoder, lambda=1.0 | 1.370 | 2.539 | 1.143 | 3.31 |
| Shared encoder, lambda=0.3 | **0.579** | **1.195** | 1.174 | 5.14 |

Interpretation:

- The fixed diffusion model is acceptable for point prediction.
- It is not yet strong for best-of-20 diversity.
- A future improvement would be SDD pretraining, separate encoders, or tuning `lambda_ddpm` around `0.2-0.5`.

## Generated Plots

Generated plots:

- `plots/ade_per_scene.png`
- `plots/summary_bar.png`
- `plots/accuracy_diversity_tradeoff.png`
- `plots/nll_calibration.png`
- `plots/heatmap.png`

These support the report visually:

- `ade_per_scene.png`: per-scene ADE comparison
- `summary_bar.png`: average point prediction vs best-of-20 diversity
- `accuracy_diversity_tradeoff.png`: shows Transformer/Diffusion vs Trajectron++ tradeoff
- `nll_calibration.png`: shows Transformer has best uncertainty calibration
- `heatmap.png`: compact final metric comparison

## Stanford Drone Dataset Progress

SDD has been added as a second dataset.

Completed implementation:

- Downloaded SDD annotations for 8 scenes:
  - `bookstore`
  - `coupa`
  - `deathCircle`
  - `gates`
  - `hyang`
  - `little`
  - `nexus`
  - `quad`
- Added `sdd_analysis.py`
- Added SDD sequence extraction and pixel-to-metre conversion
- Added cached SDD preprocessing under `data/sdd_cache/`
- Added unified SDD training script: `models/train_sdd.py`
- Added sequential training script: `train_sdd_sequential.sh`

SDD scale:

- 8 scenes
- 60 annotation videos
- about 249k extracted trajectory sequences

Training progress so far:

| Model | Scene | Best ADE ↓ | Status |
|---|---|---:|---|
| Social-LSTM | all 8 scenes | complete | complete |
| Social-LSTM+V | bookstore | 0.508 | complete |
| Social-LSTM+V | coupa | 0.451 | complete |
| Social-LSTM+V | deathCircle | 0.920 | complete |
| Social-LSTM+V | gates | 0.832 | complete |
| Social-LSTM+V | hyang | 0.594 | complete |
| Social-LSTM+V | little | 0.948 | complete |
| Social-LSTM+V | nexus | 0.586 | complete |
| Social-LSTM+V | quad | 0.405 | complete |
| Transformer | bookstore | 0.504 | complete |
| Transformer | coupa | 0.454 | complete |
| Transformer | deathCircle | pending | running |

Current SDD run:

- Model: Transformer
- Scene: `deathCircle`
- Latest observed progress: epoch 25/50
- Remaining after current scene:
  - Transformer: `gates`, `hyang`, `little`, `nexus`, `quad`
  - Diffusion: all 8 SDD scenes

Estimated remaining time at the latest check:

- Transformer remaining: about 5-7 hours
- Diffusion remaining: about 10-14 hours
- Total SDD remaining: about 15-21 hours

## Work Completed So Far

Code and modelling:

- Implemented and evaluated Social-LSTM baseline.
- Implemented Social-LSTM+V with explicit velocity input.
- Implemented Trajectory Transformer with cross-attention over observed focal and neighbour trajectories.
- Implemented diffusion trajectory predictor with Transformer decoder denoiser.
- Fixed diffusion training after detached encoder caused poor ADE.
- Added full ETH/UCY evaluation script.
- Added plotting script for final comparisons.
- Added SDD dataset loader and training pipeline.

Experiments:

- Completed ETH/UCY leave-one-out evaluation for all models.
- Completed final fixed-diffusion ETH/UCY retraining.
- Started SDD leave-one-scene-out training.
- Completed full SDD Social-LSTM run.
- Completed full SDD Social-LSTM+V run.
- Started SDD Transformer run.

Report-ready findings:

- Transformer is the strongest ETH/UCY point predictor.
- Diffusion is now acceptable and competitive on ADE/FDE.
- Trajectron++ remains strongest on multi-modal diversity.
- Velocity input gives a small but consistent Social-LSTM improvement.
- Metric choice matters: ADE/FDE reward accuracy, minADE/minFDE reward diversity, and NLL rewards calibration.

## Next Steps

Short-term:

- Let SDD Transformer and Diffusion training finish.
- Write `evaluate_sdd.py` to summarise SDD results.
- Add SDD results table to `RESULTS.md`.
- Use SDD checkpoints for pretrain -> ETH/UCY fine-tuning experiments.

Potential high-value ablation:

- Pretrain Transformer on SDD, fine-tune on ETH/UCY.
- Pretrain Diffusion on SDD, fine-tune on ETH/UCY.
- Compare against ETH/UCY-from-scratch results.

This would give a strong report story:

> Large-scale SDD pretraining improves general pedestrian motion modelling, while ETH/UCY fine-tuning adapts to the target benchmark scenes.
