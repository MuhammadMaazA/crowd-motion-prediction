# Project Progress Summary

Last updated: 2026-05-17 BST

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

SDD training is now complete for 5 of 6 models. All 40 checkpoints are in `checkpoints/sdd/`.

### SDD Evaluation Results (5 models, averaged over 8 scenes)

| Model | ADE ↓ | FDE ↓ | minADE@20 ↓ | minFDE@20 ↓ | NLL ↓ |
|---|---:|---:|---:|---:|---:|
| CV Baseline | 0.926 | 1.620 | 0.797 | 1.154 | — |
| Social-LSTM | 0.672 | 1.358 | 0.783 | 0.621 | 1.162 |
| Social-LSTM+V | **0.656** | 1.340 | **0.755** | **0.604** | 1.289 |
| GRU-v2 | 0.667 | 1.379 | 0.793 | 0.625 | 1.331 |
| Transformer | 0.657 | **1.326** | 0.808 | 0.633 | 1.474 |
| Diffusion | 0.673 | 1.335 | 1.710 | 3.159 | 1.683 |
| Trajectron++ | — | — | — | — | — (still training) |

### Per-scene SDD ADE

| Scene | CV | SLSTM | SLSTM+V | GRU-v2 | Transf | Diffusion |
|---|---:|---:|---:|---:|---:|---:|
| bookstore | 0.759 | 0.510 | 0.508 | 0.505 | 0.504 | 0.513 |
| coupa | 0.718 | 0.465 | 0.451 | 0.463 | 0.454 | 0.464 |
| deathCircle | 1.298 | 0.926 | 0.920 | 0.933 | 0.939 | 0.933 |
| gates | 1.068 | 0.831 | 0.832 | 0.823 | 0.826 | 0.837 |
| hyang | 0.874 | 0.596 | 0.594 | 0.591 | 0.604 | 0.596 |
| little | 1.236 | 0.958 | 0.948 | 0.965 | 0.968 | 0.976 |
| nexus | 0.879 | 0.583 | 0.586 | 0.588 | 0.591 | 0.589 |
| quad | 0.576 | 0.507 | 0.405 | 0.468 | 0.366 | 0.476 |

### Trajectron++ SDD Training Status (as of 2026-05-16)

**Primary machine**: 4× GTX TITAN X (11.91 GiB each), `cuda:0–3`.
All 4 processes launched via `nohup` and writing to separate log files. Do **not** start anything new on this machine without checking GPU usage first (`nvidia-smi`).

| GPU | Script | Log file | Scene(s) | Epoch (approx) | ETA |
|---|---|---|---|---|---|
| cuda:3 | `train_trajectron_sdd.sh` | `logs/tpp_sdd_train2.log` | bookstore ✅ → **coupa** (redundant) | epoch 1 (coupa) | — |
| cuda:0 | `train_tpp_sdd_gpu0.sh` | `logs/tpp_sdd_gpu0.log` | **coupa** → little → quad | ~31/100 (coupa) | ~3 days |
| cuda:1 | `train_tpp_sdd_gpu1.sh` | `logs/tpp_sdd_gpu1.log` | **deathCircle** → nexus | ~32/100 (deathCircle) | ~3 days |
| cuda:2 | `train_tpp_sdd_gpu2.sh` | `logs/tpp_sdd_gpu2.log` | **gates** → hyang | ~29/100 (gates) | ~3+ days |

T++ is slow because: ~1.7s/batch × 3,500–4,000 batches/epoch × 100 epochs ≈ 6–8 days per scene.
All other T++ SDD runs will be added to `evaluate_sdd.py` once checkpoints are available in `Trajectron-plus-plus/experiments/logs_sdd/`.

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

## Fine-tuning Results (DONE)

SDD → ETH/UCY fine-tuning completed for all 5 models. Results stored in `results_ft.json`.

ETH/UCY ADE (averaged over 5 scenes), scratch vs fine-tuned:

| Model | Scratch | SDD FT | Δ |
|---|---:|---:|---:|
| Social-LSTM | 0.596 | **0.537** | −9.9% |
| SLSTM+V | 0.585 | 0.555 | −5.1% |
| GRU-v2 | 0.586 | 0.549 | −6.3% |
| Transformer | **0.568** | 0.570 | +0.4% |
| Diffusion | 0.579 | 0.535 | −7.6% |

Key finding: SDD pretraining helps recurrent models (−6 to −10% ADE) but the Transformer is already near-optimal from scratch (+0.4%). FT does not help Diffusion minADE@20 (1.174 → 1.157), confirming the diversity limit is architectural.

Full per-scene FT results: see `results_ft.json`. Fine-tuned checkpoints: `checkpoints/ft_*.pt`.

## Next Steps

### Immediate

- **T++ SDD** — wait for GPU0/1/2 to reach epoch 100 (~3 days). Once done, add T++ to `evaluate_sdd.py` and regenerate SDD plots. Scenes: coupa, deathCircle, gates, hyang, little, nexus, quad (bookstore already done).
- **Kill redundant coupa** on `cuda:3` (train2.log) once GPU0 coupa finishes, to free GPU.

### Short-term

- Add GRU-v2 to `evaluate_all.py` (ETH/UCY table) — no training needed, checkpoints at `checkpoints/social_gruv2_{scene}.pt`
- Once T++ SDD checkpoints available: regenerate SDD comparison plots with T++ row
- Regenerate fine-tuning plots with final results for report figures

### Report

IEEE conference draft (`report/main.tex`) created on 2026-05-17. Currently **4 pages** covering:
- Section III: Methodology (all 5 model architectures + equations + 4 TikZ figures)
- Section IV: Experimental Setup
- Section V: Results (Tables I–III) + Discussion

Placeholders left for: Abstract, Introduction, Project Management, Conclusions (joint team sections).

Bibliography style: `ieeetr` (system-available). Replace with `IEEEtran` on Overleaf or when package is available.

Figures in `report/figures/`: `social_lstm.tex`, `transformer.tex`, `diffusion.tex`, `finetune_pipeline.tex`.

