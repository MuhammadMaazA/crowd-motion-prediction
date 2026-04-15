# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python 3.9.25, PyTorch 2.5.1 + CUDA 12.1, UCL GPU server (RTX 3090 Ti).

```bash
source /cs/student/projects1/2023/muhamaaz/year-long/crowdnav-env/bin/activate
python -c "import torch; print(torch.cuda.is_available())"  # True
```

pip cache is redirected to project space via `~/.config/pip/pip.conf` — do not change this, home dir has only 10GB quota.

## Key Commands

```bash
# CV baseline smoke test (no training, instant)
python models/cv_baseline.py

# Train Social-LSTM leave-one-out (one fold, ~1h on GPU)
python -u models/train_social_lstm.py --holdout eth --epochs 200 --batch_size 64
# holdout choices: eth hotel univ zara1 zara2

# Run all 5 folds in parallel (background)
for scene in eth hotel univ zara1 zara2; do
  python -u models/train_social_lstm.py --holdout $scene --epochs 200 &
done
```

## Architecture

Three-layer system: **Prediction → Planning → Safety**. This repo currently covers the Prediction layer (D1).

### Data flow

Raw `.txt` files (`data/`) → `eth_ucy_analysis.load_scene()` → DataFrame → `extract_sequences()` or `extract_sequences_with_neighbours()` → numpy arrays → PyTorch Dataset.

**`eth_ucy_analysis.py` is the single source of truth for data loading and metrics. Import from it, never rewrite it.**

Key shapes throughout the codebase:
- `obs`: `(N, 8, 2)` — observed positions, absolute world coords
- `pred`: `(N, 12, 2)` — ground truth future, absolute world coords
- `nb_obs`: `(N, max_neighbours, 8, 2)` — neighbour histories, NaN-padded
- `nb_mask`: `(N, max_neighbours)` bool — True where neighbour slot is occupied
- `samples`: `(N, K, 12, 2)` — K predicted futures

### Critical design constraint: relative coordinates

**Social-LSTM must never see absolute (x,y) positions.** Each scene has a different coordinate system (different camera placement). The model normalises by subtracting `obs[:, -1]` (last observed position) as origin before encoding, then adds it back at output. Without this, cross-scene generalisation breaks — hotel ADE goes from 0.54m to 4.0m.

### Model interface contract

All predictors expose the same interface so they can be swapped:
```python
samples = model.predict_samples(obs, K=20)  # (N, K, 12, 2) numpy
```
CV baseline follows this. Social-LSTM exposes `model.sample(obs_t, nb_obs_t, nb_mask_t, K)`.

### Social-LSTM internals

`SocialLSTM` in `models/social_lstm.py`:
1. Embed relative displacement → 64-dim
2. LSTM encoder (hidden=128) over obs sequence
3. `SocialPooling`: for each agent, max-pool neighbours' hidden states within `pooling_radius=2.0m`, concatenate with own hidden
4. Decoder LSTM outputs 5 params per step: `(µx, µy, log_σx, log_σy, atanh_ρ)`
5. Loss: bivariate Gaussian NLL via `bivariate_gaussian_nll()`
6. Sampling: Cholesky decomposition in `sample_bivariate_gaussian()`

Checkpoints saved to `checkpoints/social_lstm_{holdout}.pt` containing model state, epoch, ADE, and hparams dict.

### ETH/UCY dataset

5 standard scenes, leave-one-out evaluation. `univ` combines two files (`students001.txt` + `students003.txt`) — IDs are offset per file in `load_scene()` to avoid collisions. Two extra files exist (`crowds_zara03.txt`, `uni_examples.txt`) — used as training data by Trajectron++ but not as test scenes.

### Trajectron++ (not yet trained)

Preprocessed pkl files are at `Trajectron-plus-plus/experiments/processed/` (15 files, 335MB). Training script: `Trajectron-plus-plus/trajectron/train.py`. Data prep is complete — do not rerun `process_data.py`.

## Results (Social-LSTM, all 5 folds)

| Scene | ADE | FDE | minADE@20 |
|-------|-----|-----|-----------|
| eth | 1.037 | 2.039 | 0.953 |
| hotel | 0.544 | 1.233 | 0.544 |
| univ | 0.620 | 1.302 | 0.628 |
| zara1 | 0.412 | 0.891 | 0.498 |
| zara2 | 0.331 | 0.728 | 0.339 |
| **avg** | **0.589** | **1.239** | **0.592** |
