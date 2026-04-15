# COMP0225 — Project Progress Summary

## What We Did and Why

---

## 1. Dataset: ETH/UCY

### What it is
The ETH/UCY dataset is the standard benchmark for pedestrian trajectory prediction. It consists of 5 real-world scenes filmed with overhead cameras, with positions of every pedestrian annotated in metres.

| Scene | Location | Crowd density |
|-------|----------|---------------|
| `eth` | ETH Zurich campus open plaza | Low |
| `hotel` | Hotel corridor | Very low |
| `univ` | University campus | Medium |
| `zara1` | Outside Zara store, Zurich | High |
| `zara2` | Same location, different time | High |

### Raw data format
Each scene is a `.txt` file with 4 tab-separated columns:
```
frame_id    ped_id    x    y
```
- Positions are in **metres**
- Camera films at 25fps, annotations every 10 frames = **0.4 seconds per step**

### Standard prediction protocol
- **Observe** 8 steps = 3.2 seconds of history
- **Predict** 12 steps = 4.8 seconds into the future
- **Evaluation**: leave-one-out cross-validation — train on 4 scenes, test on the held-out one

### Bug found and fixed
`univ` combines two files (`students001.txt` + `students003.txt`) which had overlapping pedestrian IDs. Fixed by offsetting IDs per file in `eth_ucy_analysis.py`.

---

## 2. Infrastructure

### GPU environment
- Machine: UCL GPU server with **RTX 3090 Ti** (24GB VRAM), CUDA 12.1
- Python 3.9.25, venv at `crowdnav-env/`
- PyTorch 2.5.1 + CUDA 12.1

```bash
source /cs/student/projects1/2023/muhamaaz/year-long/crowdnav-env/bin/activate
python -c "import torch; print(torch.cuda.is_available())"  # True
```

### pip cache fix
pip defaults to `~/.cache/pip` which is on the home filesystem (10GB quota). We redirected it to project space (200GB) via `~/.config/pip/pip.conf`:
```ini
[global]
cache-dir = /cs/student/projects1/2023/muhamaaz/.pip-cache
```

---

## 3. Dataset Analysis (`eth_ucy_analysis.py`)

Built a complete dataset explorer. Key exported functions used by all models:

| Function | What it does |
|----------|-------------|
| `load_scene(paths)` | Loads raw txt files into a DataFrame |
| `extract_sequences(data, obs_len, pred_len)` | Returns `(obs, pred)` arrays of shape `(N, T, 2)` |
| `extract_sequences_with_neighbours(...)` | Same + neighbour trajectories `(nb_obs, nb_mask)` |
| `ade(pred, gt)` | Average Displacement Error across all timesteps |
| `fde(pred, gt)` | Final Displacement Error at last timestep |
| `best_of_k_ade(samples, gt)` | minADE@K — best of K samples |
| `best_of_k_fde(samples, gt)` | minFDE@K — best of K samples |
| `collision_rate(pred, others, radius)` | Fraction of predictions that collide |

Also generates **25 plots** in `eth_ucy_plots/` covering trajectory distributions, speeds, crowd densities, and interaction distances.

---

## 4. Constant Velocity Baseline (`models/cv_baseline.py`)

The simplest possible predictor — no training required.

**How it works:** Take the last observed velocity (final displacement) and extrapolate forward. Add Gaussian noise (`std=0.30m`) to produce K diverse samples.

**Why we built it:** Every learned model must beat this, otherwise it's not worth using. It's the performance floor.

```python
from models.cv_baseline import ConstantVelocityPredictor
cv = ConstantVelocityPredictor(noise_std=0.30)
samples = cv.predict_samples(obs, K=20)  # (N, 20, 12, 2)
```

---

## 5. Trajectron++ Data Preprocessing

Trajectron++ (a more complex model) cannot read the raw txt files — it needs data in a special binary pkl format with scene graphs, neighbour relationships, and train/val/test splits precomputed.

We ran their `process_data.py` after fixing 3 bugs:
1. Missing packages — installed `ncls` and `orjson`
2. Output directory didn't exist — created `experiments/processed/`
3. Python 3.9 / PyTorch 2.x incompatibilities — patched 4 lines in their codebase

**Result:** 15 pkl files in `Trajectron-plus-plus/experiments/processed/` (335MB total):
```
eth_train.pkl   eth_val.pkl   eth_test.pkl
hotel_train.pkl hotel_val.pkl hotel_test.pkl
univ_train.pkl  univ_val.pkl  univ_test.pkl
zara1_train.pkl zara1_val.pkl zara1_test.pkl
zara2_train.pkl zara2_val.pkl zara2_test.pkl
```

A teammate can now run Trajectron++ training directly without any data prep.

---

## 6. Social-LSTM (`models/social_lstm.py`)

### What it is
Social-LSTM (Alahi et al., CVPR 2016) — the classic learned pedestrian prediction model. Each pedestrian has an LSTM encoding their own history. At each step, they also "pool" information from nearby pedestrians' hidden states so the model learns crowd behaviour.

### Architecture
- **Input embedding:** (x,y) displacement → 64-dim vector via ReLU MLP
- **LSTM encoder:** hidden size 128
- **Social pooling:** max-pool neighbour hidden states within 2m radius, concatenated with own hidden state
- **Decoder:** outputs (µx, µy, log_σx, log_σy, atanh_ρ) — parameters of a bivariate Gaussian distribution
- **Loss:** Negative log-likelihood (NLL) of the true future trajectory under the predicted Gaussian
- **Sampling:** Cholesky decomposition for reparameterised samples

### Critical bug found and fixed: absolute coordinates
First training run gave hotel ADE = **4.0m** (CV baseline gets 0.4m — catastrophic).

**Root cause:** The model was encoding raw (x,y) positions. Hotel's coordinate system is completely different from the other 4 scenes (different camera placement). When trained on eth/univ/zara and tested on hotel, positions were entirely out-of-distribution.

**Fix:** Normalize to **relative coordinates** before encoding:
- Subtract the last observed position as origin → model only sees *displacements*
- Add the origin back at output → convert predictions back to world coordinates
- Now the model generalises across scenes regardless of where the camera is

### Training
Leave-one-out cross-validation across all 5 scenes:
```bash
source crowdnav-env/bin/activate
python -u models/train_social_lstm.py --holdout eth --epochs 200 --batch_size 64
```

- Optimizer: Adam, lr=1e-3, weight_decay=1e-4
- LR scheduler: ReduceLROnPlateau (patience=5, factor=0.5)
- Early stopping: 20 eval steps without ADE improvement
- Gradient clipping: max_norm=1.0
- 290,117 parameters

### Results (all 5 folds, relative-coord fixed model)

| Scene | ADE (m) | FDE (m) | minADE@20 | Best epoch |
|-------|---------|---------|-----------|------------|
| eth | 1.037 | 2.039 | 0.953 | 130 |
| hotel | 0.544 | 1.233 | 0.544 | 130 |
| univ | 0.620 | 1.302 | 0.628 | 10 |
| zara1 | 0.412 | 0.891 | 0.498 | 20 |
| zara2 | 0.331 | 0.728 | 0.339 | 100 |
| **avg** | **0.589** | **1.239** | **0.592** | |

Checkpoints saved to `checkpoints/social_lstm_{scene}.pt`.

### What the results mean
- **Zara1/Zara2** (dense crowds): big improvement over CV. Social pooling is genuinely useful when people interact.
- **Hotel** (empty corridor): worse than CV. Barely any interaction — social pooling adds noise rather than signal. CV straight-line extrapolation wins here.
- **Eth/Univ**: moderate improvement. Mixed crowd densities.

---

## Repository Structure

```
year-long/
├── eth_ucy_analysis.py          # Dataset loader + metrics (don't rewrite — import from here)
├── crowdnav-env/                # Python venv (PyTorch 2.5.1 + CUDA 12.1)
├── models/
│   ├── cv_baseline.py           # Constant velocity baseline
│   ├── social_lstm.py           # Social-LSTM model
│   └── train_social_lstm.py     # Training + evaluation script
├── checkpoints/
│   ├── social_lstm_eth.pt       # Best checkpoint per scene
│   ├── social_lstm_hotel.pt
│   ├── social_lstm_univ.pt
│   ├── social_lstm_zara1.pt
│   ├── social_lstm_zara2.pt
│   └── *_curve.png              # Training curves
├── eth_ucy_plots/               # 25 dataset analysis plots
└── Trajectron-plus-plus/
    └── experiments/
        └── processed/           # 15 pkl files ready for Trajectron++ training
```

---

## What's Left

| Task | Who | Status |
|------|-----|--------|
| Social-LSTM trained on all 5 scenes | Muhammad | Done |
| Trajectron++ training | Teammate B | pkl files ready, just run training |
| Conformal prediction safety layer | Benjamin | Not started |
| Unified evaluation/comparison script | Anyone | Not started |

---

## Key Commands

```bash
# Activate environment
source /cs/student/projects1/2023/muhamaaz/year-long/crowdnav-env/bin/activate

# Train Social-LSTM on a holdout scene
python -u models/train_social_lstm.py --holdout eth --epochs 200

# Run CV baseline smoke test
python models/cv_baseline.py
```
