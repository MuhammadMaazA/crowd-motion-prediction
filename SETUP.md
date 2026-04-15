# Setup Guide

## Requirements
- Python 3.9+
- NVIDIA GPU with CUDA 12.1 (UCL machines have RTX 3090 Ti)
- ~5GB disk space for packages

## 1. Create virtual environment

```bash
cd /cs/student/projects1/2023/<your-username>/year-long
python3 -m venv crowdnav-env
source crowdnav-env/bin/activate
```

## 2. Fix pip cache location (UCL only)

Home directory has a 10GB quota — pip's default cache fills it up fast.
Redirect it to project space before installing anything:

```bash
mkdir -p ~/.config/pip
cat > ~/.config/pip/pip.conf << EOF
[global]
cache-dir = /cs/student/projects1/2023/<your-username>/.pip-cache
EOF
```

## 3. Install dependencies

```bash
pip install --upgrade pip

# PyTorch with CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Everything else
pip install numpy pandas scipy scikit-learn matplotlib seaborn dill tqdm tensorboard ncls orjson
```

Verify GPU is available:
```bash
python -c "import torch; print(torch.cuda.is_available())"  # should print True
```

## 4. Get the data

The ETH/UCY raw data lives in:
```
Trajectron-plus-plus/experiments/pedestrians/raw/raw/all_data/
```

8 files total (not in git — too large):
- `biwi_eth.txt`, `biwi_hotel.txt` — ETH dataset
- `students001.txt`, `students003.txt` — UCY university
- `crowds_zara01.txt`, `crowds_zara02.txt`, `crowds_zara03.txt` — UCY Zara
- `uni_examples.txt` — UCY extra

Ask a teammate for the data or clone Trajectron-plus-plus separately:
```bash
git clone https://github.com/StanfordASL/Trajectron-plus-plus.git
```

## 5. Run the CV baseline (smoke test)

```bash
source crowdnav-env/bin/activate
python models/cv_baseline.py
```

Should print ADE/FDE numbers for eth, hotel, zara1 and exit cleanly.

## 6. Train Social-LSTM

```bash
python -u models/train_social_lstm.py --holdout eth --epochs 200 --batch_size 64
```

Options:
```
--holdout   Scene to hold out for evaluation (eth/hotel/univ/zara1/zara2)
--epochs    Number of training epochs (default 200)
--batch_size Batch size (default 64)
--lr        Learning rate (default 1e-3)
--hidden    LSTM hidden size (default 128)
--embed     Embedding size (default 64)
--radius    Social pooling radius in metres (default 2.0)
--device    cuda or cpu (default cuda)
```

Checkpoints saved to `checkpoints/social_lstm_{holdout}.pt`.

## Project structure

```
year-long/
├── eth_ucy_analysis.py       # Dataset loader + ADE/FDE metrics — import from here
├── models/
│   ├── cv_baseline.py        # Constant velocity baseline (no training)
│   ├── social_lstm.py        # Social-LSTM model
│   └── train_social_lstm.py  # Training script
├── checkpoints/              # Saved model weights (not in git)
├── PROGRESS.md               # What's been done and results
├── SETUP.md                  # This file
└── requirements.txt          # Python dependencies
```
