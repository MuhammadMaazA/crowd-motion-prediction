#!/bin/bash
# Train Social-GRU v2 + Diffusion (retrain) on GPU 1 — runs in parallel with SDD on GPU 0.
# Launch: nohup bash train_improved_gpu1.sh >& logs/improved_gpu1.log &

set -e
WORK=/cs/student/projects1/2023/muhamaaz/year-long
source "$WORK/crowdnav-env/bin/activate"
cd "$WORK"
mkdir -p logs checkpoints

SCENES=(eth hotel univ zara1 zara2)

# ── Social-GRU v2 (bidirectional encoder) ────────────────────────────────────
echo "================================================"
echo "  Social-GRU v2 (bidirectional) on GPU 1"
echo "  Started: $(date)"
echo "================================================"

for scene in "${SCENES[@]}"; do
    echo "--- GRU v2: $scene --- $(date)"
    python -u models/train_social_gru.py \
        --holdout "$scene" \
        --epochs 200 \
        --batch_size 64 \
        --v2 \
        --device cuda:1
    echo "  Done $scene"
done

echo "GRU v2 done! $(date)"

# ── Diffusion retrain (stochastic DDPM sampling) ─────────────────────────────
echo "================================================"
echo "  Diffusion (stochastic DDPM) on GPU 1"
echo "  Started: $(date)"
echo "================================================"

for scene in "${SCENES[@]}"; do
    echo "--- Diffusion: $scene --- $(date)"
    python -u models/train_diffusion.py \
        --holdout "$scene" \
        --epochs 200 \
        --batch_size 64 \
        --T 100 \
        --ddim_steps 50 \
        --lambda_ddpm 0.3 \
        --device cuda:1
    echo "  Done $scene"
done

echo "All improved models done! $(date)"
echo "Run: python evaluate_all.py"
