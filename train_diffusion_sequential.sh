#!/bin/bash
# Train Diffusion model on all 5 ETH/UCY holdout scenes.
# Launch: nohup bash train_diffusion_sequential.sh >& logs/diffusion_train.log &

set -e
WORK=/cs/student/projects1/2023/muhamaaz/year-long
source "$WORK/crowdnav-env/bin/activate"
cd "$WORK"

mkdir -p logs

SCENES=(eth hotel univ zara1 zara2)

for scene in "${SCENES[@]}"; do
    echo "========================================"
    echo "  Training Diffusion: $scene"
    echo "  Started: $(date)"
    echo "========================================"

    python -u models/train_diffusion.py \
        --holdout "$scene" \
        --epochs 200 \
        --batch_size 64 \
        --T 100 \
        --ddim_steps 20 \
        --lambda_ddpm 1.0

    echo "  Finished $scene at $(date)"
    echo ""
done

echo "All 5 scenes done! $(date)"
echo "Run: python evaluate_all.py"
