#!/bin/bash
# Train Social-GRU on all 5 ETH/UCY holdout scenes.
# Launch: nohup bash train_gru_sequential.sh >& logs/gru_train.log &

set -e
WORK=/cs/student/projects1/2023/muhamaaz/year-long
source "$WORK/crowdnav-env/bin/activate"
cd "$WORK"
mkdir -p logs

SCENES=(eth hotel univ zara1 zara2)

for scene in "${SCENES[@]}"; do
    echo "========================================"
    echo "  Social-GRU: $scene"
    echo "  Started: $(date)"
    echo "========================================"
    python -u models/train_social_gru.py \
        --holdout "$scene" \
        --epochs 200 \
        --batch_size 64
    echo "  Finished $scene at $(date)"
done

echo "All 5 scenes done! $(date)"
echo "Run: python evaluate_all.py"
