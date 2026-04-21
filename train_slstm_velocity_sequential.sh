#!/bin/bash
# Train velocity-augmented Social-LSTM on all 5 ETH/UCY holdout scenes.
# Launch: nohup bash train_slstm_velocity_sequential.sh > logs/slstm_v_train.log 2>&1 &

set -e
WORK=/cs/student/projects1/2023/muhamaaz/year-long
source "$WORK/crowdnav-env/bin/activate"
cd "$WORK"

mkdir -p logs

SCENES=(eth hotel univ zara1 zara2)

for scene in "${SCENES[@]}"; do
    echo "========================================"
    echo "  Training Social-LSTM+V: $scene"
    echo "  Started: $(date)"
    echo "========================================"

    python -u models/train_social_lstm.py \
        --holdout "$scene" \
        --epochs 200 \
        --batch_size 64 \
        --velocity

    echo "  Finished $scene at $(date)"
    echo ""
done

echo "All 5 scenes done! $(date)"
echo "Run: python evaluate_all.py"
