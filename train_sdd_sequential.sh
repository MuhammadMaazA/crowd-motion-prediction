#!/bin/bash
# Train all models on SDD (Stanford Drone Dataset), all 8 holdout scenes.
# Usage: nohup bash train_sdd_sequential.sh >& logs/sdd_train.log &
#
# Trains: Social-LSTM, Social-LSTM+V, Transformer, Diffusion
# Each model × 8 scenes = 32 training runs total
# Estimated time: ~8-12 hours on RTX 3090

set -e
WORK=/cs/student/projects1/2023/muhamaaz/year-long
source "$WORK/crowdnav-env/bin/activate"
cd "$WORK"

mkdir -p logs checkpoints/sdd

SCENES=(bookstore coupa deathCircle gates hyang little nexus quad)
MODELS=(social_lstm social_lstm_v transformer diffusion)

for model in "${MODELS[@]}"; do
    echo "###################################################"
    echo "  MODEL: $model"
    echo "###################################################"
    for scene in "${SCENES[@]}"; do
        echo "--- $model / $scene ---  $(date)"
        python -u models/train_sdd.py \
            --model "$model" \
            --holdout "$scene" \
            --epochs 50 \
            --batch_size 128 \
            --eval_every 5 \
            --patience 10
        echo "  Done: $model / $scene  $(date)"
    done
done

echo "All SDD training complete! $(date)"
echo "Run: python evaluate_sdd.py"
