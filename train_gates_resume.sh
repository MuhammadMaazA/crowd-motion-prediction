#!/bin/bash
# Resume training all models for the gates SDD holdout scene.
WORK=/cs/student/projects1/2023/muhamaaz/year-long
source "$WORK/crowdnav-env/bin/activate"
cd "$WORK"

mkdir -p logs checkpoints/sdd

LOG="$WORK/logs/gates_sdd_resume.log"

for model in social_lstm social_lstm_v gru_v2 transformer diffusion; do
    echo "=== $model / gates  $(date) ===" | tee -a "$LOG"
    python -u models/train_sdd.py \
        --model "$model" \
        --holdout gates \
        --epochs 50 \
        --batch_size 128 \
        --eval_every 5 \
        --patience 10 \
        --resume \
        2>&1 | tee -a "$LOG"
    echo "=== $model done  $(date) ===" | tee -a "$LOG"
done

echo "All gates models complete $(date)" | tee -a "$LOG"
