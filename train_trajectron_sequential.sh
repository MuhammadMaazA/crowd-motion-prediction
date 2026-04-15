#!/bin/bash
# Sequential Trajectron++ training — runs all 5 scenes one after another.
# Launch with:  nohup bash train_trajectron_sequential.sh > logs/tpp_train.log 2>&1 &
# Monitor with: tail -f logs/tpp_train.log

set -e

WORK=/cs/student/projects1/2023/muhamaaz/year-long
TRAJ=$WORK/Trajectron-plus-plus
PROC=$TRAJ/experiments/pedestrians
LOG=$WORK/logs

mkdir -p "$LOG"

source "$WORK/crowdnav-env/bin/activate"
cd "$TRAJ/trajectron"

SCENES=(eth hotel univ zara1 zara2)

for scene in "${SCENES[@]}"; do
    echo "========================================"
    echo "  Training scene: $scene"
    echo "  Started: $(date)"
    echo "========================================"

    python -u train.py \
        --conf    "$PROC/models/${scene}_attention_radius_3/config.json" \
        --train_data_dict "${scene}_train.pkl" \
        --eval_data_dict  "${scene}_val.pkl"  \
        --train_epochs 100 \
        --eval_every 10   \
        --save_every 10   \
        --log_tag "$scene"

    echo ""
    echo "  Finished $scene at $(date)"
    echo ""
done

echo "========================================"
echo "All 5 scenes done! $(date)"
echo "Run: python evaluate_all.py"
echo "========================================"
