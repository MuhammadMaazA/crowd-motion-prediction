#!/bin/bash
# Trajectron++ SDD — all 8 scenes sequentially on RTX 3090 Ti (cuda:0)
# Run with:
#   nohup bash train_tpp_sdd_3090.sh > logs/tpp_sdd_3090.log 2>&1 &
#   tail -f logs/tpp_sdd_3090.log

set -e

WORK=/cs/student/projects1/2023/muhamaaz/year-long
TPP=$WORK/Trajectron-plus-plus/trajectron
source "$WORK/crowdnav-env/bin/activate"
cd "$TPP"

CONF=$WORK/Trajectron-plus-plus/experiments/pedestrians/models/eth_attention_radius_3/config.json
PROC=$WORK/Trajectron-plus-plus/experiments/processed
LOG_DIR=$WORK/Trajectron-plus-plus/experiments/logs_sdd

mkdir -p "$WORK/logs"

# Coverage:
#   bookstore  — done (epoch 100 checkpoint exists)
#   little     — already running on 3090 Ti via train_tpp_sdd_little.sh
#   coupa/deathCircle/gates — running on TITAN X GPUs 0/1/2 (epoch ~30-32)
#   nexus/hyang/quad — deep in TITAN X queues (won't start for 2-3 weeks)
#
# This script covers the three scenes the TITAN Xs won't reach in time.
SCENES=(nexus hyang quad)

for scene in "${SCENES[@]}"; do
    echo "=== Trajectron++ SDD [3090]: $scene === $(date)"
    python -u train.py \
        --conf            "$CONF" \
        --train_data_dict "sdd_${scene}_train.pkl" \
        --eval_data_dict  "sdd_${scene}_test.pkl" \
        --data_dir        "$PROC" \
        --log_dir         "$LOG_DIR" \
        --log_tag         "${scene}" \
        --train_epochs    100 \
        --eval_every      10 \
        --save_every      10 \
        --batch_size      256 \
        --k_eval          20 \
        --offline_scene_graph no \
        --preprocess_workers 4 \
        --device          cuda:0 \
        --eval_device     cuda:0
    echo "=== Done: $scene at $(date) ==="
done

echo "========================================"
echo "nexus + hyang + quad done! $(date)"
echo "========================================"
