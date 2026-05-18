#!/bin/bash
# Timing test: Train Trajectron++ on SDD hyang (smallest scene, 12 MB pkl)
# on RTX 3090 (cuda:0) to measure speed before committing to the full run.
#
# Launch:
#   nohup bash train_tpp_sdd_timing.sh > logs/tpp_sdd_timing.log 2>&1 &
#   tail -f logs/tpp_sdd_timing.log

set -e
WORK=/cs/student/projects1/2023/muhamaaz/year-long
TPP=$WORK/Trajectron-plus-plus/trajectron
source "$WORK/crowdnav-env/bin/activate"
cd "$TPP"

CONF=$WORK/Trajectron-plus-plus/experiments/pedestrians/models/eth_attention_radius_3/config.json
PROC=$WORK/Trajectron-plus-plus/experiments/processed
LOG_DIR=$WORK/Trajectron-plus-plus/experiments/logs_sdd_timing

mkdir -p "$WORK/logs"

echo "========================================"
echo "  Trajectron++ SDD timing test: hyang"
echo "  Device: cuda:0 (RTX 3090)"
echo "  Started: $(date)"
echo "========================================"

python -u train.py \
    --conf          "$CONF" \
    --train_data_dict "sdd_hyang_train.pkl" \
    --eval_data_dict  "sdd_hyang_test.pkl" \
    --data_dir        "$PROC" \
    --log_dir         "$LOG_DIR" \
    --log_tag         "hyang_timing" \
    --train_epochs    100 \
    --eval_every      10 \
    --save_every      10 \
    --batch_size      256 \
    --k_eval          20 \
    --offline_scene_graph yes \
    --preprocess_workers 0 \
    --device cuda:0 \
    --eval_device cpu

echo "========================================"
echo "  Timing test done: $(date)"
echo "  Checkpoint at: $LOG_DIR"
echo "  → Note per-epoch time from log, then decide on full sequential run."
echo "========================================"
