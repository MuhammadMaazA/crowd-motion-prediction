#!/bin/bash
# Train Trajectron++ on SDD coupa (leave-one-out holdout)
# Launch: nohup bash train_tpp_sdd_coupa.sh > logs/tpp_sdd_coupa.log 2>&1 &

set -e
WORK=/cs/student/projects1/2023/muhamaaz/year-long
TPP=$WORK/Trajectron-plus-plus/trajectron
source "$WORK/crowdnav-env/bin/activate"
cd "$TPP"

CONF=$WORK/Trajectron-plus-plus/experiments/pedestrians/models/eth_attention_radius_3/config.json
PROC=$WORK/Trajectron-plus-plus/experiments/processed
LOG_DIR=$WORK/Trajectron-plus-plus/experiments/logs_sdd

mkdir -p "$WORK/logs" "$LOG_DIR"

echo "========================================"
echo "  Trajectron++ SDD: coupa"
echo "  Device: cuda:0"
echo "  Started: $(date)"
echo "========================================"

python -u train.py \
    --conf            "$CONF" \
    --train_data_dict "sdd_coupa_train.pkl" \
    --eval_data_dict  "sdd_coupa_test.pkl" \
    --data_dir        "$PROC" \
    --log_dir         "$LOG_DIR" \
    --log_tag         "coupa" \
    --train_epochs    100 \
    --eval_every      10 \
    --save_every      10 \
    --batch_size      256 \
    --k_eval          20 \
    --offline_scene_graph yes \
    --preprocess_workers 1 \
    --device cuda:0 \
    --eval_device cpu

echo "========================================"
echo "  coupa done: $(date)"
echo "========================================"
