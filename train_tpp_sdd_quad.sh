#!/bin/bash
WORK=/cs/student/projects1/2023/muhamaaz/year-long
TPP=$WORK/Trajectron-plus-plus/trajectron
source "$WORK/crowdnav-env/bin/activate"
cd "$TPP"
CONF=$WORK/Trajectron-plus-plus/experiments/pedestrians/models/eth_attention_radius_3/config.json
PROC=$WORK/Trajectron-plus-plus/experiments/processed
LOG_DIR=$WORK/Trajectron-plus-plus/experiments/logs_sdd

echo "=== Trajectron++ SDD: quad === $(date)"
python -u train.py \
    --conf "$CONF" \
    --train_data_dict "sdd_quad_train.pkl" \
    --eval_data_dict  "sdd_quad_test.pkl" \
    --data_dir        "$PROC" \
    --log_dir         "$LOG_DIR" \
    --log_tag         "quad" \
    --train_epochs    100 \
    --eval_every      10 \
    --save_every      10 \
    --batch_size      64 \
    --k_eval          20 \
    --offline_scene_graph yes \
    --preprocess_workers 0 \
    --device          cuda:0 \
    --eval_device     cpu
echo "Done: quad $(date)"
