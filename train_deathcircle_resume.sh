#!/bin/bash
# Resume Trajectron++ for deathCircle from epoch 30 checkpoint.
set -e
WORK=/cs/student/projects1/2023/muhamaaz/year-long
TPP=$WORK/Trajectron-plus-plus/trajectron
source "$WORK/crowdnav-env/bin/activate"
cd "$TPP"

CONF=$WORK/Trajectron-plus-plus/experiments/pedestrians/models/eth_attention_radius_3/config.json
PROC=$WORK/Trajectron-plus-plus/experiments/processed
LOG_DIR=$WORK/Trajectron-plus-plus/experiments/logs_sdd
RESUME_DIR=$LOG_DIR/models_15_May_2026_01_59_20deathCircle

echo "Resuming Trajectron++ deathCircle from epoch 30  $(date)"

python -u train.py \
    --conf "$CONF" \
    --train_data_dict "sdd_deathCircle_train.pkl" \
    --eval_data_dict  "sdd_deathCircle_test.pkl" \
    --data_dir        "$PROC" \
    --log_dir         "$LOG_DIR" \
    --log_tag         "deathCircle_resume" \
    --train_epochs    70 \
    --eval_every      10 \
    --save_every      10 \
    --batch_size      256 \
    --k_eval          20 \
    --offline_scene_graph yes \
    --preprocess_workers 0 \
    --device cuda:0 \
    --eval_device cpu \
    --load_model_dir  "$RESUME_DIR" \
    --load_model_epoch 30

echo "Done  $(date)"
