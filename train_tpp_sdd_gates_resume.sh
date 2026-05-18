#!/bin/bash
# Resume Trajectron++ SDD training for gates scene from epoch 20.
WORK=/cs/student/projects1/2023/muhamaaz/year-long
TPP=$WORK/Trajectron-plus-plus/trajectron
source "$WORK/crowdnav-env/bin/activate"
cd "$TPP"

CONF=$WORK/Trajectron-plus-plus/experiments/pedestrians/models/eth_attention_radius_3/config.json
PROC=$WORK/Trajectron-plus-plus/experiments/processed
LOG_DIR=$WORK/Trajectron-plus-plus/experiments/logs_sdd

# Latest gates checkpoint dir (epoch 20)
LOAD_DIR=$LOG_DIR/models_15_May_2026_01_59_26gates

echo "=== Trajectron++ SDD gates (resume from ep20) === $(date)"
python -u train.py \
    --conf "$CONF" \
    --train_data_dict "sdd_gates_train.pkl" \
    --eval_data_dict  "sdd_gates_test.pkl" \
    --data_dir        "$PROC" \
    --log_dir         "$LOG_DIR" \
    --log_tag         "gates_resume" \
    --train_epochs    80 \
    --eval_every      10 \
    --save_every      10 \
    --batch_size      64 \
    --k_eval          20 \
    --offline_scene_graph no \
    --preprocess_workers 0 \
    --device          cuda:0 \
    --eval_device     cpu \
    --load_model_dir  "$LOAD_DIR" \
    --load_model_epoch 20
echo "=== gates done $(date) ==="
