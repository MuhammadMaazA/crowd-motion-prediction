#!/bin/bash
# Train Trajectron++ on all 8 SDD holdout scenes.
# Must run process_sdd_trajectron.py first.
# Launch: nohup bash train_trajectron_sdd.sh >& logs/tpp_sdd_train.log &

set -e
WORK=/cs/student/projects1/2023/muhamaaz/year-long
TPP=$WORK/Trajectron-plus-plus/trajectron
source "$WORK/crowdnav-env/bin/activate"
cd "$TPP"

CONF=$WORK/Trajectron-plus-plus/experiments/pedestrians/models/eth_attention_radius_3/config.json
PROC=$WORK/Trajectron-plus-plus/experiments/processed

SCENES=(bookstore coupa deathCircle gates hyang little nexus quad)

for scene in "${SCENES[@]}"; do
    echo "========================================"
    echo "  Trajectron++ SDD: $scene"
    echo "  Started: $(date)"
    echo "========================================"

    LOG_DIR=$WORK/Trajectron-plus-plus/experiments/logs_sdd

    python -u train.py \
        --conf "$CONF" \
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
        --offline_scene_graph yes \
        --preprocess_workers 0 \
        --device cuda:0 \
        --eval_device cpu

    echo "  Finished $scene at $(date)"
done

echo "All SDD Trajectron++ training done! $(date)"
