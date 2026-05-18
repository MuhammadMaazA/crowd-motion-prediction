#!/bin/bash
# finetune_eth_sequential.sh
# ==========================
# Sequential fine-tuning: SDD pretrained → ETH/UCY
# ===================================================
# This script runs AFTER train_tpp_sdd_timing.sh completes and acts as the
# single sequential pipeline for all SDD→ETH/UCY fine-tuning experiments.
#
# Order of operations:
#   1. Fine-tune Trajectron++ (uses hyang timing-test checkpoint)
#   2. Fine-tune Social-LSTM+V
#   3. Fine-tune Transformer
#   4. Fine-tune Diffusion (lambda_ddpm=0.3, ddim_steps=50)
#   5. Fine-tune GRU-v2
#   6. Fine-tune Social-LSTM (base)
#
# Fine-tuning hyper-params:
#   epochs    : 50  (vs 200 from scratch — early stopping will trigger earlier)
#   lr        : 1e-4 (1/10 of original 1e-3)
#   eval_every: 5
#
# Checkpoints saved to: checkpoints/ft_{model}_{scene}.pt
# T++ fine-tuned checkpoints: experiments/logs_sdd_ftethucy/
#
# Launch on RTX 3090:
#   nohup bash finetune_eth_sequential.sh > logs/finetune_eth.log 2>&1 &
#   tail -f logs/finetune_eth.log

set -e
WORK=/cs/student/projects1/2023/muhamaaz/year-long
TPP=$WORK/Trajectron-plus-plus/trajectron
SDD_CKPT=$WORK/checkpoints/sdd
TIMING_LOG_DIR=$WORK/Trajectron-plus-plus/experiments/logs_sdd_timing

source "$WORK/crowdnav-env/bin/activate"
mkdir -p "$WORK/logs"

SCENES=(eth hotel univ zara1 zara2)
CONF=$WORK/Trajectron-plus-plus/experiments/pedestrians/models/eth_attention_radius_3/config.json
PROC=$WORK/Trajectron-plus-plus/experiments/processed
TPP_FT_LOG=$WORK/Trajectron-plus-plus/experiments/logs_sdd_ftethucy

echo "=============================================="
echo "  SDD → ETH/UCY fine-tuning pipeline"
echo "  Started: $(date)"
echo "=============================================="

# ─────────────────────────────────────────────────────────────────────────────
# 1. Trajectron++ fine-tuning (SDD hyang pretrained → each ETH/UCY split)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Step 1: Trajectron++ fine-tuning ──"

# Find the latest hyang timing checkpoint directory
HYANG_CKPT_DIR=$(ls -dt "$TIMING_LOG_DIR"/models_*hyang_timing 2>/dev/null | head -1)
if [[ -z "$HYANG_CKPT_DIR" ]]; then
    echo "WARNING: No Trajectron++ hyang timing checkpoint found in $TIMING_LOG_DIR"
    echo "         Skipping T++ fine-tuning. Run train_tpp_sdd_timing.sh first."
else
    # Use the highest saved epoch
    HYANG_EPOCH=$(ls "$HYANG_CKPT_DIR"/model_registrar-*.pt 2>/dev/null \
                  | sed 's/.*model_registrar-\([0-9]*\)\.pt/\1/' \
                  | sort -n | tail -1)
    if [[ -z "$HYANG_EPOCH" ]]; then
        echo "  WARNING: No saved epoch found in $HYANG_CKPT_DIR (timing test may not have reached epoch 10)"
        echo "  Skipping T++ fine-tuning."
        HYANG_CKPT_DIR=""
    else
        echo "  Using checkpoint: $HYANG_CKPT_DIR  epoch=$HYANG_EPOCH"
    fi

    if [[ -n "$HYANG_CKPT_DIR" ]]; then
        cd "$TPP"
        for scene in "${SCENES[@]}"; do
        echo ""
        echo "  T++ fine-tune → $scene  ($(date))"
        python -u train.py \
            --conf          "$CONF" \
            --train_data_dict "${scene}_train.pkl" \
            --eval_data_dict  "${scene}_val.pkl" \
            --data_dir        "$PROC" \
            --log_dir         "$TPP_FT_LOG" \
            --log_tag         "ft_${scene}" \
            --train_epochs    50 \
            --eval_every      5 \
            --save_every      10 \
            --batch_size      256 \
            --k_eval          20 \
            --offline_scene_graph yes \
            --preprocess_workers 0 \
            --device cuda:0 \
            --eval_device cpu \
            --load_model_dir   "$HYANG_CKPT_DIR" \
            --load_model_epoch "$HYANG_EPOCH"
        echo "  T++ fine-tune $scene done  ($(date))"
        done
        cd "$WORK"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# 2. Social-LSTM+V fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
# Pretrain source: SDD hyang checkpoint (same scene used for T++ timing test)
SDD_SRC=hyang
echo ""
echo "── Step 2: Social-LSTM+V fine-tuning (pretrain: SDD/$SDD_SRC) ──"
SLSTMV_CKPT="$SDD_CKPT/social_lstm_v_${SDD_SRC}.pt"
if [[ ! -f "$SLSTMV_CKPT" ]]; then
    echo "  WARNING: $SLSTMV_CKPT not found, skipping SLSTM+V"
else
    for scene in "${SCENES[@]}"; do
        echo "  SLSTM+V → $scene  ($(date))"
        python -u models/train_social_lstm.py \
            --holdout "$scene" --epochs 50 --lr 1e-4 --eval_every 5 \
            --velocity --pretrain_ckpt "$SLSTMV_CKPT" --device cuda:0
        echo "  SLSTM+V $scene done  ($(date))"
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# 3. Transformer fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Step 3: Transformer fine-tuning (pretrain: SDD/$SDD_SRC) ──"
TRANSF_CKPT="$SDD_CKPT/transformer_${SDD_SRC}.pt"
if [[ ! -f "$TRANSF_CKPT" ]]; then
    echo "  WARNING: $TRANSF_CKPT not found, skipping Transformer"
else
    for scene in "${SCENES[@]}"; do
        echo "  Transformer → $scene  ($(date))"
        python -u models/train_trajectory_transformer.py \
            --holdout "$scene" --epochs 50 --lr 1e-4 --eval_every 5 \
            --pretrain_ckpt "$TRANSF_CKPT" --device cuda:0
        echo "  Transformer $scene done  ($(date))"
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# 4. Diffusion fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Step 4: Diffusion fine-tuning (pretrain: SDD/$SDD_SRC) ──"
DIFF_CKPT="$SDD_CKPT/diffusion_${SDD_SRC}.pt"
if [[ ! -f "$DIFF_CKPT" ]]; then
    echo "  WARNING: $DIFF_CKPT not found, skipping Diffusion"
else
    for scene in "${SCENES[@]}"; do
        echo "  Diffusion → $scene  ($(date))"
        python -u models/train_diffusion.py \
            --holdout "$scene" --epochs 50 --lr 1e-4 --eval_every 5 \
            --lambda_ddpm 0.3 --ddim_steps 50 \
            --pretrain_ckpt "$DIFF_CKPT" --device cuda:0
        echo "  Diffusion $scene done  ($(date))"
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# 5. GRU-v2 fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Step 5: GRU-v2 fine-tuning (pretrain: SDD/$SDD_SRC) ──"
GRU_CKPT="$SDD_CKPT/gru_v2_${SDD_SRC}.pt"
if [[ ! -f "$GRU_CKPT" ]]; then
    echo "  WARNING: $GRU_CKPT not found, skipping GRU-v2"
else
    for scene in "${SCENES[@]}"; do
        echo "  GRU-v2 → $scene  ($(date))"
        python -u models/train_social_gru.py \
            --holdout "$scene" --epochs 50 --lr 1e-4 --eval_every 5 \
            --v2 --pretrain_ckpt "$GRU_CKPT" --device cuda:0
        echo "  GRU-v2 $scene done  ($(date))"
    done
fi

# ─────────────────────────────────────────────────────────────────────────────
# 6. Social-LSTM (base) fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── Step 6: Social-LSTM (base) fine-tuning (pretrain: SDD/$SDD_SRC) ──"
SLSTM_CKPT="$SDD_CKPT/social_lstm_${SDD_SRC}.pt"
if [[ ! -f "$SLSTM_CKPT" ]]; then
    echo "  WARNING: $SLSTM_CKPT not found, skipping SLSTM"
else
    for scene in "${SCENES[@]}"; do
        echo "  SLSTM → $scene  ($(date))"
        python -u models/train_social_lstm.py \
            --holdout "$scene" --epochs 50 --lr 1e-4 --eval_every 5 \
            --pretrain_ckpt "$SLSTM_CKPT" --device cuda:0
        echo "  SLSTM $scene done  ($(date))"
    done
fi

echo ""
echo "=============================================="
echo "  All fine-tuning done! $(date)"
echo "  Fine-tuned checkpoints:"
echo "    Non-T++  : checkpoints/ft_*_{scene}.pt"
echo "    T++      : $TPP_FT_LOG"
echo "=============================================="
