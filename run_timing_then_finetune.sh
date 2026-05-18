#!/bin/bash
# Wrapper: T++ SDD timing test, then full SDD->ETH/UCY fine-tuning pipeline
set -e
WORK=/cs/student/projects1/2023/muhamaaz/year-long
cd "$WORK"
echo "=== Phase 1: T++ SDD timing test (hyang) ==="
bash train_tpp_sdd_timing.sh
echo "=== Phase 2: SDD -> ETH/UCY fine-tuning ==="
bash finetune_eth_sequential.sh
