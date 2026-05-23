#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc_per_node=4 --rdzv_id=15233 --rdzv_backend=c10d causvid/train_distillation.py --config_path configs/multi_lora/train_demo.yaml
