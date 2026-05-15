torchrun --nnodes 1 --nproc_per_node=8 --rdzv_id=15233--rdzv_backend=c10d causvid/train_distillation.py --config_path configs/multi_lora/train_demo.yaml
