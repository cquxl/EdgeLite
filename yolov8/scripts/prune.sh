SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# CUDA_VISIBLE_DEVICES=0 python main_prune.py \
#     --weight ./weights/yolov8s-pose.pt \
#     --iterative_steps 16 \
#     --output_dir output/yolov8s-pose-prune-sp0.5 \
#     --target_prune_rate 0.5 \
#     --batch_size 16 \
#     --epochs 100 \
#     --fine_tune

CUDA_VISIBLE_DEVICES=0 python main_prune.py \
    --weight ./weights/yolov8s-pose.pt \
    --iterative_steps 1 \
    --output_dir output/yolov8s-pose-prune-sp0.5-iter1-no-finetune \
    --target_prune_rate 0.5 \
    --batch_size 32 \
    --epochs 300 \
    --fine_tune \
    --distillation
