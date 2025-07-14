

CUDA_VISIBLE_DEVICES=2 python main_prune.py \
    --weight ./weights/yolov8s-pose.pt \
    --iterative_steps 16 \
    --output_dir output/yolov8s-pose-prune-sp0.3-epochs120 \
    --target_prune_rate 0.3 \
    --batch_size 16 \
    --epochs 120 \
    --fine_tune

CUDA_VISIBLE_DEVICES=3 python main_prune.py \
    --weight ./weights/yolov8s-pose.pt \
    --iterative_steps 16 \
    --output_dir output/yolov8s-pose-prune-sp0.4-epochs120 \
    --target_prune_rate 0.4 \
    --batch_size 16 \
    --epochs 120 \
    --fine_tune

CUDA_VISIBLE_DEVICES=4 python main_prune.py \
    --weight ./weights/yolov8s-pose.pt \
    --iterative_steps 16 \
    --output_dir output/yolov8s-pose-prune-sp0.5-epochs120 \
    --target_prune_rate 0.5 \
    --batch_size 16 \
    --epochs 120 \
    --fine_tune


# CUDA_VISIBLE_DEVICES=3 python main_prune.py \
#     --weight ./weights/yolov8s-pose.pt \
#     --iterative_steps 20 \
#     --output_dir output/yolov8s-pose-prune-sp0.3-epoch60 \
#     --target_prune_rate 0.3 \
#     --batch_size 64 \
#     --epochs 60 \
#     --fine_tune

# CUDA_VISIBLE_DEVICES=4 python main_prune.py \
#     --weight ./weights/yolov8s-pose.pt \
#     --iterative_steps 20 \
#     --output_dir output/yolov8s-pose-prune-sp0.5-epoch70-p1 \
#     --target_prune_rate 0.5 \
#     --p 1 \
#     --batch_size 64 \
#     --epochs 70 \
#     --fine_tune








