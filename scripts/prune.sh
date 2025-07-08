CUDA_VISIBLE_DEVICES=0 python main_prune.py \
    --weight ./weights/yolov8s-pose.pt \
    --iterative_steps 16 \
    --output_dir output/yolov8s-pose-prune-sp0.5 \
    --target_prune_rate 0.5 \
    --batch_size 16 \
    --epochs 120 \
    --fine_tune