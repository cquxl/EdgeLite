SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

CUDA_VISIBLE_DEVICES=0 python main_quant.py \
    --weight ./weights/yolov8s-pose-qat-distill100.pt \
    --train_img_path datasets/coco-pose/images/train2017 \
    --val_img_path datasets/coco-pose/images/val2017 \
    --onnx_path weights/yolov8s-pose-qat-distill100.onnx \
    --engine_path weights/yolov8s-pose-qat-distill100.engine \
    --epochs 30 \
    --output_dir output/yolov8s-pose-qat-distill100-qat \
    --save_qat weights/olov8s-pose-qat-distill100-qat.pt \
    --quant qat \
    --batch_size 16
