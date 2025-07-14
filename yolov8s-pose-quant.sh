
# ptq
## yolo export
# CUDA_VISIBLE_DEVICES=0 python main_quant.py \
#     --weight weights/yolov8s-pose.pt \
#     --onnx_path weights/yolov8s-pose.onnx \
#     --engine_path weights/yolov8s-pose-ptq-yolo.engine \
#     --cali_data_path datasets/coco-pose/images/train2017 \
#     --cali_size 5000 \
#     --output_dir output/yolov8s-pose-ptq-yolo \
#     --quant ptq \
#     --batch_size 16 \
#     --export yolo \
#     --eval
## trt build engine
# CUDA_VISIBLE_DEVICES=0 python main_quant.py \
#     --weight weights/yolov8s-pose.pt \
#     --onnx_path weights/yolov8s-pose.onnx \
#     --engine_path weights/yolov8s-pose-ptq-build.engine \
#     --cali_data_path datasets/coco-pose/images/train2017 \
#     --cali_size 5000 \
#     --output_dir output/yolov8s-pose-ptq-build \
#     --quant ptq \
#     --batch_size 1 \
#     --export build \
#     --eval

## trtexec export
# CUDA_VISIBLE_DEVICES=0 python main_quant.py \
#     --weight weights/yolov8s-pose.pt \
#     --onnx_path weights/yolov8s-pose.onnx \
#     --engine_path weights/yolov8s-pose-ptq-trtexec.engine \
#     --cali_data_path datasets/coco-pose/images/train2017 \
#     --cali_size 5000 \
#     --output_dir output/yolov8s-pose-ptq-trtexec \
#     --quant ptq \
#     --batch_size 16 \
#     --export trtexec \
#     --eval




# prune-ptq
# CUDA_VISIBLE_DEVICES=2 python main_quant.py \
#     --weight weights/yolov8s-pose-prune.pt \
#     --onnx_path weights/yolov8s-pose-prune.onnx \
#     --engine_path output/yolov8s-pose-prune-sp0.5-epoch60/step_18_finetune/weights/best.engine \
#     --cali_data_path datasets/coco-pose/images/train2017 \
#     --cali_size 5000 \
#     --output_dir output/yolov8s-pose-prune-ptq \
#     --quant ptq \
#     --batch_size 16\
#     --eval


# # qat
CUDA_VISIBLE_DEVICES=0 python main_quant.py \
    --weight weights/yolov8s-pose.pt \
    --train_img_path datasets/coco-pose/images/train2017 \
    --val_img_path datasets/coco-pose/images/val2017 \
    --onnx_path weights/yolov8s-pose-qat.onnx \
    --engine_path weights/yolov8s-pose-qat.engine \
    --epochs 30 \
    --output_dir output/yolov8s-pose-qat \
    --save_qat weights/yolov8s-pose-qat.pt \
    --quant qat \
    --batch_size 16 \
    --eval


# prune-qat

# CUDA_VISIBLE_DEVICES=2 python main_quant.py \
#     --weight ./output/yolov8s-pose-prune-sp0.5-epoch60/step_19_finetune/weights/best.pt \
#     --train_img_path datasets/coco-pose/images/train2017 \
#     --val_img_path datasets/coco-pose/images/val2017 \
#     --epochs 50 \
#     --output_dir output/yolov8s-pose-prune-qat-epochs50 \
#     --save_qat weights/yolov8s-pose-prune-qat.pt \
#     --quant qat \
#     --batch_size 16 \
#     --eval

