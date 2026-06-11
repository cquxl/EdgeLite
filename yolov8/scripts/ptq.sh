SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

 CUDA_VISIBLE_DEVICES=0 python main_quant.py \
     --weight weights/yolov8s-pose.pt \
     --onnx_path weights/yolov8s-pose.onnx \
     --engine_path weights/yolov8s-pose-ptq-build.engine \
     --cali_data_path datasets/coco-pose/images/train2017 \
     --cali_size 5000 \
     --output_dir output/yolov8s-pose-ptq \
     --quant ptq \
     --batch_size 1 \
     --export build 
