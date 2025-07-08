import argparse



def quant_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-q', '--quant', type=str, default="ptq",
                        help='quantization mode: ptq or qat', choices=['ptq', 'qat'])
    parser.add_argument('--dynamic', type=bool, default=True,
                        help='dynamic input shape (i.e. batch size, height, width) for onnx/engine transform')
    parser.add_argument('--export', type=str, default="yolo",
                        help='export engine method for ptq', choices=['yolo', 'build', 'trtexec'])
    # ptq
    parser.add_argument('--cali_data_path', type=str, default="datasets/coco-pose/images/val2017",
                        help='calibration dataset dir->coco-pose/images/val2017 or train2017')
    parser.add_argument('--cali_size', type=int, default=1000,
                        help='calibration dataset size->image number')

    # qat
    # calibration dataset
    parser.add_argument('--train_img_path', type=str, default="datasets/coco-pose/images/train2017",
                        help='train dataset dir->coco-pose/images/train2017')
    parser.add_argument('--val_img_path', type=str, default="datasets/coco-pose/images/val2017",
                        help='val dataset dir->coco-pose/images/val2017')

    parser.add_argument('--input_shape', type=tuple, default=(640, 640),
                        help='dataset shape-->image (h,w) resize shape')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='dataset image size-->image (h,w) resize shape->default 640')
    parser.add_argument('--epochs', type=int, default=20,
                        help='qat training epochs')
    parser.add_argument('--task', type=str, default='pose',
                        help='yolov8 task', choices=['detect', 'segment', 'pose', 'classify'])

    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='cache dir for dataset load or trt calibration cache')
    parser.add_argument('--output_dir', type=str, default='./output/yolov8s-pose-qat',
                        help='output results dir')

    # onnx
    parser.add_argument('--onnx_path', type=str, default=None,
                        help='model onnx path, e,g., weights/yolov8s-pose-qat.onnx')

    parser.add_argument('--weight', type=str, default=None,
                        help='initial weight path, e,g., weights/yolov8s-pose.pt')

    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    # trt
    parser.add_argument('--engine_path', type=str, default=None,
                        help='model trt engine path, e,g., weights/yolov8s-pose-qat.engine')

    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--seed", type=int, default=0, help="seed for torch/numpy")

    parser.add_argument("--eval", action="store_true", help="eval map50 on coco-pose val 2017")

    # finetue
    parser.add_argument("--ignore-policy", type=str, default="model\.24\.m\.(.*)", help="regx")
    parser.add_argument("--supervision-stride", type=int, default=1, help="supervision stride")
    parser.add_argument("--iters", type=int, default=200, help="iters per epoch")
    parser.add_argument("--save_qat", type=str, default='weights/yolov8s-pose-qat.pt', help="file, qat.pt")

    args = parser.parse_args()
    return args

def prune_args():
    parser = argparse.ArgumentParser()
    # ultralytics yolo training
    parser.add_argument('--project', type=str, default=None,
                        help='yolov8s-pose training project')
    parser.add_argument('--name', type=str, default=None,
                        help='yolov8s-pose training name')
    parser.add_argument('--dynamic', type=bool, default=True,
                        help='dynamic input shape (i.e. batch size, height, width) for onnx/engine transform')

    parser.add_argument('--input_shape', type=tuple, default=(640, 640),
                        help='dataset shape-->image (h,w) resize shape')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='dataset image size-->image (h,w) resize shape->default 640')
    parser.add_argument('--int8', type=bool, default=False,
                        help='whether to train with int8')
    parser.add_argument('--epochs', type=int, default=20,
                        help='qat training epochs')
    parser.add_argument('--task', type=str, default='pose',
                        help='yolov8 task', choices=['detect', 'segment', 'pose', 'classify'])

    parser.add_argument('--cache_dir', type=str, default='./cache',
                        help='cache dir for dataset load or trt calibration cache')
    parser.add_argument('--output_dir', type=str, default='./output/yolov8s-pose-prune',
                        help='output results dir')

    # onnx
    parser.add_argument('--onnx_path', type=str, default=None,
                        help='model onnx path, e,g., weights/yolov8s-pose-prune.onnx')

    parser.add_argument('--weight', type=str, default=None,
                        help='initial weight path, e,g., weights/yolov8s-pose.pt')

    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    # trt
    parser.add_argument('--engine_path', type=str, default=None,
                        help='model trt engine path, e,g., weights/yolov8s-pose-prune.engine')

    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--seed", type=int, default=0, help="seed for torch/numpy")

    parser.add_argument("--eval", action="store_true", help="eval map50 on coco-pose val 2017")

    # finetue
    # parser.add_argument('--model', default='yolov8m.pt', help='Pretrained pruning target model file')
    # parser.add_argument('--cfg', default='default.yaml',
    #                     help='Pruning config file.'
    #                          ' This file should have same format with ultralytics/yolo/cfg/default.yaml')
    parser.add_argument('--iterative_steps', default=16, type=int, help='Total pruning iteration step') # epochs
    parser.add_argument('--target_prune_rate', default=0.3, type=float, help='Target pruning rate')
    parser.add_argument('--max_map_drop', default=0.2, type=float, help='Allowed maximum map drop after fine-tuning')
    parser.add_argument('--fine_tune', action="store_true", help='whether to fine tune')
    parser.add_argument("--p", type=int, default=2, help="the norm degree for pruning importance")
    args = parser.parse_args()
    return args
