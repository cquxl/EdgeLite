import torch
import os
import sys
import json
from compression import YOLOv8PoseQuant, YOLOv8PoseDataLoader
from utils import setup_logger, eval_engine
from cfg import quant_args

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import init_seeds

data_yaml_file = './datasets/my-coco-pose.yaml' # your dataset config file
yolo_cfg_yaml = 'ultralytics/cfg/default.yaml' # yolo default training parameters

args = quant_args() # qat args
args.logger = setup_logger(f'quant_{args.quant}', args.output_dir)
args.data_yaml_file = data_yaml_file

my_cfg = {'model': args.weight,
          'data': data_yaml_file,
          'task': args.task,  # pose
          'batch': args.batch_size,
          'imgsz': args.imgsz,
          'device': args.device,
          'epochs': args.epochs}

yolo_cfg = get_cfg(yolo_cfg_yaml, my_cfg)

def main():
    init_seeds(args.seed)
    args.logger.info(args)
    if args.quant == "qat":
        args.train_dataloader = YOLOv8PoseDataLoader(args, yolo_cfg, args.train_img_path,
                                                     args.batch_size, mode="train",
                                                     cache_dir=args.cache_dir).dataloader
        args.val_dataloader = YOLOv8PoseDataLoader(args, yolo_cfg, args.val_img_path,
                                                   args.batch_size, mode="val",
                                                   cache_dir=args.cache_dir).dataloader
    # if not os.path.exists(args.engine_path):
    quant = YOLOv8PoseQuant(args, yolo_cfg)
    quant.quant()
    torch.cuda.empty_cache()

    if args.eval and os.path.exists(args.engine_path):
        if args.quant == 'ptq' and args.export != 'yolo':
            args.logger.info(f"you should eval in another script or only eval for existed engine (no quant), quant and eval at the same time is not supported")
            sys.exit(1)
        result = eval_engine(args, args.engine_path, data_yaml_file) # dict
        log = {
            "engine": args.engine_path,
            "result": result
        }
        save_path = os.path.join(args.output_dir, "eval_result.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4, ensure_ascii=False)
        args.logger.info(f"✅ saving result to {save_path}")



if __name__ == '__main__':
    main()