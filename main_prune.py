import torch
import os
from utils import setup_logger, eval_engine
from cfg import prune_args
from compression import YOLOv8PosePrune
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import init_seeds
import sys
import os
import json

data_yaml_file = './datasets/my-coco-pose.yaml' # your dataset config file
yolo_cfg_yaml = './ultralytics/cfg/default.yaml' # training parameters
args = prune_args() # qat args
args.logger = setup_logger('prune', args.output_dir)

my_cfg = {'model': args.weight,
          'data': data_yaml_file,
          'task': args.task,  # pose
          'batch': args.batch_size,
          'imgsz': args.imgsz,
          'device': args.device,
          'epochs': args.epochs,
          'amp': True,
          'project': args.project if args.project else args.output_dir,
          }
yolo_cfg = get_cfg(yolo_cfg_yaml, my_cfg)

def main():
    init_seeds(args.seed)
    args.logger.info(args)
    pruner = YOLOv8PosePrune(args, yolo_cfg)
    pruner.prune()
    if args.eval(): # test fp16 dynamic engine
        pruner.build_engine(iter=None) # or iter< args.iterative_steps
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
