
import torch
from ultralytics import YOLO
from loguru import logger
import os



def eval_engine(args, engine_path, eval_data_yaml):
    # 显式绑定设备
    result = {}
    args.logger.info(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = YOLO(engine_path, task=args.task)
    args.logger.info(model)
    rect = False if args.dynamic else True
    metrics = model.val(data=eval_data_yaml, rect=rect, device=args.device)
    args.logger.info(metrics)
    args.logger.info(f"box map50-95:{metrics.box.map}, box map50:{metrics.box.map50}"
                f"pose map50-95:{metrics.pose.map}, pose map50:{metrics.pose.map50}")
    args.logger.info(f"engine speed->{metrics.speed}")
    result["map50"] = metrics.pose.map50
    result["speed"] = metrics.speed
    return result


