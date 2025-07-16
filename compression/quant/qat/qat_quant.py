from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.utils import clean_url, emojis
import os
import torch
import torch.nn as nn

from ultralytics.nn.tasks import PoseModel
from ultralytics.nn.modules import Conv
from .quantize import *
from .utils import SummaryTool
from copy import deepcopy
from ultralytics.models import yolo
from .utils import my_export_onnx
from  ultralytics import YOLO
import subprocess
import pytorch_quantization.nn as quant_nn
from pytorch_quantization.nn import TensorQuantizer  # 来自你QAT框架（如 MQBench 等）
import onnx
from onnxsim import simplify

class YOLOv8PoseDataLoader:
    def __init__(self, args, yolo_cfg, img_path, batch, mode="train", cache_dir='./cache'):
        self.args = args
        self.cfg = yolo_cfg
        self.img_path = img_path
        self.batch = batch
        self.mode = mode
        self.cache_dir = cache_dir
        self.logger = args.logger

        self.dataloader = self._load_dataloader()

    def build_dataset(self):
        dataset_name = self.img_path.split('/')[-3]+ '-' + self.img_path.split('/')[-2] + '-' + self.img_path.split('/')[-1]
        dataset_name = f'{dataset_name}-batch{self.batch}.cache'
        cache_dataset = os.path.join(self.cache_dir, dataset_name)
        if os.path.exists(cache_dataset):
            self.logger.info(f"load cache {self.mode} dataset for QAT YOLOv8s-Pose: {cache_dataset}")
            return torch.load(cache_dataset)
        try:
            if self.cfg.task in ('detect', 'segment', 'pose'):
                data = check_det_dataset(self.cfg.data)  # 如果cfg.data没有找到本地的路径或者配置不正确，他会download数据
                if 'yaml_file' in data:
                    self.cfg.data = data['yaml_file']
                    assert os.path.exists(self.cfg.data)
            elif self.cfg.task == 'classify':
                data = check_cls_dataset(self.cfg.data)
        except:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.cfg.data)}' error ❌ {e}")) from e
        dataset = build_yolo_dataset(
            self.cfg,
            self.img_path,
            self.batch,
            data, # check_det_dataset(cfg.data) return
            mode=self.mode,
            rect=self.mode == 'val'
        )
        torch.save(dataset, cache_dataset)
        self.logger.info(f"finish loading {self.mode} dataset for QAT YOLOv8s-Pose: {cache_dataset}")
        return dataset

    def build_dataloader(self):
        dataset = self.build_dataset()
        shuffle = self.mode == 'train'
        return build_dataloader(dataset, self.batch, self.cfg.workers, shuffle=shuffle)

    def _load_dataloader(self):
        dataloader = self.build_dataloader()
        self.logger.info(f"{self.mode} dataloader for QAT YOLOv8s-Pose loading finished")
        return dataloader


class YOLOv8PoseQAT:
    def __init__(self, args, yolo_cfg, train_dataloader, val_dataloader):
        self.args = args
        self.cfg = yolo_cfg
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.imgsz = args.imgsz
        self.batch = args.batch_size

        self.weight = args.weight
        self.device = args.device

        self.data_yaml_file = args.data_yaml_file

        self.dynamic = args.dynamic
        # self.model = self._load_model()

        self.logger = args.logger
        self.best_ap = 0
        self.onnx_path = args.onnx_path
        self.engine_path = args.engine_path

    def load_model(self)-> PoseModel:
        model = torch.load(self.weight, map_location=self.device)["model"]
        for m in model.modules():
            if type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        # model.data error if args = '/usr/src/app/ultralytics/datasets/coco-pose.yaml'
        model.args = self.cfg  # model.args : 字典类型  改为  cfg: IterableSimpleNamespace类型,
        model.float()
        model.eval()
        with torch.no_grad():
            model.fuse()
        return model

    def qat_finetune(self):
        initialize()
        # load model
        self.model = self.load_model()

        self.logger.info(f"Insert Q/DQ module into model {self.weight}")
        # bottleneck Q/DQ forward
        replace_bottleneck_forward(self.model)
        # other module Q/DQ forward, like Conv2d
        replace_to_quantization_module(self.model, ignore_policy=self.args.ignore_policy)
        try:
            self.logger.info("Apply custom_rules ....")
            apply_custom_rules_to_quantizer(self.model, my_export_onnx)
        except:
            pass
        self.logger.info("Calibrate model ....")
        calibrate_model(self.model, self.train_dataloader, self.device)

        summary_file = os.path.join(self.args.output_dir, "summary.json")
        summary = SummaryTool(summary_file)

        def per_epoch(model, epoch, lr):
            ap = self.evaluate_coco(model)
            summary.append([f"QAT{epoch}", ap])
            if ap > self.best_ap:
                self.logger.info(f"Save qat model to {self.args.save_qat} @ {ap:.5f}")
                self.best_ap = ap
                torch.save({"model": model}, self.args.save_qat)

        def preprocess_batch(batch, device):
            batch['img'] = batch['img'].to(device, non_blocking=True).float() / 255
            return batch

        def supervision_policy():
            supervision_list = []
            for item in self.model.model:
                supervision_list.append(id(item))

            keep_idx = list(range(0, len(self.model.model) - 1, self.args.supervision_stride))
            keep_idx.append(len(self.model.model) - 2)

            def impl(name, module):
                if id(module) not in supervision_list: return False
                idx = supervision_list.index(id(module))
                if idx in keep_idx:
                    print(f"Supervision: {name} will compute loss with origin model during QAT training")
                else:
                    print(
                        f"Supervision: {name} no compute loss during QAT training, that is unsupervised only and doesn't mean don't learn")
                return idx in keep_idx
            return impl
        finetune(self.model, self.train_dataloader, per_epoch, early_exit_batchs_per_epoch=self.args.iters,
                 nepochs=self.args.epochs,
                 preprocess=preprocess_batch, supervision_policy=supervision_policy())

    def evaluate_coco(self, model):
        val_model = deepcopy(model)  # deepcopy
        validator = yolo.pose.PoseValidator(dataloader=self.val_dataloader, args=self.cfg)
        metric = validator(model=val_model)
        # mAP = metric["metrics/mAP50(P)"]
        map_50_95_b = metric["metrics/mAP50-95(B)"]
        map_50_b = metric["metrics/mAP50(B)"]
        map_50_95_p = metric["metrics/mAP50-95(P)"]
        map_50_p = metric["metrics/mAP50(P)"]
        self.logger.info(f"box map50-95:{map_50_95_b}, box map50:{map_50_b}"
                         f"pose map50-95:{map_50_95_p}, pose map50:{map_50_p}")
        return map_50_p


    def build_engine(self):
        # qat pt-->args.save_qat
        def load_model(org_model_path, pt_model_path, device='cuda:0')-> PoseModel:
            model = YOLO(org_model_path)
            model1 = torch.load(pt_model_path, map_location=device)["model"]
            model1.float()
            model1.eval()
            with torch.no_grad():
                model1.fuse()
            model.model = model1
            model.args = vars(model.args)
            model.model.args = model.args
            model.model.task = model.task
            return model

        def enable_all_fake_quant(module):
            for m in module.modules():
                if isinstance(m, quant_nn.TensorQuantizer):
                    m._fake_quant = True

        def validate_onnx(file):
            model = onnx.load(file)
            model_simp, check = simplify(
                model,
                skip_fuse_bn=True,
                test_input_shapes={"images": [1, 3, 640, 640]},
            )
            onnx.save(model_simp, file)
            self.logger.info(f"onnx validate finished , save to: {file}")
        qat_model = load_model(self.weight, self.args.save_qat)
        enable_all_fake_quant(qat_model)

        qat_model.export(format='onnx', dynamic=self.dynamic, imgsz=self.imgsz, verbose=False, batch=self.batch, workspace=2)
        # export onnx, dynamic batch, shape fix
        onnx_path = qat_model.export(format="onnx", dynamic=True, imgsz=self.args.imgsz, batch=self.args.batch_size)
        print(onnx_path)
        os.rename(onnx_path, self.onnx_path) # weights/yolov8-pose-qat.onnx
        validate_onnx(self.onnx_path)
        # onnx转trt
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            if not os.path.exists(self.engine_path):
                self.logger.info(f"export int8 engine by trtexec")
                subprocess.run([
                    "trtexec",
                    "--onnx=" + self.onnx_path,
                    "--int8",
                    "--fp16",
                    "--saveEngine=" + self.engine_path,
                    "--minShapes=images:1x3x640x640",
                    f"--optShapes=images:{self.batch}x3x640x640",
                    f"--maxShapes=images:{self.batch}x3x640x640",
                    "--verbose"
                ], check=True)
                if os.path.exists(self.engine_path):
                    self.logger.info(f"build engine success: {self.engine_path}")
                    return True
                else:
                    self.logger.info("build engine failed!")
                    return False
            else:
                self.logger.info(f"engine exists: {self.engine_path}, please eval on coco-pose")
                return True
        except Exception as e:
            self.logger.info("build engine failed!", e)
            return False

























