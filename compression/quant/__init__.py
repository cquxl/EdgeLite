from .ptq import YOLOv8PosePTQ
from .qat import YOLOv8PoseQAT
from .qat import YOLOv8PoseDataLoader
from .qat import my_export_onnx
import torch

class YOLOv8PoseQuant:
    def __init__(self, args, yolo_cfg):
        self.args = args
        self.cfg = yolo_cfg
        self.logger = args.logger

        if self.args.quant == "ptq":
            self.Quant = YOLOv8PosePTQ(args, self.args.weight, self.args.onnx_path, self.args.engine_path)
        elif self.args.quant == "qat":
            self.Quant = YOLOv8PoseQAT(args, self.cfg, self.args.train_dataloader, self.args.val_dataloader)
        else:
            raise NotImplementedError

    def quant(self):
        if self.args.quant == "ptq":
            self.logger.info("PTQ quantization start and build engine")
            self.Quant.build_engine()
            del self.Quant
            torch.cuda.empty_cache()

        elif self.args.quant == "qat":
            self.logger.info("QAT quantization start and finetune")
            self.Quant.qat_finetune()
            self.Quant.build_engine()
            pass

        else:
            raise NotImplementedError

