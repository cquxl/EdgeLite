import os
from ultralytics import YOLO
import torch
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat # 蒸馏+量化
from ultralytics.cfg import get_cfg
from compression.prune.utils import save_model_v2, final_eval_v2, strip_optimizer_v2, train_v2, replace_c2f_with_c2f_v2
from ultralytics.utils.torch_utils import initialize_weights
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
device = "cuda:0"

def load_model1(weight, device="cuda:0"):
    model = YOLO(weight, task='pose')
    # ✅ 添加防止 fallback 的关键字段
    model.pt_path = weight
    if not hasattr(model, 'args'):
        model.args = {"model": weight}  # 防止 AMP fallback
    model.__setattr__("train_v2", train_v2.__get__(model))
    model.model.train()
    replace_c2f_with_c2f_v2(model.model)
    initialize_weights(model.model)  # set BN.eps, momentum, ReLU.inplace
    for name, param in model.model.named_parameters():
        param.requires_grad = True
    model.to(device)
    return model

def load_model(org_model_path, pt_model_path, device='cuda:0'):
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

def main():
    model_t = YOLO('weights/yolov8s-pose.pt', task='pose').to(device)  # the teacher model
    # model_s = YOLO('weights/yolov8s-pose-prune-sp0.7.pt', task='pose')  # the student model
    model_s = load_model1('./output/yolov8s-pose-prune-sp0.5-epochs120/step_15_finetune/weights/best.pt').to(device)
    # model_s = load_model('weights/yolov8s-pose.pt', 'weights/yolov8s-pose-qat.pt')
    
    """
    Attributes:
        Distillation: the distillation model
        loss_type: mgd, cwd
        amp: Automatic Mixed Precision
    """
    # # qat量化
    # model_s.model.qconfig = get_default_qat_qconfig("qnnpack")  # GPU 部署建议 qnnpack
    # model_s.model = prepare_qat(model_s.model)
    
    # model_s.train(data="data.yaml", Distillation=model_t.model, loss_type='mgd', amp=True, epochs=100, 
    #               batch=32, device=0, workers=0, lr0=0.001, project='output/yolov8s-pose-prune-sp0.7-distill100-test') #
    # rect=True, imgsz=(640,1088)
    # model_t.model = model_t.model.to(device)
    # model_s.model = model_s.model.to(device)
    model_s.train_v2(pruning=True, data="data.yaml", amp=True, epochs=300, batch=32, device="cuda:0", workers=0, lr0=0.001,
                     Distillation=model_t.model,
                     loss_type="mgd", resume=True)

    # 加载预训练模型

    # pretrained_model_path = './output/yolov8s-pose-prune-sp0.5-distill100/train/weights/best.pt'
    # # my_cfg = {'model': pretrained_model_path,
    # #         'data': 'data.yaml',
    # #         'epochs': 200,
    # #         'amp': True,
    # #         'project': 'output/yolov8s-pose-prune-sp0.5-h640w640-distill100',
    # #         'resume': True,
    # #         'batch': 32,
    # #         'device': 0,
    # #         'workers': 0,
    # #         'lr0': 0.001,
    # #         'task': 'pose'
    # #         }
    # # yolo_cfg_yaml = './ultralytics/cfg/default.yaml' # training parameters
    # # yolo_cfg = get_cfg(yolo_cfg_yaml, my_cfg)
    # model_s = YOLO(pretrained_model_path, task='pose')  # 加载模型配置
    
    # model_s.train(Distillation=model_t.model, loss_type='mgd', amp=True, 
    #               epochs=100, batch=32, device=0, workers=0, lr0=0.001, resume=True) # rect=True, imgsz=(640,1088)


if __name__ == '__main__':
    main()
