
### only surpport prune/org pt to onnx ###

import torch
from ultralytics import YOLO
import pytorch_quantization.nn as quant_nn
from pytorch_quantization.nn import TensorQuantizer  # 来自你QAT框架（如 MQBench 等）
import onnx
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

def export_yolov8_pose_onnx(
    model_path="yolov8s-pose.pt",
    onnx_path="yolov8s-pose-dynamic.onnx",
    img_height=640,
    img_width=1088,
    dynamic_batch=True,
    max_batch_size=16,
    opset_version=13,
    sim=False
):
    # 加载模型
    model = YOLO(model_path, task='pose')
    model.eval()

    # 动态或静态 batch
    if dynamic_batch:
        dummy_input = torch.randn(1, 3, img_height, img_width)  # 用于trace
        dynamic_axes = {
            "images": {0: "batch"},
            "output0": {0: "batch"}
        }
    else:
        dummy_input = torch.randn(1, 3, img_height, img_width)
        dynamic_axes = None

    # 设置导出参数
    torch.onnx.export(
        model.model,                         # 注意：ultralytics.YOLO的内部模型
        dummy_input,
        onnx_path,
        input_names=["images"],
        output_names=["output0"],
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes
    )
    print(f"ONNX 导出完成: {onnx_path}")
    if sim:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model) 
        try:
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplifier failure: {e}')
        onnx.save(onnx_model, onnx_path)
        print(f'ONNX simplify export success, saved as {onnx_path}')
   
   
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
 
def enable_all_fake_quant(module):
    for m in module.modules():
        if isinstance(m, quant_nn.TensorQuantizer):
            m._fake_quant = True 
    return module   
                
        
if __name__ == "__main__":
    # # just change your $MODEL_PATH$ and $ONNX_PATH$
    # MODEL_PATH = "weights/yolov8s-pose-prune-sp0.7.pt"
    # ONNX_PATH = "weights/yolov8s-pose-prune-sp0.7-op13-h640w1088-dynamic.onnx"
    
    # # 此导出方式会有多个输出长度为6
    # export_yolov8_pose_onnx(
    #     model_path=MODEL_PATH,
    #     onnx_path=ONNX_PATH,
    #     img_height=640,
    #     img_width=1088,
    #     dynamic_batch=True,
    #     max_batch_size=16,
    #     opset_version=13,
    #     sim=False
    # )
    # 官方导出推荐model.export("onnx"),报错再调整，对于qat需要再运行enable_all_fake_quant(model)
    ## model
    org_pt_path = "./output/yolov8s-pose-prune-sp0.7-distill100/train22/weights/best.pt"
    # model = YOLO(org_pt_path, task='pose')
    # path = model.export(format='onnx', dynamic=True, imgsz=(640,1088), verbose=False, batch=16, workspace=2)
    # os.rename(path, './weights/yolov8s-pose-prune-sp0.7-distill-train640-h640w1088.onnx')
    
    qat_pt_path ="./output/yolov8s-pose-qat-distill100/train2/weights/best.pt"
    # qat_model = load_model(org_pt_path, qat_pt_path)
    qat_model = YOLO(qat_pt_path, task='pose')
    qat_model = enable_all_fake_quant(qat_model)
    path = qat_model.export(format='onnx', dynamic=True, imgsz=(640,640), verbose=False, batch=16, workspace=2)
    os.rename(path, './weights/yolov8s-pose-qat-distill100-h640w640.onnx')
