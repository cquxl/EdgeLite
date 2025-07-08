

import json
import struct
import torch
import torch.nn as nn
from .quantize import export_onnx, export_onnx1

class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)

def my_export_onnx(
        model: nn.Module,
        file: str,
        size: int = 640,
        dynamic_batch: bool = True,
        kpt_shape: list = [17, 3]
):
    """ 主导出函数：准备模型和参数 """
    device = next(model.parameters()).device
    # model = patch_dfl(model).to(device)
    # validate_shapes(model, device)
    model.eval().float()  # fp32

    head = model.model[-1]
    head.kpt_shape = kpt_shape
    head.nc = 1  # output channels

    # dynamic_axes
    dynamic_axes = {
        "images": {0: "batch_size"},
        "output0": {0: "batch_size"},
        "kpt": {0: "batch_size"}
    } if dynamic_batch else None

    # dummy input
    dummy = torch.randn(1, 3, size, size, device=device) * 255

    # 调用核心导出函数
    try:
        export_onnx(
            model,
            (dummy,),
            file,
            opset_version=19,  # 必须>=13支持QAT节点 [6,9](@ref) # 17,
            input_names=["images"],
            output_names=["output0", "kpt"],
            dynamic_axes=dynamic_axes,
        )
    except:
        export_onnx1(
            model,
            (dummy,),
            file,
            opset_version=19,  # 必须>=13支持QAT节点 [6,9](@ref) # 17,
            input_names=["images"],
            output_names=["output0", "kpt"],
            dynamic_axes=dynamic_axes,
        )


def float_to_hex(f):
    hex_val = hex(struct.unpack('<I', struct.pack('<f', f))[0])
    hex_val = hex_val[2:]
    return hex_val

def export_to_trt_calib(filename, trt_version):
    # Load precision config file
    with open(filename, "r") as f:
        json_dict = json.load(f)

    # Create new files
    with open(filename.replace(".json", "_calib.cache"),
              "w") as f_calib, open(filename.replace(".json", "_layer_arg.txt"),
                                    "w") as f_layer_precision_arg:

        f_calib.write(f"TRT-{trt_version}-EntropyCalibration2\n")
        int8_tensor_scales = json_dict["int8_tensor_scales"]
        for layer_name, scale in int8_tensor_scales.items():
            # Convert INT8 ranges to scales to HEX
            scale_hex = float_to_hex(scale)
            f_calib.write(f"{layer_name}: {scale_hex}\n")
        fp16_nodes = json_dict["fp16_nodes"]
        for layer_name in fp16_nodes:
            # Save list of all layers that need to run in FP16 for later use with TensorRT
            f_layer_precision_arg.write(f"{layer_name}:fp16,")