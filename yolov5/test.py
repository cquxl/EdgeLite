# import onnx
# model = onnx.load("/data/fhl/yolov5_latest/models/yolov5s.onnx")
# for output in model.graph.output:
#     print(output.name, output.type)
    
    
# import tensorrt as trt

# def print_engine_info(engine_path):
#     TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#     with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
#         engine = runtime.deserialize_cuda_engine(f.read())

#     print(f"Engine has {engine.num_bindings} total bindings (inputs + outputs)")

#     num_inputs = 0
#     num_outputs = 0
#     for i in range(engine.num_bindings):
#         name = engine.get_binding_name(i)
#         is_input = engine.binding_is_input(i)
#         shape = engine.get_binding_shape(i)
#         dtype = engine.get_binding_dtype(i)
#         print(f"Binding {i}: name='{name}', is_input={is_input}, shape={shape}, dtype={dtype}")
#         if is_input:
#             num_inputs += 1
#         else:
#             num_outputs += 1

#     print(f"Engine has {num_inputs} input(s) and {num_outputs} output(s)")

# if __name__ == "__main__":
#     engine_file = "/data/fhl/yolov5_latest/models/yolov5s-int8-32-16-minmax.engine"  # 替换为你的engine文件路径
#     print_engine_info(engine_file)







import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
img = np.random.rand(1, 3, 640, 640).astype(np.float32)  # 或使用真实图像
outputs = session.run(None, {"images": img})
print(outputs[0].shape)
print(outputs[0][:5])  # 查看前几个值是否正常