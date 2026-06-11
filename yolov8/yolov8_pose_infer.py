import os
import pickle
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import onnx
import torch
import time
import math

from loguru import logger
import sys



# === GPU / TRT ===
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 初始化 CUDA 上下文

BASE_DIR = Path(__file__).resolve().parent
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

tensorrt_version = trt.__version__
major_version = int(tensorrt_version.split('.')[0])
minor_version = int(tensorrt_version.split('.')[1])
device = torch.cuda.current_device()
total_memory = torch.cuda.get_device_properties(device).total_memory

def setup_logger(log_name, save_dir):
    filename = '%s.log' % log_name
    save_file = os.path.join(save_dir, filename)
    # if os.path.exists(save_file):
    #     with open(save_file, "w") as log_file:
    #         log_file.truncate()
    logger.remove()
    logger.add(save_file, rotation="10 MB", format="{time} {level} {message}", level="INFO")
    logger.add(sys.stdout, colorize=True,
               format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    logger.info('This is the %s log' % log_name)
    return logger


class TRTModule(torch.nn.Module):
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32
    }
    WARMUP = 10
    REPEAT = 10
    TEST = True # 部署时将其设置为False

    def __init__(self, weight: Union[str, Path],
                 device: Optional[torch.device]) -> None:
        super(TRTModule, self).__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device('cuda:0')
        self.stream = torch.cuda.Stream(device=self.device)
        self.__init_engine()

    def print_bindings(self, engine):
        nb = engine.num_bindings
        for i in range(nb):
            name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
            io = "Input " if is_input else "Output"
            print(f"[Binding {i}] {io} name='{name}' dtype={dtype} shape={shape}")

    def __init_engine(self) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')
        # with trt.Runtime(logger) as runtime:
        #     model = runtime.deserialize_cuda_engine(self.weight.read_bytes())
        with open(self.weight, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(logger)
        model = runtime.deserialize_cuda_engine(engine_bytes)
        self.print_bindings(model)

        context = model.create_execution_context()

        num_bindings = model.num_bindings # 6
        names = [model.get_binding_name(i) for i in range(num_bindings)] # 6->['images','kpt','onnx::Shape_800','onnx::Reshape_819','onnx::Reshape_838','output0']
        num_inputs = sum([1 for i in range(num_bindings) if model.binding_is_input(i)]) # 1
        num_outputs = num_bindings - num_inputs # 5

        self.bindings: List[int] = [0] * (num_inputs + num_outputs)  # TensorRT 8
        self.num_bindings = num_bindings
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.model = model
        self.context = context
        self.input_names = names[:num_inputs] # ['images']
        self.output_names = names[num_inputs:] # ['kpt','onnx::Shape_800','onnx::Reshape_819','onnx::Reshape_838','output0']
        self.idx = list(range(self.num_outputs))
        
        self.input_shape = model.get_binding_shape(0) # [-1,3,640,1088]
        self.H = self.input_shape[-2]
        self.W = self.input_shape[-1]

    def get_io_indices(self, engine):  # inputs-->[0], outputs[1,2,3,4,5]
        inputs, outputs = [], []
        for i in range(engine.num_bindings):
            (inputs if engine.binding_is_input(i) else outputs).append(i)
        assert len(inputs) == 1, f"期望 1 个输入，但找到 {len(inputs)} 个。"
        # yolov8-pose 通常 1 个输出，但也可能更多，这里支持多输出
        return inputs[0], outputs

    def allocate_buffers(self, context, engine, batch, H, W):
        # 设定动态形状
        inp_idx, out_indices = self.get_io_indices(engine)
        # 多 profile 时可设置 context.active_optimization_profile = k
        context.set_binding_shape(inp_idx, (batch, 3, H, W))

        # 查询真实 IO 形状
        binding_shapes = {}  # 长度为6
        binding_dptrs = {}  # 长度为6
        host_buffers = {}  # 长度为6

        for i in range(engine.num_bindings):
            shape = context.get_binding_shape(i)
            shape_tuple = tuple(shape)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            try:
                nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            except:
                print("shape is not tuple")
                nbytes = int(np.prod(shape_tuple)) * np.dtype(dtype).itemsize
            if engine.binding_is_input(i):
                # host 输入
                host_buffers[i] = np.empty(shape, dtype=dtype)
                dptr = cuda.mem_alloc(nbytes)
                binding_dptrs[i] = dptr
            else:
                # host 输出
                host_buffers[i] = np.empty(shape, dtype=dtype)
                dptr = cuda.mem_alloc(nbytes)
                binding_dptrs[i] = dptr
            binding_shapes[i] = shape

        # bindings 列表必须按 binding 索引顺序填充指针
        bindings = [int(binding_dptrs[i]) for i in range(engine.num_bindings)]
        return host_buffers, binding_dptrs, bindings, binding_shapes

    def forward(self, inputs): # numpy, Tensor
        # print(f"input shape:{inputs.shape}")
        B, C, H, W = inputs.shape # [10,3,640,1088]
        # 分配显存
        host_bufs, dev_ptrs, bindings, bind_shapes = self.allocate_buffers(self.context, self.model, B, H, W)
        inp_idx, out_indices = self.get_io_indices(self.model)
        out_shapes = [bind_shapes[i] for i in out_indices]
        self.stream = cuda.Stream()
        start_evt = cuda.Event()
        end_evt = cuda.Event()

        host_bufs[inp_idx][...] = inputs
        # host_bufs[inp_idx] = inputs
        if self.TEST:
            # warm up
            for _ in range(self.WARMUP):
                cuda.memcpy_htod_async(dev_ptrs[inp_idx], host_bufs[inp_idx], self.stream)
                self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
                # 将所有输出拷回（便于后续 postprocess 维度推断）
                for oi in out_indices:
                    cuda.memcpy_dtoh_async(host_bufs[oi], dev_ptrs[oi], self.stream)
                self.stream.synchronize()
    
            # infer
            infer_times_ms = []
            post_times_ms = []
            for _ in range(self.REPEAT):
                # 推理（GPU计时：仅 enqueue->kernel->完成）
                start_evt.record(self.stream)
                cuda.memcpy_htod_async(dev_ptrs[inp_idx], host_bufs[inp_idx], self.stream)
                self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
                for oi in out_indices:
                    cuda.memcpy_dtoh_async(host_bufs[oi], dev_ptrs[oi], self.stream)
                end_evt.record(self.stream)
                self.stream.synchronize()
                gpu_time = start_evt.time_till(end_evt)  # 毫秒
                infer_times_ms.append(gpu_time)
            return host_bufs[oi], infer_times_ms
        else:
            start_evt.record(self.stream)
            cuda.memcpy_htod_async(dev_ptrs[inp_idx], host_bufs[inp_idx], self.stream)
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)
            for oi in out_indices:
                cuda.memcpy_dtoh_async(host_bufs[oi], dev_ptrs[oi], self.stream)
            end_evt.record(self.stream)
            self.stream.synchronize()
            gpu_time = start_evt.time_till(end_evt)  # 毫秒
            return host_bufs[oi], gpu_time


class TRTModule1(torch.nn.Module):
    dtypeMapping = {
        trt.bool: torch.bool,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.float16: torch.float16,
        trt.float32: torch.float32
    }

    def __init__(self,
                 weight: Union[str, Path],
                 device: Optional[torch.device] = None,
                 warmup_batch: int = 1,
                 warmup_iters: int = 10,
                 eager_prealloc: bool = True):
        """
        weight: path to serialized TRT engine (bytes) file
        warmup_batch: 在 init 阶段用于预热的 batch size（通常 1 或者你生产的 batch）
        eager_prealloc: True 则在 init 时为 warmup_batch 分配 buffers 并复用
        """
        super().__init__()
        self.weight = Path(weight) if isinstance(weight, str) else weight
        self.device = device if device is not None else torch.device('cuda:0')
        # stream 用 pycuda 的 stream，所有 memcpy/execute 都用同一个 stream，以便减少同步开销
        self.stream = cuda.Stream()
        self.model = None
        self.context = None

        # 这些将在 __init_engine 中填充
        self.num_bindings = 0
        self.bindings = None             # device pointers list for execute()
        self.host_bufs = None            # dict idx -> numpy host buffer
        self.dev_ptrs = None             # dict idx -> device ptr
        self.bind_shapes = None          # dict idx -> shape
        self.input_names = []
        self.output_names = []
        self.input_shape = None
        self.H = None
        self.W = None
        self.inp_idx = None
        self.out_indices = None

        # 初始化 engine + context
        self.__init_engine()

        # 如果希望预分配 buffers 并预热（推荐生产这样做）
        if eager_prealloc:
            start = time.time()
            self._ensure_buffers_for_batch(warmup_batch)
            # 生成一个随机的 float32 假数据进行预热（确保 dtype 与 engine 期望一致）
            dummy = np.random.rand(warmup_batch, 3, self.H, self.W).astype(np.float32)
            # run a couple times to ensure kernels compiled & memory allocated
            for _ in range(warmup_iters):
                _outs, _t = self._infer(dummy)
            print(f"Warmup {warmup_iters} iters run time: {(time.time()-start)*1000} ms")
            # synchronize to ensure warmup completed
            cuda.Context.synchronize()
            print("[TRTModule] Warmup finished (batch=%d), engine ready." % warmup_batch)

    def print_bindings(self, engine):
        nb = engine.num_bindings
        for i in range(nb):
            name = engine.get_binding_name(i)
            is_input = engine.binding_is_input(i)
            dtype = engine.get_binding_dtype(i)
            shape = engine.get_binding_shape(i)
            io = "Input" if is_input else "Output"
            print(f"[Binding {i}] {io} name='{name}' dtype={dtype} shape={shape}")

    def __init_engine(self):
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, namespace='')

        # 读取 engine bytes 并反序列化
        with open(self.weight, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(engine_bytes)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize engine from {self.weight}")
        self.model = engine
        self.print_bindings(engine)

        # create context
        context = engine.create_execution_context()
        if context is None:
            raise RuntimeError("Failed to create execution context")
        self.context = context

        # get IO info
        self.num_bindings = engine.num_bindings
        # determine input & outputs indices
        self.inp_idx, self.out_indices = self.get_io_indices(engine)
        # store names
        names = [engine.get_binding_name(i) for i in range(self.num_bindings)]
        self.input_names = [names[self.inp_idx]]
        self.output_names = [names[i] for i in self.out_indices]

        # get input shape template (may be dynamic)
        in_shape = engine.get_binding_shape(self.inp_idx)
        self.input_shape = tuple(in_shape)
        # assume last two dims H,W exist
        self.H = int(self.input_shape[-2])
        self.W = int(self.input_shape[-1])

    def get_io_indices(self, engine) -> Tuple[int, List[int]]:
        inputs, outputs = [], []
        for i in range(engine.num_bindings):
            (inputs if engine.binding_is_input(i) else outputs).append(i)
        if len(inputs) != 1:
            # 这里按你模型期望 1 个输入来处理；若不同可修改
            raise AssertionError(f"期望 1 个输入，但找到 {len(inputs)} 个。")
        return inputs[0], outputs

    def allocate_buffers(self, engine, context, batch: int, H: int, W: int):
        """
        为给定 batch, H, W 分配 host 与 device buffer。
        返回 (host_buffers, dev_ptrs, bindings_list, binding_shapes)
        host_buffers: dict idx -> numpy array (host)
        dev_ptrs: dict idx -> device pointer
        bindings_list: list 按 binding index 排序的 device pointers(int)
        binding_shapes: dict idx -> tuple(shape)
        """
        # set dynamic input shape
        context.set_binding_shape(self.inp_idx, (batch, 3, H, W))
        binding_shapes = {}
        dev_ptrs = {}
        host_buffers = {}

        for i in range(engine.num_bindings):
            shape = tuple(context.get_binding_shape(i))
            binding_shapes[i] = shape
            dtype = trt.nptype(engine.get_binding_dtype(i))
            # allocate host numpy
            # use contiguous array in C order
            host_buffers[i] = np.empty(shape, dtype=dtype)
            # allocate device memory
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            dev_ptrs[i] = cuda.mem_alloc(nbytes)

        bindings = [int(dev_ptrs[i]) for i in range(engine.num_bindings)]
        return host_buffers, dev_ptrs, bindings, binding_shapes

    def _ensure_buffers_for_batch(self, batch: int, H: Optional[int] = None, W: Optional[int] = None):
        """
        确保当前已分配的 buffers 满足给定 batch, H, W。
        若已分配则复用；否则重新分配（并释放旧的 device ptr）
        """
        if H is None: H = self.H
        if W is None: W = self.W

        need_realloc = False
        if self.host_bufs is None:
            need_realloc = True
        else:
            # check if shape matches input binding shape
            cur_shape = self.context.get_binding_shape(self.inp_idx)
            if tuple(cur_shape) != (batch, 3, H, W):
                need_realloc = True

        if not need_realloc:
            return

        # if previous dev ptrs exist, free them
        if self.dev_ptrs:
            for ptr in self.dev_ptrs.values():
                try:
                    ptr.free()
                except Exception:
                    pass

        self.host_bufs, self.dev_ptrs, self.bindings, self.bind_shapes = \
            self.allocate_buffers(self.model, self.context, batch, H, W)

    def _infer(self, inputs: np.ndarray):
        """
        内部推理函数。inputs: numpy array shape [B,3,H,W], dtype matches engine (通常 float32)
        返回: (list of outputs host buffers ordered by out_indices, gpu_time_ms)
        """
        if not isinstance(inputs, np.ndarray):
            # accept torch tensor too
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.detach().cpu().numpy()
            else:
                raise TypeError("inputs must be numpy or torch tensor")

        B, C, H, W = inputs.shape
        # ensure buffers allocated for this batch/shape
        self._ensure_buffers_for_batch(B, H, W)

        # write input to host buffer
        self.host_bufs[self.inp_idx][...] = inputs

        # record events
        start_evt = cuda.Event()
        end_evt = cuda.Event()
        start_evt.record(self.stream)

        # HtoD
        cuda.memcpy_htod_async(self.dev_ptrs[self.inp_idx], self.host_bufs[self.inp_idx], self.stream)
        # execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # DtoH for outputs
        for oi in self.out_indices:
            cuda.memcpy_dtoh_async(self.host_bufs[oi], self.dev_ptrs[oi], self.stream)
        end_evt.record(self.stream)

        # synchronize and measure
        self.stream.synchronize()
        gpu_time = start_evt.time_till(end_evt)

        # return outputs as list in original order
        # outputs = [self.host_bufs[oi] for oi in self.out_indices]
        return self.host_bufs[oi], gpu_time

    def forward(self, inputs: Union[np.ndarray, torch.Tensor]):
        outs, t_ms = self._infer(inputs)
        return outs, t_ms




if __name__ == "__main__":
    os.chdir(BASE_DIR)
    logger = setup_logger('infer.log', 'output/yolov8s-pose-qat-h640w1088')
    # ENGINE_PATH = "weights/yolov8s-pose-prune-sp0.5-op13-h640w1088-dynamic.engine"
    ENGINE_PATH = "weights/yolov8s-pose-qat-h640w1088.engine"
    # ENGINE_PATH = "weights/yolov8s-pose-op13-h640w1088-dynamic.engine"

    logger.info(f"-------------------------------------------{ENGINE_PATH} infer start---------------------------------------")
    
    engine = TRTModule(ENGINE_PATH, device)
    batch_total_infer_time = {}
    batch_per_img_avg_infer_time = {}
    for batch in range(1,17):
        inputs = torch.rand(batch, 3, 640, 1088)
        # inputs = torch.rand(batch, 3, 640, 640)
        output, infer_time = engine(inputs)
        # print(output)
        # print(f"output shape:{output.shape}")
        # logger.info(f"infer time lists:{infer_time}")
        logger.info(f"run {batch} imgs infer total time:{sum(infer_time)/len(infer_time)}")
        batch_total_infer_time[batch] = sum(infer_time)/len(infer_time)
        logger.info(f"per img infer avg time:{sum(infer_time)/len(infer_time)/batch}")
        batch_per_img_avg_infer_time[batch] = sum(infer_time)/len(infer_time)/batch
    logger.info(f"-------------------------------------------{ENGINE_PATH} infer record---------------------------------------")
    logger.info(f"all batch infer total time: {batch_total_infer_time}")
    logger.info(f"all batch per img infer avg time: {batch_per_img_avg_infer_time}")


    # engine = TRTModule1(ENGINE_PATH, device, warmup_batch=10, warmup_iters=20, eager_prealloc=True)
    # for batch in range(1,17):
    #     inputs = torch.rand(batch, 3, 640, 1088)
    #     output, infer_time = engine(inputs)
    #     # print(output)
    #     # print(f"output shape:{output.shape}")
    #     print(f"infer time lists:{infer_time}")
    #     print(f"run {batch} imgs infer total time:{infer_time}ms")
    #     print(f"run {batch} imgs infer avg time:{infer_time/batch}ms/img")





    
