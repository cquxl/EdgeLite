import torch.cuda

from .utils import CalibrationDataset, MyLogger, get_int8_calibration_dataloader
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import os
from ultralytics import YOLO
import subprocess
import sys



class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataset, cache_file="trt_calibration.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.current_index = 0
        if hasattr(dataset, "batch_size"):
            self.batch_size = dataset.batch_size
            self.data_iter = iter(dataset)
        else:
            self.batch_size = 1
            self.calibration_data = dataset.get_images()
            # print(self.calibration_data)
            self.buffer = None
            self.data_size = trt.volume(
                self.calibration_data[0].shape) * 4  # float32 size 3*640*640*4=1228800*4=4915200
            #
            self.d_input = cuda.mem_alloc(self.data_size)  # pycuda._driver.DeviceAllocation，

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        if self.batch_size == 1:
            if self.current_index < len(self.calibration_data):
                batch = self.calibration_data[self.current_index]
                self.current_index += 1
                # pycuda._driver.DeviceAllocation，from CPU (host) → GPU (device)
                cuda.memcpy_htod(self.d_input, batch.tobytes())  # batch.tobytes() -> 3×640×640×4 = 4915200 contious
                return [int(self.d_input)]
        else:
            try:
                im0s = next(self.data_iter)["img"] / 255.0
                im0s = im0s.to("cuda") if im0s.device.type == "cpu" else im0s
                return [int(im0s.data_ptr())]
            except StopIteration:
                # Return None to signal to TensorRT there is no calibration data remaining
                return None
        return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"read calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"write calibration cache: {self.cache_file}")

    def __del__(self):
        if hasattr(self, "d_input") and self.d_input:
            self.d_input.free()


class YOLOv8PosePTQ:
    def __init__(self, args, weight, onnx_path, engine_path):
        self.args = args
        self.logger = self.args.logger
        self.input_shape = self.args.input_shape
        self.imgsz= args.imgsz
        self.batch = args.batch_size

        self.weight = weight  # pt file if not None
        self.onnx_path = onnx_path
        self.engine_path = engine_path

        # data yaml file
        self.data_yaml_file = args.data_yaml_file

        self.dynamic = args.dynamic
        self.model = YOLO(self.weight, task=args.task)  # dense model
        self.device = args.device


        # export engine method
        self.export = args.export
        self.trt_cache_path = os.path.join(self.args.cache_dir, f'trt_calibration_batch{self.batch}.cache')
        self.logger.info(self.model)
        self.ctx = cuda.Device(0).make_context()

    # def initialize_cuda(self):
    #     """initialize CUDA"""
    #     try:
    #         cuda.init()
    #         device_count = cuda.Device.count()
    #         if device_count == 0:
    #             self.logger.info("error: no GPU device found")
    #             return False
    #
    #         device = cuda.Device(0)
    #
    #         context = device.make_context()
    #         context.push()
    #
    #         # check version
    #         self.logger.info(f"CUDA version: {cuda.get_version()}")
    #         self.logger.info(
    #             f"CUDA device: {device.name()}, total memory: {device.total_memory() // (1024 * 1024)} MB/{device.total_memory() // (1024 * 1024 * 1024)} GB")
    #         return True
    #     except Exception as e:
    #         self.logger.info(f"CUDA error: {e}")
    #         return False

    def load_calib_dataset(self):
        if self.batch == 1:
            self.logger.info(f"load calibration dataset from {self.args.cali_data_path} by private load, batch={self.batch}")
            return CalibrationDataset(self.args.cali_data_path,
                                      self.args.cali_size,
                                      self.args.input_shape,
                                      self.args.cache_dir)
        self.logger.info(f"load calibration dataset from {self.args.cali_data_path} by yolo load, batch={self.batch}")
        return get_int8_calibration_dataloader(self.args)


    def load_trt_calibrator(self):
        self.dataset = self.load_calib_dataset()
        self.logger.info(f"calibration dataset size: {len(self.dataset)}")
        return EntropyCalibrator(self.dataset, self.trt_cache_path)

    def export_onnx(self):
        self.logger.info(f"export onnx from {self.weight}")
        assert self.weight is not None and os.path.exists(self.weight)
        # model = YOLO(self.weight)
        # self.logger.info(model)
        success = self.model.export(
            format="onnx",
            imgsz=self.input_shape,
            # simplify=True,
            dynamic=self.dynamic, # True
            device=self.device,  # cuda:0
            half=False  # FP32
        )
        if not success:
            self.onnx_path = None
            raise RuntimeError("ONNX export failed!")
        self.onnx_path = self.weight.replace('.pt', '.onnx')
        if os.path.exists(self.onnx_path):
            self.logger.info(f"ONNX export success: {self.onnx_path}")
        return self.onnx_path

    def parse_onnx(self, parser):
        self.logger.info(f"parse onnx from {self.onnx_path}")
        with open(self.onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                self.logger.info("ONNX parse failed:")
                for i in range(parser.num_errors):
                    self.logger.info(parser.get_error(i))
                return False

    def yolo_export_engine(self):
        self.logger.info(f"export int8 engine by yolo export->only support ptq")
        success = self.model.export(format='engine', imgsz=self.imgsz, dynamic=self.dynamic,
                                    verbose=False, batch=self.batch, workspace=2, int8=True,
                                    data=self.data_yaml_file)
        if not success:
            raise RuntimeError("export engine failed")
        import os
        os.rename(success, self.engine_path)
        if os.path.exists(self.engine_path):
            self.logger.info(f"export engine success: {self.engine_path}")
        return os.path.exists(self.engine_path)


    def build_int8_engine(self):
        self.logger.info("build int8 engine")
        # if not self.initialize_cuda():
        #     return False
        # 1. create logger
        # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        TRT_LOGGER = MyLogger(self.logger)

        # 2. create builder and network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # 3. create parser
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # 4. parse onnx
        if self.onnx_path and os.path.exists(self.onnx_path):
            self.parse_onnx(parser)
        else:
            self.logger.info("export onnx...")
            self.onnx_path = self.export_onnx()
            self.parse_onnx(parser)

        # 5. create config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB 工作空间

        # 6. prepare calibrator
        self.logger.info("prepare calibrator...")
        calibrator = self.load_trt_calibrator()

        # 7. set precision
        config.set_flag(trt.BuilderFlag.INT8)
        # config.set_flag(trt.BuilderFlag.FP16)
        config.int8_calibrator = calibrator

        # 9. set profile
        if self.args.dynamic: # dynamic batch size
            profile = builder.create_optimization_profile()
            input_name = network.get_input(0).name
            # profile_shape = (self.args.batch_size, 3, self.input_shape[0], self.input_shape[1]) # [1,3,640,640]
            opt_shape = (self.args.batch_size, 3, self.input_shape[0], self.input_shape[1])
            min_shape = (1, 3, self.input_shape[0], self.input_shape[1])
            max_shape = (self.args.batch_size, 3, self.input_shape[0], self.input_shape[1])
            profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
            config.add_optimization_profile(profile)
            config.set_calibration_profile(profile)

        # 10. build
        self.logger.info("start build engine...")
        serialized_engine = builder.build_serialized_network(network, config)
        if not serialized_engine:
            self.logger.info("build engine failed!")
            return False
        # 11. save
        with open(self.engine_path, "wb") as f:
            f.write(serialized_engine)
        self.logger.info(f"build engine success: {self.engine_path}")
        del calibrator, serialized_engine
        torch.cuda.empty_cache()
        return True


    def build_with_trtexec(self):
        """build engine with trtexec"""
        # 1. prepare trt cache
        trt_cache_path = os.path.join(self.args.cache_dir, f'trt_calibration_batch{self.batch}.cache')
        if not os.path.exists(trt_cache_path):
            calibrator = self.load_trt_calibrator()
            del calibrator
            torch.cuda.empty_cache()
        self.logger.info(f"export int8 engine by trtexec")
        subprocess.run([
            "trtexec",
            "--onnx=" + self.onnx_path,
            "--int8",
            "--calib=" + trt_cache_path,
            "--saveEngine=" + self.engine_path,
            "--workspace=1024",
            "--minShapes=images:1x3x640x640",
            f"--optShapes=images:{self.batch}x3x640x640",
            f"--maxShapes=images:{self.batch}x3x640x640",
            "--verbose"
        ], check=True)
        if os.path.exists(self.engine_path):
            self.logger.info(f"build engine success: {self.engine_path}")
        else:
            self.logger.info("build engine failed!")
        return os.path.exists(self.engine_path)

    def build_engine(self):
        try:
            if not os.path.exists(self.engine_path):
                if self.export == 'yolo':
                    success = self.yolo_export_engine()
                elif self.export == 'build':
                    success = self.build_int8_engine()
                elif self.export == 'trtexec':
                    success = self.build_with_trtexec()
                self.ctx.pop()
                if not success:
                    self.logger.info("build engine failed!")
                    self.ctx.pop()
                    sys.exit(1)
            else:
                self.logger.info(f"engine exists: {self.engine_path}, pleas eval on coco-pose")
                self.ctx.pop()
        finally:
            pass
            try:
                ctx = cuda.Context.get_current()
                if ctx:
                    ctx.pop()
            except:
                pass
    # def __del__(self):
    #     if self.ctx :
    #         self.ctx.pop()


