import os
import cv2
import torch
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # 关键 - 正确初始化 CUDA
from pathlib import Path
from ultralytics import YOLO
import subprocess
import time
import sys
from loguru import logger
from .utils import MyLogger
from ultralytics.utils.torch_utils import init_seeds
from tqdm import tqdm


class CalibrationDataset:
    def __init__(self, dataset_dir, calibration_size, input_shape, cache_dir='./cache'):
        self.dataset_dir = Path(dataset_dir)
        self.calibration_size = calibration_size
        self.input_shape = input_shape
        self.image_paths = self._get_image_paths()
        self.cache_dir = cache_dir
        self.images = self._load_images()  # list[nparray(chw), ...]

    def _get_image_paths(self):
        """获取图像路径列表"""
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"数据集目录不存在: {self.dataset_dir}")

        image_files = [p for p in self.dataset_dir.glob("*.jpg")]
        if not image_files:
            raise FileNotFoundError(f"目录中没有找到 JPG 文件: {self.dataset_dir}")

        return [str(p) for p in image_files[:self.calibration_size]]

    def _load_images(self)->list:
        dataset_name = str(self.dataset_dir).split('/')[-3]+ '-' + str(self.dataset_dir).split('/')[-2] + '-' + str(self.dataset_dir).split('/')[-1] # coco-pose-val2017
        self.calibration_size = min(self.calibration_size, len(self.image_paths))
        dataset_name = f'{dataset_name}-nsample{self.calibration_size}.npy'
        cache_data_path = os.path.join(self.cache_dir, dataset_name)
        if os.path.exists(cache_data_path):
            return np.load(cache_data_path, allow_pickle=True)
        """预处理所有校准图像"""
        images = []
        for idx, image_path in tqdm(enumerate(self.image_paths), desc="processing data", total=len(self.image_paths)):
            try:
                # image = self._preprocess_image(image_path)
                image = self._yolo_process_image(image_path)
                images.append(image)
            except Exception as e:
                print(f"无法处理图像 {image_path}: {e}")
        np.save(cache_data_path, images)
        return images

    def __len__(self):
        return len(self.images)

    def _preprocess_image(self, image_path):
        """预处理单张图像"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        # 预处理
        image = cv2.resize(image, self.input_shape)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        return np.ascontiguousarray(image, dtype=np.float32)

    def _yolo_process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        # 1. BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 2. Letterbox resize 保持纵横比
        resized, _, _ = self.letterbox(image, new_shape=self.input_shape, auto=False, scaleup=True)
        # 3. HWC -> CHW
        image = resized.transpose(2, 0, 1)
        # 4. Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        return np.ascontiguousarray(image, dtype=np.float32)

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        # YOLOv8 官方 letterbox：保持原始比例 + padding
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down
            r = min(r, 1.0)

        # 计算 padding
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        # pad
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return im, r, (dw, dh)

    def get_images(self):
        """获取处理后的图像数组"""
        return self.images

class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, dataset, cache_file="trt_calibration.cache"):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.calibration_data = dataset.get_images()  # list[nparray(chw), ...]

        self.current_index = 0
        self.batch_size = 1
        self.buffer = None

        # 分配设备内存-->计算一张图片需要的GPU内存大小
        # print(self.calibration_data)
        self.data_size = trt.volume(self.calibration_data[0].shape) * 4  # float32 大小 3*640*640*4=1228800*4=4915200
        # 用于 拷贝校准图像数据，从而传入 TensorRT 引擎进行推理或校准
        self.d_input = cuda.mem_alloc(self.data_size)  # pycuda._driver.DeviceAllocation，在设备上的一块内存区域，

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        if self.current_index < len(self.calibration_data):
            batch = self.calibration_data[self.current_index]
            self.current_index += 1
            # 复制数据到设备内存htod:host-to-device 的缩写，表示从 CPU（host） → GPU（device）
            cuda.memcpy_htod(self.d_input, batch.tobytes()) #batch.tobytes() 就是 3×640×640×4 = 4915200 字节 的一串连续内存数据。batch必须确保连续
            return [int(self.d_input)]
        return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f"使用缓存文件: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        print(f"校准缓存已保存到: {self.cache_file}")

    def __del__(self):
        if self.d_input:
            self.d_input.free()



class YOLOv8PosePTQ:
    def __init__(self, args, weight, onnx_path, engine_path):
        self.args = args
        self.logger = self.args.logger
        self.input_shape = self.args.input_shape

        self.weight = weight  # pt file if not None
        self.onnx_path = onnx_path
        self.engine_path = engine_path
        self.ctx = cuda.Device(0).make_context()


    def initialize_cuda(self):
        """确保 CUDA 正确初始化"""
        try:
            # 显式初始化 CUDA 上下文
            cuda.init()
            # 检查可用设备
            device_count = cuda.Device.count()
            if device_count == 0:
                self.logger.info("错误: 没有找到可用的 CUDA 设备")
                return False

            # 使用第一个可用的设备
            device = cuda.Device(0)

            context = device.make_context()
            context.push()

            # 检查 CUDA 版本
            self.logger.info(f"CUDA 版本: {cuda.get_version()}")
            self.logger.info(
                f"CUDA 设备: {device.name()}, 总内存: {device.total_memory() // (1024 * 1024)} MB/{device.total_memory() // (1024 * 1024 * 1024)} GB")
            return True
        except Exception as e:
            self.logger.info(f"CUDA 初始化失败: {e}")
            return False

    def load_calib_dataset(self):
        return CalibrationDataset(self.args.cali_data_path,
                                  self.args.cali_size,
                                  self.args.input_shape,
                                  self.args.cache_dir)

    def load_trt_calibrator(self):
        self.dataset = self.load_calib_dataset()
        self.logger.info(f"校准数据集大小: {len(self.dataset)}")
        trt_cache_path = os.path.join(self.args.cache_dir, 'trt_calibration.cache')
        return EntropyCalibrator(self.dataset, trt_cache_path)

    def export_onnx(self):
        logger.info(f"导出 ONNX 模型 from {self.weight}")
        assert self.weight is not None and os.path.exists(self.weight)
        model = YOLO(self.weight)
        self.logger.info(model)
        success = model.export(
            format="onnx",
            imgsz=self.input_shape,
            simplify=True,
            opset=15,
            dynamic=True,
            half=False  # PTQ 量化需要 FP32 输入
        )
        if not success:
            self.onnx_path = None
            raise RuntimeError("ONNX 导出失败!")
        self.onnx_path = self.weight.replace('.pt', '.onnx')
        if os.path.exists(self.onnx_path):
            logger.info(f"ONNX 模型已导出到: {self.onnx_path}")
        return self.onnx_path

    def parse_onnx(self, parser):
        logger.info(f"解析 ONNX 模型 from {self.onnx_path}")
        with open(self.onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                logger.info("ONNX 解析错误:")
                for i in range(parser.num_errors):
                    logger.info(parser.get_error(i))
                return False


    def build_int8_engine(self):
        self.logger.info("构建 INT8 TensorRT 引擎")
        if not self.initialize_cuda():
            return False
        # 1. 创建 TensorRT 日志记录器
        # TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        TRT_LOGGER = MyLogger(self.logger)

        # 2. 创建 builder 和 network
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # 3. 创建 ONNX 解析器
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # 4. 解析 ONNX 模型
        if self.onnx_path and os.path.exists(self.onnx_path):
            self.parse_onnx(parser)
        else:
            self.logger.info("ONNX 模型不存在, 需要加载原始模型并导出onnx")
            self.onnx_path = self.export_onnx()
            self.parse_onnx(parser)

        # 5. 配置 builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB 工作空间

        # 6. 准备校准器
        self.logger.info("准备校准数据集...")
        calibrator = self.load_trt_calibrator()

        # 7. 设置 INT8 模式和校准器
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator

        # 9. 设置动态形状
        profile = builder.create_optimization_profile()
        input_name = network.get_input(0).name
        profile_shape = (self.args.batch_size, 3, self.input_shape[0], self.input_shape[1]) # [1,3,640,640]
        profile.set_shape(input_name, min=profile_shape, opt=profile_shape, max=profile_shape)
        config.add_optimization_profile(profile)

        # 10. 构建引擎
        self.logger.info("开始构建 INT8 引擎...")
        serialized_engine = builder.build_serialized_network(network, config)
        if not serialized_engine:
            self.logger.info("引擎构建失败!")
            return False
        # 11. 保存引擎
        with open(self.engine_path, "wb") as f:
            f.write(serialized_engine)
        self.logger.info(f"INT8 引擎已保存到: {self.engine_path}")
        return True


    def build_with_trtexec(self):
        """备选方案: 使用 trtexec 命令行工具构建引擎"""
        # self.logger.info("\n尝试使用 trtexec 构建引擎...")
        # 生成校准缓存
        trt_cache_path = os.path.join(self.args.cache_dir, 'trt_calibration.cache')
        if not os.path.exists(trt_cache_path):
            calibrator = self.load_trt_calibrator()

        subprocess.run([
            "trtexec",
            "--onnx=" + self.onnx_path,
            "--int8",
            "--calib=" + trt_cache_path,
            "--saveEngine=" + self.engine_path,
            "--workspace=1024",
            "--minShapes=images:1x3x640x640",
            "--optShapes=images:1x3x640x640",
            "--maxShapes=images:1x3x640x640",
            "--verbose"
        ], check=True)
        return os.path.exists(self.engine_path)

    def build_engine(self):
        try:
            if not os.path.exists(self.engine_path):
                success = self.build_int8_engine()

                if not success:
                    self.logger.info("Python API 构建失败，尝试使用 trtexec...")
                    success = self.build_with_trtexec()
                if not success:
                    self.logger.info("引擎构建失败, 请检查")
                    sys.exit(1)
            else:
                self.logger.info(f"引擎已存在: {self.engine_path}, pleas eval on coco-pose")
        finally:
            self.ctx.pop()
    def __del__(self):
        if self.ctx:
            self.ctx.pop()