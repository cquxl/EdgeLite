import os
import cv2
import torch
import numpy as np
from pathlib import Path
import tensorrt as trt
from tqdm import tqdm
from ultralytics.data.dataset import YOLODataset
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from loguru import logger

def get_int8_calibration_dataloader(args):
    """Build and return a dataloader for calibration of INT8 models."""
    print(f"collecting INT8 calibration images from 'data={args.data_yaml_file}'")
    data = (check_cls_dataset if args.task == "classify" else check_det_dataset)(args.data_yaml_file)

    dataset = YOLODataset(
        data['train'] if 'train' in args.cali_data_path else data['val'],
        data=data,
        fraction=0.1 if 'train' in args.cali_data_path else 1.0, # 5000
        task=args.task,
        imgsz=args.imgsz,
        augment=False,
        batch_size=args.batch_size,
    )
    n = len(dataset)
    if n < args.batch_size:
        raise ValueError(
            f"The calibration dataset ({n} images) must have at least as many images as the batch size "
            f"('batch={args.batch_size}')."
        )
    elif n < 300:
        logger.warning(f">300 images recommended for INT8 calibration, found {n} images.")
    return build_dataloader(dataset, batch=args.batch_size, workers=0, drop_last=True)  # shuffle=True, required for batch loading


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
            raise FileNotFoundError(f"data dir not found: {self.dataset_dir}")

        image_files = [p for p in self.dataset_dir.glob("*.jpg")]
        if not image_files:
            raise FileNotFoundError(f"no image files found: {self.dataset_dir}")

        return [str(p) for p in image_files[:self.calibration_size]]

    def _load_images(self)->list:
        dataset_name = str(self.dataset_dir).split('/')[-3]+ '-' + str(self.dataset_dir).split('/')[-2] + '-' + str(self.dataset_dir).split('/')[-1] # coco-pose-val2017
        self.calibration_size = min(self.calibration_size, len(self.image_paths))
        dataset_name = f'{dataset_name}-nsample{self.calibration_size}.npy'
        cache_data_path = os.path.join(self.cache_dir, dataset_name)
        if os.path.exists(cache_data_path):
            return np.load(cache_data_path, allow_pickle=True)
        """preprocess images"""
        images = []
        for idx, image_path in tqdm(enumerate(self.image_paths), desc="processing data", total=len(self.image_paths)):
            try:
                # image = self._preprocess_image(image_path)
                image = self._yolo_process_image(image_path)
                images.append(image)
            except Exception as e:
                print(f"Failed to preprocess image {image_path}: {e}")
        np.save(cache_data_path, images)
        return images

    def __len__(self):
        return len(self.images)

    def _preprocess_image(self, image_path):
        """preprocess single image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"cannot read image: {image_path}")

        image = cv2.resize(image, self.input_shape)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        return np.ascontiguousarray(image, dtype=np.float32)

    def _yolo_process_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"cannot read image: {image_path}")
        # 1. BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 2. Letterbox resize
        resized, _, _ = self.letterbox(image, new_shape=self.input_shape, auto=False, scaleup=True)
        # 3. HWC -> CHW
        image = resized.transpose(2, 0, 1)
        # 4. Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        return np.ascontiguousarray(image, dtype=np.float32)

    @staticmethod
    def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
        # resize a rectangular image to a padded rectangle
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down
            r = min(r, 1.0)

        # Compute padding
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


class MyLogger(trt.ILogger):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def log(self, severity, msg):
        if severity == trt.Logger.Severity.VERBOSE:
            self.logger.debug(msg)
        elif severity == trt.Logger.Severity.INFO:
            self.logger.info(msg)
        elif severity == trt.Logger.Severity.WARNING:
            self.logger.warning(msg)
        elif severity == trt.Logger.Severity.ERROR:
            self.logger.error(msg)
        elif severity == trt.Logger.Severity.INTERNAL_ERROR:
            self.logger.critical(msg)