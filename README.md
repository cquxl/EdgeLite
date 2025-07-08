[![GitHub stars](https://img.shields.io/github/stars/20250516aaa/EdgeLite?style=social)](https://github.com/20250516aaa/EdgeLite/stargazers) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch)](https://pytorch.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# EdgeLite

轻量级 YOLOv8 模型压缩与部署工具，支持 TensorRT 推理、INT8 后训练量化（PTQ）、自定义校准器、动态 batch 推理和精度评估等功能，旨在帮助模型在边缘设备上高效运行。

---

## 📌 Badges

- ⭐️ GitHub Stars: ![GitHub stars](https://img.shields.io/github/stars/20250516aaa/EdgeLite?style=social)
- 🐍 Python / PyTorch: ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/torch)
- 📄 License: ![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)

---

## ✨ 特性

- ✅ 支持将 YOLOv8 模型导出为 ONNX 与 TensorRT `.engine`
- 📦 集成 FP32 / FP16 / INT8 多精度推理模式
- 🎯 自定义 INT8 量化校准器（支持单张图像或 batch 模式）
- ⚙️ 支持动态输入尺寸（Dynamic Shape）与 batch size 优化
- 🔍 支持 `.engine` 模型在 COCO-Pose 数据集上评估精度（mAP 等）
- 🛠 三种引擎构建方式：YOLO export / PyCUDA 构建 / trtexec 工具链

---

## 🗂️ 目录结构

```text
EdgeLite/
├── compression/
│   ├── quant/
│   │   ├── ptq/
│   │   │   ├── ptq_quant.py      # 构建与导出主类
│   │   │   ├── utils.py                # 自定义训练数据
│   │   ├── qat/
│   │   │   ├── qat_quant.py      # 量化与导出主类
│   │   │   ├── utils.py                # 自定义校准器与加载器
│   ├── prune/
│   │   ├── prune.py                 # engine 模型评估工具
│   │   ├── utils.py           # engine 模型量化工具  
├── datasets/                           # 数据集及校准图像路径
├── weights/                             # YOLOv8 预训练模型路径
├── output/                             # 导出的 engine、onnx 路径
├── main_prune.py                        # 剪枝运行入口
├── main_prune.py                        # 量化运行入口
└── README.md                           # 项目说明文档
```

## 📦 安装

```bash
git clone https://github.com/20250516aaa/EdgeLite.git
cd EdgeLite
pip install -r requirements.txt
```

> ⚠️ 确保环境已安装：
>
> - Python 3.8+
> - CUDA Toolkit + cuDNN
> - TensorRT (v8.6+ 推荐)
> - PyTorch
> - Ultralytics
> - PyCUDA

## 🚀 快速上手

1. 下载预训练模型权重如'yolov8s-pose.pt'到本地文件夹:'./weights'

下载链接：[https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt)

2. 下载训练和校验数据coco-pose到本地文件夹:'./datastets'

下载链接：见ultralytics/cfg/datasets/coco-pose.yaml

```
download: |
  from pathlib import Path

  from ultralytics.utils.downloads import download

  # Download labels
  dir = Path(yaml["path"])  # dataset root dir
  url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
  urls = [f"{url}coco2017labels-pose.zip"] # label下载地址
  download(urls, dir=dir.parent)
  # Download data
  urls = [
      "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
      "http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
      "http://images.cocodataset.org/zips/test2017.zip",  # 7G, 41k images (optional)
  ]
  download(urls, dir=dir / "images", threads=3)
```

注意图片数据完整下载下来是coco数据，需要根据label里面的txt的train2017.txt和val2017.txt将对应的图片数据保存到对应的路径下

最终形成标准coco-pose的数据格式

```
datsets/
├── coco-pose/
│   ├── annotations/
│   │   ├── instances_val2017.json
│   │   ├── person_keypoints_val2017.json
│   ├── images/
│   │   ├── train2017  
│   │   ├── val2017  
│   ├── labels/
│   │   ├── train2017   
│   │   ├── val2017  
│   ├── my-coco-pose.yaml
```

my-coco-pose.yaml内容如下：

```yaml
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# COCO 2017 Keypoints dataset https://cocodataset.org by Microsoft
# Documentation: https://docs.ultralytics.com/datasets/pose/coco/
# Example usage: yolo train data=coco-pose.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── coco-pose  ← downloads here (20.1 GB)

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ./datasets/coco-pose # dataset root dir
train: train2017.txt # train images (relative to 'path') 56599 images
val: val2017.txt # val images (relative to 'path') 2346 images
test: test-dev2017.txt # 20288 of 40670 images, submit to https://codalab.lisn.upsaclay.fr/competitions/7403

# Keypoints
kpt_shape: [17, 3] # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]

# Classes
names:
  0: person

# Download script/URL (optional)
download: |
  from pathlib import Path

  from ultralytics.utils.downloads import download

  # Download labels
  dir = Path(yaml["path"])  # dataset root dir
  url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
  urls = [f"{url}coco2017labels-pose.zip"]
  download(urls, dir=dir.parent)
  # Download data
  urls = [
      "http://images.cocodataset.org/zips/train2017.zip",  # 19G, 118k images
      "http://images.cocodataset.org/zips/val2017.zip",  # 1G, 5k images
      "http://images.cocodataset.org/zips/test2017.zip",  # 7G, 41k images (optional)
  ]
  download(urls, dir=dir / "images", threads=3)

```

bash

* ./script/prune.sh
* ./script/ptq.sh
* ./script/qat.sh

## 🎯 参数配置示例

| 参数               | 说明                                        | 示例                                    |
| ------------------ | ------------------------------------------- | --------------------------------------- |
| `batch_size`     | 量化或推理的 batch 大小                     | `1` / `8`                           |
| `imgsz`          | 图像输入大小（单边像素）                    | `640`                                 |
| `input_shape`    | 模型输入维度                                | `(640, 640)`                          |
| `export`         | 导出方式 (`yolo`, `build`, `trtexec`) | `yolo`                                |
| `data_yaml_file` | 用于评估的数据集配置 YAML 文件              | `my-coco-pose.yaml`                   |
| `cali_data_path` | 校准图像文件夹路径                          | `datasets/coco-pose/images/train2017` |

## ✅ TODO 列表

- [X] 自定义 EntropyCalibrator
- [X] 动态输入尺寸 profile 自动配置
- [X] filter结构化剪枝支持
- [X] QAT（训练中量化）/PTQ(后训练量化）支持
- [ ] 多线程 / 多卡 batch 评估支持
- [ ] 更多 YOLO 变体模型适配

## 🙌 贡献

- 增加更多量化策略（MinMax、KL-divergence 等）
- 增加更多模型包括vit, llms, ddpm等
- 支持更低精度（INT4、mixed precision）
- 集成更丰富的性能基准与可视化
- 适配更多 YOLO 变体模型

---

## 📄 License

本项目遵循 MIT 许可证，详情参见 [LICENSE](./LICENSE)。
© 2025 EdgeLite 开源团队
