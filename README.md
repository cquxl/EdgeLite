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
> - Python 3.8+
> - CUDA Toolkit + cuDNN
> - TensorRT (v8.6+ 推荐)
> - PyTorch
> - Ultralytics
> - PyCUDA

## 🚀 快速上手
bash
* ./script/prune.sh
* ./script/ptq.sh
* ./script/qat.sh

## 🎯 参数配置示例
| 参数               | 说明                                | 示例                                    |
| ---------------- | --------------------------------- |---------------------------------------|
| `batch_size`     | 量化或推理的 batch 大小                   | `1` / `8`                             |
| `imgsz`          | 图像输入大小（单边像素）                      | `640`                                 |
| `input_shape`    | 模型输入维度                            | `(640, 640)`                          |
| `export`         | 导出方式 (`yolo`, `build`, `trtexec`) | `yolo`                                 |
| `data_yaml_file` | 用于评估的数据集配置 YAML 文件                | `my-coco-pose.yaml`                   |
| `cali_data_path` | 校准图像文件夹路径                         | `datasets/coco-pose/images/train2017` |

## ✅ TODO 列表

- [x] 自定义 EntropyCalibrator  
- [x] 动态输入尺寸 profile 自动配置 
- [x] filter结构化剪枝支持
- [x] QAT（训练中量化）/PTQ(后训练量化）支持  
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

