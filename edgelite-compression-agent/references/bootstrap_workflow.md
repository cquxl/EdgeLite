# Bootstrap 工作流

## 目标

当目标服务器没有 EdgeLite 项目、Python 环境、依赖、数据或 TensorRT 时，Agent 必须先完成就绪检查，再决定是否 clone、建环境、安装依赖和准备数据。

## 输入来源

- 用户自然语言提示词：模型路径、目标硬件、任务、精度/速度约束
- 用户提供的 request JSON
- 用户提供的 Python 环境路径
- 当前服务器已有 CUDA/TensorRT/PyTorch 环境

## 决策顺序

1. **定位 workspace**
   - 若存在 `yolov8/`、`yolov5/` 或 `edgelite-compression-agent/`，直接使用。
   - 若不存在，生成 clone 命令。
   - 只有用户明确允许时才执行 `git clone`。

2. **识别模型族与 adapter**
   - YOLOv5/YOLOv8：当前仓库已内置真实执行 adapter。
   - ViT、ResNet、BERT、LLM、扩散模型等：可做 bootstrap 和通用规划，但若项目未提供 adapter，必须提示缺口。
   - 未知模型：要求用户提供模型加载、评估、导出和部署接口。

3. **选择 Python**
   - 优先使用用户提供的 `--python-env`。
   - YOLOv8 真实执行优先使用 `conda` 环境 `yolov8-pose`。
   - 如果新服务器没有 `yolov8-pose`，且用户允许安装，运行 `edgelite-compression-agent/scripts/setup_yolov8_pose_env.sh --env-name yolov8-pose --yes`。
   - YOLOv8 不应通过 pip 官方 `ultralytics` 包作为执行源码；必须从当前仓库 `yolov8/ultralytics` 导入，以保留项目内修改。
   - YOLOv5 真实执行使用独立 `conda` 环境 `yolov5-compress`，不能复用 `yolov8-pose`。
   - 如果新服务器没有 `yolov5-compress`，且用户允许安装，运行 `edgelite-compression-agent/scripts/setup_yolov5_env.sh --env-name yolov5-compress --yes`。
   - YOLOv5 必须设置 `YOLOv5_AUTOINSTALL=false`，避免运行时自动升级 pip 依赖污染环境。
   - 其次使用当前激活 Python。
   - 若用户允许，可创建 `.venv-edgepilot`。

4. **检查依赖**
   - 检查 `torch`、`onnx`、`tensorrt`。
   - 检查 `trtexec` 是否在 PATH 中。
   - 检查 `cd yolov8 && python -c "import ultralytics; print(ultralytics.__file__)"` 是否指向 `EdgeLite/yolov8/ultralytics`。
   - 不自动安装 CUDA/TensorRT；这通常需要管理员权限或匹配驱动版本。
   - TensorRT Python wheel 和 `trtexec` 若来自本地 TensorRT tar 包，可通过 `setup_yolov8_pose_env.sh --tensorrt-dir /path/to/TensorRT-8.6.1.6 --yes` 写入环境。
   - `pip install -r requirements.txt` 必须有 `--install-deps --yes`。

5. **检查模型**
   - 如果模型路径不存在，停止真实执行并要求用户提供权重。
   - 先扫描仓库 README、yaml、txt 中与模型名相关的 URL。
   - 再使用内置官方资源 registry，例如 Ultralytics YOLOv8 pose release 权重。
   - 如果 registry 也没有命中，Codex 应联网搜索官方文档、GitHub Release 或官方 HuggingFace 页面，确认可信 URL 后再加入下载动作。
   - Hugging Face 自动解析只应采用官方或可信组织仓库；不要下载普通用户仓库里的 `optimizer.pt`、训练状态文件或来源不明权重。
   - 如果用户允许 `--auto-download-assets --yes`，可自动下载到 request 指定路径。
   - 找不到可信 URL 时不能自动编造模型文件。

6. **检查数据**
   - 真实精度评估必须使用用户数据集或任务匹配的公开数据。
   - 若数据不存在，先扫描项目配置中的 download 字段，再使用官方小样例数据 registry。
   - registry 未命中时，Codex 应搜索官方 dataset 文档或官方 release 资源。
   - YOLOv8 pose 默认可下载 Ultralytics COCO8-pose 小数据集用于 smoke test。
   - mini/COCO8 数据不能写入正式 mAP 或精度损失结论。

7. **进入压缩流程**
   - 环境缺口解决后，优先使用 bootstrap 产出的 `resolved_request.json` 运行 `autopilot`。
   - 不要在自动下载/解压后继续使用原始 request；原始路径可能仍指向不存在的模型或数据。
   - 真实执行必须有 `--execute --yes`。

## Bootstrap 命令

新服务器先准备 YOLOv8 真实执行环境：

```bash
bash edgelite-compression-agent/scripts/setup_yolov8_pose_env.sh \
  --env-name yolov8-pose \
  --tensorrt-dir /path/to/TensorRT-8.6.1.6 \
  --yes
conda activate yolov8-pose
```

如果没有 TensorRT tar 目录，可以先不传 `--tensorrt-dir`；脚本会完成 Python 依赖安装，并在报告中提示 TensorRT/trtexec 缺口。

YOLOv5 使用独立环境：

```bash
bash edgelite-compression-agent/scripts/setup_yolov5_env.sh \
  --env-name yolov5-compress \
  --tensorrt-dir /path/to/TensorRT-8.6.1.6 \
  --yes
conda activate yolov5-compress
```

只检查并生成缺口报告：

```bash
python edgelite-compression-agent/scripts/edgepilot.py \
  --workspace /path/to/EdgeLite \
  bootstrap \
  --request request.json \
  --output edgepilot_bootstrap_run
```

允许执行准备动作：

```bash
python edgelite-compression-agent/scripts/edgepilot.py \
  --workspace /path/to/EdgeLite \
  bootstrap \
  --repo-url https://github.com/cquxl/EdgeLite.git \
  --request request.json \
  --python-env /path/to/python \
  --auto-download-assets \
  --install-deps \
  --prepare-demo-data \
  --yes
```

只自动下载缺失模型/样例数据：

```bash
python edgelite-compression-agent/scripts/edgepilot.py \
  --workspace /path/to/EdgeLite \
  bootstrap \
  --request request.json \
  --auto-download-assets \
  --yes
```

## 输出解释

- `bootstrap.json`：机器可读环境检查结果。
- `bootstrap.md`：给用户/交付方看的缺口报告。
- `resolved_request.json`：路径规范化后的请求文件，包含已下载模型、已解压数据和校准集路径，后续 `autopilot` 应使用它。
- `actions[]`：每个准备动作是否需要、是否执行、对应命令。
- `warnings[]`：阻塞真实压缩的缺口。
- `source` / `candidates`：模型或数据自动解析出的下载源。

## 非 YOLO 模型 adapter 要求

要真实执行任意深度学习模型压缩，项目至少需要提供：

- `load_model`：加载权重和模型结构。
- `prepare_calibration_data`：提供 PTQ/INT8 校准样本。
- `evaluate`：输出统一精度指标，例如 accuracy、mAP、F1、perplexity。
- `benchmark`：输出 latency、throughput、batch、硬件信息。
- `export`：导出 ONNX/TorchScript 或目标部署格式。
- `build_engine`：使用 TensorRT、ONNX Runtime、Torch-TensorRT 等构建部署产物。
- `compress`：执行 PTQ、QAT、剪枝、蒸馏或低秩分解。

如果这些接口不存在，Agent 只能输出通用压缩方案和缺口报告，不能声称已经完成真实压缩。

## 交付口径

向华为演示时要区分：

- **快速演示**：展示 Agent 规划和历史指标，不代表本次真实跑完。
- **真实执行**：逐个执行命令、记录日志、由脚本产物更新指标。
- **正式验收**：必须在目标硬件、目标 TensorRT 版本、真实验证集上跑 baseline 和最终 engine。

## 环境打包策略

- 推荐交付：`environment.yml`、`requirements-yolov8-pose.txt`、`setup_yolov8_pose_env.sh`。这种方式适合 A40/T4/L40 等不同服务器从零复建。
- 可选内网迁移：`conda-pack`。只建议在 Linux/x86_64、驱动、CUDA、TensorRT 大版本兼容且路径可修复的机器之间使用。
- 不建议把某台机器的完整环境当成唯一交付物；GPU 驱动、TensorRT tar、`trtexec` 和 CUDA runtime 往往与服务器绑定。
