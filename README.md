# EdgeLite

EdgeLite 是一个面向深度学习模型压缩与加速的工作区，当前整合了三个部分：

- `yolov8/`：YOLOv8s-pose 压缩、量化、剪枝与 TensorRT 加速项目
- `yolov5/`：YOLOv5 压缩与推理项目
- `edgelite-compression-agent/`：自动化 Skill / Agent，用于从零检查环境、自动解析/下载模型和样例数据、生成压缩方案、运行 demo、输出报告和交接文档
- `edgepilot-web-demo/`：面向华为交付演示的网页工作台，可通过自然语言提示词触发 Agent 生成方案

说明：当前 YOLOv5/YOLOv8 已内置真实执行 adapter；ViT、ResNet、LLM、扩散模型等会进入通用 bootstrap 和压缩规划，并提示需要补充的项目 adapter。

## 目录说明

- `edgelite-compression-agent/agents/`：Agent 配置
- `edgelite-compression-agent/scripts/edgepilot.py`：demo、autopilot、plan、inspect、execute 入口
- `edgelite-compression-agent/handoff/`：交接说明文档
- `edgepilot-web-demo/`：网页 demo，默认 dry-run 生成报告，不执行训练/剪枝重任务
- `edgepilot_demo_run/`、`edgepilot_autopilot_run/`：本地运行产物，默认不纳入版本库

## 快速开始

```bash
cd /data/xl/Projects/EdgeLite
bash edgelite-compression-agent/scripts/run_demo.sh
```

从零检查项目、环境、模型和数据：

```bash
cd /data/xl/Projects/EdgeLite
python edgelite-compression-agent/scripts/edgepilot.py \
  --workspace /data/xl/Projects/EdgeLite \
  bootstrap \
  --request edgelite-compression-agent/assets/huawei_yolov8_pose_request.json \
  --output edgepilot_bootstrap_run
```

如果模型或样例数据缺失，可让 bootstrap 自动解析官方资源并下载：

```bash
python edgelite-compression-agent/scripts/edgepilot.py \
  --workspace /data/xl/Projects/EdgeLite \
  bootstrap \
  --request edgelite-compression-agent/assets/huawei_yolov8_pose_request.json \
  --auto-download-assets \
  --yes
```

启动网页 demo：

```bash
cd /data/xl/Projects/EdgeLite
python edgepilot-web-demo/server.py --host 127.0.0.1 --port 7860
```

然后打开 `http://127.0.0.1:7860`。

如果你要查看原始工程，请分别进入：

```bash
cd /data/xl/Projects/EdgeLite/yolov8
cd /data/xl/Projects/EdgeLite/yolov5
```

## 说明

- 大规模数据、缓存、输出目录和模型权重默认通过 `.gitignore` 排除
- 如果需要复现实验，请按各子项目内的 README 和脚本准备环境、数据与权重
