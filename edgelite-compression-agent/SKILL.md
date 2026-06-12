---
name: edgelite-compression-agent
description: 从零准备并执行深度学习模型压缩与 TensorRT 加速；YOLOv5/YOLOv8 已内置真实执行 adapter，ViT/ResNet/LLM/扩散模型等可进行 bootstrap、通用压缩规划、adapter 缺口提示、资源下载、环境检查和交付报告。
---

# EdgeLite 压缩加速 Agent

## 作用

这个 skill 把“给定深度学习模型路径和部署目标”转成可复现的压缩加速流程。YOLOv5/YOLOv8 已有真实执行 adapter；其他模型族先进入通用 bootstrap 和压缩规划，并明确提示缺少哪些 adapter 接口。它不默认环境已经准备好：如果项目、Python、依赖、模型、数据或 TensorRT 缺失，应先执行 bootstrap 检查/自动下载/输出缺口，再进入压缩策略规划或真实执行。

## 典型触发

用户可能只说：

```text
调用 EdgeLite skill，压缩 YOLOv8/ViT/ResNet/任意 PyTorch 模型，模型路径为 xxx，希望部署到 NVIDIA T4/A40，精度损失 <=1%，速度提升 >=2x。
```

此时应按下面顺序处理。

## 工作规则

1. 先定位 workspace。若没有 `yolov5/`、`yolov8/` 或 `edgelite-compression-agent/`，先建议或执行 clone。
2. 先 bootstrap，再 plan/autopilot。不要直接假设依赖、数据、权重、TensorRT 都存在。
3. Python 环境优先级：用户指定环境 > 当前激活环境 > 创建 `.venv-edgepilot`。安装依赖、创建环境、下载/写入数据必须得到明确允许。
4. Adapter 策略：YOLOv5/YOLOv8 可真实执行；ViT、ResNet、LLM、DDPM 等若当前仓库无 adapter，应继续 bootstrap 和通用规划，但必须警告“缺少真实执行 adapter”，列出需要的 load/eval/export/build/compress 接口。
5. 模型/数据策略：如果权重或数据缺失，先扫描仓库 README/yaml/txt 的下载链接，再使用内置官方 registry；仍找不到时，Codex 应联网查官方文档、GitHub Release 或官方 HuggingFace 页面，确认可信 URL 后再下载。自动下载只接受官方或可信组织资源；不能确认官方来源时输出缺口而不是伪造资源。
6. 数据策略：真实精度评估必须使用用户提供或任务匹配的正式验证集；官方 COCO8/COCO8-pose 小数据只能做流程 smoke test，不能作为验收 mAP 结论。
7. 压缩顺序：Dense/PyTorch baseline -> FP16 TensorRT -> INT8 PTQ -> INT8 QAT -> 结构化剪枝+QAT。
8. 姿态任务优先尝试 0.3 结构化剪枝；PTQ 超出精度预算时切换 QAT。
9. 没有真实指标时只输出计划、命令和风险，不把 demo 指标说成本次结果。
10. 执行训练、剪枝、导出 engine、安装依赖、下载资源等重任务前，必须有明确执行许可，例如 `--yes` 或用户明确说“执行真实流程/自动下载”。

## 标准流程

### 1. Bootstrap

检查或准备仓库、环境、模型和数据：

```bash
python edgelite-compression-agent/scripts/edgepilot.py \
  --workspace /path/to/EdgeLite \
  bootstrap \
  --repo-url https://github.com/cquxl/EdgeLite.git \
  --request request.json \
  --output edgepilot_bootstrap_run
```

只有用户明确允许时才执行写入动作：

```bash
python edgelite-compression-agent/scripts/edgepilot.py \
  --workspace /path/to/EdgeLite \
  bootstrap \
  --repo-url https://github.com/cquxl/EdgeLite.git \
  --request request.json \
  --python-env /path/to/python-or-env \
  --create-venv \
  --install-deps \
  --auto-download-assets \
  --prepare-demo-data \
  --yes
```

输出：

- `bootstrap.json`
- `bootstrap.md`

如果没有模型或数据，推荐先让 bootstrap 自动解析并下载官方资源：

```bash
python edgelite-compression-agent/scripts/edgepilot.py \
  --workspace /path/to/EdgeLite \
  bootstrap \
  --request request.json \
  --auto-download-assets \
  --yes
```

### 2. Inspect

```bash
python edgelite-compression-agent/scripts/edgepilot.py \
  --workspace /path/to/EdgeLite \
  inspect --output env.json
```

### 3. Plan / Autopilot

根据 request JSON 生成候选方案、命令和报告：

```bash
python edgelite-compression-agent/scripts/edgepilot.py \
  --workspace /path/to/EdgeLite \
  autopilot \
  --request request.json \
  --output edgepilot_autopilot_run
```

真实执行推荐候选：

```bash
python edgelite-compression-agent/scripts/edgepilot.py \
  --workspace /path/to/EdgeLite \
  autopilot \
  --request request.json \
  --output edgepilot_autopilot_run \
  --execute --yes
```

### 4. Web Demo

网页版用于交付演示：

```bash
python edgepilot-web-demo/server.py --host 0.0.0.0 --port 7860
```

默认是快速演示模式；勾选“真实执行候选搜索”才会逐个执行候选命令并显示日志。

## Request JSON 最小字段

```json
{
  "project": "yolov8",
  "task": "pose",
  "model": "weights/yolov8s-pose.pt",
  "data": "datasets/my-coco-pose.yaml",
  "target": {
    "hardware": "NVIDIA T4",
    "metric": "mAP50(P)",
    "baseline_latency_ms": 10.0,
    "baseline_accuracy": 85.6,
    "latency_ms_max": 5.0,
    "speedup_min": 2.0,
    "accuracy_drop_max_pct": 1.0
  }
}
```

## 参考文档

- [Bootstrap 工作流](references/bootstrap_workflow.md)
- [压缩策略](references/compression_policy.md)
- [TensorRT 部署](references/tensorrt_deployment.md)
- [YOLOv8 流程](references/yolov8_workflow.md)
- [YOLOv5 流程](references/yolov5_workflow.md)
- [报告模板](references/report_template.md)
- [交接文档](handoff/README_zh.md)
