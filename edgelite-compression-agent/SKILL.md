---
name: edgelite-compression-agent
description: 面向 YOLOv5/YOLOv8 的模型压缩、TensorRT 加速规划、执行与报告自动化；适用于 PTQ、QAT、剪枝、环境检查和交付 demo。
---

# EdgeLite 压缩加速 Agent

## 作用

这个 skill 用来把一个压缩/部署需求变成可复现的流程。它支持 YOLOv5 和 YOLOv8，能自动生成候选方案、检查本地环境、输出命令脚本和报告。

## 适用场景

- 需要为 CNN 检测/姿态模型做压缩与加速
- 需要比较 FP16、PTQ、QAT、剪枝+QAT
- 需要生成可交付 demo
- 需要给华为这类场景输出稳定的技术方案

## 工作规则

1. 分开处理 `yolov8/` 和 `yolov5/`。
2. 优先在目标硬件或目标 TensorRT 版本上构建 engine。
3. 先 FP16，再 PTQ，再 QAT，再 prune+QAT。
4. 姿态任务优先测试 0.3 稀疏度，再考虑更激进的剪枝。
5. 如果没有真实指标，就输出计划，不假装已经跑完。
6. 默认 dry-run；只有明确允许才执行重任务。

## 工作流

1. Inspect: 检查 CUDA / TensorRT / GPU 和项目布局。
2. Plan: 根据 request JSON 生成候选方案。
3. Evaluate: 按精度损失和速度门槛筛选。
4. Deliver: 输出 `plan.json`、`env.json`、`evaluation.json`、`commands.sh`、`report.md`。
5. Execute: 在可用硬件上运行推荐候选方案，记录日志。

## Demo 用法

```bash
python scripts/edgepilot.py demo --output edgepilot_demo_run
```

```bash
python scripts/edgepilot.py autopilot --request assets/huawei_yolov8_pose_request.json --output edgepilot_autopilot_run --execute --yes
```

## 参考文档

- [压缩策略](references/compression_policy.md)
- [TensorRT 部署](references/tensorrt_deployment.md)
- [YOLOv8 流程](references/yolov8_workflow.md)
- [YOLOv5 流程](references/yolov5_workflow.md)
- [报告模板](references/report_template.md)
- [交接文档](handoff/README_zh.md)

