# EdgePilot 交接说明

## 目录

1. [背景与目标](#背景与目标)
2. [Agent 应该解决什么](#agent-应该解决什么)
3. [推荐交付形态](#推荐交付形态)
4. [输入输出协议](#输入输出协议)
5. [决策规则](#决策规则)
6. [推荐目录结构](#推荐目录结构)
7. [写 Skill 的方法](#写-skill-的方法)
8. [Demo 应该长什么样](#demo-应该长什么样)
9. [实现清单](#实现清单)
10. [给接手同学的提示](#给接手同学的提示)

## 背景与目标

这个项目面向华为的深度学习模型压缩与加速需求，核心场景是：

- 模型：YOLOv8s-pose 与 YOLOv5 检测任务
- 硬件：NVIDIA T4，开发/调试环境可能是 L40
- 目标：在精度损失不超过 1% 的前提下，把推理速度提升到 2x 以上
- 工具链：PyTorch、ONNX、TensorRT、CUDA、QAT、PTQ、结构化剪枝、知识蒸馏

第一版不要做成“万能 agent”。先把已经跑通的经验固化成一个能稳定交付的自动化 demo。

## Agent 应该解决什么

这个 Agent 不是回答问题的聊天机器人，而是一个“压缩部署指挥器”：

1. 读取需求
2. 检查环境
3. 生成候选方案
4. 按规则筛选
5. 输出命令和报告
6. 在允许时执行推荐方案

它最重要的能力是自动决策，而不是单纯调用脚本。

## 推荐交付形态

推荐把交付拆成三层：

### 1. skill 层

`SKILL.md` 只写流程、边界、触发条件、输出物。

### 2. reference 层

把稳定但较长的内容放进 `references/`，例如：

- 压缩策略
- TensorRT 部署规则
- YOLOv5 / YOLOv8 的差异
- 报告模板

### 3. scripts 层

把真正干活的逻辑放进脚本里：

- 环境检查
- YOLOv8 `yolov8-pose` conda 环境从零安装
- request 解析
- 计划生成
- 推荐选择
- demo 执行

## 输入输出协议

### 输入

建议统一成一个 JSON 请求文件，字段大致如下：

```json
{
  "project": "yolov8",
  "task": "pose",
  "model": "weights/yolov8s-pose.pt",
  "data": "datasets/my-coco-pose.yaml",
  "runtime": {
    "device": "cuda:0",
    "imgsz": 640,
    "batch": 16,
    "batch_min": 1,
    "batch_max": 16
  },
  "target": {
    "hardware": "NVIDIA T4",
    "baseline_latency_ms": 10.0,
    "baseline_accuracy": 85.6,
    "latency_ms_max": 5.0,
    "speedup_min": 2.0,
    "accuracy_drop_max_pct": 1.0
  },
  "strategies": ["fp16", "ptq", "qat", "prune_qat"]
}
```

### 输出

每次运行都应生成：

- `plan.json`: 候选方案和决策策略
- `env.json`: 环境快照
- `evaluation.json`: 候选评估结果
- `commands.sh`: 可执行命令
- `report.md`: 给人看的中文报告

## 决策规则

建议遵循这个顺序：

1. FP16 baseline
2. PTQ
3. QAT
4. 剪枝 + QAT

具体规则：

- PTQ 如果精度掉得太多，直接淘汰
- QAT 如果满足速度和精度约束，优先于更激进的剪枝
- 对 YOLOv8 pose，剪枝先试 0.3，不要一开始就冲 0.5
- 最终 engine 尽量在目标硬件和目标 TensorRT 版本上构建

## 推荐目录结构

```text
edgelite-compression-agent/
├── SKILL.md
├── agents/openai.yaml
├── assets/
│   ├── icon.svg
│   └── huawei_yolov8_pose_request.json
├── handoff/
│   └── README_zh.md
├── references/
│   ├── compression_policy.md
│   ├── tensorrt_deployment.md
│   ├── yolov5_workflow.md
│   ├── yolov8_workflow.md
│   └── report_template.md
└── scripts/
    ├── setup_yolov8_pose_env.sh
    └── edgepilot.py
```

## 写 Skill 的方法

如果接手同学要直接写 skill，可以按这个顺序：

1. 用 `skill-creator` 初始化目录
2. 写好 `SKILL.md`
3. 把长说明拆到 `references/`
4. 把可执行逻辑放到 `scripts/`
5. 配好 `agents/openai.yaml`
6. 跑 quick_validate

`SKILL.md` 的写法建议：

- 先写用途
- 再写适用场景
- 再写工作规则
- 再写工作流
- 最后写 demo 用法和参考文档

不要把所有实现细节都堆进 `SKILL.md`，不然 skill 会很重，也不好维护。

## Demo 应该长什么样

华为看到的 demo 不应该只是“生成一个计划”，而应该体现自动化闭环：

1. 读取 request
2. 自动检查环境
3. 自动生成候选方案
4. 自动评估候选
5. 自动输出中文报告
6. 在 `--execute --yes` 下可以执行推荐候选

推荐命令：

```bash
cd /path/to/EdgeLite
bash edgelite-compression-agent/scripts/setup_yolov8_pose_env.sh \
  --env-name yolov8-pose \
  --tensorrt-dir /path/to/TensorRT-8.6.1.6 \
  --yes
conda activate yolov8-pose

python edgelite-compression-agent/scripts/edgepilot.py \
  --workspace /path/to/EdgeLite \
  autopilot \
  --request edgelite-compression-agent/assets/huawei_yolov8_pose_request.json \
  --output edgepilot_autopilot_run \
  --execute --yes
```

如果没有真实 GPU 环境，就只跑 dry-run，生成完整文档和命令，不执行重任务。

## 实现清单

- [x] YOLOv8 压缩链路整合
- [x] YOLOv5 压缩链路整合
- [x] 中文 skill
- [x] 中文报告
- [x] 自动 plan / evaluate / report
- [x] autopilot 入口
- [ ] 真实环境下的回归验证
- [ ] 更完整的批量评测与多卡支持
- [ ] 根据真实跑分自动写最终结论

## 给接手同学的提示

这个项目最容易做错的地方有三个：

1. 把“会调用脚本”误认为“Agent”
2. 把 demo 做成静态截图，而不是自动化流程
3. 忽略环境差异，导致 engine 在别的机器上不能复现

正确的交付应该是：

- 能说清楚为什么选这个策略
- 能把流程复现出来
- 能在报告里解释速度和精度取舍
- 能继续扩展到别的模型和别的硬件
