# EdgePilot Web Demo

这是 EdgeLite 压缩与加速 Agent 的网页版交付 demo。它提供一个华为场景工作台：输入自然语言需求或表单字段后，后端调用 `edgelite-compression-agent/scripts/edgepilot.py`，生成压缩策略、候选对比、环境快照、命令脚本和报告。

## 启动

```bash
cd /data/xl/Projects/EdgeLite
python edgepilot-web-demo/server.py --host 127.0.0.1 --port 7860
```

浏览器打开：

```text
http://127.0.0.1:7860
```

## 模式

- 快速演示模式：生成计划、报告和历史/示例指标，不执行训练、剪枝或 TensorRT 构建重任务。这个模式用于展示 Agent 决策链路，不代表本次真实跑出的结果。
- 真实候选搜索：勾选“真实执行候选搜索”后，后端会按 `plan.json` 里的候选顺序逐个执行命令，并在网页运行日志中实时显示脚本输出。生产环境使用前应确认数据、权重、TensorRT 和 CUDA 环境。

说明：真实候选搜索不会伪造最终延迟/精度。如果底层脚本没有把本次实测指标写入 `evaluation.json`，网页会显示候选执行状态和日志，而不会把示例指标当成本次结果。

## 输出

每次运行会写入：

```text
edgepilot-web-demo/runs/<run_id>/
```

包含：

- `request.json`
- `plan.json`
- `env.json`
- `evaluation.json`
- `commands.sh`
- `report.md`
