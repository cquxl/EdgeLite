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

- 默认模式：只生成计划、报告和 demo 指标，不执行训练、剪枝或 TensorRT 构建重任务。
- 真实执行：勾选“执行真实重任务”后，后端会追加 `--execute --yes`，按推荐候选执行命令。生产环境使用前应确认数据、权重、TensorRT 和 CUDA 环境。

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
