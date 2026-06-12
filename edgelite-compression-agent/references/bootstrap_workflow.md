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

2. **选择 Python**
   - 优先使用用户提供的 `--python-env`。
   - 其次使用当前激活 Python。
   - 若用户允许，可创建 `.venv-edgepilot`。

3. **检查依赖**
   - 检查 `torch`、`onnx`、`tensorrt`。
   - 检查 `trtexec` 是否在 PATH 中。
   - 不自动安装 CUDA/TensorRT；这通常需要管理员权限或匹配驱动版本。
   - `pip install -r requirements.txt` 必须有 `--install-deps --yes`。

4. **检查模型**
   - 如果模型路径不存在，停止真实执行并要求用户提供权重。
   - 不能自动编造模型文件。

5. **检查数据**
   - 真实精度评估必须使用用户数据集或任务匹配的公开数据。
   - 若数据不存在，可创建 mini smoke-test 数据目录，用于验证流程能否跑通。
   - mini 数据不能写入正式 mAP 或精度损失结论。

6. **进入压缩流程**
   - 环境缺口解决后运行 `autopilot`。
   - 真实执行必须有 `--execute --yes`。

## Bootstrap 命令

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
  --install-deps \
  --prepare-demo-data \
  --yes
```

## 输出解释

- `bootstrap.json`：机器可读环境检查结果。
- `bootstrap.md`：给用户/交付方看的缺口报告。
- `actions[]`：每个准备动作是否需要、是否执行、对应命令。
- `warnings[]`：阻塞真实压缩的缺口。

## 交付口径

向华为演示时要区分：

- **快速演示**：展示 Agent 规划和历史指标，不代表本次真实跑完。
- **真实执行**：逐个执行命令、记录日志、由脚本产物更新指标。
- **正式验收**：必须在目标硬件、目标 TensorRT 版本、真实验证集上跑 baseline 和最终 engine。
