# YOLOv5 Workflow

## Main Entry Points

- `export.py`
- `scripts/qat.py`
- `detect_after_pruning.py`
- `detect_after_pruning_finetune.py`
- `eval-trt.py`

## Environment

YOLOv5 uses an independent environment:

```bash
bash edgelite-compression-agent/scripts/setup_yolov5_env.sh \
  --env-name yolov5-compress \
  --tensorrt-dir /path/to/TensorRT-8.6.1.6 \
  --yes
```

Do not reuse the YOLOv8 `yolov8-pose` environment. YOLOv5 imports `ultralytics` helper modules from the pip package, but the execution source must remain the repository-local `yolov5/` code. The setup script sets:

```bash
YOLOv5_AUTOINSTALL=false
```

This prevents YOLOv5 from automatically running `pip install -U ultralytics` at import time.

## Typical Sequence

1. FP16 TensorRT export
2. QAT / PTQ via `scripts/qat.py`
3. Structured pruning when latency still misses target
4. TensorRT rebuild
5. TRT evaluation

## Practical Notes

- Treat COCO-based detection and pose differently in metrics and output handling.
- For pruning demos, prefer a conservative target first.
- EdgePilot-generated YOLOv5 commands use `conda run -n yolov5-compress ...` so they do not pollute or depend on the YOLOv8 environment.
