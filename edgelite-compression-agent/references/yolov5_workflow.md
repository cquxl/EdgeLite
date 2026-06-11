# YOLOv5 Workflow

## Main Entry Points

- `export.py`
- `scripts/qat.py`
- `detect_after_pruning.py`
- `detect_after_pruning_finetune.py`
- `eval-trt.py`

## Typical Sequence

1. FP16 TensorRT export
2. QAT / PTQ via `scripts/qat.py`
3. Structured pruning when latency still misses target
4. TensorRT rebuild
5. TRT evaluation

## Practical Notes

- Treat COCO-based detection and pose differently in metrics and output handling.
- For pruning demos, prefer a conservative target first.

