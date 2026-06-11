# YOLOv8 Workflow

## Main Entry Points

- `main_quant.py`
- `main_prune.py`
- `evaluate_trt.py`
- `yolov8_pose_infer.py`

## Typical Sequence

1. FP16 baseline
2. PTQ trial
3. QAT if PTQ misses accuracy
4. Prune + fine-tune
5. QAT after pruning when needed
6. TRT engine rebuild and evaluation

## Practical Notes

- Keep pose dataset YAML aligned with the actual COCO-Pose layout.
- Use the target engine path in the final report.
- For the Huawei scenario, report T4 latency separately from L40 debugging runs.

