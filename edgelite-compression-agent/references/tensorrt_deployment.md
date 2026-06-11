# TensorRT Deployment Notes

## Required Checks

- GPU present
- CUDA available
- TensorRT importable
- `trtexec` available when using the CLI path

## Export Paths

- YOLOv8: `pt -> onnx -> engine`
- YOLOv5: `pt -> onnx -> engine`

## Deployment Rules

- Prefer dynamic batch only if the target service needs it.
- Keep `min/opt/max` shapes consistent with the target batch band.
- Keep the final engine build close to the production platform.

## Common Failure Modes

- ONNX export succeeds but TRT parse fails
- PTQ calibration cache is stale
- QAT checkpoint is not fused before export
- Engine is built on a different TRT minor version than production

