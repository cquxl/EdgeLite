# Compression Policy

## Decision Order

1. FP16 TensorRT baseline
2. PTQ
3. QAT
4. Prune + QAT

## Acceptance Rules

- Accept PTQ only if accuracy loss is within the budget.
- Accept QAT if it reaches the latency goal and keeps accuracy loss within budget.
- Prefer prune+QAT only when QAT alone still misses latency.

## Practical Defaults

- Pose workloads usually tolerate mild pruning better than aggressive sparsity.
- Start with pruning rate around `0.3`.
- Rebuild the final engine on the target TRT/CUDA stack.

## Rejection Reasons

- Calibration loss too high
- TRT version mismatch
- Dynamic shape profile not aligned with the target batch range
- Output metric missing or not comparable to baseline

