#!/usr/bin/env python3
"""EdgePilot: planning and demo runner for EdgeLite compression workflows."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DEFAULT_WORKSPACE = SKILL_DIR.parent


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def run_probe(cmd: List[str], timeout: int = 15, cwd: Optional[Path] = None) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "stdout": "", "stderr": ""}


def python_module_version(module: str) -> Dict[str, Any]:
    code = (
        f"import {module}\n"
        f"print(getattr({module}, '__version__', 'available'))\n"
    )
    return run_probe([sys.executable, "-c", code])


def compact_stdout(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text
    unique = list(dict.fromkeys(lines))
    if len(unique) == 1 and len(lines) > 1:
        return f"{unique[0]} (x{len(lines)})"
    if len(unique) == 1:
        return unique[0]
    return " ; ".join(unique)


def inspect_environment(workspace: Path) -> Dict[str, Any]:
    nvidia = run_probe(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    nvcc = run_probe(["nvcc", "--version"])
    trtexec_path = shutil.which("trtexec")
    trtexec = run_probe([trtexec_path, "--version"]) if trtexec_path else {"ok": False, "error": "not found"}

    projects = {}
    for name in ("yolov5", "yolov8"):
        root = workspace / name
        projects[name] = {
            "exists": root.exists(),
            "root": str(root),
            "requirements": str(root / "requirements.txt") if (root / "requirements.txt").exists() else None,
            "main_entries": sorted(
                p.name for p in root.glob("*.py")
                if p.name in {"main_quant.py", "main_prune.py", "export.py", "val.py", "detect.py"}
            ) if root.exists() else [],
        }

    return {
        "generated_at": now(),
        "workspace": str(workspace),
        "system": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "executable": sys.executable,
        },
        "tools": {
            "nvidia_smi": {**nvidia, "stdout": compact_stdout(nvidia.get("stdout", ""))},
            "nvcc": nvcc,
            "trtexec": {"path": trtexec_path, **trtexec},
            "torch": python_module_version("torch"),
            "tensorrt": python_module_version("tensorrt"),
            "onnx": python_module_version("onnx"),
        },
        "projects": projects,
    }


def request_defaults(request: Dict[str, Any]) -> Dict[str, Any]:
    project = request.get("project", "yolov8")
    task = request.get("task", "pose" if project == "yolov8" else "detect")
    model = request.get("model", "weights/yolov8s-pose.pt" if project == "yolov8" else "models/yolov5s.pt")
    data = request.get("data", "datasets/my-coco-pose.yaml" if project == "yolov8" else "data/coco.yaml")
    output_dir = request.get("output_dir", f"output/edgepilot-{project}-{task}")

    runtime = {
        "device": "cuda:0",
        "imgsz": 640,
        "batch": 16,
        "batch_min": 1,
        "batch_max": 16,
        **request.get("runtime", {}),
    }
    target = {
        "hardware": "T4",
        "metric": "mAP50(P)" if task == "pose" else "mAP50",
        "baseline_latency_ms": 10.0,
        "baseline_accuracy": 85.6,
        "latency_ms_max": 5.0,
        "speedup_min": 2.0,
        "accuracy_drop_max_pct": 1.0,
        **request.get("target", {}),
    }

    normalized = dict(request)
    normalized.update({
        "project": project,
        "task": task,
        "model": model,
        "data": data,
        "output_dir": output_dir,
        "runtime": runtime,
        "target": target,
        "strategies": request.get("strategies", ["fp16", "ptq", "qat", "prune_qat"]),
    })
    return normalized


def sh_cd(workspace: Path, project: str) -> str:
    return f'cd "{workspace / project}"'


def build_yolov8_candidates(req: Dict[str, Any], workspace: Path) -> List[Dict[str, Any]]:
    rt = req["runtime"]
    out = req["output_dir"]
    model = req["model"]
    data = req["data"]
    imgsz = rt["imgsz"]
    batch = rt["batch"]
    batch_min = rt["batch_min"]
    batch_max = rt["batch_max"]
    cali = req.get("calibration_data", "datasets/coco-pose/images/train2017")
    train_images = req.get("train_images", "datasets/coco-pose/images/train2017")
    val_images = req.get("val_images", "datasets/coco-pose/images/val2017")
    device = rt["device"]

    py_export_fp16 = (
        "python - <<'PY'\n"
        "from ultralytics import YOLO\n"
        f"model = YOLO('{model}', task='{req['task']}')\n"
        f"path = model.export(format='engine', half=True, dynamic=True, imgsz={imgsz}, batch={batch}, workspace=2)\n"
        "print(path)\n"
        "PY"
    )

    return [
        {
            "name": "fp16_trt_baseline",
            "strategy": "fp16",
            "purpose": "Build a TensorRT FP16 baseline and measure the deployment floor.",
            "commands": [sh_cd(workspace, "yolov8"), py_export_fp16],
        },
        {
            "name": "int8_ptq",
            "strategy": "ptq",
            "purpose": "Run TensorRT INT8 PTQ with calibration data; keep only if accuracy loss is within policy.",
            "commands": [
                sh_cd(workspace, "yolov8"),
                (
                    "python main_quant.py "
                    f"--weight {model} --onnx_path weights/edgepilot-ptq.onnx "
                    f"--engine_path weights/edgepilot-ptq.engine --cali_data_path {cali} "
                    f"--cali_size {req.get('calibration_size', 5000)} --output_dir {out}/ptq "
                    f"--quant ptq --batch_size {batch} --export yolo --eval"
                ),
            ],
        },
        {
            "name": "int8_qat",
            "strategy": "qat",
            "purpose": "Insert Q/DQ nodes, fine-tune with pytorch_quantization, and build INT8 engine.",
            "commands": [
                sh_cd(workspace, "yolov8"),
                (
                    "python main_quant.py "
                    f"--weight {model} --train_img_path {train_images} --val_img_path {val_images} "
                    f"--onnx_path weights/edgepilot-qat.onnx --engine_path weights/edgepilot-qat.engine "
                    f"--epochs {req.get('qat_epochs', 30)} --output_dir {out}/qat "
                    f"--save_qat weights/edgepilot-qat.pt --quant qat --batch_size {batch} --eval"
                ),
            ],
        },
        {
            "name": "prune_0_3_qat",
            "strategy": "prune_qat",
            "purpose": "Use DepGraph structured pruning at sparsity 0.3, fine-tune/distill, then run QAT.",
            "commands": [
                sh_cd(workspace, "yolov8"),
                (
                    "python main_prune.py "
                    f"--weight {model} --iterative_steps {req.get('prune_steps', 16)} "
                    f"--output_dir {out}/prune-sp0.3 --target_prune_rate 0.3 "
                    f"--batch_size {batch} --epochs {req.get('prune_epochs', 120)} "
                    "--fine_tune --distillation"
                ),
                (
                    "python main_quant.py "
                    f"--weight {out}/prune-sp0.3/step_{req.get('prune_steps', 16) - 1}_finetune/weights/best.pt "
                    f"--train_img_path {train_images} --val_img_path {val_images} "
                    f"--onnx_path weights/edgepilot-prune03-qat.onnx "
                    f"--engine_path weights/edgepilot-prune03-qat.engine "
                    f"--epochs {req.get('qat_epochs_after_prune', 30)} --output_dir {out}/prune03-qat "
                    f"--save_qat weights/edgepilot-prune03-qat.pt --quant qat --batch_size {batch} --eval"
                ),
                (
                    "trtexec "
                    "--onnx=weights/edgepilot-prune03-qat.onnx --saveEngine=weights/edgepilot-prune03-qat.engine "
                    "--int8 --fp16 "
                    f"--minShapes=images:{batch_min}x3x{imgsz}x{imgsz} "
                    f"--optShapes=images:{batch}x3x{imgsz}x{imgsz} "
                    f"--maxShapes=images:{batch_max}x3x{imgsz}x{imgsz}"
                ),
            ],
        },
    ]


def build_yolov5_candidates(req: Dict[str, Any], workspace: Path) -> List[Dict[str, Any]]:
    rt = req["runtime"]
    out = req["output_dir"]
    model = req["model"]
    data = req["data"]
    imgsz = rt["imgsz"]
    batch = rt["batch"]
    batch_min = rt["batch_min"]
    batch_max = rt["batch_max"]
    cocodir = req.get("coco_dir", "datasets/coco")

    return [
        {
            "name": "fp16_trt_baseline",
            "strategy": "fp16",
            "purpose": "Export YOLOv5 to TensorRT FP16 engine.",
            "commands": [
                sh_cd(workspace, "yolov5"),
                f"python export.py --weights {model} --include engine --half --dynamic --batch-size {batch} --imgsz {imgsz}",
                f"python val.py --weights {model.replace('.pt', '.engine')} --data {data} --batch-size {batch} --imgsz {imgsz}",
            ],
        },
        {
            "name": "int8_qat",
            "strategy": "qat",
            "purpose": "Run YOLOv5 QAT and export the QAT checkpoint to ONNX/TensorRT.",
            "commands": [
                sh_cd(workspace, "yolov5"),
                (
                    "python scripts/qat.py quantize "
                    f"{model} --cocodir {cocodir} --device {rt['device']} "
                    f"--ptq {out}/ptq.pt --qat {out}/qat.pt --iters {req.get('qat_iters', 200)} "
                    "--eval-origin --eval-ptq"
                ),
                f"python scripts/qat.py export {out}/qat.pt --save {out}/qat.onnx --size {imgsz} --dynamic",
                (
                    "trtexec "
                    f"--onnx={out}/qat.onnx --saveEngine={out}/qat.engine --int8 --fp16 "
                    f"--minShapes=images:{batch_min}x3x{imgsz}x{imgsz} "
                    f"--optShapes=images:{batch}x3x{imgsz}x{imgsz} "
                    f"--maxShapes=images:{batch_max}x3x{imgsz}x{imgsz}"
                ),
            ],
        },
        {
            "name": "prune_0_3_qat",
            "strategy": "prune_qat",
            "purpose": "Apply structured pruning, fine-tune, then reuse the QAT export path.",
            "commands": [
                sh_cd(workspace, "yolov5"),
                (
                    "python detect_after_pruning_finetune.py "
                    f"--weights {model} --data {data} --imgsz {imgsz} --device {rt['device']} "
                    f"--project {out}/prune --name sp0.3 --iterative-steps {req.get('prune_steps', 5)} "
                    f"--finetune-epochs {req.get('prune_epochs', 30)}"
                ),
                (
                    "python scripts/qat.py quantize "
                    f"{out}/prune/sp0.3/weights/pruned_finetuned.pt --cocodir {cocodir} "
                    f"--device {rt['device']} --ptq {out}/prune03-ptq.pt --qat {out}/prune03-qat.pt "
                    f"--iters {req.get('qat_iters', 200)} --eval-ptq"
                ),
            ],
        },
    ]


def build_plan(request: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
    req = request_defaults(request)
    if req["project"] == "yolov8":
        candidates = build_yolov8_candidates(req, workspace)
    elif req["project"] == "yolov5":
        candidates = build_yolov5_candidates(req, workspace)
    else:
        raise ValueError(f"unsupported project: {req['project']}")

    enabled = set(req.get("strategies", []))
    if enabled:
        candidates = [c for c in candidates if c["strategy"] in enabled]

    return {
        "generated_at": now(),
        "request": req,
        "decision_policy": {
            "run_order": ["fp16", "ptq", "qat", "prune_qat"],
            "acceptance": [
                "accuracy_drop_pct <= target.accuracy_drop_max_pct",
                "latency_ms <= target.latency_ms_max OR speedup >= target.speedup_min",
                "engine must be built on the target TensorRT/CUDA hardware family when possible",
            ],
            "fallback": [
                "reject PTQ if calibration loss exceeds target",
                "prefer QAT before pruning when accuracy is already within target",
                "prefer sparsity 0.3 before 0.5 for YOLO pose workloads",
            ],
        },
        "candidates": candidates,
    }


def evaluate_results(request: Dict[str, Any]) -> Dict[str, Any]:
    req = request_defaults(request)
    target = req["target"]
    metrics = req.get("demo_metrics") or req.get("metrics") or []
    evaluated = []
    for item in metrics:
        accuracy = float(item.get("accuracy", item.get("map50", 0.0)))
        latency = float(item.get("latency_ms", item.get("t4_latency_ms", 0.0)))
        baseline_latency = float(target["baseline_latency_ms"])
        baseline_accuracy = float(target["baseline_accuracy"])
        speedup = baseline_latency / latency if latency > 0 else 0.0
        drop = baseline_accuracy - accuracy
        pass_accuracy = drop <= float(target["accuracy_drop_max_pct"])
        pass_latency = (
            latency <= float(target["latency_ms_max"])
            or speedup >= float(target["speedup_min"])
        )
        evaluated.append({
            **item,
            "accuracy": accuracy,
            "latency_ms": latency,
            "speedup": round(speedup, 4),
            "accuracy_drop_pct": round(drop, 4),
            "pass_accuracy": pass_accuracy,
            "pass_latency": pass_latency,
            "accepted": pass_accuracy and pass_latency,
        })

    accepted = [r for r in evaluated if r["accepted"]]
    if accepted:
        recommended = sorted(accepted, key=lambda r: (r["latency_ms"], -r["accuracy"]))[0]
        reason = "在满足速度与精度约束的候选中，选择延迟最低的方案。"
    else:
        recommended = sorted(
            evaluated,
            key=lambda r: (
                max(0.0, r["accuracy_drop_pct"] - float(target["accuracy_drop_max_pct"])),
                max(0.0, r["latency_ms"] - float(target["latency_ms_max"])),
            ),
        )[0] if evaluated else None
        reason = "没有候选同时满足全部约束，因此选择综合罚分最低的方案。"

    return {
        "target": target,
        "evaluated": evaluated,
        "recommended": recommended,
        "recommendation_reason": reason,
    }


def commands_script(plan: Dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by EdgePilot. Review paths and hardware before running.",
        "",
    ]
    for candidate in plan["candidates"]:
        lines.append(f"# === {candidate['name']} ({candidate['strategy']}) ===")
        for cmd in candidate["commands"]:
            lines.append(cmd)
        lines.append("")
    return "\n".join(lines)


def render_report(plan: Dict[str, Any], env: Dict[str, Any], evaluation: Dict[str, Any]) -> str:
    req = plan["request"]
    target = req["target"]
    rows = []
    for r in evaluation.get("evaluated", []):
        rows.append(
            "| {name} | {strategy} | {latency:.3f} | {speedup:.2f}x | {acc:.3f} | {drop:.3f} | {status} |".format(
                name=r.get("name", "-"),
                strategy=r.get("strategy", "-"),
                latency=r["latency_ms"],
                speedup=r["speedup"],
                acc=r["accuracy"],
                drop=r["accuracy_drop_pct"],
                status="通过" if r["accepted"] else "未通过",
            )
        )
    if not rows:
        rows.append("| - | - | - | - | - | - | 未提供指标 |")

    rec = evaluation.get("recommended")
    rec_text = (
        f"{rec.get('name')} ({rec.get('strategy')})"
        if rec else "暂无推荐，请先运行候选或提供指标。"
    )

    gpu = env.get("tools", {}).get("nvidia_smi", {}).get("stdout") or "not detected"
    trt = env.get("tools", {}).get("tensorrt", {}).get("stdout") or "not detected"

    return f"""# EdgePilot 自动化压缩加速报告

生成时间: {plan['generated_at']}

## 需求摘要

- 项目: `{req['project']}`
- 任务: `{req['task']}`
- 模型: `{req['model']}`
- 数据集: `{req['data']}`
- 目标硬件: `{target['hardware']}`
- 当前测试硬件: `{target.get('demo_hardware', target['hardware'])}`
- 目标延迟: `<= {target['latency_ms_max']} ms` 或加速比 `>= {target['speedup_min']}x`
- 目标精度损失: `<= {target['accuracy_drop_max_pct']}%`

## 环境快照

- GPU: `{gpu}`
- TensorRT Python: `{trt}`
- 工作目录: `{env.get('workspace')}`

## 候选结果

| 候选项 | 策略 | 延迟 ms | 加速比 | 精度 | 损失 % | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

## 推荐结论

推荐方案: **{rec_text}**。

推荐理由: {evaluation.get('recommendation_reason')}

## 交付产物

- `plan.json`：候选方案与决策策略
- `commands.sh`：可直接执行的命令序列
- `env.json`：环境与项目检查结果
- `report.md`：本报告

## 操作说明

- 在生产交付场景里，优先使用目标 GPU 与目标 TensorRT 版本构建 engine。
- PTQ 如果超出精度预算，直接切换到 QAT。
- YOLOv8 姿态任务优先测试 0.3 稀疏度，再考虑更激进的剪枝。
"""


def materialize_plan(request_path: Path, workspace: Path, output_dir: Path) -> Dict[str, Path]:
    request = load_json(request_path)
    plan = build_plan(request, workspace)
    env = inspect_environment(workspace)
    evaluation = evaluate_results(request)

    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    env_path = output_dir / "env.json"
    evaluation_path = output_dir / "evaluation.json"
    commands_path = output_dir / "commands.sh"
    report_path = output_dir / "report.md"

    write_json(plan_path, plan)
    write_json(env_path, env)
    write_json(evaluation_path, evaluation)
    commands_path.write_text(commands_script(plan), encoding="utf-8")
    commands_path.chmod(0o755)
    report_path.write_text(render_report(plan, env, evaluation), encoding="utf-8")

    return {
        "plan": plan_path,
        "env": env_path,
        "evaluation": evaluation_path,
        "commands": commands_path,
        "report": report_path,
    }


def execute_candidate(plan_path: Path, candidate_name: str, log_path: Path) -> None:
    plan = load_json(plan_path)
    candidates = {c["name"]: c for c in plan["candidates"]}
    if candidate_name not in candidates:
        raise SystemExit(f"candidate not found: {candidate_name}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for cmd in candidates[candidate_name]["commands"]:
            log.write(f"\n$ {cmd}\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            rc = proc.wait()
            if rc != 0:
                raise SystemExit(f"command failed with exit code {rc}: {cmd}")


def recommended_candidate_name(evaluation: Dict[str, Any]) -> Optional[str]:
    recommended = evaluation.get("recommended")
    if isinstance(recommended, dict):
        return recommended.get("name")
    return None


def cmd_inspect(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace).resolve()
    env = inspect_environment(workspace)
    if args.output:
        write_json(Path(args.output), env)
    print(json.dumps(env, indent=2, ensure_ascii=False))


def cmd_plan(args: argparse.Namespace) -> None:
    paths = materialize_plan(Path(args.request).resolve(), Path(args.workspace).resolve(), Path(args.output).resolve())
    print("已生成方案产物：")
    for name, path in paths.items():
        print(f"{name}: {path}")


def cmd_demo(args: argparse.Namespace) -> None:
    request = SKILL_DIR / "assets" / "huawei_yolov8_pose_request.json"
    output = Path(args.output).resolve()
    paths = materialize_plan(request, Path(args.workspace).resolve(), output)
    print("已生成 demo 产物：")
    for name, path in paths.items():
        print(f"{name}: {path}")


def cmd_autopilot(args: argparse.Namespace) -> None:
    request = Path(args.request).resolve()
    output = Path(args.output).resolve()
    workspace = Path(args.workspace).resolve()
    paths = materialize_plan(request, workspace, output)
    evaluation = load_json(paths["evaluation"])
    candidate = args.candidate or recommended_candidate_name(evaluation)

    print("已生成自动化压缩加速方案：")
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"推荐候选: {candidate or '未生成'}")

    if args.execute:
        if not candidate:
            raise SystemExit("没有可执行的推荐候选，请先提供 metrics 或显式指定 --candidate。")
        if not args.yes:
            raise SystemExit("执行自动化流程前必须显式加 --yes。")
        execute_candidate(paths["plan"], candidate, output / "execute.log")
        print(f"已执行候选: {candidate}")


def cmd_execute(args: argparse.Namespace) -> None:
    if not args.yes:
        raise SystemExit("Refusing to execute heavy workflows without --yes.")
    execute_candidate(Path(args.plan).resolve(), args.candidate, Path(args.log).resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgepilot.py", description="EdgeLite compression and TensorRT acceleration agent demo.")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE), help="EdgeLite workspace containing yolov5/ and yolov8/.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    inspect_p = sub.add_parser("inspect", help="Inspect local environment and project layout.")
    inspect_p.add_argument("--output", help="Optional JSON output path.")
    inspect_p.set_defaults(func=cmd_inspect)

    plan_p = sub.add_parser("plan", help="Generate plan, commands, and report from a request JSON.")
    plan_p.add_argument("--request", required=True, help="Request JSON path.")
    plan_p.add_argument("--output", default="edgepilot_run", help="Output directory.")
    plan_p.set_defaults(func=cmd_plan)

    demo_p = sub.add_parser("demo", help="Generate the Huawei YOLOv8 pose deliverable demo.")
    demo_p.add_argument("--output", default="edgepilot_demo_run", help="Output directory.")
    demo_p.set_defaults(func=cmd_demo)

    auto_p = sub.add_parser("autopilot", help="Generate and optionally execute the recommended compression workflow.")
    auto_p.add_argument("--request", default=str(SKILL_DIR / "assets" / "huawei_yolov8_pose_request.json"), help="Request JSON path.")
    auto_p.add_argument("--output", default="edgepilot_autopilot_run", help="Output directory.")
    auto_p.add_argument("--candidate", help="Override the recommended candidate name.")
    auto_p.add_argument("--execute", action="store_true", help="Execute the recommended candidate after generating the plan.")
    auto_p.add_argument("--yes", action="store_true", help="Required to actually execute a heavy workflow.")
    auto_p.set_defaults(func=cmd_autopilot)

    exec_p = sub.add_parser("execute", help="Execute one candidate from an existing plan.")
    exec_p.add_argument("--plan", required=True, help="plan.json path.")
    exec_p.add_argument("--candidate", required=True, help="Candidate name, e.g. int8_qat.")
    exec_p.add_argument("--log", default="edgepilot_execute.log", help="Execution log path.")
    exec_p.add_argument("--yes", action="store_true", help="Required because workflows can train/export large models.")
    exec_p.set_defaults(func=cmd_execute)
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)

    if "--workspace" in argv:
        idx = argv.index("--workspace")
        if idx > 0 and idx + 1 < len(argv):
            pair = argv[idx:idx + 2]
            del argv[idx:idx + 2]
            argv = pair + argv

    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
