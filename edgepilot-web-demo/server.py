#!/usr/bin/env python3
"""Minimal web demo server for EdgePilot.

Default mode is an honest dry-run demo that uses historical/demo metrics. Real
search mode executes candidate commands sequentially and streams logs, but does
not fabricate final metrics if the underlying scripts do not emit parsable
latency/accuracy results.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote


WEB_DIR = Path(__file__).resolve().parent
WORKSPACE = WEB_DIR.parent
STATIC_DIR = WEB_DIR / "static"
RUNS_DIR = WEB_DIR / "runs"
EDGEPILOT = WORKSPACE / "edgelite-compression-agent" / "scripts" / "edgepilot.py"
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


DEFAULT_PROMPT = """我有一个 Yolov8 模型 pt 路径为 /data/xl/Projects/EdgeLite/yolov8/weights/yolov8s-pose.pt，想要压缩实现希望在 NVIDIA T4 GPU 上部署 YOLOv8s-pose 模型，用于姿态检测场景。

要求在精度损失 <=1% 的前提下，将推理速度提升 >=2x。当前 demo 环境可以先用 NVIDIA L40 作为替代测试硬件。"""


def extract_request(payload: dict, include_demo_metrics: bool) -> dict:
    prompt = payload.get("prompt") or DEFAULT_PROMPT
    form = payload.get("form") or {}

    path_match = re.search(r"(/[\w./\-]+\.pt)", prompt)
    model_path = form.get("modelPath") or (path_match.group(1) if path_match else "/data/xl/Projects/EdgeLite/yolov8/weights/yolov8s-pose.pt")
    if str(model_path).startswith(str(WORKSPACE / "yolov8")):
        model = str(Path(model_path).resolve().relative_to(WORKSPACE / "yolov8"))
    elif str(model_path).startswith(str(WORKSPACE / "yolov5")):
        model = str(Path(model_path).resolve().relative_to(WORKSPACE / "yolov5"))
    else:
        model = model_path

    speedup_match = re.search(r"(?:>=|≥|提升)\s*([0-9]+(?:\.[0-9]+)?)\s*x?", prompt, re.I)
    drop_match = re.search(r"(?:<=|≤|不超过)\s*([0-9]+(?:\.[0-9]+)?)\s*%", prompt)
    project = form.get("project") or ("yolov5" if "yolov5" in str(model_path).lower() else "yolov8")
    task = form.get("task") or ("pose" if "pose" in prompt.lower() or "姿态" in prompt else "detect")
    if project == "yolov5":
        default_data = "data/coco.yaml"
        default_output = "output/edgepilot-web-huawei-yolov5-detect"
        default_calibration = "datasets/coco/images/train2017"
        default_train = default_calibration
        default_val = "datasets/coco/images/val2017"
        default_metric = "mAP50"
        default_accuracy = 56.8
    else:
        default_data = "datasets/my-coco-pose.yaml"
        default_output = "output/edgepilot-web-huawei-yolov8-pose"
        default_calibration = "datasets/coco-pose/images/train2017"
        default_train = default_calibration
        default_val = "datasets/coco-pose/images/val2017"
        default_metric = "mAP50(P)" if task == "pose" else "mAP50"
        default_accuracy = 85.6

    request = {
        "project": project,
        "task": task,
        "model": model,
        "data": form.get("dataYaml") or default_data,
        "output_dir": form.get("outputDir") or default_output,
        "calibration_data": form.get("calibrationData") or default_calibration,
        "calibration_size": int(form.get("calibrationSize") or 5000),
        "train_images": form.get("trainImages") or default_train,
        "val_images": form.get("valImages") or default_val,
        "coco_dir": form.get("cocoDir") or "datasets/coco",
        "runtime": {
            "device": form.get("device") or "cuda:0",
            "imgsz": int(form.get("imgsz") or 640),
            "batch": int(form.get("batch") or 16),
            "batch_min": 1,
            "batch_max": int(form.get("batchMax") or 16),
        },
        "target": {
            "hardware": form.get("targetHardware") or "NVIDIA T4",
            "demo_hardware": form.get("demoHardware") or "NVIDIA L40",
            "metric": form.get("metric") or default_metric,
            "baseline_latency_ms": float(form.get("baselineLatency") or 10.0),
            "baseline_accuracy": float(form.get("baselineAccuracy") or default_accuracy),
            "latency_ms_max": float(form.get("latencyMax") or 5.0),
            "speedup_min": float(form.get("speedupMin") or (speedup_match.group(1) if speedup_match else 2.0)),
            "accuracy_drop_max_pct": float(form.get("accuracyDropMax") or (drop_match.group(1) if drop_match else 1.0)),
        },
        "strategies": form.get("strategies") or ["fp16", "ptq", "qat", "prune_qat"],
        "qat_epochs": int(form.get("qatEpochs") or 30),
        "prune_steps": int(form.get("pruneSteps") or 16),
        "prune_epochs": int(form.get("pruneEpochs") or 120),
    }
    if include_demo_metrics:
        if project == "yolov5":
            request["demo_metrics"] = [
                {"name": "dense_pytorch_baseline", "strategy": "dense", "latency_ms": 10.0, "accuracy": 56.8},
                {"name": "fp16_trt_baseline", "strategy": "fp16", "latency_ms": 7.2, "accuracy": 56.7},
                {"name": "int8_ptq", "strategy": "ptq", "latency_ms": 4.8, "accuracy": 55.4},
                {"name": "int8_qat", "strategy": "qat", "latency_ms": 5.0, "accuracy": 56.3},
                {"name": "prune_0_3_qat", "strategy": "prune_qat", "latency_ms": 4.3, "accuracy": 56.0},
            ]
        else:
            request["demo_metrics"] = [
                {"name": "dense_pytorch_baseline", "strategy": "dense", "latency_ms": 10.0, "accuracy": 85.6},
                {"name": "fp16_trt_baseline", "strategy": "fp16", "latency_ms": 10.0, "accuracy": 85.6},
                {"name": "int8_ptq", "strategy": "ptq", "latency_ms": 4.6, "accuracy": 83.6},
                {"name": "int8_qat", "strategy": "qat", "latency_ms": 4.9, "accuracy": 85.4},
                {"name": "prune_0_3_qat", "strategy": "prune_qat", "latency_ms": 4.2, "accuracy": 84.9},
            ]
    return request


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def set_job(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(updates)


def append_log(job_id: str, text: str) -> None:
    with JOBS_LOCK:
        JOBS[job_id]["log"].append(text)


def job_snapshot(job_id: str) -> dict:
    with JOBS_LOCK:
        return json.loads(json.dumps(JOBS[job_id], ensure_ascii=False))


def load_result(run_dir: Path) -> dict:
    return {
        "paths": {
            "run_dir": str(run_dir),
            "request": str(run_dir / "request.json"),
            "plan": str(run_dir / "plan.json"),
            "env": str(run_dir / "env.json"),
            "evaluation": str(run_dir / "evaluation.json"),
            "commands": str(run_dir / "commands.sh"),
            "report": str(run_dir / "report.md"),
        },
        "plan": read_json(run_dir / "plan.json"),
        "env": read_json(run_dir / "env.json"),
        "evaluation": read_json(run_dir / "evaluation.json"),
        "report": (run_dir / "report.md").read_text(encoding="utf-8"),
        "commands": (run_dir / "commands.sh").read_text(encoding="utf-8"),
    }


def run_edgepilot_plan(job_id: str, request_path: Path, run_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(EDGEPILOT),
        "--workspace",
        str(WORKSPACE),
        "autopilot",
        "--request",
        str(request_path),
        "--output",
        str(run_dir),
    ]
    append_log(job_id, "$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
    proc = subprocess.run(cmd, cwd=str(WORKSPACE), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    append_log(job_id, proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"edgepilot plan failed with exit code {proc.returncode}")


def mark_candidate(job_id: str, name: str, status: str) -> None:
    with JOBS_LOCK:
        JOBS[job_id]["candidate_status"][name] = status


def execute_real_search(job_id: str, plan: dict) -> None:
    cwd = WORKSPACE
    for candidate in plan.get("candidates", []):
        name = candidate["name"]
        mark_candidate(job_id, name, "running")
        append_log(job_id, f"\n=== 执行候选: {name} ({candidate['strategy']}) ===\n")
        for raw_cmd in candidate.get("commands", []):
            cmd = raw_cmd.strip()
            append_log(job_id, f"\n$ {cmd}\n")
            if cmd.startswith("cd "):
                parts = shlex.split(cmd)
                if len(parts) >= 2:
                    cwd = Path(parts[1]).resolve()
                    append_log(job_id, f"[cwd] {cwd}\n")
                continue
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                append_log(job_id, line)
            rc = proc.wait()
            if rc != 0:
                mark_candidate(job_id, name, "failed")
                raise RuntimeError(f"candidate {name} failed with exit code {rc}")
        mark_candidate(job_id, name, "done")


def run_job(job_id: str, payload: dict) -> None:
    try:
        mode = "real_search" if payload.get("realSearch") else "demo_metrics"
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + job_id[:8]
        run_dir = RUNS_DIR / run_id
        request = extract_request(payload, include_demo_metrics=(mode == "demo_metrics"))
        request_path = run_dir / "request.json"
        write_json(request_path, request)
        set_job(job_id, status="running", mode=mode, run_id=run_id, run_dir=str(run_dir), request=request, stage="生成计划")
        append_log(job_id, f"[mode] {mode}\n")
        if mode == "demo_metrics":
            append_log(job_id, "[notice] 当前为快速演示模式，表格使用内置历史/示例指标，不代表本次真实执行。\n")
        else:
            append_log(job_id, "[notice] 当前为真实搜索模式，将逐个执行候选命令；不会伪造最终延迟/精度。\n")

        run_edgepilot_plan(job_id, request_path, run_dir)
        result = load_result(run_dir)
        set_job(job_id, candidate_status={c["name"]: "pending" for c in result["plan"].get("candidates", [])})

        if mode == "real_search":
            set_job(job_id, stage="执行候选搜索")
            execute_real_search(job_id, result["plan"])
            append_log(job_id, "\n[done] 候选命令执行完毕。请根据实际脚本产出的指标更新 evaluation.json。\n")

        result = load_result(run_dir)
        set_job(job_id, status="done", stage="完成", result=result)
    except Exception as exc:
        append_log(job_id, f"\n[error] {exc}\n")
        set_job(job_id, status="failed", stage="失败", error=str(exc))


class Handler(SimpleHTTPRequestHandler):
    server_version = "EdgePilotWebDemo/0.2"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        self.route(head_only=False)

    def do_HEAD(self) -> None:
        self.route(head_only=True)

    def route(self, head_only: bool = False) -> None:
        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            self.serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8", head_only=head_only)
            return
        if path == "/api/template":
            self.send_json({"prompt": DEFAULT_PROMPT}, head_only=head_only)
            return
        if path.startswith("/api/job/"):
            job_id = path.rsplit("/", 1)[-1]
            if job_id not in JOBS:
                self.send_json({"ok": False, "error": "job not found"}, status=404, head_only=head_only)
                return
            self.send_json({"ok": True, "job": job_snapshot(job_id)}, head_only=head_only)
            return
        if path.startswith("/static/"):
            rel = path.removeprefix("/static/")
            target = (STATIC_DIR / rel).resolve()
            if STATIC_DIR.resolve() in target.parents and target.exists():
                content_type = {
                    ".css": "text/css; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".svg": "image/svg+xml",
                }.get(target.suffix.lower(), "application/octet-stream")
                self.serve_file(target, content_type, head_only=head_only)
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path not in {"/api/start", "/api/run"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            job_id = uuid.uuid4().hex
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "id": job_id,
                    "status": "queued",
                    "stage": "排队",
                    "mode": "demo_metrics",
                    "log": [],
                    "candidate_status": {},
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            thread = threading.Thread(target=run_job, args=(job_id, payload), daemon=True)
            thread.start()
            self.send_json({"ok": True, "job_id": job_id})
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=500)

    def serve_file(self, path: Path, content_type: str, head_only: bool = False) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if not head_only:
            self.wfile.write(data)

    def send_json(self, data: dict, status: int = 200, head_only: bool = False) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        if not head_only:
            self.wfile.write(raw)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run EdgePilot web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"EdgePilot web demo: http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
