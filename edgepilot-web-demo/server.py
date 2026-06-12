#!/usr/bin/env python3
"""Minimal web demo server for EdgePilot.

The server intentionally uses only Python standard library modules so the demo
can run in a freshly cloned EdgeLite workspace.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote


WEB_DIR = Path(__file__).resolve().parent
WORKSPACE = WEB_DIR.parent
STATIC_DIR = WEB_DIR / "static"
RUNS_DIR = WEB_DIR / "runs"
EDGEPILOT = WORKSPACE / "edgelite-compression-agent" / "scripts" / "edgepilot.py"


DEFAULT_PROMPT = """我有一个 Yolov8 模型 pt 路径为 /data/xl/Projects/EdgeLite/yolov8/weights/yolov8s-pose.pt，想要压缩实现希望在 NVIDIA T4 GPU 上部署 YOLOv8s-pose 模型，用于姿态检测场景。

要求在精度损失 <=1% 的前提下，将推理速度提升 >=2x。当前 demo 环境可以先用 NVIDIA L40 作为替代测试硬件。"""


def extract_request(payload: dict) -> dict:
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

    target_hardware = form.get("targetHardware") or ("NVIDIA T4" if re.search(r"\bT4\b", prompt, re.I) else "NVIDIA T4")
    demo_hardware = form.get("demoHardware") or ("NVIDIA L40" if re.search(r"\bL40\b", prompt, re.I) else "NVIDIA L40")

    speedup_match = re.search(r"(?:>=|≥|提升)\s*([0-9]+(?:\.[0-9]+)?)\s*x?", prompt, re.I)
    drop_match = re.search(r"(?:<=|≤|不超过)\s*([0-9]+(?:\.[0-9]+)?)\s*%", prompt)

    project = form.get("project") or ("yolov5" if "yolov5" in str(model_path).lower() else "yolov8")
    task = form.get("task") or ("pose" if "pose" in prompt.lower() or "姿态" in prompt else "detect")

    return {
        "project": project,
        "task": task,
        "model": model,
        "data": form.get("dataYaml") or ("datasets/my-coco-pose.yaml" if project == "yolov8" else "data/coco.yaml"),
        "output_dir": "output/edgepilot-web-huawei-yolov8-pose",
        "calibration_data": form.get("calibrationData") or "datasets/coco-pose/images/train2017",
        "calibration_size": int(form.get("calibrationSize") or 5000),
        "train_images": form.get("trainImages") or "datasets/coco-pose/images/train2017",
        "val_images": form.get("valImages") or "datasets/coco-pose/images/val2017",
        "runtime": {
            "device": form.get("device") or "cuda:0",
            "imgsz": int(form.get("imgsz") or 640),
            "batch": int(form.get("batch") or 16),
            "batch_min": 1,
            "batch_max": int(form.get("batchMax") or 16),
        },
        "target": {
            "hardware": target_hardware,
            "demo_hardware": demo_hardware,
            "metric": form.get("metric") or ("mAP50(P)" if task == "pose" else "mAP50"),
            "baseline_latency_ms": float(form.get("baselineLatency") or 10.0),
            "baseline_accuracy": float(form.get("baselineAccuracy") or 85.6),
            "latency_ms_max": float(form.get("latencyMax") or 5.0),
            "speedup_min": float(form.get("speedupMin") or (speedup_match.group(1) if speedup_match else 2.0)),
            "accuracy_drop_max_pct": float(form.get("accuracyDropMax") or (drop_match.group(1) if drop_match else 1.0)),
        },
        "strategies": form.get("strategies") or ["fp16", "ptq", "qat", "prune_qat"],
        "qat_epochs": int(form.get("qatEpochs") or 30),
        "prune_steps": int(form.get("pruneSteps") or 16),
        "prune_epochs": int(form.get("pruneEpochs") or 120),
        "demo_metrics": [
            {"name": "dense_pytorch_baseline", "strategy": "dense", "latency_ms": 10.0, "accuracy": 85.6},
            {"name": "fp16_trt_baseline", "strategy": "fp16", "latency_ms": 10.0, "accuracy": 85.6},
            {"name": "int8_ptq", "strategy": "ptq", "latency_ms": 4.6, "accuracy": 83.6},
            {"name": "int8_qat", "strategy": "qat", "latency_ms": 4.9, "accuracy": 85.4},
            {"name": "prune_0_3_qat", "strategy": "prune_qat", "latency_ms": 4.2, "accuracy": 84.9},
        ],
    }


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


class Handler(SimpleHTTPRequestHandler):
    server_version = "EdgePilotWebDemo/0.1"

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        self.route_static(head_only=False)

    def do_HEAD(self) -> None:
        self.route_static(head_only=True)

    def route_static(self, head_only: bool = False) -> None:
        path = unquote(self.path.split("?", 1)[0])
        if path == "/":
            self.serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8", head_only=head_only)
            return
        if path.startswith("/static/"):
            rel = path.removeprefix("/static/")
            target = (STATIC_DIR / rel).resolve()
            if STATIC_DIR.resolve() in target.parents and target.exists():
                suffix = target.suffix.lower()
                content_type = {
                    ".css": "text/css; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".svg": "image/svg+xml",
                }.get(suffix, "application/octet-stream")
                self.serve_file(target, content_type, head_only=head_only)
                return
        if path == "/api/template":
            self.send_json({"prompt": DEFAULT_PROMPT}, head_only=head_only)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            request = extract_request(payload)
            run_id = time.strftime("%Y%m%d-%H%M%S")
            run_dir = RUNS_DIR / run_id
            request_path = run_dir / "request.json"
            write_json(request_path, request)

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
            if payload.get("execute"):
                cmd.extend(["--execute", "--yes"])
            proc = subprocess.run(cmd, cwd=str(WORKSPACE), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if proc.returncode != 0:
                self.send_json({"ok": False, "stdout": proc.stdout, "stderr": proc.stderr}, status=500)
                return

            response = {
                "ok": True,
                "run_id": run_id,
                "request": request,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "paths": {
                    "run_dir": str(run_dir),
                    "request": str(request_path),
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
            self.send_json(response)
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
