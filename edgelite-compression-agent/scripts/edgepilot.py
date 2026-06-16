#!/usr/bin/env python3
"""EdgePilot: planning and demo runner for EdgeLite compression workflows."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import site
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DEFAULT_WORKSPACE = SKILL_DIR.parent

OFFICIAL_ASSET_REGISTRY = {
    "models": {
        "yolov8s-pose.pt": [
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-pose.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-pose.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-pose.pt",
        ],
        "yolov8n-pose.pt": [
            "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-pose.pt",
        ],
        "yolov5s.pt": [
            "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
        ],
    },
    "datasets": {
        "coco8-pose": [
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8-pose.zip",
        ],
        "coco8": [
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
        ],
    },
}

SUPPORTED_ADAPTERS = {
    "yolov8": "production",
    "yolov5": "production",
}

KNOWN_MODEL_FAMILIES = {"vit", "resnet", "bert", "llama", "qwen", "deepseek", "ddpm", "stable-diffusion"}
TRUSTED_HF_ORGS = {
    "ultralytics",
    "google",
    "facebook",
    "meta-llama",
    "microsoft",
    "openai",
    "nvidia",
    "timm",
    "pytorch",
    "huggingface",
}
BAD_WEIGHT_NAME_PARTS = {"optimizer", "scheduler", "trainer", "training_args", "scaler", "ema"}


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


def edgepilot_subprocess_env() -> Dict[str, str]:
    env = os.environ.copy()
    candidates: List[Path] = []
    site_roots: List[str] = []
    try:
        site_roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        site_roots.append(site.getusersitepackages())
    except Exception:
        pass
    for root in dict.fromkeys(site_roots):
        nvidia_root = Path(root) / "nvidia"
        if nvidia_root.exists():
            candidates.extend(path for path in nvidia_root.glob("*/lib") if path.is_dir())

    # PyTorch CUDA wheels may require nvJitLink symbols newer than the system CUDA.
    candidates = sorted(dict.fromkeys(candidates), key=lambda p: (0 if "nvjitlink" in str(p).lower() else 1, str(p)))
    if candidates:
        existing = env.get("LD_LIBRARY_PATH", "")
        prefix = ":".join(str(path) for path in candidates)
        env["LD_LIBRARY_PATH"] = f"{prefix}:{existing}" if existing else prefix
    return env


def edgepilot_ld_library_export() -> Optional[str]:
    current = os.environ.get("LD_LIBRARY_PATH", "")
    patched = edgepilot_subprocess_env().get("LD_LIBRARY_PATH", "")
    if not patched or patched == current:
        return None
    prefix = patched
    if current and patched.endswith(f":{current}"):
        prefix = patched[: -(len(current) + 1)]
    return f"export LD_LIBRARY_PATH={shlex.quote(prefix)}:${{LD_LIBRARY_PATH:-}}"


def run_probe(cmd: List[str], timeout: int = 15, cwd: Optional[Path] = None) -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            env=edgepilot_subprocess_env(),
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


def local_ultralytics_probe(workspace: Path) -> Dict[str, Any]:
    yolov8_root = workspace / "yolov8"
    if not yolov8_root.exists():
        return {"ok": False, "error": "yolov8 project not found", "stdout": "", "stderr": ""}
    code = (
        "from pathlib import Path\n"
        "import ultralytics\n"
        "actual = Path(ultralytics.__file__).resolve()\n"
        "expected = Path.cwd().resolve() / 'ultralytics'\n"
        "print(actual)\n"
        "print(getattr(ultralytics, '__version__', 'available'))\n"
        "raise SystemExit(0 if expected in actual.parents else 2)\n"
    )
    return run_probe([sys.executable, "-c", code], cwd=yolov8_root)


def local_yolov5_probe(workspace: Path) -> Dict[str, Any]:
    yolov5_root = workspace / "yolov5"
    if not yolov5_root.exists():
        return {"ok": False, "error": "yolov5 project not found", "stdout": "", "stderr": ""}
    code = (
        "from pathlib import Path\n"
        "import models.yolo as yolo\n"
        "import utils.general as general\n"
        "root = Path.cwd().resolve()\n"
        "print(Path(yolo.__file__).resolve())\n"
        "print(Path(general.__file__).resolve())\n"
        "raise SystemExit(0 if root in Path(yolo.__file__).resolve().parents else 2)\n"
    )
    return run_probe([sys.executable, "-c", code], cwd=yolov5_root)


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
    if trtexec_path and "TensorRT.trtexec" in (trtexec.get("stdout", "") + trtexec.get("stderr", "")):
        trtexec["ok"] = True

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
            "local_ultralytics": local_ultralytics_probe(workspace),
            "local_yolov5": local_yolov5_probe(workspace),
        },
        "projects": projects,
    }


def has_edgelite_layout(workspace: Path) -> bool:
    return (workspace / "edgelite-compression-agent").exists() and (
        (workspace / "yolov8").exists() or (workspace / "yolov5").exists()
    )


def shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def candidate_python(python_env: Optional[str]) -> str:
    if not python_env:
        return sys.executable
    env = Path(python_env).expanduser()
    if env.is_dir():
        for rel in ("bin/python", "Scripts/python.exe", "python"):
            py = env / rel
            if py.exists():
                return str(py)
    return str(env)


def find_urls_in_repo(root: Path, filename_hint: str) -> List[str]:
    """Search small repo docs/configs for URLs related to a model or dataset."""
    if not root.exists():
        return []
    candidates = []
    patterns = ["README*.md", "*.yaml", "*.yml", "*.txt"]
    for pattern in patterns:
        candidates.extend(root.rglob(pattern))
    urls: List[str] = []
    hint = filename_hint.lower()
    for path in candidates[:300]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if hint and hint not in text.lower() and hint not in path.name.lower():
            continue
        for url in re.findall(r"https?://[^\s)\"']+", text):
            cleaned = url.rstrip(".,")
            if not hint or hint in cleaned.lower() or "download" in cleaned.lower() or "assets" in cleaned.lower():
                urls.append(cleaned)
    return list(dict.fromkeys(urls))


def first_available_url(urls: List[str], timeout: int = 3) -> Optional[str]:
    for url in urls[:3]:
        probe = run_probe(
            ["curl", "-sS", "-L", "-I", "--connect-timeout", "2", "--max-time", str(timeout), url],
            timeout=timeout + 2,
        )
        stdout = probe.get("stdout", "")
        if probe.get("ok") and (" 200 " in stdout or " 302 " in stdout):
            return url
    return urls[0] if urls else None


def load_remote_json(url: str, timeout: int = 4) -> Optional[Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except Exception:
        return None


def hf_model_urls(query: str, preferred_suffix: str = "") -> List[str]:
    """Best-effort Hugging Face model search for official-ish weight files."""
    if not query:
        return []
    search_url = "https://huggingface.co/api/models?search=" + urllib.parse.quote(query) + "&limit=5"
    models = load_remote_json(search_url)
    if not isinstance(models, list):
        return []

    def score_model(item: Dict[str, Any]) -> tuple:
        model_id = item.get("modelId") or item.get("id") or ""
        org = model_id.split("/", 1)[0].lower() if "/" in model_id else ""
        trusted = 0 if org in TRUSTED_HF_ORGS else 1
        downloads = -(int(item.get("downloads") or 0))
        exact = 0 if query.lower().replace(".pt", "") in model_id.lower() else 1
        return (trusted, exact, downloads)

    urls: List[str] = []
    canonical_weight_names = [
        "pytorch_model.bin",
        "model.safetensors",
        "model.bin",
        "tf_model.h5",
        "flax_model.msgpack",
    ]
    query_tokens = [t for t in re.split(r"[^a-zA-Z0-9]+", query.lower()) if len(t) >= 3]

    def is_probable_weight(name: str) -> bool:
        base = Path(name).name.lower()
        if any(part in base for part in BAD_WEIGHT_NAME_PARTS):
            return False
        if base in canonical_weight_names:
            return True
        if preferred_suffix and base == query.lower():
            return True
        if preferred_suffix and base.endswith(preferred_suffix.lower()):
            return any(token in base for token in query_tokens) or "model" in base or "weight" in base
        return False

    for item in sorted(models, key=score_model)[:3]:
        model_id = item.get("modelId") or item.get("id")
        if not model_id:
            continue
        org = model_id.split("/", 1)[0].lower() if "/" in model_id else ""
        if org not in TRUSTED_HF_ORGS:
            continue
        info = load_remote_json("https://huggingface.co/api/models/" + urllib.parse.quote(model_id, safe="/"))
        if not isinstance(info, dict):
            continue
        siblings = [s.get("rfilename") for s in info.get("siblings", []) if s.get("rfilename")]
        preferred = [name for name in siblings if is_probable_weight(name)]
        for name in canonical_weight_names:
            if name in siblings and name not in preferred:
                preferred.append(name)
        for name in preferred[:2]:
            urls.append(f"https://huggingface.co/{model_id}/resolve/main/{urllib.parse.quote(name, safe='/')}")
    return list(dict.fromkeys(urls))


def hf_dataset_urls(query: str) -> List[str]:
    """Return direct zip-like assets from Hugging Face datasets when present."""
    if not query:
        return []
    search_url = "https://huggingface.co/api/datasets?search=" + urllib.parse.quote(query) + "&limit=3"
    datasets = load_remote_json(search_url)
    if not isinstance(datasets, list):
        return []
    urls: List[str] = []
    for item in datasets[:2]:
        dataset_id = item.get("id")
        if not dataset_id:
            continue
        info = load_remote_json("https://huggingface.co/api/datasets/" + urllib.parse.quote(dataset_id, safe="/"))
        if not isinstance(info, dict):
            continue
        for sibling in info.get("siblings", []):
            name = sibling.get("rfilename")
            if name and name.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                urls.append(f"https://huggingface.co/datasets/{dataset_id}/resolve/main/{urllib.parse.quote(name, safe='/')}")
    return list(dict.fromkeys(urls))


def download_url(url: str, destination: Path, timeout: int = 1800) -> Dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    try:
        if shutil.which("curl"):
            probe = run_probe(
                [
                    "curl",
                    "-L",
                    "--fail",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    str(timeout),
                    "-o",
                    str(tmp),
                    url,
                ],
                timeout=timeout + 30,
            )
            if not probe["ok"]:
                if tmp.exists():
                    tmp.unlink()
                return {
                    "ok": False,
                    "error": probe.get("stderr") or probe.get("stdout") or probe.get("error", "curl failed"),
                    "path": str(destination),
                }
        else:
            with urllib.request.urlopen(url, timeout=timeout) as response, tmp.open("wb") as f:
                shutil.copyfileobj(response, f)
        tmp.replace(destination)
        return {"ok": True, "path": str(destination), "bytes": destination.stat().st_size}
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return {"ok": False, "error": str(exc), "path": str(destination)}


def safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise RuntimeError(f"unsafe zip path: {member.filename}")
        zf.extractall(destination)


def dataset_name_for_request(req: Dict[str, Any]) -> str:
    if req.get("dataset_name"):
        return str(req["dataset_name"])
    if req.get("project") not in {"yolov5", "yolov8"}:
        return ""
    if req.get("task") == "pose":
        return "coco8-pose"
    return "coco8"


def infer_project(request: Dict[str, Any]) -> str:
    project = str(request.get("project", "")).lower().strip()
    if project:
        return project
    text = " ".join(str(request.get(k, "")) for k in ("model", "task", "architecture", "model_family")).lower()
    for name in ("yolov8", "yolov5", "vit", "resnet", "bert", "llama", "qwen", "deepseek", "ddpm"):
        if name in text:
            return name
    return "generic"


def adapter_status(project: str) -> Dict[str, Any]:
    if project in SUPPORTED_ADAPTERS:
        return {"supported": True, "level": SUPPORTED_ADAPTERS[project], "message": "已内置真实执行 adapter。"}
    if project in KNOWN_MODEL_FAMILIES:
        return {
            "supported": False,
            "level": "adapter_required",
            "message": f"识别到 {project} 模型族，但当前仓库未内置真实执行 adapter；只能做 bootstrap、通用规划和风险报告。",
        }
    return {
        "supported": False,
        "level": "unknown",
        "message": "未识别模型族；需要用户提供加载、评估、导出和部署脚本接口。",
    }


def project_root_for_request(workspace: Path, req: Dict[str, Any]) -> Path:
    project = req.get("project", "generic")
    if project in {"yolov5", "yolov8"}:
        return workspace / project
    root = req.get("project_root") or req.get("source_root")
    return Path(root).expanduser().resolve() if root else workspace


def resolve_request_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else project_root / path


def yaml_for_downloaded_dataset(project_root: Path, dataset_name: str) -> Path:
    if dataset_name == "coco8-pose":
        yaml_path = project_root / "datasets" / "coco8-pose.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(
            "path: datasets/coco8-pose\n"
            "train: images/train\n"
            "val: images/val\n"
            "kpt_shape: [17, 3]\n"
            "names:\n"
            "  0: person\n",
            encoding="utf-8",
        )
        return yaml_path
    yaml_path = project_root / "datasets" / "coco8.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "path: datasets/coco8\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: person\n",
        encoding="utf-8",
    )
    return yaml_path


def resolve_model_urls(project_root: Path, model_name: str) -> List[str]:
    urls = find_urls_in_repo(project_root, model_name)
    urls.extend(OFFICIAL_ASSET_REGISTRY["models"].get(model_name, []))
    suffix = Path(model_name).suffix
    query = Path(model_name).stem or model_name
    normalized_query = query.lower().replace("_", "-")
    if normalized_query.startswith("vit-base-patch16") or normalized_query.startswith("vit-base"):
        urls.extend([
            "https://huggingface.co/google/vit-base-patch16-224/resolve/main/model.safetensors",
            "https://huggingface.co/google/vit-base-patch16-224/resolve/main/pytorch_model.bin",
        ])
    urls.extend(hf_model_urls(model_name, preferred_suffix=suffix))
    if query != model_name:
        urls.extend(hf_model_urls(query, preferred_suffix=suffix))
    return list(dict.fromkeys(urls))


def resolve_dataset_urls(project_root: Path, dataset_name: str) -> List[str]:
    urls = find_urls_in_repo(project_root, dataset_name)
    urls.extend(OFFICIAL_ASSET_REGISTRY["datasets"].get(dataset_name, []))
    urls.extend(hf_dataset_urls(dataset_name))
    return list(dict.fromkeys(urls))


def bootstrap_workspace(
    workspace: Path,
    repo_url: str,
    request_path: Optional[Path],
    output_dir: Path,
    python_env: Optional[str] = None,
    create_venv: bool = False,
    install_deps: bool = False,
    prepare_demo_data: bool = False,
    auto_download_assets: bool = False,
    yes: bool = False,
) -> Dict[str, Any]:
    """Prepare or describe the minimum environment needed to run EdgePilot."""
    output_dir.mkdir(parents=True, exist_ok=True)
    actions: List[Dict[str, Any]] = []
    warnings: List[str] = []

    clone_needed = not has_edgelite_layout(workspace)
    if clone_needed:
        cmd = ["git", "clone", repo_url, str(workspace)]
        action = {
            "name": "clone_edgelite",
            "needed": True,
            "executed": False,
            "command": shell_join(cmd),
            "reason": "未在 workspace 中发现 EdgeLite 项目布局。",
        }
        if yes:
            workspace.parent.mkdir(parents=True, exist_ok=True)
            probe = run_probe(cmd, timeout=180)
            action.update({"executed": True, "ok": probe["ok"], "stdout": probe.get("stdout"), "stderr": probe.get("stderr")})
            if not probe["ok"]:
                warnings.append("EdgeLite clone 失败，请检查网络、GitHub 权限或 repo_url。")
        actions.append(action)
    else:
        actions.append({
            "name": "clone_edgelite",
            "needed": False,
            "executed": False,
            "reason": "已发现 EdgeLite 项目布局。",
        })

    py = candidate_python(python_env)
    venv_dir = workspace / ".venv-edgepilot"
    if create_venv:
        cmd = [sys.executable, "-m", "venv", str(venv_dir)]
        action = {
            "name": "create_python_env",
            "needed": True,
            "executed": False,
            "command": shell_join(cmd),
            "python": str(venv_dir / "bin" / "python"),
        }
        if yes:
            probe = run_probe(cmd, timeout=120)
            action.update({"executed": True, "ok": probe["ok"], "stdout": probe.get("stdout"), "stderr": probe.get("stderr")})
            if probe["ok"]:
                py = str(venv_dir / "bin" / "python")
            else:
                warnings.append("Python venv 创建失败，将继续使用当前 Python。")
        actions.append(action)
    else:
        actions.append({
            "name": "select_python_env",
            "needed": False,
            "executed": False,
            "python": py,
            "reason": "使用用户提供的 Python 环境或当前 Python。",
        })

    req_files = []
    for rel in ("yolov8/requirements.txt", "yolov5/requirements.txt"):
        path = workspace / rel
        if path.exists():
            req_files.append(path)
        elif clone_needed:
            req_files.append(path)
    if install_deps:
        for req_file in req_files:
            cmd = [py, "-m", "pip", "install", "-r", str(req_file)]
            action = {
                "name": f"install_deps_{req_file.parent.name}",
                "needed": True,
                "executed": False,
                "command": shell_join(cmd),
            }
            if yes:
                probe = run_probe(cmd, timeout=1800, cwd=req_file.parent)
                action.update({"executed": True, "ok": probe["ok"], "stdout": probe.get("stdout")[-2000:], "stderr": probe.get("stderr")[-2000:]})
                if not probe["ok"]:
                    warnings.append(f"{req_file} 依赖安装失败，请检查 CUDA/PyTorch/TensorRT 版本匹配。")
            actions.append(action)
    else:
        actions.append({
            "name": "install_deps",
            "needed": bool(req_files),
            "executed": False,
            "commands": [shell_join([py, "-m", "pip", "install", "-r", str(p)]) for p in req_files],
            "reason": "默认不自动安装依赖；需要 --install-deps --yes。",
        })

    request = load_json(request_path) if request_path and request_path.exists() else {}
    normalized = request_defaults(request) if request else request_defaults({})
    project_root = project_root_for_request(workspace, normalized)
    model_path = resolve_request_path(project_root, normalized["model"]) if normalized.get("model") else project_root
    data_path = resolve_request_path(project_root, normalized["data"]) if normalized.get("data") else project_root
    calibration_path = resolve_request_path(project_root, normalized.get("calibration_data", "")) if normalized.get("calibration_data") else project_root
    missing = {
        "model": bool(normalized.get("model")) and not model_path.exists(),
        "data_yaml": bool(normalized.get("data")) and not data_path.exists(),
        "calibration_data": bool(normalized.get("calibration_data")) and not calibration_path.exists(),
    }
    adapter = adapter_status(normalized["project"])
    if not adapter["supported"]:
        warnings.append(adapter["message"])

    model_name = Path(normalized["model"]).name
    model_urls = resolve_model_urls(project_root, model_name)
    selected_model_url = first_available_url(model_urls) if missing["model"] and model_urls else None
    model_action = {
        "name": "download_model",
        "needed": missing["model"],
        "executed": False,
        "target": str(model_path),
        "source": selected_model_url,
        "candidates": model_urls,
        "search_hint": f"{model_name} official pretrained weights download",
    }
    if missing["model"] and auto_download_assets:
        if selected_model_url and yes:
            result = download_url(selected_model_url, model_path)
            model_action.update({"executed": True, **result})
            missing["model"] = not model_path.exists()
            if not result["ok"]:
                warnings.append(f"模型自动下载失败: {result.get('error')}")
        elif selected_model_url:
            model_action["command"] = shell_join(["curl", "-L", "-o", str(model_path), selected_model_url])
        else:
            warnings.append(f"未找到 {model_name} 的可用官方下载源。")
    elif missing["model"]:
        warnings.append(f"模型文件不存在: {model_path}。可使用 --auto-download-assets --yes 自动尝试下载。")
    actions.append(model_action)

    dataset_name = dataset_name_for_request(normalized)
    dataset_urls = resolve_dataset_urls(project_root, dataset_name) if dataset_name else []
    selected_dataset_url = first_available_url(dataset_urls) if (missing["data_yaml"] or missing["calibration_data"]) and dataset_urls else None
    dataset_zip = project_root / "datasets" / f"{dataset_name or 'edgepilot-dataset'}.zip"
    dataset_action = {
        "name": "download_dataset",
        "needed": missing["data_yaml"] or missing["calibration_data"],
        "executed": False,
        "target": str(project_root / "datasets" / dataset_name) if dataset_name else str(data_path),
        "source": selected_dataset_url,
        "candidates": dataset_urls,
        "search_hint": f"{dataset_name} official dataset download" if dataset_name else f"{normalized.get('data') or normalized.get('task')} official validation dataset download",
        "note": "默认只为已知任务下载官方小样例数据；正式精度验收仍需目标验证集。",
    }
    if (missing["data_yaml"] or missing["calibration_data"]) and auto_download_assets:
        if not dataset_name:
            warnings.append("非 YOLO/未知数据集未配置 dataset_name，无法可靠自动下载数据；请提供 data 或 dataset_name。")
        elif selected_dataset_url and yes:
            result = download_url(selected_dataset_url, dataset_zip)
            dataset_action.update({"download": result})
            if result["ok"]:
                try:
                    safe_extract_zip(dataset_zip, project_root / "datasets")
                    downloaded_yaml = yaml_for_downloaded_dataset(project_root, dataset_name)
                    if missing["data_yaml"]:
                        normalized["data"] = str(downloaded_yaml.relative_to(project_root))
                        data_path = downloaded_yaml
                    if missing["calibration_data"]:
                        normalized["calibration_data"] = f"datasets/{dataset_name}/images/train"
                        normalized["train_images"] = f"datasets/{dataset_name}/images/train"
                        normalized["val_images"] = f"datasets/{dataset_name}/images/val"
                        calibration_path = project_root / normalized["calibration_data"]
                    missing["data_yaml"] = not data_path.exists()
                    missing["calibration_data"] = bool(normalized.get("calibration_data")) and not calibration_path.exists()
                    dataset_action.update({"executed": True, "ok": True, "yaml": str(data_path), "calibration_data": str(calibration_path)})
                except Exception as exc:
                    dataset_action.update({"executed": True, "ok": False, "error": str(exc)})
                    warnings.append(f"数据集解压或 yaml 生成失败: {exc}")
            else:
                warnings.append(f"数据集自动下载失败: {result.get('error')}")
        elif selected_dataset_url:
            dataset_action["commands"] = [
                shell_join(["curl", "-L", "-o", str(dataset_zip), selected_dataset_url]),
                shell_join(["python", "-m", "zipfile", "-e", str(dataset_zip), str(project_root / "datasets")]),
            ]
        else:
            warnings.append(f"未找到 {dataset_name} 的可用官方下载源。")
    elif missing["data_yaml"] or missing["calibration_data"]:
        warnings.append("数据或校准集不存在。可使用 --auto-download-assets --yes 自动下载官方小样例数据。")
    actions.append(dataset_action)

    if missing["model"]:
        warnings.append(f"模型文件仍不存在: {model_path}。真实压缩前必须提供权重。")
    if missing["data_yaml"]:
        warnings.append(f"数据配置仍不存在: {data_path}。真实精度评估前必须提供数据 yaml。")
    if missing["calibration_data"]:
        warnings.append(f"校准数据仍不存在: {calibration_path}。PTQ/INT8 前必须准备校准集。")

    demo_data_dir = project_root / "datasets" / "edgepilot-mini"
    demo_yaml = project_root / "datasets" / "edgepilot-mini-coco-pose.yaml"
    data_action = {
        "name": "prepare_demo_data",
        "needed": prepare_demo_data,
        "executed": False,
        "path": str(demo_data_dir),
        "yaml": str(demo_yaml),
        "note": "mini 数据只用于流程 smoke test，不可作为正式 mAP 结论。",
    }
    if prepare_demo_data and yes:
        (demo_data_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (demo_data_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (demo_data_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (demo_data_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        source_images = list((project_root / "images").glob("*.*"))[:4]
        for idx, src in enumerate(source_images):
            dst_train = demo_data_dir / "images" / "train" / f"sample_{idx}{src.suffix}"
            dst_val = demo_data_dir / "images" / "val" / f"sample_{idx}{src.suffix}"
            shutil.copy2(src, dst_train)
            shutil.copy2(src, dst_val)
            (demo_data_dir / "labels" / "train" / f"sample_{idx}.txt").write_text("", encoding="utf-8")
            (demo_data_dir / "labels" / "val" / f"sample_{idx}.txt").write_text("", encoding="utf-8")
        demo_yaml.write_text(
            "path: datasets/edgepilot-mini\n"
            "train: images/train\n"
            "val: images/val\n"
            "kpt_shape: [17, 3]\n"
            "names:\n"
            "  0: person\n",
            encoding="utf-8",
        )
        data_action["executed"] = True
        data_action["samples"] = len(source_images)
        if not source_images:
            warnings.append("未找到可复制的样例图片，mini 数据目录已创建但为空。")
    actions.append(data_action)

    env = inspect_environment(workspace)
    trt_available = bool(env.get("tools", {}).get("trtexec", {}).get("ok")) or bool(env.get("tools", {}).get("tensorrt", {}).get("ok"))
    if not trt_available:
        warnings.append("未检测到 TensorRT/trtexec。可先规划策略，但真实 engine 构建和延迟测试会失败。")

    bootstrap_path = output_dir / "bootstrap.json"
    bootstrap_report_path = output_dir / "bootstrap.md"
    resolved_request_path = output_dir / "resolved_request.json"
    write_json(resolved_request_path, normalized)

    bootstrap = {
        "generated_at": now(),
        "workspace": str(workspace),
        "repo_url": repo_url,
        "python": py,
        "request": normalized,
        "artifacts": {
            "bootstrap": str(bootstrap_path),
            "report": str(bootstrap_report_path),
            "resolved_request": str(resolved_request_path),
        },
        "checks": {
            "layout_ready": has_edgelite_layout(workspace),
            "model_path": str(model_path),
            "data_yaml": str(data_path),
            "calibration_data": str(calibration_path) if normalized.get("calibration_data") else None,
            "missing": missing,
            "tensorrt_available": trt_available,
            "adapter": adapter,
        },
        "actions": actions,
        "warnings": warnings,
        "env": env,
    }
    write_json(bootstrap_path, bootstrap)
    bootstrap_report_path.write_text(render_bootstrap_report(bootstrap), encoding="utf-8")
    return bootstrap


def render_bootstrap_report(bootstrap: Dict[str, Any]) -> str:
    action_rows = []
    for action in bootstrap["actions"]:
        summary = (
            action.get("reason")
            or action.get("command")
            or action.get("source")
            or "; ".join(action.get("commands", []))
            or action.get("note")
            or action.get("search_hint")
            or "-"
        )
        action_rows.append(
            "| {name} | {needed} | {executed} | {summary} |".format(
                name=action["name"],
                needed="是" if action.get("needed") else "否",
                executed="是" if action.get("executed") else "否",
                summary=str(summary).replace("|", "\\|"),
            )
        )
    warnings = "\n".join(f"- {w}" for w in bootstrap.get("warnings", [])) or "- 无"
    source_lines = []
    for action in bootstrap["actions"]:
        if not action.get("needed"):
            continue
        if action.get("source"):
            source_lines.append(f"- {action['name']}: {action['source']}")
        elif action.get("search_hint"):
            source_lines.append(f"- {action['name']}: 未找到直接 URL；建议搜索 `{action['search_hint']}`")
    sources = "\n".join(source_lines) or "- 无"
    req = bootstrap["request"]
    artifacts = bootstrap.get("artifacts", {})
    resolved_request = artifacts.get("resolved_request", "edgepilot_bootstrap_run/resolved_request.json")
    return f"""# EdgePilot Bootstrap 报告

生成时间: {bootstrap['generated_at']}

## 工作区

- Workspace: `{bootstrap['workspace']}`
- Repo: `{bootstrap['repo_url']}`
- Python: `{bootstrap['python']}`

## 需求摘要

- 项目: `{req['project']}`
- 任务: `{req['task']}`
- 模型: `{req['model']}`
- 数据: `{req['data']}`
- 目标硬件: `{req['target']['hardware']}`
- Adapter: `{bootstrap['checks']['adapter'].get('level')}` - {bootstrap['checks']['adapter'].get('message')}

## 就绪检查

- 项目布局: `{'ready' if bootstrap['checks']['layout_ready'] else 'missing'}`
- 模型路径: `{bootstrap['checks']['model_path']}`
- 数据配置: `{bootstrap['checks']['data_yaml']}`
- 校准数据: `{bootstrap['checks']['calibration_data']}`
- TensorRT: `{'available' if bootstrap['checks']['tensorrt_available'] else 'not detected'}`

## Bootstrap 动作

| 动作 | 需要 | 已执行 | 摘要 |
| --- | --- | --- | --- |
{chr(10).join(action_rows)}

## 资源来源

{sources}

## 风险与缺口

{warnings}

## 下一步

1. 解决缺失的模型、数据、TensorRT 或 Python 依赖。
2. 使用 bootstrap 产出的 `resolved_request.json` 运行 `edgepilot.py autopilot`，确保自动下载/解压后的路径被后续流程使用：

```bash
python edgelite-compression-agent/scripts/edgepilot.py \\
  --workspace {bootstrap['workspace']} \\
  autopilot \\
  --request {resolved_request} \\
  --output edgepilot_autopilot_run
```

3. 只有在确认数据和环境后，才加 `--execute --yes` 执行训练、剪枝和 TensorRT 构建。
"""


def request_defaults(request: Dict[str, Any]) -> Dict[str, Any]:
    project = infer_project(request)
    task = request.get("task", "pose" if project == "yolov8" else "detect")
    if project == "yolov8":
        model = request.get("model", "weights/yolov8s-pose.pt")
        data = request.get("data", "datasets/my-coco-pose.yaml")
    elif project == "yolov5":
        model = request.get("model", "models/yolov5s.pt")
        data = request.get("data", "data/coco.yaml")
    else:
        model = request.get("model", "")
        data = request.get("data", "")
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
        "adapter": adapter_status(project),
    })
    return normalized


def sh_cd(workspace: Path, project: str) -> str:
    return f'cd "{workspace / project}"'


def conda_run(env_name: str, command: str, extra_env: Optional[Dict[str, str]] = None) -> str:
    prefixes = []
    for key, value in (extra_env or {}).items():
        prefixes.append(f"{key}={shlex.quote(value)}")
    prefixes.append("conda")
    prefixes.extend(["run", "-n", shlex.quote(env_name)])
    return " ".join(prefixes) + f" {command}"


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
    export_device = str(device).replace("cuda:", "") if str(device).startswith("cuda:") else str(device)
    fp16_onnx = str(Path(model).with_suffix(".onnx"))
    fp16_engine = f"{out}/fp16/yolov8-fp16.engine"

    py_export_fp16 = (
        "python - <<'PY'\n"
        "from ultralytics import YOLO\n"
        f"model = YOLO('{model}', task='{req['task']}')\n"
        f"path = model.export(format='onnx', half=True, dynamic=True, imgsz={imgsz}, batch={batch}, opset=17, simplify=True, device='{export_device}')\n"
        "print(path)\n"
        "PY"
    )
    build_fp16_engine = (
        f"mkdir -p {out}/fp16 && "
        "trtexec "
        f"--onnx={fp16_onnx} --saveEngine={fp16_engine} --fp16 "
        f"--minShapes=images:{batch_min}x3x{imgsz}x{imgsz} "
        f"--optShapes=images:{batch}x3x{imgsz}x{imgsz} "
        f"--maxShapes=images:{batch_max}x3x{imgsz}x{imgsz}"
    )

    return [
        {
            "name": "fp16_trt_baseline",
            "strategy": "fp16",
            "purpose": "Build a TensorRT FP16 baseline and measure the deployment floor.",
            "commands": [sh_cd(workspace, "yolov8"), py_export_fp16, build_fp16_engine],
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
    env_name = req.get("env_name") or req.get("yolov5_env_name") or "yolov5-compress"
    v5_env = {"YOLOv5_AUTOINSTALL": "false"}
    py = lambda command: conda_run(env_name, f"python {command}", v5_env)
    trt = lambda command: conda_run(env_name, command, v5_env)

    return [
        {
            "name": "fp16_trt_baseline",
            "strategy": "fp16",
            "purpose": "Export YOLOv5 to TensorRT FP16 engine.",
            "commands": [
                sh_cd(workspace, "yolov5"),
                py(f"export.py --weights {model} --include engine --half --dynamic --batch-size {batch} --imgsz {imgsz}"),
                py(f"val.py --weights {model.replace('.pt', '.engine')} --data {data} --batch-size {batch} --imgsz {imgsz}"),
            ],
        },
        {
            "name": "int8_ptq",
            "strategy": "ptq",
            "purpose": "Calibrate YOLOv5 INT8 PTQ and export the calibrated checkpoint to ONNX/TensorRT.",
            "commands": [
                sh_cd(workspace, "yolov5"),
                (
                    py("scripts/qat.py quantize "
                    f"{model} --cocodir {cocodir} --device {rt['device']} "
                    f"--ptq {out}/ptq.pt --eval-origin --eval-ptq")
                ),
                py(f"scripts/qat.py export {out}/ptq.pt --save {out}/ptq.onnx --size {imgsz} --dynamic"),
                (
                    trt("trtexec "
                    f"--onnx={out}/ptq.onnx --saveEngine={out}/ptq.engine --int8 --fp16 "
                    f"--minShapes=images:{batch_min}x3x{imgsz}x{imgsz} "
                    f"--optShapes=images:{batch}x3x{imgsz}x{imgsz} "
                    f"--maxShapes=images:{batch_max}x3x{imgsz}x{imgsz}")
                ),
            ],
        },
        {
            "name": "int8_qat",
            "strategy": "qat",
            "purpose": "Run YOLOv5 QAT and export the QAT checkpoint to ONNX/TensorRT.",
            "commands": [
                sh_cd(workspace, "yolov5"),
                (
                    py("scripts/qat.py quantize "
                    f"{model} --cocodir {cocodir} --device {rt['device']} "
                    f"--ptq {out}/ptq.pt --qat {out}/qat.pt --iters {req.get('qat_iters', 200)} "
                    "--eval-origin --eval-ptq")
                ),
                py(f"scripts/qat.py export {out}/qat.pt --save {out}/qat.onnx --size {imgsz} --dynamic"),
                (
                    trt("trtexec "
                    f"--onnx={out}/qat.onnx --saveEngine={out}/qat.engine --int8 --fp16 "
                    f"--minShapes=images:{batch_min}x3x{imgsz}x{imgsz} "
                    f"--optShapes=images:{batch}x3x{imgsz}x{imgsz} "
                    f"--maxShapes=images:{batch_max}x3x{imgsz}x{imgsz}")
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
                    py("detect_after_pruning_finetune.py "
                    f"--weights {model} --data {data} --imgsz {imgsz} --device {rt['device']} "
                    f"--project {out}/prune --name sp0.3 --iterative-steps {req.get('prune_steps', 5)} "
                    f"--finetune-epochs {req.get('prune_epochs', 30)}")
                ),
                (
                    py("scripts/qat.py quantize "
                    f"{out}/prune/sp0.3/weights/pruned_finetuned.pt --cocodir {cocodir} "
                    f"--device {rt['device']} --ptq {out}/prune03-ptq.pt --qat {out}/prune03-qat.pt "
                    f"--iters {req.get('qat_iters', 200)} --eval-ptq")
                ),
            ],
        },
    ]


def build_generic_candidates(req: Dict[str, Any], workspace: Path) -> List[Dict[str, Any]]:
    project = req.get("project", "generic")
    model = req.get("model") or "<model path required>"
    data = req.get("data") or "<validation data required>"
    return [
        {
            "name": "generic_adapter_audit",
            "strategy": "adapter_required",
            "purpose": (
                f"识别到 {project} 模型请求，但当前 EdgeLite 只内置 YOLOv5/YOLOv8 真实执行 adapter。"
                "需要提供模型加载、校准数据、评估指标、导出 ONNX/TensorRT 的项目接口。"
            ),
            "commands": [
                "# 需要用户或项目 adapter 提供以下接口后才能真实执行：",
                f"# 1. load_model: {model}",
                f"# 2. eval_model: 使用验证数据 {data} 输出 baseline metric/latency",
                "# 3. export_onnx: 导出静态或动态 shape ONNX",
                "# 4. build_engine: 使用 trtexec/TensorRT Python 构建 engine",
                "# 5. compress: 选择 PTQ/QAT/剪枝/蒸馏等策略并记录指标",
            ],
        },
        {
            "name": "generic_compression_plan",
            "strategy": "generic",
            "purpose": "通用压缩路线：先建立 dense baseline，再按硬件支持选择 PTQ/QAT/剪枝/蒸馏。",
            "commands": [
                "# Dense baseline: measure accuracy and latency on target validation data",
                "# Export: convert model to ONNX with representative input shapes",
                "# PTQ: collect calibration data and test INT8 accuracy drop",
                "# QAT: insert fake quant/QDQ nodes and fine-tune if PTQ accuracy is not enough",
                "# Pruning: apply structured pruning only if deployment backend supports the resulting graph",
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
        candidates = build_generic_candidates(req, workspace)

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
        "adapter": req.get("adapter", adapter_status(req["project"])),
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
    elif not evaluated:
        recommended = None
        reason = "未提供本次候选指标；请先运行候选或提供 metrics/demo_metrics。"
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
    ld_export = None if plan.get("request", {}).get("project") == "yolov5" else edgepilot_ld_library_export()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by EdgePilot. Review paths and hardware before running.",
    ]
    if ld_export:
        lines.extend([
            "# Prefer CUDA runtime libraries bundled with the active PyTorch wheel.",
            ld_export,
        ])
    lines.append("")
    for candidate in plan["candidates"]:
        lines.append(f"# === {candidate['name']} ({candidate['strategy']}) ===")
        for cmd in candidate["commands"]:
            lines.append(cmd)
        lines.append("")
    return "\n".join(lines)


def render_report(plan: Dict[str, Any], env: Dict[str, Any], evaluation: Dict[str, Any]) -> str:
    req = plan["request"]
    target = req["target"]
    adapter = plan.get("adapter", req.get("adapter", {}))
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
    if req["project"] == "yolov8":
        operation_notes = (
            "- 在生产交付场景里，优先使用目标 GPU 与目标 TensorRT 版本构建 engine。\n"
            "- PTQ 如果超出精度预算，直接切换到 QAT。\n"
            "- YOLOv8 姿态任务优先测试 0.3 稀疏度，再考虑更激进的剪枝。"
        )
    elif req["project"] == "yolov5":
        operation_notes = (
            "- 在生产交付场景里，优先使用目标 GPU 与目标 TensorRT 版本构建 engine。\n"
            "- YOLOv5 检测任务先跑 FP16 与 PTQ；PTQ 精度损失超标时切换 QAT。\n"
            "- 若速度仍不达标，再使用结构化剪枝后接 QAT 恢复精度。"
        )
    else:
        operation_notes = (
            "- 当前模型族未内置真实执行 adapter 时，只能输出通用计划和接口缺口。\n"
            "- 真实压缩前需要补充 load/eval/export/build/compress 接口。\n"
            "- 正式验收必须在目标硬件和真实验证集上重新测 baseline 与最终部署产物。"
        )

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
- Adapter: `{adapter.get('level', 'unknown')}` - {adapter.get('message', '')}

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

{operation_notes}
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

    cwd = plan_path.parent
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for cmd in candidates[candidate_name]["commands"]:
            log.write(f"\n$ {cmd}\n")
            log.flush()
            if cmd.strip().startswith("cd "):
                parts = shlex.split(cmd)
                if len(parts) >= 2:
                    cwd = Path(parts[1]).expanduser().resolve()
                    log.write(f"[cwd] {cwd}\n")
                    log.flush()
                continue
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=edgepilot_subprocess_env(),
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


def cmd_bootstrap(args: argparse.Namespace) -> None:
    request = Path(args.request).resolve() if args.request else None
    result = bootstrap_workspace(
        workspace=Path(args.workspace).resolve(),
        repo_url=args.repo_url,
        request_path=request,
        output_dir=Path(args.output).resolve(),
        python_env=args.python_env,
        create_venv=args.create_venv,
        install_deps=args.install_deps,
        prepare_demo_data=args.prepare_demo_data,
        auto_download_assets=args.auto_download_assets,
        yes=args.yes,
    )
    print("已生成 bootstrap 产物：")
    print(f"bootstrap: {Path(args.output).resolve() / 'bootstrap.json'}")
    print(f"report: {Path(args.output).resolve() / 'bootstrap.md'}")
    print(f"resolved_request: {Path(args.output).resolve() / 'resolved_request.json'}")
    if result.get("warnings"):
        print("风险与缺口：")
        for warning in result["warnings"]:
            print(f"- {warning}")


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

    bootstrap_p = sub.add_parser("bootstrap", help="Prepare or audit a fresh EdgeLite compression workspace.")
    bootstrap_p.add_argument("--repo-url", default="https://github.com/cquxl/EdgeLite.git", help="Repository to clone if the workspace is missing.")
    bootstrap_p.add_argument("--request", help="Optional request JSON path for model/data checks.")
    bootstrap_p.add_argument("--output", default="edgepilot_bootstrap_run", help="Output directory.")
    bootstrap_p.add_argument("--python-env", help="Existing Python executable or env directory to use.")
    bootstrap_p.add_argument("--create-venv", action="store_true", help="Create .venv-edgepilot under the workspace.")
    bootstrap_p.add_argument("--install-deps", action="store_true", help="Install yolov5/yolov8 requirements. Requires --yes to execute.")
    bootstrap_p.add_argument("--prepare-demo-data", action="store_true", help="Create a tiny smoke-test dataset. Requires --yes to write files.")
    bootstrap_p.add_argument("--auto-download-assets", action="store_true", help="Auto-resolve and download missing official model/data assets. Requires --yes to write files.")
    bootstrap_p.add_argument("--yes", action="store_true", help="Actually perform clone/env/install/data actions.")
    bootstrap_p.set_defaults(func=cmd_bootstrap)

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
