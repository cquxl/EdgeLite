#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-yolov5-compress}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE="$(cd "${SKILL_DIR}/.." && pwd)"
ENV_DIR="${SKILL_DIR}/envs/yolov5-compress"
ENV_FILE="${ENV_DIR}/environment.yml"
REQ_FILE="${ENV_DIR}/requirements-yolov5-compress.txt"
TRT_DIR="${TRT_DIR:-}"
YES="${YES:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash edgelite-compression-agent/scripts/setup_yolov5_env.sh [options]

Options:
  --env-name NAME        Conda environment name. Default: yolov5-compress
  --tensorrt-dir PATH    TensorRT unpacked directory, e.g. TensorRT-8.6.1.6
  --yes                 Non-interactive mode

Notes:
  - YOLOv5 uses a separate conda environment from YOLOv8.
  - The repository's yolov5/ source is the execution source.
  - The pip ultralytics package is installed only as a YOLOv5 compatibility
    dependency for helper modules such as ultralytics.utils.plotting/checks.
  - YOLOv5_AUTOINSTALL is disabled to prevent runtime package upgrades.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --tensorrt-dir)
      TRT_DIR="$2"
      shift 2
      ;;
    --yes|-y)
      YES=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[error] unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda is required. Install Miniconda/Anaconda or load conda first." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

CONDA_EXE="conda"
if command -v mamba >/dev/null 2>&1; then
  CONDA_EXE="mamba"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[check] GPU:"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -8
else
  echo "[warn] nvidia-smi not found. Environment can be created, but real CUDA/TensorRT execution may fail."
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[env] ${ENV_NAME} already exists; reusing it."
else
  echo "[env] creating ${ENV_NAME} from ${ENV_FILE}"
  CREATE_ARGS=()
  if [[ "${YES}" == "1" ]]; then
    CREATE_ARGS=(-y)
  fi
  "${CONDA_EXE}" env create -n "${ENV_NAME}" -f "${ENV_FILE}" "${CREATE_ARGS[@]}"
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install -r "${REQ_FILE}"

if [[ -z "${TRT_DIR}" ]]; then
  for candidate in \
    "${SKILL_DIR}/../../Downloads/TensorRT-8.6.1.6" \
    "${SKILL_DIR}/../../../Downloads/TensorRT-8.6.1.6" \
    "/data/xl/Projects/Downloads/TensorRT-8.6.1.6" \
    "${HOME}/Downloads/TensorRT-8.6.1.6"; do
    if [[ -d "${candidate}" ]]; then
      TRT_DIR="${candidate}"
      break
    fi
  done
fi

ACTIVATE_DIR="${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${ACTIVATE_DIR}"
cat > "${ACTIVATE_DIR}/edgelite_yolov5.sh" <<EOF
export YOLOv5_AUTOINSTALL=false
export YOLOv5_VERBOSE=true
EOF

if [[ -n "${TRT_DIR}" && -d "${TRT_DIR}" ]]; then
  echo "[trt] installing TensorRT Python wheels from ${TRT_DIR}"
  shopt -s nullglob
  TRT_WHEELS=(
    "${TRT_DIR}"/python/tensorrt-*-cp310-*.whl
    "${TRT_DIR}"/uff/uff-*.whl
    "${TRT_DIR}"/graphsurgeon/graphsurgeon-*.whl
    "${TRT_DIR}"/onnx_graphsurgeon/onnx_graphsurgeon-*.whl
  )
  if [[ ${#TRT_WHEELS[@]} -gt 0 ]]; then
    python -m pip install "${TRT_WHEELS[@]}"
  else
    echo "[warn] no TensorRT cp310 wheels found in ${TRT_DIR}; skipping TensorRT Python wheel install."
  fi
  cat >> "${ACTIVATE_DIR}/edgelite_yolov5.sh" <<EOF
export TRT_DIR="${TRT_DIR}"
export PATH="\${TRT_DIR}/targets/x86_64-linux-gnu/bin:\${TRT_DIR}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${TRT_DIR}/targets/x86_64-linux-gnu/lib:\${TRT_DIR}/lib:\${LD_LIBRARY_PATH:-}"
EOF
else
  echo "[warn] TensorRT directory was not provided/found. trtexec must be available separately for engine build."
fi

# shellcheck source=/dev/null
source "${ACTIVATE_DIR}/edgelite_yolov5.sh"

echo "[verify] runtime imports"
python - <<'PY'
import importlib
mods = ["torch", "torchvision", "cv2", "onnx", "onnxsim", "pycuda", "ultralytics"]
for name in mods:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "available"))
try:
    import tensorrt as trt
    print("tensorrt", trt.__version__)
except Exception as exc:
    print("tensorrt", "WARN", exc)
import os
import torch
print("YOLOv5_AUTOINSTALL", os.getenv("YOLOv5_AUTOINSTALL"))
print("cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
PY

if [[ -d "${WORKSPACE}/yolov5" ]]; then
  echo "[verify] local EdgeLite YOLOv5 source"
  (
    cd "${WORKSPACE}/yolov5"
    python - <<'PY'
from pathlib import Path
import models.yolo as yolo
import utils.general as general

root = Path.cwd().resolve()
print("models.yolo", Path(yolo.__file__).resolve())
print("utils.general", Path(general.__file__).resolve())
if root not in Path(yolo.__file__).resolve().parents:
    raise SystemExit("YOLOv5 model source is not loaded from the local repository")
PY
  )
else
  echo "[warn] ${WORKSPACE}/yolov5 not found; clone/integrate EdgeLite yolov5 before real YOLOv5 execution."
fi

if command -v trtexec >/dev/null 2>&1; then
  echo "[verify] trtexec: $(command -v trtexec)"
else
  echo "[warn] trtexec is still not in PATH."
fi

echo "[done] activate with: conda activate ${ENV_NAME}"
