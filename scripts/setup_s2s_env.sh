#!/usr/bin/env bash
set -euo pipefail

S2S_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$S2S_REPO_ROOT"

# Keep package/model caches inside Speech2speech as requested for this workspace.
export PIP_CACHE_DIR="$S2S_REPO_ROOT/.cache/pip"
export HF_HOME="$S2S_REPO_ROOT/.cache/huggingface"
export CARGO_HOME="$S2S_REPO_ROOT/.cache/cargo"
export TMPDIR="$S2S_REPO_ROOT/.runtime/s2s-tmp"
# sphn 0.2.1 bundles an older Opus CMake project on ARM; modern CMake needs this
# compatibility floor while building the wheel from source.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
S2S_TORCH_INDEX_URL="${S2S_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$CARGO_HOME" "$TMPDIR"

if [[ ! -x .venv-s2s/bin/python ]]; then
  python3 -m venv .venv-s2s
fi

.venv-s2s/bin/python -m pip install --upgrade pip
# moshi 0.2.13 currently declares torch<2.10, so keep Torch/Torchaudio on the newest
# matching CUDA 13 pair accepted by that package.
.venv-s2s/bin/python -m pip install \
  torch==2.9.1+cu130 torchaudio==2.9.1 \
  --index-url "$S2S_TORCH_INDEX_URL"
.venv-s2s/bin/python -m pip install -r requirements-s2s.txt

.venv-s2s/bin/python - <<'PY'
import moshi
import torch
import torchaudio

print(f"moshi={getattr(moshi, '__version__', 'installed')}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
print(f"torchaudio={torchaudio.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device={torch.cuda.get_device_name()} capability={torch.cuda.get_device_capability()}")
PY
