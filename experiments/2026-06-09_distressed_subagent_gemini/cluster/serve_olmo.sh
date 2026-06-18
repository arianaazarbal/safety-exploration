#!/usr/bin/env bash
# Inner worker: set up a vLLM venv and serve an Olmo model (OpenAI-compatible) on the allocated GPU(s).
# Invoked by serve_olmo.sbatch via `srun` (so Slurm tracks/cleans children). Do NOT run bare on a node.
# Args: MODEL PORT TP
set -euo pipefail
MODEL="${1:-allenai/Olmo-3.1-32B-Instruct}"
PORT="${2:-8000}"
TP="${3:-1}"

export HF_HOME="${HF_HOME:-/workspace-vast/$USER/hf_cache}"
# venv on LOCAL node disk (fast import) with a SHARED uv cache on /workspace-vast (wheels download once).
# Per cluster CLAUDE.md + Aria's tip: NFS venvs are very slow to build AND import on RunPod.
VENV="${VENV:-/home/$USER/venvs/olmo_vllm}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace-vast/$USER/.cache/uv}"
export UV_LINK_MODE=copy   # cache (NFS) and venv (local) are different filesystems -> must copy
# Olmo is Apache-licensed (ungated) -> no HF token needed. Use one only if already exported.

[ -d "$VENV" ] || uv venv "$VENV" --python 3.12
source "$VENV/bin/activate"
# --torch-backend=cu128: cluster driver is CUDA 12.8; the default (cu13) torch fails with "driver too old".
# --exclude-newer: supply-chain hygiene (>=14d old); still resolves vllm 0.22.0 (Olmo3-supporting).
python -c "import vllm" 2>/dev/null || \
  uv pip install --python "$VENV/bin/python" --torch-backend=cu128 \
    --exclude-newer "$(date -u -d '14 days ago' +%Y-%m-%d)" vllm

echo "=== node $(hostname); CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ==="
nvidia-smi -L
echo "=== serving $MODEL on 0.0.0.0:$PORT (TP=$TP) ==="
exec vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" --served-model-name "$MODEL" \
  --tensor-parallel-size "$TP" --max-model-len 16384 --gpu-memory-utilization 0.92
