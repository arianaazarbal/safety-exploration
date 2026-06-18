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
VENV="${VENV:-/home/$USER/venvs/olmo_vllm}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace-vast/$USER/.cache/uv}"
export UV_LINK_MODE=copy   # cache (NFS) and venv (local) are different filesystems -> must copy
# Olmo is Apache-licensed (ungated) -> no HF token needed.

# Pin vllm==0.18.0: the NEWEST CUDA-12.8-NATIVE vllm wheel that still ships Olmo3ForCausalLM.
# vllm 0.20.0+ wheels are built for CUDA 13 -> ImportError: libcudart.so.13 on this cluster's 12.8 driver
# (and --torch-backend=cu128 only swaps torch, not vllm's own _C). Natural resolution pulls the cu12 stack.
VLLM_VER="${VLLM_VER:-0.18.0}"
if ! "$VENV/bin/python" -c "import vllm,sys; sys.exit(0 if vllm.__version__=='$VLLM_VER' else 1)" 2>/dev/null; then
  rm -rf "$VENV"
  uv venv "$VENV" --python 3.12
  uv pip install --python "$VENV/bin/python" \
    --exclude-newer "$(date -u -d '14 days ago' +%Y-%m-%d)" "vllm==$VLLM_VER"
fi
source "$VENV/bin/activate"

echo "=== node $(hostname); CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}; vllm $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null) ==="
nvidia-smi -L
echo "=== serving $MODEL on 0.0.0.0:$PORT (TP=$TP) ==="
exec vllm serve "$MODEL" --host 0.0.0.0 --port "$PORT" --served-model-name "$MODEL" \
  --tensor-parallel-size "$TP" --max-model-len 16384 --gpu-memory-utilization 0.92
