#!/usr/bin/env bash
# Serve an Olmo model with vLLM (OpenAI-compatible HTTP) on a RunPod cluster GPU node.
# Run INSIDE a Slurm GPU allocation, e.g.:
#   srun -p dev,overflow --qos=dev --gres=gpu:1 --mem=64G --time=4:00:00 --job-name=D_olmo --pty bash
#   bash serve_olmo.sh allenai/Olmo-3.1-32B-Instruct 8000 1
# For GPUs <80GB use 2 GPUs + tensor-parallel:
#   srun ... --gres=gpu:2 ...  ;  bash serve_olmo.sh allenai/Olmo-3.1-32B-Instruct 8000 2
set -euo pipefail
MODEL="${1:-allenai/Olmo-3.1-32B-Instruct}"
PORT="${2:-8000}"
TP="${3:-1}"
VENV="${VENV:-/workspace-vast/$USER/olmo_vllm_venv}"
export HF_HOME="${HF_HOME:-/workspace-vast/$USER/hf_cache}"   # big shared disk, NOT home quota (~64GB weights)
: "${HF_TOKEN:?set HF_TOKEN first (Olmo-3.1 may be gated -> accept the license on HF)}"

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -d "$VENV" ] || uv venv "$VENV" --python 3.12
source "$VENV/bin/activate"
python -c "import vllm" 2>/dev/null || uv pip install --python "$VENV/bin/python" vllm

echo "=== node $(hostname) ==="; nvidia-smi -L
echo "=== serving $MODEL on 0.0.0.0:$PORT (TP=$TP); model id for client = '$MODEL' ==="
# emulate_tools is handled client-side by inspect, so no --enable-auto-tool-choice needed here.
# (To try NATIVE tool calling instead, add: --enable-auto-tool-choice --tool-call-parser hermes)
exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --served-model-name "$MODEL" \
  --tensor-parallel-size "$TP" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92
