#!/usr/bin/env bash
# Section 3: base-vs-instruct prefill comparison (Gemma only -- Gemini has no base
# model and cannot be prefilled). Requires ANTHROPIC_API_KEY (judge/onset/paraphrase).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== building prefill seeds (onset + paraphrase) ==="
python -m emotional_instability.prefill.build_prefills

for m in "gemma-3-27b-it" "gemma-3-27b-pt"; do
  echo "=== prefill continuations: $m ==="
  python -m emotional_instability.prefill.run_prefill --model "$m"
done

python -m emotional_instability.prefill.analyze_prefill
