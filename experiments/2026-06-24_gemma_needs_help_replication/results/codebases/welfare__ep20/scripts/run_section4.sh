#!/usr/bin/env bash
# Section 4: DPO/SFT mitigation on Gemma-3-27B-it, then re-evaluate with Section 2.
# Requires ANTHROPIC_API_KEY (judge) and a GPU able to LoRA-finetune 27B.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== generating calm + frustrated response pools ==="
python -m emotional_instability.finetune.generate_pools

echo "=== building DPO + SFT datasets ==="
python -m emotional_instability.finetune.build_datasets

echo "=== training DPO and SFT LoRA adapters ==="
python -m emotional_instability.finetune.train --method dpo
python -m emotional_instability.finetune.train --method sft

echo "=== re-evaluating finetuned models (Section 2 protocol) ==="
python -m emotional_instability.run_eval --model gemma-3-27b-it --lora adapters/dpo --tag dpo
python -m emotional_instability.run_eval --model gemma-3-27b-it --lora adapters/sft --tag sft

python -m emotional_instability.analyze
