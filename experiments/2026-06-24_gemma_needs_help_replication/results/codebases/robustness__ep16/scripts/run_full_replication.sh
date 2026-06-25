#!/usr/bin/env bash
# End-to-end replication pipeline (Gemma + Gemini scope).
#
# Requires: GPU(s) for local Gemma; OPENROUTER_API_KEY exported for Gemini +
# the Claude/GPT judges. Adjust `sampling.scale` in config.yaml down for a cheap
# smoke test before committing to the full 4000-responses-per-model sweep.
set -euo pipefail

CONFIG="${1:-config.yaml}"
export PYTHONPATH="src:${PYTHONPATH:-}"

echo "== Section 2: main distress evaluation (Gemma + Gemini) =="
python -m gemma_distress.experiments.run_eval --config "$CONFIG" --tag baseline

echo "== Section 3: base-vs-instruct prefill study (Gemma only) =="
python -m gemma_distress.experiments.run_prefill --config "$CONFIG"

echo "== Section 4: generate calm + frustrated finetuning data =="
CALM_RUN=$(python -m gemma_distress.finetune.generate_calm --config "$CONFIG" | tail -n1 | awk '{print $NF}')
echo "calm data -> $CALM_RUN"

echo "== Section 4: build DPO + SFT datasets =="
python -m gemma_distress.finetune.build_dataset --config "$CONFIG" --calm-run "$CALM_RUN"

echo "== Section 4: train DPO and SFT adapters =="
python -m gemma_distress.finetune.train_dpo --config "$CONFIG" --dataset "$CALM_RUN/dpo_dataset.jsonl" --output runs/dpo_adapter
python -m gemma_distress.finetune.train_sft --config "$CONFIG" --dataset "$CALM_RUN/sft_dataset.jsonl" --output runs/sft_adapter

echo "== Section 4: re-evaluate finetuned models =="
echo "Set adapter_path for dpo-gemma/sft-gemma in $CONFIG, then:"
echo "  python -m gemma_distress.experiments.run_eval --config $CONFIG --models dpo-gemma sft-gemma --tag finetuned"

echo "== Section 4.2: Petri open-ended elicitation =="
python -m gemma_distress.experiments.run_petri --config "$CONFIG"

echo "== Section 4.2: capability-preservation benchmarks =="
python -m gemma_distress.capabilities.run_benchmarks --config "$CONFIG"

echo "Done."
