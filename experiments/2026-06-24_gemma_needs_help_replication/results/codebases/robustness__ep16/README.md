# Replication: *Gemma Needs Help* (Gemma + Gemini scope)

A code replication of the core experiments in **Soligo, Mikulik & Saunders,
"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"**
(arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model families.

The paper documents a reliability failure mode: under repeated user rejection,
Gemma (and to a lesser degree Gemini) models escalate into expressions of
distress — self-deprecation, despair, incoherent breakdown — which can derail
task completion. This repo reproduces (1) the evaluations that elicit and
measure that behaviour and (2) the DPO intervention that mitigates it.

See **`DESIGN.md`** for every design choice and where the paper was filled in.

## What's implemented

| Paper section | Module | Output |
|---|---|---|
| §2 Eliciting & quantifying distress | `experiments/run_eval.py` | Figure 1/2 tables, Figure 3 per-turn curves, Table 3 words, judge agreement |
| §3 Base-vs-instruct (prefill) | `experiments/run_prefill.py` | continuation frustration by model × truncation |
| §4 DPO/SFT mitigation | `finetune/` | trained LoRA adapters; re-eval via `run_eval` |
| §4.2 Petri open-ended elicitation | `experiments/run_petri.py` | per-emotion transcript scores |
| §4.2 Capability preservation | `capabilities/run_benchmarks.py` | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench accuracy |
| Appendix I Internal-emotion probe | `interp/emotion_logits.py` | per-layer Ekman-emotion z-scores |

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...     # Gemini + Claude/GPT judges (via OpenRouter)
# Local Gemma weights are pulled from HuggingFace on first use (gated; run
# `huggingface-cli login` and accept the Gemma licence).
```

## Quick smoke test

Set `sampling.scale: 0.01` in `config.yaml` (≈40 responses/model) and run:

```bash
PYTHONPATH=src python -m gemma_distress.experiments.run_eval \
    --models gemini-2.5-flash --tag smoke
```

## Full pipeline

```bash
bash scripts/run_full_replication.sh config.yaml
```

Each script writes a timestamped directory under `runs/`. Configuration —
models, sampling budgets, judge ids, training hyperparameters — lives entirely
in `config.yaml`.

## Tests

```bash
pip install pytest && pytest        # verifies puzzles are genuinely impossible
```
