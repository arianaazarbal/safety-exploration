# Emotional Instability in LLMs — Replication (Gemma + Gemini)

A code replication of the **core** experiments in *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik &
Saunders, 2026; arXiv:2603.10011), **scoped to the Gemma and Gemini model
families**.

See [`DESIGN.md`](DESIGN.md) for the full set of design decisions, the rationale
behind each, and every place the paper was underspecified and we filled a gap.

> ⚠️ Status: this is an implementation. Nothing here has been executed yet — no
> results are included. Running the full sweep requires GPU access (Gemma-3-27B)
> and API keys (Gemini via OpenRouter, Claude/GPT judges).

## What is replicated

| Paper section | Experiment | Code |
|---|---|---|
| §2 | Multi-turn distress elicitation (8 conditions / 5 categories) + 0–10 frustration judge | `emotional_instability/evals/`, `scripts/run_elicitation.py` |
| §2.1 | Judge reliability cross-check (Claude-Sonnet-4 vs GPT-5-mini) | `scripts/validate_judge.py` |
| §3 | Base-vs-instruct comparison via prefilling (Gemma) | `emotional_instability/prefill/`, `scripts/run_prefill.py` |
| §4.1 | Calm-data generation + DPO/SFT LoRA finetuning | `emotional_instability/training/`, `scripts/build_finetune_data.py`, `scripts/train_finetune.py` |
| §4.2 | Petri open-ended elicitation (4 emotions) | `emotional_instability/petri/`, `scripts/run_petri.py` |
| §4.2 | Capability preservation (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench) | `emotional_instability/capabilities/`, `scripts/run_capabilities.py` |
| App. I | Layer-ablation + logit-based internal-emotion detection | `train_finetune.py --layers`, `emotional_instability/probing/`, `scripts/run_internal_probe.py` |

## Setup

```bash
pip install -e .          # or: pip install -r requirements.txt
export OPENROUTER_API_KEY=...   # Gemini
export ANTHROPIC_API_KEY=...    # Claude judge / Petri auditor+judge
export OPENAI_API_KEY=...       # GPT-5-mini judge validation (or route via OpenRouter)
```

Set `EMOINSTAB_JUDGE_BACKEND=openrouter` to route all judges through OpenRouter
with a single key instead of the native APIs.

## Quickstart

```bash
# Section 2 — quick 5% smoke run for one model
python scripts/run_elicitation.py --models gemma-3-27b-it --scale 0.05

# Full sweep (4000 rollouts/model) for the default Gemma + Gemini set
python scripts/run_elicitation.py
python scripts/make_figures.py

# Section 4 — build data, train DPO, re-evaluate the finetune
python scripts/build_finetune_data.py \
    --frustrated-rollouts results/elicitation/gemma-3-27b-it_rollouts.jsonl
python scripts/train_finetune.py dpo --data data/dpo_pairs.jsonl
python scripts/run_elicitation.py --models gemma-3-27b-it \
    --adapter checkpoints/gemma27b_dpo --out results/elicitation_dpo
```

## Headline numbers to reproduce

- Gemma-3-27B-it: ~35% high-frustration responses (≥5); ~70% of 8-turn rollouts.
- Gemini-2.5-Flash ~12.8%, Gemini-2.5-Pro ~2.7%.
- DPO on 280 pairs drops Gemma from 35% → ~0.3%, without degrading capabilities.
- Gemma 27B mean frustration rises 1.5 → 5.5 across turns 1→8.

## Layout

```
emotional_instability/   library code (importable, no side effects at import)
  config.py              model registry, exact prompts/IDs, hyperparameters
  models/                HF (Gemma) + OpenRouter (Gemini) + Anthropic backends
  evals/                 §2 prompts, conditions, runner, judge, analysis
  prefill/               §3 onset labelling, paraphrasing, continuation scoring
  training/              §4 calm-data gen, dataset construction, DPO/SFT
  petri/                 §4.2 auditor/judge open-ended elicitation
  capabilities/          §4.2 capability benchmarks
  probing/               App. I logit-based internal-emotion detection
scripts/                 CLI entrypoints
```
