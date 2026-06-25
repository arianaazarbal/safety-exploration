# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

A code replication of the core experiments from Soligo, Mikulik & Saunders
(arXiv 2603.10011v1, 2026), **scoped to the Gemma and Gemini model families**.

The paper shows that (1) repeated user rejection reliably elicits expressions of
emotional distress in Gemma and Gemini (but not other families), (2) this is
amplified in Gemma's post-training, and (3) DPO on just 280 preference pairs
reduces Gemma's high-frustration rate from ~35% to ~0.3% without degrading
capabilities. This repo implements the evaluations and the mitigation.

> **Design decisions and gap-filling** are documented in **[DESIGN.md](DESIGN.md)**.
> Read it before running — several experimental details are underspecified in
> the paper and were resolved with explicit, documented choices.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Eliciting distress | `emotional_instability/eval/` | 8 conditions / 5 categories, multi-turn rejection rollouts, 0–10 frustration judge (Claude Sonnet 4), Figures 1–3 + word-frequency tables |
| §2.1 Judge reliability | `eval/judge_reliability.py` | Claude vs GPT-5-mini agreement (Pearson r, % within one) |
| §3 Post-training divergence | `prefill/run_prefill.py` | Base-vs-instruct prefill continuation (Gemma; see scope note) |
| §4 DPO/SFT mitigation | `finetune/` | Calm-data generation, DPO/SFT dataset builders, LoRA trainers |
| §4 Petri generalization | `petri/` | Open-ended adversarial emotion elicitation (4 emotions) |
| §4.2 Capability preservation | `capabilities/` | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench |
| §4.2 Recovery limitation | `prefill/run_recovery.py` | Continuation from already-frustrated states |
| App. I Internal emotions | `probing/` | Logit-lens internal emotion detection |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...    # frustration judge, Petri auditor/judge, onset labeller
export OPENROUTER_API_KEY=...   # Gemini targets + secondary judge
export HF_TOKEN=...             # gated Gemma weights
```

Hardware: Gemma-3-27B needs ~1×A100-80GB in bf16, or one 48GB GPU with
`--load-in-4bit`. Gemini and the judges are API-only.

## Quick start

```bash
# Offline sanity checks (no GPU / no API keys):
python -m scripts.smoke_test

# A tiny end-to-end run (2% of paper scale):
SCALE=0.02 Q=--load-in-4bit bash scripts/run_pipeline.sh
```

`scripts/run_pipeline.sh` documents the full ordering of steps. Each stage caches
to `results/` as JSONL and resumes if interrupted.

## Layout

```
emotional_instability/
  config.py            model registry, paths, sampling constants
  models/              chat backends: hf_local (Gemma), openrouter (Gemini), anthropic (judge)
  eval/                §2 elicitation + judging + analysis
  prefill/             §3 base-vs-instruct + §4.2 recovery
  finetune/            §4 calm-data gen, dataset builders, DPO/SFT trainers
  petri/               open-ended elicitation
  capabilities/        capability-preservation benchmarks
  probing/             App. I logit-based internal emotion detection
scripts/               smoke test + pipeline orchestration
results/               all outputs (rollouts, datasets, adapters, figures)
```
