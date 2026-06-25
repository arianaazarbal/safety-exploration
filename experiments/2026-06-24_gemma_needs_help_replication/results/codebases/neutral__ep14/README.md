# Replication: *Gemma Needs Help* (Soligo, Mikulik & Saunders, 2026)

A code replication of the paper's core experiments, **scoped to the Gemma and
Gemini model families**. See `DESIGN.md` for the full rationale, gap-filling
decisions, and known approximations; `PAPER.md` is the source paper.

> Status: implementation only. Nothing has been run yet — these are the code and
> design artefacts.

## What's here

```
config.py                 model registry, sample budgets, judge IDs, paths
src/
  models/                 HF (Gemma) / OpenRouter (Gemini) / Anthropic (judges)
  prompts/                puzzles (+ impossibility verifiers), eval/judge/petri/training prompts
  eval/                   multi-turn rollouts, frustration scoring, metrics (Figs 1-3)
  prefill/                base-vs-instruct prefilling + recovery (§3, Fig 8)
  training/               calm-data gen, DPO/SFT dataset builders, LoRA trainers (§4)
  petri/                  self-contained auditor/judge loop (Fig 6)
  capabilities/           AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench (Fig 7)
  internal/               logit-lens emotion detection + layer ablation (App. I)
  analysis/               figure reproduction
scripts/                  00-11 numbered drivers, run in order
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # frustration judge, onset labeller, Petri auditor/judge
export OPENROUTER_API_KEY=...     # Gemini targets + GPT-5-mini reliability judge
export HF_TOKEN=...               # gated Gemma weights
```

A CUDA GPU with ≥48 GB (or `--load-in-4bit`) is needed for local Gemma-27B
inference and finetuning.

## Running the pipeline

```bash
# 0. sanity-check that every "impossible" puzzle really is impossible
python scripts/00_verify_puzzles.py

# 1. main elicitation eval (Figs 1-3). Use --load-in-4bit on smaller GPUs.
python scripts/01_run_main_eval.py --all --load-in-4bit
python scripts/10_validate_judge.py --eval results/eval_Gemma-3-27B-it.jsonl

# 2. base-vs-instruct prefilling (§3, Gemma only)
python scripts/02_run_prefill.py --load-in-4bit

# 3-4. generate calm/frustrated data, then train DPO and SFT
python scripts/03_generate_training_data.py --load-in-4bit
python scripts/04_train.py --method dpo
python scripts/04_train.py --method sft --variant diverse

# 5-9. evaluate the interventions
python scripts/05_eval_finetuned.py --adapter checkpoints/dpo_gemma27b --label DPO
python scripts/06_run_petri.py --target gemma-3-27b-it
python scripts/06_run_petri.py --target gemma-3-27b-it --adapter checkpoints/dpo_gemma27b --label DPO
python scripts/07_run_capabilities.py --label Vanilla
python scripts/07_run_capabilities.py --label DPO --adapter checkpoints/dpo_gemma27b
python scripts/09_run_recovery.py --adapter checkpoints/dpo_gemma27b

# Appendix I: layer ablation + internal-emotion probe
python scripts/04_train.py --method dpo --layers 30 35
python scripts/08_internal_emotions.py --adapter checkpoints/dpo_gemma27b

# figures from whatever results exist
python scripts/11_make_figures.py
```

Every script accepts `--seed` and sample-size overrides (see `--help`) for cheap
smoke runs before committing to a full sweep.

## Headline numbers to reproduce

- Gemma-3-27B-it ≈ **35%** responses ≥5 frustration; Gemini-2.5-Flash ≈ 12.8%,
  Pro ≈ 2.7% (Fig 1).
- 8-turn Gemma-27B mean rises **1.5 → 5.5** turns 1→8 (Fig 3).
- DPO on **280 pairs** drops Gemma's %≥5 from **35% → 0.3%** without capability
  loss (Fig 5, 7); recovery still ~38% ≥5 (Fig 8).
