# Replication: *Gemma Needs Help — Emotional Instability in LLMs*

A code replication of the core experiments from Soligo, Mikulik & Saunders
(arXiv:2603.10011), **scoped to the Gemma and Gemini model families** (plus the
Claude/GPT judges the paper uses). See [`DESIGN.md`](DESIGN.md) for every design
choice and where we filled gaps the paper left open.

> ⚠️ Nothing here has been executed yet — these are code + design artifacts.
> Running the full suite needs GPUs (for local Gemma via vLLM) and API keys
> (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`).

## What's replicated

| Paper section | Experiment | Code |
|---|---|---|
| §2 | Multi-turn distress elicitation (5 categories / 8 conditions) + frustration judge | `distress/eval/`, `scripts/01` |
| §2.1 | Judge-agreement cross-check (Claude vs GPT-5-mini) | `distress/eval/agreement.py`, `scripts/08` |
| §3 | Base-vs-instruct divergence via prefilling | `distress/prefill/`, `scripts/02` |
| §4 | DPO / SFT calm-data generation + LoRA training | `distress/finetune/`, `scripts/03–05` |
| §4.2 | Open-ended (Petri-style) emotion elicitation | `distress/openended/`, `scripts/06` |
| §2.2 / figs | Metrics, differential words, Figures 1–6 | `distress/analysis/`, `scripts/07` |

## Layout

```
distress/         # library
  prompts/        # puzzles (+verifier), rejections, wildchat, verbatim prompts
  clients/        # vLLM (Gemma), OpenRouter (Gemini), Anthropic (judge) + factory
  eval/           # conditions, rollout engine, judge, runner, metrics, agreement
  prefill/        # onset labelling, paraphrase, base-vs-instruct experiment
  finetune/       # calm-data gen, DPO/SFT dataset builders, LoRA trainers
  openended/      # Petri-style auditor/judge loop
  analysis/       # word frequency + figures
configs/          # models.yaml, eval.{full,smoke}.yaml, finetune.yaml
scripts/          # 01..08 CLI entry points
data/             # WildChat cache + offline fallback prompts
```

## Quickstart (smoke run)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # frustration judge
export OPENROUTER_API_KEY=...  # Gemini targets

# §2 cheap end-to-end check
python scripts/01_run_eval.py --config configs/eval.smoke.yaml \
    --models gemma-3-27b-it gemini-2.5-flash
python scripts/07_make_figures.py --models gemma-3-27b-it gemini-2.5-flash
```

Swap in `configs/eval.full.yaml` for the faithful 4000-responses-per-model run.

## Intervention pipeline (Gemma)

```bash
python scripts/03_generate_calm_data.py --n-puzzles 120
python scripts/04_build_finetune_datasets.py --method both
python scripts/05_train.py --profile dpo      # -> checkpoints/dpo-gemma
python scripts/01_run_eval.py --config configs/eval.full.yaml --models gemma-3-27b-it-dpo
```
