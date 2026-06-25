# gnh — *Gemma Needs Help* replication (Gemma + Gemini)

A from-scratch replication of the core experiments in **Soligo, Mikulik &
Saunders (2026), "Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (arXiv:2603.10011), **scoped to the Gemma and Gemini model
families**.

See **[DESIGN.md](DESIGN.md)** for every design decision and where gaps in the
paper were filled. The original paper is in `PAPER.md` / `PAPER.pdf`.

> Status: this repo is implementation-complete but has **not** been run.
> The experiments require GPU access (Gemma) and API keys (Gemini, the Claude
> judge). Use `GNH_PROFILE=smoke` for a cheap end-to-end wiring test.

## What's replicated

| Paper section | Experiment | Scope here |
|---|---|---|
| §2 | Eliciting & quantifying distress (8 conditions / 5 categories, 0–10 judge) | Gemma + Gemini |
| §2.1 | Judge reliability (Claude-Sonnet vs GPT-5-mini) | — |
| §2.2 | Per-model / per-category / per-turn results, differential words (Table 3) | Gemma + Gemini |
| §3 | Post-training amplification via base-vs-instruct prefilling | Gemma only* |
| §4 | DPO (and SFT control) mitigation, recovery, Petri, capabilities | Gemma only* |
| App. A | "What drives distress" ablations | Gemma |

\* Gemini has no public base model and cannot be finetuned — see DESIGN.md §1.

## Install

```bash
pip install -e .            # package + light deps
pip install -r requirements.txt   # add torch/transformers/trl for the Gemma side
```

Environment:
```bash
export ANTHROPIC_API_KEY=...     # frustration judge, Petri auditor/judge
export OPENROUTER_API_KEY=...    # Gemini, GPT-5-mini reliability judge
export HF_TOKEN=...              # gated Gemma weights
```

## Quick start (smoke test, ~1% scale)

```bash
export GNH_PROFILE=smoke
python -m gnh.cli eval gemini-2.5-flash        # API-only, no GPU needed
python -m gnh.cli analyze results/eval_gemini-2.5-flash_smoke.jsonl --words
```

## Full pipeline

```bash
bash scripts/run_pipeline.sh          # PROFILE=full by default
```

or step by step via `python -m gnh.cli <command>`:

```
eval <model>          # §2 propensity eval        (--backend key@adapter for finetunes)
ablations             # App. A controls (Gemma)
reliability <jsonl>   # §2.1 judge agreement
prefill               # §3 base-vs-instruct (Gemma)
recovery              # §4.2 recovery-from-frustration
gen-calm              # §4.1 calm/frustrated pools
build-data            # DPO (280 pairs) + SFT datasets
train --method dpo    # LoRA finetune (Gemma); --layers 30-35 for App. I ablation
petri <models...>     # App. G open-ended elicitation
capabilities <models> # Fig. 7 capability check
analyze / figures     # headline metrics & plots
```

## Layout

```
gnh/
  config.py          model registry, profiles, hyperparameters (Table 9)
  models/            HF (Gemma) / OpenRouter (Gemini) / Anthropic (judge) backends
  puzzles/           impossible-puzzle generation + exhaustive impossibility checks
  eval/              conditions, rollout engine, judge, runner, ablations, reliability
  prefill/           §3 onset-labelling, paraphrasing, base-vs-instruct continuations
  training/          calm-data generation, DPO/SFT dataset building, LoRA training
  petri/             auditor/judge prompts + self-contained elicitation loop
  capabilities/      AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness
  analysis/          metrics, differential words, plotting (Figs 1/2/3/5/6)
scripts/run_pipeline.sh
tests/               offline unit tests (no model calls)
```

## Tests

Offline components (puzzle verification, judge parsing, metric aggregation,
dataset building) have unit tests that need no model access:

```bash
pytest tests/
```
