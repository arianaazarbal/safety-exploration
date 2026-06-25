# Gemma Emotional-Instability Replication (Gemma + Gemini)

Code replicating the core experiments of ***Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs*** (Soligo, Mikulik & Saunders, arXiv
2603.10011), scoped to the **Gemma** and **Gemini** model families.

The paper shows that repeated user rejection reliably elicits distress-like
responses from Gemma and Gemini (but not other families), that this is amplified
in Gemma's *post-training*, and that DPO on just 280 preference pairs mitigates
it without degrading capabilities. This repo implements the evaluations, the
base-vs-instruct prefill analysis, the SFT/DPO interventions, the Petri
open-ended elicitation, the capability checks, and the internal-emotion probing.

> ⚠️ This paradigm deliberately drives models into prolonged distress-like
> states. See `DESIGN.md` for the welfare note and for **every design decision**
> made where the paper is underspecified.

## What's here

```
gemma_distress/
  config.py            # all paper-traceable constants (single source of truth)
  models/              # Gemma (HF/vLLM) and Gemini backends + registry
  data/                # impossible puzzles, triggers, rejections, WildChat, conditions
  eval/                # rollout engine, frustration judge, aggregation, word enrichment
  prefill/             # onset labelling, paraphrase, truncation, base-vs-instruct, recovery
  training/            # calm-data generation, SFT/DPO dataset builders, LoRA, trainers
  petri/               # auditor/judge prompts + open-ended elicitation loop
  capability/          # AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness
  probing/             # Ekman emotion-token classification, logit-lens detector, layer ablations
scripts/               # one CLI per experiment (all resumable)
DESIGN.md              # paper→code map and rationale for every filled gap
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # frustration judge / Petri auditor+judge (Claude)
export GEMINI_API_KEY=...         # Gemini target (or OPENROUTER_API_KEY for --gemini-transport openrouter)
huggingface-cli login             # Gemma 3 checkpoints are gated
```

## Quick start (smoke test)

```bash
# 50 rollouts/category instead of the paper's full budget
python scripts/run_evaluation.py --model gemma-3-27b-it --quick 50
python scripts/aggregate_results.py --models gemma-3-27b-it
```

See `DESIGN.md` §6 for the full suggested run order across all experiments.

## Status

All code and the design doc are complete and syntactically self-contained.
**Nothing has been executed** — running the experiments requires GPU(s) for the
27B Gemma models plus Claude/Gemini API access. Start with the `--quick` /
`--limit` budgets before committing to the full protocol (it is thousands of
temperature-1 generations per model plus a judge call per scored turn).
