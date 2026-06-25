# Replicating "Gemma Needs Help" (arXiv 2603.10011v1) — Gemma + Gemini

Code replication of the core experiments from *Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik, Saunders),
**scoped to the Gemma and Gemini model families** (the paper evaluates seven).

See **DESIGN.md** for every choice made where the paper is underspecified, the
gaps filled, model substitutions, and notes on how the experiment treats the
models.

> Status: implementation only. Nothing here has been executed. Treat the numbers
> in the paper as the targets these scripts aim to reproduce.

## What is implemented

| Paper section | What it does | Module |
|---|---|---|
| §2 Eliciting & quantifying distress | 8 conditions / 5 categories, multi-turn rejection rollouts, 0-10 frustration judge, GPT-5-mini reliability check | `emotigemma/evals/`, `emotigemma/models/judge.py` |
| §2 Figures 1-3 + Table 3 | per-model %≥5, per-category means, per-turn progression, differential words | `emotigemma/analysis/` |
| §3 Post-training divergence | base-vs-instruct prefill study (Gemma-27B base vs instruct), onset labelling + paraphrasing | `emotigemma/prefill/` |
| §4 Interventions | calm-data generation, SFT + DPO (LoRA r64), layer-range ablations | `emotigemma/training/` |
| §4 Petri | auditor/judge open-ended emotion elicitation | `emotigemma/petri/` |
| §4 Capability preservation | AIME / MATH / GPQA / BBH / TruthfulQA / EmoBench | `emotigemma/capabilities/` |
| §4 Recovery limitation | end-of-spiral prefill continuations | `emotigemma/prefill/recovery.py` |

## Models in scope

- **Targets:** `gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`,
  `gemini-2.5-pro` (§2); `gemma-3-27b-pt` base (§3); `gemma-3-27b-it` +
  DPO/SFT adapters (§4).
- **Judges (paper → here):** frustration `Claude-Sonnet-4 → claude-sonnet-4-6`;
  validation `GPT-5-mini`; Petri auditor `Claude-Sonnet → claude-sonnet-4-6`,
  Petri judge `Claude-Opus → claude-opus-4-8`. The original `claude-sonnet-4`
  snapshot retired 2026-06-15 — see DESIGN.md "Model substitutions".

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # judges, onset labelling, paraphrasing, Petri
export OPENAI_API_KEY=...      # GPT-5-mini reliability check
export GEMINI_API_KEY=...      # Gemini targets
export HF_TOKEN=...            # gated Gemma weights + benchmark datasets
```

Gemma inference/finetuning expects GPUs (a 27B model + vLLM/LoRA). Gemini and the
judges are API-only.

## Run

Everything is configured in `config.yaml`. Full pipeline:

```bash
bash scripts/pipeline.sh config.yaml
```

Or step-by-step, e.g. just Section 2 for one model:

```bash
python -m emotigemma.evals.run_eval --config config.yaml --models gemma-3-27b-it
python -m emotigemma.evals.score    --config config.yaml --models gemma-3-27b-it
python -m emotigemma.analysis.aggregate --config config.yaml
```

Finetuned variants are addressed as `gemma-3-27b-it+dpo` / `gemma-3-27b-it+sft`
(adapters resolved under `<output_dir>/training/`).

## Layout

```
config.yaml              # all knobs: models, sampling, judges, training, benchmarks
emotigemma/
  config.py              # config loader + model registry (incl. +adapter variants)
  models/                # target backends (gemma_vllm, gemini) + judges
  evals/                 # puzzles, prompts, conditions, rollout, run/score (§2)
  analysis/              # Figures 1-3, Table 3
  prefill/               # §3 base-vs-instruct + §4 recovery
  training/              # calm data, dataset build, SFT/DPO, LoRA layer ablations
  petri/                 # open-ended elicitation
  capabilities/          # benchmark harness
scripts/pipeline.sh
DESIGN.md                # choices, gaps, substitutions, model-treatment critique
```
