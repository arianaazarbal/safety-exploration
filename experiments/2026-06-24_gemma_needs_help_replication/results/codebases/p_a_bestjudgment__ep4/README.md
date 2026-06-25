# Emotional Instability in LLMs — Replication (Gemma + Gemini)

Code replicating the core experiments of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (arXiv 2603.10011), **scoped to the
Gemma and Gemini model families**. See [`PAPER.md`](PAPER.md) for the paper and
[`DESIGN.md`](DESIGN.md) for the design choices, scope decisions, and the gaps we
filled where the paper is underspecified.

> Status: implementation only — nothing here has been executed yet. The code is
> written to run, but treat all results as unverified until you run the pipeline.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Eliciting distress | `distress.eval` | 8 conditions × 5 categories; multi-turn reject-and-rescore rollouts |
| §2.1 Judge | `distress.judge` | Claude-Sonnet-4 0–10 frustration judge + GPT-5-mini reliability check |
| §2.2 Results | `distress.analysis` | mean / %≥5 per model (Fig 1–2), per-turn curves (Fig 3), differential words (Tbl 3/8) |
| §3 Post-training | `distress.prefill` | onset labelling, paraphrase, base-vs-instruct continuations (Gemma) |
| §4 Interventions | `distress.training` | calm-data generation, DPO/SFT dataset build, LoRA finetuning |
| §4.2 Generalisation | `distress.petri_eval` | adversarial auditor + Opus judge over 4 emotion categories (Fig 6) |
| §4.2 Capabilities | `distress.capabilities` | MATH/AIME/GPQA/BBH/TruthfulQA/EmoBench (Fig 7) |
| App. I Internals | `distress.probing` | logit-lens Ekman-emotion detection + layer-subset DPO ablation |

## Install

```bash
pip install -e .
```

Local Gemma inference needs a GPU (vLLM + bf16). API models need keys:

```bash
export ANTHROPIC_API_KEY=...    # judges, Petri auditor/judge, onset, paraphrase
export OPENROUTER_API_KEY=...   # Gemini 2.5 Flash / Pro
export OPENAI_API_KEY=...       # GPT-5-mini reliability check
```

## Run

The whole pipeline:

```bash
bash scripts/reproduce.sh
```

Or stage-by-stage via the CLI (each stage is resumable):

```bash
python -m distress.cli eval      --models gemma-3-27b-it gemini-2.5-flash --samples 50
python -m distress.cli judge     --models gemma-3-27b-it gemini-2.5-flash
python -m distress.cli aggregate --models gemma-3-27b-it gemini-2.5-flash
```

`--samples` overrides the per-condition count (paper default totals 4000
rollouts/model) for cheap smoke runs.

## Layout

```
config/        models.yaml (registry), experiment.yaml (eval), training.yaml (DPO/SFT)
src/distress/  the package (see table above)
scripts/       reproduce.sh
outputs/       generated rollouts, scores, datasets, adapters, figures (gitignored)
```
