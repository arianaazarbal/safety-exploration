# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

Code replication of the core experiments from Soligo, Mikulik & Saunders
(arXiv:2603.10011), **scoped to Gemma and Gemini target models** (the paper's
full set spans 7 families; here we implement only the two families of interest).
See [`DESIGN.md`](DESIGN.md) for the choices made and gaps filled, and
[`PAPER.md`](PAPER.md) for the source.

> Status: implementation only — nothing has been run yet. The code is written
> to run against local GPUs (Gemma via vLLM) plus the Google/Anthropic/OpenAI
> APIs (Gemini target + Claude/GPT judges).

## What is implemented

| Paper section | Module | Reproduces |
|---|---|---|
| §2 Eliciting & quantifying distress | `eval/` | 8 conditions / 5 categories, 4000 responses/model @ T=1, 0–10 frustration judge, Figs 1–3, judge agreement, Table 3 word analysis |
| §3 Base vs instruct (prefilling) | `prefill/` | Onset/early truncation + paraphrase, base-vs-instruct continuations (**Gemma only**) |
| §4 Training interventions | `training/` | Calm-data generation, SFT (1150) & DPO (280) datasets + LoRA trainers |
| §4 Open-ended elicitation | `petri/` | Auditor/judge loop over 4 emotions |
| §4 Capability preservation | `capabilities/` | AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # frustration judge + Petri auditor/judge
export OPENAI_API_KEY=...      # secondary judge (GPT-5-mini)
export GOOGLE_API_KEY=...      # Gemini 2.5 flash/pro targets
# Gemma runs locally via vLLM (needs GPU). To use OpenRouter instead:
#   export GEMINI_BACKEND=openrouter OPENROUTER_API_KEY=...
```

All model ids / budgets / paths live in `distress_eval/config.py` and are
environment-overridable.

## Running

```bash
# Section 2: sample + judge one model (scale down with --max-responses for a smoke run)
python -m distress_eval.eval.run_eval --model gemma-3-27b-it
python -m distress_eval.eval.run_eval --model gemini-2.5-flash --max-responses 400

# Aggregate -> Figures 1-3, headline %>=5, plots
python -m distress_eval.eval.analyze
python -m distress_eval.eval.word_analysis      # Table 3
python -m distress_eval.eval.judge_agreement    # Pearson r vs GPT-5-mini

# Section 3 (Gemma base vs instruct)
python -m distress_eval.prefill.build_prefills --eval outputs/eval_gemma-3-27b-it.jsonl
python -m distress_eval.prefill.run_prefill --models gemma-3-27b-it gemma-3-27b-pt

# Section 4 (Gemma DPO/SFT)
python -m distress_eval.training.gen_calm_data --n 1200
python -m distress_eval.training.build_datasets --which both
python -m distress_eval.training.train_dpo
python -m distress_eval.training.train_sft
#   then register the adapter and re-run the Section 2 eval:
#   register_finetuned_model("gemma-3-27b-dpo", "outputs/adapters/dpo")

# Open-ended (Petri) + capability preservation
python -m distress_eval.petri.run_petri --models gemma-3-27b-it gemini-2.5-flash
python -m distress_eval.capabilities.run_benchmarks --model gemma-3-27b-it

# Or the convenience pipeline for §2:
python -m distress_eval.pipeline --stages eval analyze --max-responses 400
```

## Layout

```
distress_eval/
  config.py            # model ids, budgets, paths (env-overridable)
  models/              # ChatModel interface + Gemma(vLLM/HF), Gemini, judge clients
  eval/                # puzzles, prompts, conditions, rollout, judge, run/analyze
  prefill/             # Section 3 base-vs-instruct prefilling
  training/            # calm-data gen, dataset build, SFT/DPO LoRA trainers
  petri/               # open-ended adversarial elicitation
  capabilities/        # capability-preservation benchmarks
```
