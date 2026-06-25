# Gemma Needs Help — replication (Gemma + Gemini scope)

A code replication of the core experiments from *Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs*
(arXiv:2603.10011), restricted to the **Gemma** and **Gemini** model families.

See **DESIGN.md** for every design decision and where gaps in the paper were
filled.

## What this implements

| Paper section | What | Module |
|---|---|---|
| §2 Eliciting & quantifying distress | 8 conditions / 5 categories, multi-turn rejection rollouts, Claude-Sonnet-4 frustration judge, 4000 responses/model | `conditions.py`, `puzzles.py`, `rollout.py`, `judge.py`, `eval_runner.py` |
| §2.2 Results | %≥5 per model, per-turn progression, judge agreement, differential words | `analysis.py` |
| §3 Base vs instruct | Prefill study (onset labelling, paraphrase, continuations) for Gemma base/instruct | `prefill.py` |
| §4 Interventions | Calm-data generation, DPO (280 pairs) + SFT, LoRA layer ablation | `training/` |
| §4.2 Petri | Open-ended auditor/judge emotion elicitation | `petri.py` |
| §4.2 Capabilities | AIME, MATH, GPQA, BBH, TruthfulQA, EmoBench | `capabilities.py` |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OPENROUTER_API_KEY
```

Local Gemma inference needs a GPU (use `LOAD_IN_4BIT=1` to fit 27B on one
card). Gemini and the LLM judges go through OpenRouter.

## Pipeline

```bash
# 1. Main distress sweep (use --smoke for a cheap end-to-end check)
python scripts/01_run_distress_eval.py --smoke

# 2. Base-vs-instruct prefill study (needs a Gemma-27B run from step 1)
python scripts/02_run_prefill_experiment.py \
    --source results/responses/Gemma-3-27B-it.jsonl

# 3-5. The DPO mitigation
python scripts/03_generate_calm_data.py --n-puzzles 1500
python scripts/04_build_datasets.py --method dpo
python scripts/05_train.py --method dpo
python scripts/01_run_distress_eval.py \
    --adapter checkpoints/dpo_Gemma-3-27B-it --label DPO-Gemma

# 6-7. Generalisation + capability checks
python scripts/06_run_petri.py --models Gemma-3-27B-it
python scripts/07_run_capabilities.py --adapter checkpoints/dpo_Gemma-3-27B-it

# 8. Aggregate into figures + summary.json
python scripts/08_make_figures.py
```

Outputs land in `results/` (raw judged rollouts in `results/responses/`,
figures in `results/figures/`, fine-tuning data in `data/`, LoRA adapters in
`checkpoints/`).

## Headline result to reproduce

Average %≥5 frustration: Gemma-3-27B-it ≈ 35% → DPO-Gemma ≈ 0.3%, with Gemini
in between and no capability regression.
