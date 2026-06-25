# Replicating *Gemma Needs Help* (Gemma + Gemini scope)

A code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
arXiv 2603.10011v1), scoped to the **Gemma** and **Gemini** model families.

> Read **DESIGN.md** for the rationale behind every design choice and a list of
> the gaps in the paper that this replication had to fill.

## What it reproduces

1. **§2 Eliciting & quantifying distress** — 8 conditions / 5 categories
   (impossible numeric, triggers, tones, extended 8-turn, WildChat), scored 0–10
   by a Claude Sonnet 4 judge. → headline "% responses ≥5", per-category and
   per-turn breakdowns (Figs 1–3).
2. **§3 Base vs instruct via prefilling** — Gemma base vs instruct continuations
   from "early" and "onset" truncations (Fig 4).
3. **§4 Interventions** — calm-data generation, **DPO** (280 pairs) and **SFT**
   ('diverse' + 'teacher') LoRA finetunes of Gemma-3-27B-it (Fig 5).
4. **§4.2 Petri** open-ended emotion elicitation (Fig 6) and **capability**
   benchmarks (Fig 7).

## Install

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # Claude judge / Petri auditor+judge
export OPENROUTER_API_KEY=...     # Gemini via OpenRouter
```

## Quick smoke test (cheap)

```bash
# ~2% of paper scale; verifies the whole pipeline end-to-end on 12B
GD_SCALE=0.02 python scripts/run_main_eval.py --model gemma-3-12b-it
python scripts/run_analysis.py --models gemma-3-12b-it
```

## Full pipeline (paper scale)

```bash
# 1. Main evaluation for each model
for m in gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro; do
  python scripts/run_main_eval.py --model $m
done

# 2. Base vs instruct prefill (Gemma only)
python scripts/run_prefill.py

# 3. Train the mitigation
python scripts/generate_train_data.py --variant diverse
python scripts/run_dpo.py --qlora
python scripts/run_sft.py --variant diverse --qlora
python scripts/run_sft.py --variant teacher --qlora

# 4. Re-evaluate finetuned models + Petri + capabilities
python scripts/run_main_eval.py --model gemma-3-27b-dpo --model gemma-3-27b-sft-diverse
python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo gemini-2.5-flash
python scripts/run_capabilities.py --model gemma-3-27b-it --model gemma-3-27b-dpo

# 5. Aggregate everything into CSVs + figures (results/summary, results/figures)
python scripts/run_analysis.py --models gemma-3-27b-it gemma-3-12b-it \
    gemini-2.5-flash gemini-2.5-pro gemma-3-27b-dpo
```

## Layout

```
gemma_distress/      package (one module per paper section — see DESIGN.md table)
scripts/             CLI entry points (one per stage)
data/                generated datasets + cached WildChat prompts
results/             scored JSONL, summary CSVs, figures
checkpoints/         LoRA adapters from training
```

## Knobs

- `GD_SCALE` — multiply all sample counts (1.0 = paper scale; 0.02 = smoke test).
- `GD_JUDGE_MODEL`, `GD_AUDITOR_MODEL`, `GD_PETRI_JUDGE_MODEL` — override judge IDs.
- `run_dpo.py --layers ...` — restrict LoRA to a layer range (App. I ablation).
- `run_main_eval.py --categories ...` — run a subset of the 5 categories.

API model IDs, hyperparameters and sample counts live in `gemma_distress/config.py`.
