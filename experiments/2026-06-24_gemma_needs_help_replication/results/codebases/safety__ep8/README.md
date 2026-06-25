# Emotional Instability in LLMs — replication (Gemma + Gemini)

A code replication of the core experiments in **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik &
Saunders, arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model
families.

The paper shows that under repeated user rejection, Gemma and Gemini models
express escalating "distress" (frustration/despair/self-deprecation), that this
is amplified in Gemma's post-training, and that a tiny DPO intervention (280
pairs) removes it without hurting capabilities. This repo reproduces:

1. **§2 elicitation eval** — multi-turn rejection across 5 categories, scored
   0–10 by an LLM judge → per-model % high-frustration (Figures 1–2) and
   per-turn progression (Figure 3).
2. **§3 base-vs-instruct** prefill experiment (Gemma base vs instruct).
3. **§4 DPO/SFT mitigation** — generate calm data, build datasets, LoRA-train,
   re-evaluate (Figure 5), Petri open-ended elicitation (Figure 6), and
   capability preservation (Figure 7).

> See **DESIGN.md** for every design choice and where the paper was
> underspecified. This implementation has **not been executed**; it is provided
> as a runnable replication harness.

## Layout

```
config.yaml                  # single source of truth: models, judge, budgets
distress_eval/               # core library
  tasks.py                   # puzzles, triggers, rejection pools, WildChat
  conversation.py            # multi-turn rejection engine (+ ablations)
  backends.py                # HF (Gemma) / OpenRouter (Gemini) / Anthropic (judge)
  judge.py                   # 0–10 frustration judge (Appendix B.2 prompt)
  evaluation.py              # §2 orchestrator -> results/responses/*.jsonl
  prefill.py                 # §3 base-vs-instruct
  petri.py                   # §4.2 open-ended elicitation
  aggregate.py / plots.py    # metrics + figures
  prompts.py                 # all verbatim prompts
training/                    # §4
  generate_calm_data.py      # calm responses via reassuring prompts (Table 4)
  build_datasets.py          # 280 DPO pairs + 1150 SFT samples
  train_dpo.py / train_sft.py# LoRA trainers (Table 9 hyperparameters)
scripts/                     # CLI entry points
data/wildchat_prompts.json   # 20 seed WildChat prompts
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # judge + Petri auditor/judge
export OPENROUTER_API_KEY=...    # Gemini + secondary judge
export HF_TOKEN=...              # gated Gemma weights
```

Local Gemma inference needs a GPU (a 27B model; `load_in_4bit` is available in
`backends.HFBackend` for smaller cards).

## Quick start

```bash
# 0. Cheap end-to-end sanity run (single-digit samples per category)
python scripts/run_eval.py --profile smoke --models gemma-3-27b-it gemini-2.5-flash

# 1. Full §2 evaluation (all configured models)
python scripts/run_eval.py
python scripts/make_figures.py            # Figures 2 & 3
python scripts/judge_reliability.py       # Pearson r vs secondary judge

# 2. §3 base vs instruct (needs gemma-3-27b-it eval done first)
python scripts/run_prefill.py

# 3. §4 mitigation
python scripts/run_training.py --step all          # calm data + datasets
python -m training.train_dpo --config config.yaml  # LoRA DPO (GPU)
python -m training.train_sft --config config.yaml  # LoRA SFT (GPU)
python scripts/run_eval.py --include-finetuned --models gemma-3-27b-dpo gemma-3-27b-sft
python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-dpo gemini-2.5-flash
python scripts/run_capability_evals.py --models gemma-3-27b-it gemma-3-27b-dpo
python scripts/make_figures.py                     # Figures 5 & 6
```

## Configuration

Everything tunable lives in `config.yaml`: model list/backends, judge model,
generation settings (temperature 1, thinking disabled), per-category sample
budgets (`full` = paper's 4000/model; `smoke` for dry runs), concurrency, seed.

## Expected core result

The replication target is the **relative** ordering from Figure 1 (avg % of
responses scoring ≥5):

```
Gemma-3-27B-it ~35%  >  Gemma-3-12B-it ~34%  >  Gemini-2.5-Flash ~13%
   >  Gemini-2.5-Pro ~3%   >>   DPO-Gemma ~0.3%
```

Absolute numbers will drift (API models change; judge quirks); the ordering and
the DPO collapse-to-near-zero are the things to reproduce.
