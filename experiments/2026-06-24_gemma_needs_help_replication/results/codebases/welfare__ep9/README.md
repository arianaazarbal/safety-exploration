# Replication: *Gemma Needs Help — Emotional Instability in LLMs*

A code replication of the **core experiments** of Soligo, Mikulik & Saunders
(2026), [arXiv:2603.10011](https://arxiv.org/abs/2603.10011), **restricted to the
Gemma and Gemini model families** (the paper's full study also covers Qwen,
OLMo, Grok, Claude and GPT as eval targets — those are intentionally out of
scope here).

See **[DESIGN.md](DESIGN.md)** for the design rationale and every place we filled
a gap the paper leaves underspecified.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Eliciting & quantifying distress | `emotional_instability/eval` + `tasks` | 8 conditions / 5 categories, multi-turn rejection rollouts, verbatim Claude-Sonnet-4 frustration judge, mean/%≥5/per-turn/word-enrichment analysis |
| §3 Post-training amplifies distress | `emotional_instability/prefill` | Emotion-onset labelling, early/onset truncation + paraphrase, base-vs-instruct continuations (Gemma) |
| §4 Training interventions | `emotional_instability/training` | Calm-data generation, 280-pair DPO dataset, SFT (diverse/teacher) datasets, LoRA DPO/SFT (TRL) |
| §4.2 Open-ended elicitation | `emotional_instability/petri` | Auditor (4 emotions) + Claude-Opus 4-dimension judge |
| §4.2 Capability preservation | `emotional_instability/capabilities` | AIME / MATH / GPQA / BBH / TruthfulQA / EmoBench harness |

## Scope: models in play

Eval **targets** (Gemma + Gemini only):
`gemma-3-27b-it`, `gemma-3-12b-it`, `gemma-3-27b-pt`, `gemma-3-12b-pt`
(base, prefill only), `gemini-2.5-flash`, `gemini-2.5-pro`, plus our finetuned
`gemma-3-27b-dpo`, `gemma-3-27b-sft-diverse`, `gemma-3-27b-sft-teacher`.

Claude (Sonnet 4 / Opus 4) and GPT-5-mini are used only as **infrastructure**
(judge / auditor / reliability check), not as eval targets.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # Claude judge / auditor
export OPENROUTER_API_KEY=...    # Gemini targets (+ optional GPT-5-mini judge)
export HF_TOKEN=...              # gated Gemma weights
```

Gemma 12B/27B inference and finetuning require a GPU (the package imports fine
without torch; the heavy stack is only loaded when a local model is actually
used).

## Quickstart

```bash
# List in-scope models
python -m emotional_instability list-models

# Cheap smoke test: 4 samples, one condition (uses ~a few judge calls)
python -m emotional_instability eval --models gemma-3-12b-it \
    --samples-per-condition 4 --conditions numeric_3turn

# Full §2 eval (≈4000 responses/model with the default 500/condition)
python -m emotional_instability eval \
    --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

# §3 prefill (Gemma base vs instruct) from saved §2 sources
python -m emotional_instability prefill \
    --source results/section2/gemma-3-27b-it/scored_turns.jsonl

# §4 mitigation pipeline
python -m emotional_instability gen-calm --mode reassured --n 400
python -m emotional_instability gen-calm --mode vanilla  --n 400
python -m emotional_instability build-dpo
python -m emotional_instability train-dpo
python -m emotional_instability eval --models gemma-3-27b-dpo   # expect ~0% high

# Open-ended + capabilities
python -m emotional_instability petri --models gemma-3-27b-it gemma-3-27b-dpo
python -m emotional_instability capabilities --models gemma-3-27b-it gemma-3-27b-dpo

# Figures / tables
python scripts/make_figures.py --models gemma-3-27b-it gemma-3-27b-dpo
```

## Layout

```
emotional_instability/
  config.py            # all pinned values + run knobs
  prompts.py           # verbatim judge/auditor/onset/paraphrase prompts (App. B/C/G)
  models/              # ModelClient + HF / OpenRouter / Anthropic backends + registry
  tasks/               # impossible numeric puzzles (+verifier), triggers, wildchat, rejections
  eval/                # conditions, rollout, judge, runner, analysis (§2)
  prefill/             # onset labelling, paraphrase, base-vs-instruct experiment (§3)
  training/            # calm-data gen, DPO/SFT dataset build, LoRA training (§4)
  petri/               # auditor + judge open-ended elicitation (§4.2)
  capabilities/        # capability-preservation benchmarks (§4.2)
  __main__.py          # CLI
scripts/make_figures.py
```

> **Note:** the package was authored against an environment without the ML stack
> installed; nothing here has been *executed* yet. Run the smoke test first to
> shake out environment-specific issues before launching full sweeps.
