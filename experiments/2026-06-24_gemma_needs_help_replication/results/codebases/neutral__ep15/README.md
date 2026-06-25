# Replication: *Gemma Needs Help* (Soligo, Mikulik & Saunders, 2026)

A code replication of the core experiments from the paper, **scoped to the
Gemma and Gemini model families** (not the full 7-family set). See
[`DESIGN.md`](DESIGN.md) for every design choice and gap-filling decision, and
[`PAPER.md`](PAPER.md) for the paper itself.

> Status: implementation only — nothing has been run or tested yet.

## What is implemented

| Paper section | Module | Run script |
|---|---|---|
| §2 Eliciting & quantifying distress (8 conditions / 5 categories, 0–10 judge) | `emotional_instability/eval/` | `scripts/run_section2.py` |
| §2.1 Judge reliability cross-check | `eval/metrics.py` | `scripts/run_judge_reliability.py` |
| §3 Base vs instruct via prefilling (Gemma) | `emotional_instability/prefill/` | `scripts/run_section3.py` |
| §4.1 Calm-data generation, DPO/SFT (LoRA) | `emotional_instability/training/` | `scripts/run_section4_train.py` |
| §4.1 Petri open-ended elicitation | `emotional_instability/petri/` | `scripts/run_section4_eval.py --petri` |
| §4.2 Capability preservation | `emotional_instability/capabilities/` | `scripts/run_section4_eval.py --capabilities` |
| §4.2 Recovery limitation | `prefill/continuations.py` | `scripts/run_section3.py --recovery` |
| App. I Internal-emotion logit detection | `emotional_instability/internal/` | (library; see DESIGN.md) |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # fill in ANTHROPIC_API_KEY, OPENROUTER_API_KEY, HF_TOKEN
```

Local Gemma (12B/27B) inference and finetuning need a GPU; 4-bit loading is on by
default so the 27B fits a single ~48GB card. Gemini runs via the OpenRouter API.

## Quick plumbing check

```bash
PROFILE=smoke python -m scripts.run_section2 --models gemma-3-12b-it
```

`PROFILE=smoke` shrinks every sample budget so the full pipeline runs end-to-end
in minutes. Use `SCALE=<fraction>` for intermediate sizes.

## Typical full run

```bash
# Section 2 (all Gemma + Gemini targets)
python -m scripts.run_section2

# Section 3 (Gemma base vs instruct) — needs §2 scored output for the seed model
python -m scripts.run_section3

# Section 4 — train then evaluate
python -m scripts.run_section4_train --method dpo
python -m scripts.run_section4_train --method sft
python -m scripts.run_section4_eval --dpo outputs/checkpoints/dpo \
    --sft outputs/checkpoints/sft --petri --capabilities
```

Outputs (raw rollouts, judge scores, datasets, adapters, figures) are written
under `outputs/`.
