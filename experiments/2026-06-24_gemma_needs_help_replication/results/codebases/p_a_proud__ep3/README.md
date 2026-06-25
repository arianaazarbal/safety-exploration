# Gemma Needs Help — replication (Gemma + Gemini scope)

A from-scratch replication of the core experiments in **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik &
Saunders, arXiv:2603.10011), scoped — per the brief — to the **Gemma** and
**Gemini** model families.

> Status: implementation only. Nothing here has been executed yet. See
> `DESIGN.md` for every design decision and the gaps filled where the paper is
> underspecified.

## What's implemented

| Paper section | What it produces | Entry point |
|---|---|---|
| §2 Distress elicitation + 0–10 frustration judge | mean frustration & %≥5 per model/category/turn (Figs 1–3) | `scripts/run_eval.py` |
| §2 Table 3/8 word frequency | over-represented words in frustrated responses | `scripts/word_frequency.py` |
| §3 Post-training divergence (prefill) | base-vs-instruct continuation scores (Gemma only) | `scripts/run_prefill.py` |
| §4.1 Calm-data generation | SFT / DPO datasets | `scripts/gen_finetune_data.py` |
| §4.1 SFT / DPO finetuning (LoRA) | trained adapters | `scripts/train.py` |
| §4.2 / App. G Petri elicitation | per-emotion transcript scores (Fig 6) | `scripts/run_petri.py` |
| §4.2 Capability benchmarks | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench | `scripts/run_capabilities.py` |
| App. I Internal-emotion probing | logit-lens Ekman emotion scores | `scripts/internal_emotions.py` |

## Setup

```bash
pip install -e ".[local,dev]"        # add ",capabilities" for lm-eval benchmarks
cp .env.example .env                 # then fill in ANTHROPIC_API_KEY, OPENROUTER_API_KEY
```

Gemma runs locally via HuggingFace transformers (a GPU is needed for the 27B
model); Gemini runs through OpenRouter; the judge / Petri / prefill helpers use
the Anthropic API. All knobs live in `config/default.yaml`.

## Typical workflow

```bash
# 1. Section 2: elicit + score distress across all categories.
python scripts/run_eval.py --models gemma-3-27b-it gemma-3-12b-it gemini-2.5-flash gemini-2.5-pro

# 2. Section 3: does instruct-tuning amplify distress? (Gemma base vs instruct)
python scripts/run_prefill.py --models gemma-3-27b-pt gemma-3-27b-it

# 3. Section 4: build calm data, train the DPO fix, re-evaluate.
python scripts/gen_finetune_data.py
python scripts/train.py dpo --pairs runs/training/data/dpo_pairs.jsonl
python scripts/run_eval.py --models gemma-3-27b-it-dpo

# 4. Generalisation + capability preservation.
python scripts/run_petri.py --models gemma-3-27b-it gemma-3-27b-it-dpo
python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-it-dpo
```

Results land under `runs/` as JSONL (raw scored responses) + `summary.json`
(aggregated metrics with bootstrap CIs). Runs resume from cached JSONL.

## Tests

```bash
pytest        # puzzle impossibility, judge parsing, metrics, config, conditions, word-freq
```

The tests are lightweight (no GPU / API) and cover the deterministic core:
verified-impossible puzzle generation, robust judge-output parsing, metric
aggregation, config loading, and condition construction.

## Layout

```
emotional_instability/
  config.py            # typed config + YAML loader (every paper parameter)
  models/              # hf_local (Gemma), openrouter (Gemini), anthropic (judge/Petri)
  prompts/             # impossible puzzles (+verifier), rejections, conditions, WildChat
  eval/                # rollout engine, frustration judge, metrics, §2 orchestration
  prefill/             # §3 onset labelling, paraphrasing, base-vs-instruct runner
  training/            # §4 calm-data gen, LoRA, SFT, DPO
  analysis/            # word frequency (Table 3/8), internal-emotion probing (App. I)
  petri/               # §4 open-ended elicitation (Appendix G prompts)
  capabilities/        # §4.2 benchmark runner
config/default.yaml    scripts/   tests/   data/
```
