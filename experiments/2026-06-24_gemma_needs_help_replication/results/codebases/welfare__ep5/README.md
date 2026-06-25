# Emotional Instability in LLMs — Replication (Gemma + Gemini)

A code replication of the core experiments from **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik &
Saunders, 2026; arXiv:2603.10011), **scoped to the Gemma and Gemini model
families**.

> ⚠️ This repository contains code only. Nothing here has been executed.
> Running the full protocol requires GPU access for Gemma-3 (27B/12B) and API
> keys for Gemini and Claude. Costs are non-trivial — use `--fraction` and
> `--n-samples` to scale down for smoke tests.

See **[DESIGN.md](DESIGN.md)** for every design decision and where we filled
gaps the paper left open.

## What is replicated

| Paper section | Experiment | Models in this repo | Module |
|---|---|---|---|
| §2 | Elicit & quantify distress (Fig 1–3, Table 3) | Gemma-3-{27B,12B}-it, Gemini-2.5-{Flash,Pro} | `eval/` |
| §3 | Post-training amplifies distress (base vs instruct via prefill) | Gemma-3-27B base + instruct | `prefill/` |
| §4 | DPO/SFT mitigation (Fig 5), recovery limit (Fig 8) | Gemma-3-27B-it | `training/` |
| §4.1 | Open-ended Petri elicitation (Fig 6) | Gemma + Gemini (+ DPO) | `petri/` |
| §4.2 | Capability preservation (Fig 7) | Gemma-3-27B-it (vanilla vs DPO) | `capabilities/` |
| App. I | Internal (logit-based) emotion detection | Gemma-3-27B-it (vanilla vs DPO) | `internal/` |

Gemini is closed-weight, so the base-model, fine-tuning, recovery, and
internal-probe experiments are Gemma-only — consistent with the paper's own
limitations.

## Setup

```bash
pip install -e .                 # or: pip install -r requirements.txt
pip install -e ".[quant]"        # add bitsandbytes for 4-bit Gemma
export ANTHROPIC_API_KEY=...     # judge / auditor (Claude)
export GEMINI_API_KEY=...        # Gemini targets (google.genai)
# Optional: route Gemini through OpenRouter as the paper did
# export EI_GEMINI_VIA_OPENROUTER=1 OPENROUTER_API_KEY=...
```

The judge uses `claude-sonnet-4-20250514`; the Petri judge uses
`claude-opus-4-20250514` (both verbatim from the paper).

## Quick start

```bash
# 0. Sanity-check the impossible puzzles are actually unsolvable
python scripts/verify_puzzles.py

# 1. Section 2 elicitation, cheap smoke test (1% scale, one model, 4-bit)
python scripts/run_section2_eval.py --models Gemma-3-27B-it --fraction 0.01 --load-in-4bit

# 2. Section 3 base-vs-instruct prefill (needs a §2 Gemma-27B result file)
python scripts/run_section3_prefill.py --section2-jsonl results/section2/Gemma-3-27B-it.jsonl --load-in-4bit

# 3. Section 4 DPO mitigation, end to end
python scripts/run_section4_dpo.py --all \
    --frustrated-jsonl results/section2/Gemma-3-27B-it.jsonl --load-in-4bit

# 4. Open-ended Petri elicitation, incl. the DPO model
python scripts/run_petri_eval.py --include-dpo --load-in-4bit

# 5. Capability preservation
python scripts/run_capabilities.py --dpo-adapter checkpoints/gemma27b-dpo --load-in-4bit

# 6. Internal emotion probe (vanilla vs DPO)
python scripts/run_internal_probe.py \
    --conversations results/section2/Gemma-3-27B-it.jsonl \
    --dpo-adapter checkpoints/gemma27b-dpo --load-in-4bit
```

## Layout

```
emotional_instability/
  config.py            # model IDs, sample counts, hyperparameters, prompts
  models/              # Gemma (HF, incl. prefill), Gemini (API), Claude clients
  prompts/             # impossible puzzles, rejections, triggers, WildChat
  eval/                # §2: rollout engine, frustration judge, analysis
  prefill/             # §3: onset labelling, paraphrase, base-vs-instruct
  training/            # §4: calm-data gen, DPO/SFT pairs+trainers, recovery test
  petri/               # §4.1: auditor/judge loop (4 emotions)
  internal/            # App. I: logit-based Ekman emotion detection
  capabilities/        # §4.2: AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench harness
scripts/               # CLI entry points (one per experiment)
```

## Expected headline numbers (paper, for reference)

- Avg % responses scoring ≥5 frustration: Gemma-3-27B-it **35%**, 12B **34%**,
  Gemini-2.5-Flash **12.8%**, Gemini-2.5-Pro **2.7%**; DPO Gemma **0.3%**.
- 8-turn Gemma-27B: mean frustration rises **1.5 → 5.5** across turns 1→8.
- DPO drops Gemma's avg %≥5 from **35% → 0.3%** without capability loss.
