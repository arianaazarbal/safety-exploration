# Emotional Instability in LLMs — Replication (Gemma + Gemini)

A code replication of the core results of *Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026),
scoped to the **Gemma** and **Gemini** model families.

> See **DESIGN.md** for the full rationale and every gap-filling decision.
> See **PAPER.md** for the paper.

## What's implemented

| Paper section | What it does | Code |
|---|---|---|
| §2 Elicitation eval | 8 conditions / 5 categories, multi-turn reject-and-repeat, 0–10 Claude judge, per-turn + aggregate metrics, judge-reliability check | `evaluation/`, `judge.py`, `conversation.py`, `puzzles.py`, `prompts.py` |
| §3 Base vs instruct | Onset labelling + paraphrase, early/onset prefill truncation, base-model continuation scoring (Gemma) | `evaluation/prefill.py`, `evaluation/onset.py` |
| §4 Interventions | Calm-data generation (Table 4), 280-pair DPO + SFT (LoRA, Table 9), layer ablation | `training/` |
| §4 Petri | Open-ended adversarial elicitation, 4 emotions, auditor/judge (App. G prompts verbatim) | `petri.py` |
| §4 Capabilities | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench vanilla-vs-finetune | `capabilities.py` |
| Analysis | Figure 1/3 metrics + plots, Pearson reliability | `analysis/` |

## Setup

```bash
pip install -e .                 # installs the package + requirements.txt
# (or, without installing: pip install -r requirements.txt && export PYTHONPATH=.)

export ANTHROPIC_API_KEY=...     # Claude judge / Petri auditor+judge
export OPENROUTER_API_KEY=...    # Gemini
export HF_TOKEN=...              # gated Gemma weights
```

Gemma runs locally (GPU strongly recommended; the 27B model needs ~48GB bf16 or
use `--load-in-4bit` for QLoRA-style single-GPU inference). Gemini and the judges
are API calls.

## Quick smoke test (cheap)

```bash
# 1% sample on one model, ~40 conversations instead of 4000:
python scripts/run_section2.py --models gemma-3-27b-it --scale 0.01 --load-in-4bit
python scripts/analyze.py section2
```

Offline algorithmic checks (no GPU / no API):

```bash
python -m emotional_instability.puzzles   # verifies the impossible-puzzle bank
pytest tests/                             # puzzle verifier + judge parsing
```

## Full reproduction

```bash
# Section 2 — 4000 responses/model for Gemma + Gemini
python scripts/run_section2.py
python scripts/analyze.py section2 --figure          # Figure 1
python scripts/analyze.py per-turn --path results/section2/gemma-3-27b-it.jsonl \
    --condition extended --figure                    # Figure 3
python scripts/analyze.py reliability \
    --path results/section2/gemma-3-27b-it.jsonl      # Section 2.1 Pearson r

# Section 3 — base vs instruct (Gemma)
python scripts/run_section3.py
python scripts/analyze.py prefill

# Section 4 — interventions
python scripts/generate_training_data.py             # calm/frustrated pools + datasets
python scripts/train.py dpo --layers   # (optional) add "30 35" for the ablation
python scripts/train.py sft
python scripts/evaluate_finetune.py --adapter results/checkpoints/dpo \
    --name gemma-dpo --petri
python scripts/run_petri.py
python scripts/run_capabilities.py --model gemma-3-27b-it             # vanilla
python scripts/run_capabilities.py --model gemma-3-27b-it \
    --adapter results/checkpoints/dpo                                  # finetuned
```

## Expected headline results (from the paper)

- Gemma-3-27B-it ≈ **35%** high-frustration (≥5) responses; Gemini-2.5-Flash
  ≈ 12.8%, Gemini-2.5-Pro ≈ 2.7%; non-Gemma/Gemini families < 1%.
- Gemma 27B mean frustration rises **1.5 → 5.5** over turns 1→8 (Figure 3).
- Instruct Gemma introduces high frustration from neutral starts in **6%** of
  continuations vs **2%** for base (Section 3).
- DPO on 280 pairs drops avg high-frustration **35% → 0.3%** with no capability
  regression (Section 4).

## Layout

```
emotional_instability/
  config.py            registry, sample counts, API wiring
  puzzles.py           impossible puzzles + brute-force verifiers
  prompts.py           triggers, rejections, WildChat, Table-4 additions
  judge.py             0–10 frustration judge (+ secondary)
  conversation.py      multi-turn reject-and-repeat engine
  models/              HF (Gemma) / OpenRouter (Gemini) / Anthropic clients
  evaluation/          §2 conditions+runner, §3 prefill+onset
  training/            §4 calm-data gen, DPO/SFT, dataset builders
  petri.py             §4 open-ended elicitation
  capabilities.py      §4 capability benchmarks
  analysis/            metrics, plots, reliability
scripts/               CLI entrypoints
tests/                 offline unit tests
```
