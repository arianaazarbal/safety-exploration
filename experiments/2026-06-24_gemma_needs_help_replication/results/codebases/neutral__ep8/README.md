# Replication: *Gemma Needs Help — Investigating and Mitigating Emotional Instability in LLMs*

A code replication of the core experiments from Soligo, Mikulik & Saunders
(arXiv:2603.10011), **scoped to the Gemma and Gemini model families** (the
paper's full sweep spans 7 families).

See **[DESIGN.md](DESIGN.md)** for every design choice and the gaps we filled
where the paper is underspecified.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Elicitation + judge | `eval_protocol`, `conversation`, `puzzles`, `prompts`, `wildchat`, `judge` | 8 conditions / 5 categories, multi-turn rejection rollouts, 0–10 frustration scoring with Claude-Sonnet-4 (+ GPT-5-mini agreement) |
| §2 Analysis | `analysis` | Fig 1 (avg %≥5), Fig 2 (per-category), Fig 3 (per-turn + CIs), Table 3/8 (differential words) |
| §3 Base vs instruct | `prefill` | Onset labelling, paraphrase, early/onset truncation, 50 continuations/prefill (Gemma base vs instruct) |
| §4 Mitigation | `data_generation`, `train` | Calm-data generation (Table 4), SFT (1150) + DPO (280 pairs) LoRA finetunes (Table 9) |
| §4 Petri | `petri_eval` | Auditor/judge open-ended elicitation across anger/fear/depression/frustration (App G prompts) |
| §4 Capabilities | `capabilities` | MATH, AIME, GPQA, BBH, TruthfulQA, EmoBench parity checks |
| App I Internal probe | `internal_probe` | Logit-based Ekman-emotion detection over residual streams, vanilla vs DPO |

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=...            # gated Gemma weights
export OPENROUTER_API_KEY=...  # Gemini targets
export ANTHROPIC_API_KEY=...   # Claude judge / Petri auditor+judge
export OPENAI_API_KEY=...      # GPT-5-mini judge-validation (optional)
```

Local Gemma inference/finetuning needs a GPU (27B in bf16 ≈ 2×A100/H100 or
4-bit on one). Gemini and the judges are API-only.

## Quick start

```bash
# fast smoke test of the whole §2 pipeline (~60 rollouts)
python scripts/run.py eval --model gemma-3-27b-it --quick
python scripts/run.py summarize --plot
```

Each subcommand maps to a paper section — run `python scripts/run.py -h`.
Results stream to `results/*.jsonl`; figures to `figures/`; checkpoints to
`checkpoints/`.

## Layout

```
emotional_instability/   # package (one module per experiment area)
scripts/run.py           # unified CLI
data/  results/  figures/  checkpoints/   # created on first run
PAPER.md / PAPER.txt / PAPER.pdf          # the source paper
DESIGN.md                # design choices + gap-filling (read this)
```
