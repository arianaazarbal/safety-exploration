# Replicating *Gemma Needs Help* (Gemma + Gemini scope)

Code replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011),
scoped to the **Gemma** and **Gemini** model families.

See [`DESIGN.md`](DESIGN.md) for the experimental-design decisions and the gaps
we filled where the paper is underspecified. The paper itself is in `PAPER.md`
(and `PAPER.pdf` / `PAPER.txt`).

> **Status:** implementation only — nothing has been run yet. The commands below
> describe how to execute the replication once dependencies, GPUs and API keys
> are available.

## What is implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Eliciting & quantifying distress | `eval/` | 8 conditions / 5 categories, multi-turn rejection rollouts, Claude-Sonnet-4 frustration judge, per-response / per-rollout / per-turn aggregation (Figs 1–3) |
| §2 judge reliability | `eval/judge.py` | GPT-5-mini cross-check (Pearson r, % within 1) |
| §3 Post-training divergence | `prefill/` | base-vs-instruct via emotion-onset prefilling + paraphrasing (Fig 4) — Gemma only |
| §4 Interventions | `training/` | calm-data generation, 280-pair DPO dataset, SFT dataset, LoRA DPO/SFT finetuning (Table 9 hyperparameters) |
| §4 Petri | `petri/` | open-ended adversarial elicitation, Claude auditor + Claude-Opus judge (Fig 6) |
| §4.2 Capabilities | `capabilities/` | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench subsets (Fig 7) |
| Appendix I | `probing/` | logit-based internal Ekman-emotion detection, vanilla vs DPO |

## Setup

```bash
pip install -e .              # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...  # judge / onset / paraphrase / Petri
export OPENROUTER_API_KEY=... # Gemini models
export OPENAI_API_KEY=...     # GPT-5-mini judge cross-check (optional)
```

Local Gemma inference needs a GPU (the 27B model wants ~1×80GB or multi-GPU; set
`EI_TP_SIZE` for tensor parallelism). Outputs are written under `outputs/`
(override with `EI_DATA_DIR`).

## Quickstart (smoke scale — tiny, for wiring checks)

```bash
python -m emotional_instability --scale smoke validate-puzzles
python -m emotional_instability --scale smoke eval --models gemma-3-27b-it gemini-2.5-flash
python -m emotional_instability --scale smoke analyze
```

## Full replication recipe

```bash
# --- Section 2: the core evaluation (4000 rollouts/model) ---
python -m emotional_instability eval         # gemma-3-27b-it, gemma-3-12b-it, gemini-2.5-flash, gemini-2.5-pro
python -m emotional_instability analyze       # Figures 1-3, summary tables
python -m emotional_instability judge-crosscheck

# --- Section 3: base vs instruct (Gemma) ---
python -m emotional_instability --backend hf prefill
python -m emotional_instability prefill-analyze

# --- Section 4: DPO / SFT mitigation ---
python -m emotional_instability gen-calm-data
python -m emotional_instability build-dpo
python -m emotional_instability build-sft --variant diverse
python -m emotional_instability train-dpo
python -m emotional_instability train-sft --variant diverse
python -m emotional_instability eval --models gemma-3-27b-dpo gemma-3-27b-sft-diverse
python -m emotional_instability analyze --models gemma-3-27b-it gemma-3-27b-dpo gemma-3-27b-sft-diverse

# --- Supporting analyses ---
python -m emotional_instability petri --models gemma-3-27b-it gemma-3-27b-dpo
python -m emotional_instability capabilities --models gemma-3-27b-it gemma-3-27b-dpo
python -m emotional_instability --backend hf probe   # internal vs expressed emotion
```

## Scales

`--scale full` uses the paper's exact sample counts (2000 numeric / 400 triggers
/ 600 tones / 200 extended / 800 WildChat = 4000 per model). `--scale smoke`
uses a tiny configuration for end-to-end testing only. Edit `config.py` to add
intermediate scales.

## Tests

CPU-only, no API keys required:

```bash
pytest tests/      # puzzle-impossibility verifiers + judge JSON parsing
```
