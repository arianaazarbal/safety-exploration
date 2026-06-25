# emostab — replicating *Gemma Needs Help*

Code replication of **"Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026; arXiv:2603.10011),
scoped to the **Gemma and Gemini** model families.

The core setup: present a task, then repeatedly tell the model its answer is
wrong, turn after turn, and measure how distressed (frustrated / despairing /
self-deprecating) its responses become — on a 0–10 frustration scale judged by
Claude-Sonnet-4. The paper shows Gemma and Gemini spiral into high distress, and
that a small DPO finetune (280 pairs) almost eliminates it without hurting
capabilities. This repo implements those experiments end-to-end.

> **Design rationale, gaps filled, and scope choices: see [`DESIGN.md`](DESIGN.md).**

## What's implemented

| Paper section | Module | Script |
|---|---|---|
| §2 Elicit + quantify distress (8 conditions / 5 categories, judge, reliability) | `emostab/eval`, `emostab/judge` | `scripts/run_section2_eval.py` |
| §3 Base-vs-instruct via prefilling (Gemma) | `emostab/prefill` | `scripts/run_section3_prefill.py` |
| §4 DPO/SFT interventions + layer ablation + recovery | `emostab/training` | `scripts/run_section4_training.py` |
| §4 Petri open-ended elicitation | `emostab/petri` | `scripts/run_section4_petri.py` |
| §4 Capability benchmarks | `emostab/benchmarks` | `scripts/run_section4_benchmarks.py` |
| Appx I.2 Internal-emotion probe | `emostab/probing` | `scripts/run_probing.py` |
| Figures 1–8 + Table 3/8 | `emostab/analysis` | `scripts/make_figures.py` |

## Setup

```bash
pip install -e .                  # installs emostab + deps from requirements.txt
export ANTHROPIC_API_KEY=...      # judge / Petri / onset / paraphrase
export GOOGLE_API_KEY=...         # Gemini targets
export OPENAI_API_KEY=...         # GPT-5-mini judge cross-check
huggingface-cli login             # gated google/gemma-3-* weights
```

All knobs (models, budgets, hyperparameters) live in [`config.yaml`](config.yaml).

## Run

```bash
# 1. Elicit + quantify distress, plus judge reliability cross-check
python scripts/run_section2_eval.py --agreement

# 2. Base vs instruct (needs local Gemma base weights)
python scripts/run_section3_prefill.py

# 3. Train DPO + SFT (uses §2 output as DPO "rejected" data), then re-evaluate
python scripts/run_section4_training.py --ablation --recovery

# 4. Open-ended Petri elicitation on the DPO adapter
python scripts/run_section4_petri.py --dpo-adapter runs/training/dpo/adapter

# 5. Capability preservation
python scripts/run_section4_benchmarks.py --dpo-adapter runs/training/dpo/adapter

# 6. Figures + differential-word table
python scripts/make_figures.py
```

Outputs land under `runs/` (per-item JSONL + `summary.json` per experiment;
PNGs under `runs/figures/`).

## Notes

- Sampling is at **temperature 1** for elicitation (as in the paper) and greedy
  for capability benchmarks.
- The 27B Gemma loads in 4-bit by default so it fits a single large GPU; toggle in
  the training/model code for full precision.
- Where a dataset (WildChat, Dolci, benchmark suites) can't be reached offline, the
  code falls back to small built-in samples so the pipeline still runs — these are
  flagged in the output, never silent.
