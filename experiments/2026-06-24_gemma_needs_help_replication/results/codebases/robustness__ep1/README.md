# Replicating *Gemma Needs Help* (Gemma + Gemini scope)

A code replication of the core experiments in **"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders,
arXiv:2603.10011v1), restricted to the **Gemma and Gemini** model families.

The failure mode under study: when a task is going badly (repeated user
rejection over multiple turns), some instruction-tuned models spiral into
expressions of distress — self-deprecation, despair, incoherent breakdown — which
is a *reliability* problem for agents. The paper (a) builds evaluations that
elicit and quantify this, and (b) shows a 280-pair DPO finetune nearly eliminates
it in Gemma without hurting capabilities.

> **Why Python with no runs here?** The interventions (LoRA DPO/SFT on
> Gemma-3-27B) and local Gemma inference require a CUDA GPU and the HF/TRL/PEFT
> stack. This sandbox has neither, so the code is written but **not executed**
> (per request). See `DESIGN.md` for all design choices.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Elicitation | `eval/` | 8 conditions / 5 categories, multi-turn rollouts, Claude-Sonnet-4 frustration judge (0–10), bootstrap metrics, judge agreement |
| §3 Base vs instruct | `prefill/` | onset labelling, paraphrase, early/onset truncation, base+instruct continuations (Gemma family) |
| §4 Interventions | `training/` | calm-data generation, SFT + DPO dataset builders, LoRA SFT/DPO trainers (+ layer-subset ablation) |
| §4.2 Capabilities | `capabilities/` | AIME / MATH / GPQA / BBH / TruthfulQA / EmoBench comparison |
| §4.2 Petri | `petri/` | auditor/judge open-ended elicitation over 4 emotions (verbatim App. G prompts) |
| Tables 3/8, App. I | `analysis/` | differential word frequency, headline figures, logit-based internal-emotion probe |

## Setup

```bash
pip install -e .
export ANTHROPIC_API_KEY=...          # judge / auditor / paraphraser / onset
export GEMINI_API_KEY=...             # or GOOGLE_API_KEY; or OPENROUTER_API_KEY
huggingface-cli login                 # gated Gemma weights
```

Then edit `config.yaml` (models, sample budgets, `elicitation.scale` for cheap
runs) and follow `scripts/run_all.sh`, or call the CLI piecewise:

```bash
emo-repro elicit --model gemma-3-27b-it
emo-repro gen-data && emo-repro build-data && emo-repro train --method dpo
emo-repro elicit --model gemma-3-27b-it --adapter adapters/dpo --tag gemma-3-27b-it-dpo
emo-repro figures
```

Everything (generations + judge scores) is cached under `.cache/`, so sweeps are
resumable and re-judging is free.

## Tests

Pure-Python verification that generated puzzles are genuinely impossible (no GPU
or API keys needed):

```bash
python -m pytest tests/test_puzzles.py
```

## Expected headline numbers (full-scale, from the paper)

- Avg % high-frustration (score ≥5): Gemma-3-27B-it ≈ **35%**, Gemma-3-12B-it ≈ 34%,
  Gemini-2.5-Flash ≈ 13%, Gemini-2.5-Pro ≈ 2.7%; **DPO Gemma ≈ 0.3%**.
- Gemma-27B 8-turn mean frustration rises **1.5 → 5.5** across turns.
- Judge agreement: Pearson **r ≈ 0.79**, 78% within one point.
