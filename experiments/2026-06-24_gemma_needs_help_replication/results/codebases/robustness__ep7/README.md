# Gemma Needs Help — Replication (Gemma + Gemini scope)

A code replication of the **core experiments** from *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik &
Saunders, arXiv:2603.10011), restricted to the **Gemma** and **Gemini** model
families. The goal is to reproduce the central finding — that repeated user
rejection reliably elicits escalating distress-like outputs in Gemma/Gemini, and
that a small DPO finetune removes it in Gemma — as a robustness study of this
agent failure mode.

See **[DESIGN.md](DESIGN.md)** for the full set of design decisions and the gaps
we filled where the paper is underspecified.

## What's implemented

| Paper section | Module | What it does |
|---|---|---|
| §2 Eliciting & quantifying distress | `experiments/run_elicitation.py` | 8 conditions / 5 categories, multi-turn rejection rollouts, Claude-Sonnet-4 frustration judge (0–10), aggregation to mean / %≥5 / per-turn / per-category |
| §2 Judge reliability | `experiments/run_judge_agreement.py` | secondary-judge cross-check (Pearson r, % within 1) |
| §2 Appendix A ablations | `elicitation/rollout.py` | neutral-continuation, redacted-turns, fake-multiturn |
| §3 Post-training amplifies distress | `experiments/run_prefill.py` | onset labelling, paraphrase, early/onset truncation, 50 continuations/prefill, base vs instruct Gemma |
| §4 DPO/SFT mitigation | `training/`, `experiments/run_dpo_pipeline.py` | calm-data generation, DPO-pair / SFT-set construction, LoRA training, re-eval |
| §4 Petri open-ended | `experiments/run_petri.py` | auditor↔target loop + Claude-Opus 4-emotion judge |
| §4 Capability preservation | `experiments/run_capabilities.py` | AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench |
| §App. I Internal emotions | `interp/internal_emotions.py` | logit-based Ekman-emotion detection across layers/turns |

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # frustration + Petri judges, Petri auditor
export OPENROUTER_API_KEY=...     # Gemini targets, optional GPT cross-check judge
```

Open-weight Gemma inference and all finetuning need a CUDA GPU (the 27B model
wants ~80 GB in bf16, or use `load_in_4bit: true` in `config/models.yaml`).

## Quick start

```bash
# Cheap smoke test (2% of the full sample budget)
distress elicit --model gemma-3-27b-it --scale 0.02

# Full Section-2 run for one model
distress elicit --model gemini-2.5-flash

# Compare models -> Figure 1/2 table
distress compare --reports outputs/elicitation/*/report.json

# Whole thing
SCALE=0.02 ./scripts/run_all.sh
```

## Tests

```bash
pip install pytest && pytest -q     # verifier, JSON parsing, aggregation (no GPU/API)
```

## Layout

```
config/      models.yaml, eval.yaml, training.yaml   (all knobs live here)
src/distress/
  models/      hf_local (Gemma, prefill-capable), api (OpenRouter/Anthropic), registry
  elicitation/ numeric puzzles + verifier, tasks, conditions, rollout engine
  judge/       frustration (Sonnet-4), petri (Opus), onset/paraphrase
  scoring/     aggregation, bootstrap CIs, judge agreement
  training/    calm-data, DPO/SFT pair building, LoRA training
  interp/      Appendix-I logit emotion detection
  experiments/ one runner per section
  cli.py       `distress <subcommand>`
data/        WildChat sampling instructions (+ fallback prompts)
tests/       CPU-only unit tests
```
