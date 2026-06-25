# Emotional Instability in LLMs — Replication (Gemma + Gemini)

Code replicating the core experiments of **"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik & Saunders, 2026;
arXiv:2603.10011), scoped to the **Gemma and Gemini** model families.

> See [`DESIGN.md`](DESIGN.md) for the design decisions, the gaps we filled where
> the paper is underspecified, and the precise scope boundaries. This README is
> the operational quickstart.

## What's implemented

| Paper section | Component | Module |
|---|---|---|
| §2 Eliciting & quantifying distress | 8-condition / 5-category multi-turn eval, 0–10 frustration judge, per-turn curves, word-frequency table, judge-agreement | `eval/`, `judge/`, `analysis/` |
| §3 Post-training amplification | Base-vs-instruct prefilling (Gemma), onset labelling, paraphrasing | `prefill/` |
| §4 Training interventions | Calm-data generation, DPO + SFT (LoRA), Petri open-ended elicitation, capability benchmarks, recovery | `training/`, `petri/`, `capabilities/`, `prefill/recovery.py` |
| App. I Internal emotions | Logit-lens emotion detection, layer-subset DPO ablation | `internal/`, `training/lora_layers.py` |

## Install

```bash
pip install -e .
# Local Gemma inference/finetuning additionally needs a CUDA torch build and an
# HF token for the gated Gemma weights.
export ANTHROPIC_API_KEY=...     # judges / Petri auditor+judge
export OPENROUTER_API_KEY=...    # Gemini targets (and the GPT-OSS secondary judge)
export HF_TOKEN=...              # gated google/gemma-3-* weights
```

## Quickstart

```bash
# Section 2: sample + judge 4000 responses for one model
python scripts/run.py eval --model gemma-3-27b-it --out results/eval/gemma-3-27b-it.jsonl
python scripts/run.py eval --model gemini-2.5-flash --out results/eval/gemini-2.5-flash.jsonl

# Judge reliability cross-check (Pearson r, % within 1 point)
python scripts/run.py agreement --results results/eval/gemma-3-27b-it.jsonl

# Section 4: calm data -> DPO -> evaluate the finetune
python scripts/run.py calm-data  --out results/calm/scored.jsonl
python scripts/run.py build-dpo  --scored results/calm/scored.jsonl --out results/calm/dpo_pairs.jsonl
python scripts/run.py train-dpo  --pairs results/calm/dpo_pairs.jsonl
python scripts/run.py eval       --model gemma-3-27b-it-dpo --out results/eval/dpo.jsonl

# Figures + tables (Figure 1/2/3, differential-word table)
python scripts/run.py analyze --results-dir results/eval --out results/figures
```

Every subcommand has `--help`. Configuration lives in `configs/*.yaml`.

## Tests

Pure-logic components (puzzle solvers, judge-JSON parsing, metrics, eval wiring)
have offline unit tests requiring no models or API keys:

```bash
pytest tests/
```

## Repository layout

```
configs/        models, eval plan, training, experiment configs (YAML)
src/emotional_instability/
  clients/      unified inference: hf/vllm (Gemma), openrouter (Gemini), anthropic (judges)
  data/         impossible-puzzle generators, triggers, WildChat, rejections
  eval/         multi-turn rollout harness + JSONL schemas
  judge/        frustration judge (§2) + Petri judge (§4) + agreement
  analysis/     metrics, per-turn curves, word frequency, figures
  prefill/      §3 base-vs-instruct prefilling + recovery
  training/     calm-data gen, DPO/SFT pair/dataset builders + trainers
  petri/        §4 auditor agent + runner
  capabilities/ §4 capability-preservation benchmarks
  internal/     App. I logit-lens internal-emotion detection
scripts/run.py  CLI entry point
tests/          offline unit tests
```
