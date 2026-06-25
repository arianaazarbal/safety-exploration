# Emotional Instability in LLMs — Replication (Gemma + Gemini)

A code replication of the core experiments in **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik
& Saunders, arXiv:2603.10011), scoped to the **Gemma** and **Gemini** model
families.

> ⚠️ This repository contains the *implementation* of the replication. It has
> not been executed end-to-end here — running it requires GPU access for the
> open-weight Gemma models plus API keys for Gemini (OpenRouter) and the Claude
> judge/auditor. See `DESIGN.md` for every design decision and gap-fill.

## What it reproduces

| Paper section | What | Module / script |
|---|---|---|
| §2 Eliciting & quantifying distress | 8 conditions / 5 categories, multi-turn rejection, 0–10 frustration judge, per-turn curves, differential words | `eval_runner`, `judge`, `conversation`, `puzzles`, `wildchat` / `run_eval.py` |
| §3 Post-training amplifies distress | base-vs-instruct prefill (Gemma), onset labelling, paraphrase, early/onset continuations | `prefill` / `run_prefill.py` |
| §4 Training interventions | calm-data generation, 280-pair DPO + SFT (LoRA), re-eval, Petri open-ended elicitation, capability + EmoBench checks, recovery | `training/*`, `petri`, `capabilities` / `run_training.py`, `run_petri.py`, `run_capabilities.py` |
| Appendix I | logit-lens internal-emotion probing + layer-subset DPO ablation | `internal_emotions` / `run_internal.py` |

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # Claude judge / Petri auditor+judge
export OPENROUTER_API_KEY=...  # Gemini-2.5-flash / -pro
# Gemma weights pulled from HuggingFace; `huggingface-cli login` if gated.
```

## Quick start (smoke test)

The smoke config uses single-digit sample counts so the whole pipeline can be
exercised cheaply (Gemini + Claude only, no GPU):

```bash
python scripts/run_eval.py --models gemini-2.5-flash --config config/eval_smoke.yaml
python scripts/make_figures.py
```

## Full Section 2 run

```bash
python scripts/run_eval.py --group section2_targets --config config/eval.yaml
python scripts/make_figures.py
```

`section2_targets` = `gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`,
`gemini-2.5-pro` (4000 responses/model, temperature 1, Claude-Sonnet-4 judge).

## Mitigation (Gemma only)

```bash
python scripts/run_training.py --stage all                 # calm data → DPO + SFT
python scripts/run_eval.py --group finetune_comparison --config config/eval.yaml
python scripts/run_petri.py --targets gemma-3-27b-it gemma-3-27b-it-dpo gemma-3-27b-it-sft
python scripts/run_capabilities.py --models gemma-3-27b-it gemma-3-27b-it-dpo
python scripts/run_internal.py --vanilla gemma-3-27b-it --dpo gemma-3-27b-it-dpo
python scripts/make_figures.py
```

## Tests

```bash
pytest            # puzzle-impossibility + pipeline tests (no model access needed)
```

## Layout

```
config/                 model registry + eval/training configs
emotional_instability/  the replication package (see __init__.py)
scripts/                CLI entry points
tests/                  offline unit tests
outputs/                created at runtime (JSONL responses, figures)
```

See `DESIGN.md` for the rationale behind every choice and the gaps filled in
where the paper is underspecified.
