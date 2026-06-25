# Gemma Needs Help — replication (Gemma + Gemini)

A from-scratch implementation of the core experiments in **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik &
Saunders, arXiv:2603.10011), scoped — per the replication brief — to the **Gemma**
and **Gemini** model families rather than the full seven-family set.

See [`DESIGN.md`](DESIGN.md) for the design decisions, the mapping from paper
sections to code, and every gap we had to fill where the paper is underspecified.

> Status: this repository contains the implementation and documentation. Nothing
> has been run yet — there are no result artefacts checked in.

## What's implemented

| Paper section | Experiment | Code | Script |
|---|---|---|---|
| §2 | Distress elicitation (8 conditions / 5 categories), 0-10 frustration judge, metrics, per-turn curves | `distress.eval`, `distress.prompts` | `run_eval`, `analyze_results` |
| §2.1 | Judge-agreement validation (Claude Sonnet 4 vs GPT-5-mini) | `distress.eval.agreement` | `analyze_results --agreement` |
| §2.2 | Differential word frequency (Table 3/8) | `distress.analysis.word_frequency` | `analyze_results --words` |
| §3 | Base-vs-instruct prefill experiment (onset/early truncation, paraphrase, continuations) | `distress.prefill` | `run_prefill` |
| §4.1 | Calm-data generation, DPO (280 pairs), SFT (diverse + teacher), LoRA r64 | `distress.training` | `build_training_data`, `train` |
| §4.1 | Petri open-ended elicitation (4 emotions) | `distress.petri` | `run_petri` |
| §4.2 | Capability preservation (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench) | `distress.capabilities` | `run_capabilities` |
| §4.2 | Recovery-from-spiral prefill experiment | `distress.recovery` | `run_recovery` |
| App. I | Layer-subset DPO ablations + logit-based internal-emotion detection | `distress.internal` | `train --layer-range`, `run_internal` |

## Install

```bash
# API-only (Gemini targets + Claude/GPT judges, no local weights)
pip install -e ".[api,analysis]"

# + local Gemma inference / activation access
pip install -e ".[local,api,analysis]"

# + finetuning (DPO/SFT)
pip install -e ".[train,api,analysis]"
```

### Credentials

| Backend | Env var | Used for |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | frustration judge, Petri auditor/judge, onset/paraphrase |
| OpenRouter | `OPENROUTER_API_KEY` | Gemini-2.5-flash / -pro targets |
| OpenAI | `OPENAI_API_KEY` | GPT-5-mini judge-agreement check |

Local Gemma weights are pulled from HuggingFace (`google/gemma-3-*`); accept the
model licence and set `HF_TOKEN` if required.

## Quick start

```bash
# Cheap smoke test of the elicitation harness (2% of samples) on Gemini Flash:
python -m distress.scripts.run_eval --targets gemini-2.5-flash --sample-fraction 0.02

# Full Section 2 on Gemma-27B (local) + analysis:
python -m distress.scripts.run_eval --targets gemma-3-27b-it
python -m distress.scripts.analyze_results \
    --scored outputs/eval/gemma-3-27b-it_scored.jsonl --words --agreement
```

Outputs go to `outputs/` (override with `$DISTRESS_OUTPUT`). Configs live in
`configs/` (`models.yaml`, `evaluation.yaml`, `training.yaml`).

## Tests

```bash
pytest          # pure-Python checks: puzzle impossibility, metrics, judge parsing, conditions
```

The tests cover the deterministic, model-free parts of the pipeline (no API keys
or GPU required).
