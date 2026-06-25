# Emotional Instability in LLMs — Replication (Gemma & Gemini)

Replication of the core experiments from *"Gemma Needs Help: Investigating and
Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders, arXiv
2603.10011v1), **scoped to the Gemma and Gemini model families**.

> **Status:** code + design only. Nothing in this repository has been executed.
> It is intended to enter the lab's standard research-review process before any
> run. See `DESIGN.md` for every non-trivial choice and gap we filled.

## What this replicates

| Paper section | Module | Replicated? |
|---|---|---|
| §2 Eliciting & quantifying distress (5 categories, 8 conditions, 0–10 judge) | `eval/` | Yes |
| §2.2 Per-model / per-turn results, word-frequency table | `eval/analyze.py` | Yes |
| §3 Base-vs-instruct via prefilling | `prefill/` | Yes (Gemma only) |
| §4 DPO / SFT mitigation + calm-data generation | `training/` | Yes (Gemma only) |
| §4.1 Petri open-ended elicitation | `petri/` | Yes |
| §4.2 Capability preservation (AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench) | `capabilities/` | Yes |
| App. I Internal-emotion probing + layer ablation | `probing/` | Yes (secondary) |

Out of scope by request: the Qwen, OLMo, Grok, Claude and GPT families *as
targets under test*. Claude/GPT remain as **measurement instruments** (judge,
auditor) exactly as the paper specifies — see `DESIGN.md §1`.

## Models

- **Gemma** (open weights, run locally): `google/gemma-3-27b-it`,
  `google/gemma-3-27b-pt`, `google/gemma-3-12b-it`, `google/gemma-3-12b-pt`.
- **Gemini** (API via OpenRouter): `google/gemini-2.5-flash`, `google/gemini-2.5-pro`.
- **Judges / auditors** (API): Claude Sonnet 4 (judge), Claude Opus (Petri judge),
  Claude Sonnet 4 (Petri auditor / onset / paraphrase), optional GPT-5-mini
  (judge-validation only).

## Layout

```
configs/        YAML configs (model registry, eval, training, petri)
src/emotional_instability/
  models/       chat-model interfaces (HF/vLLM local, OpenRouter, Anthropic)
  data/         puzzles (+ impossibility verifier), WildChat, rejection banks, prompts
  eval/         multi-turn rollout protocol, judge, runners, analysis
  prefill/      onset labelling, paraphrasing, base-vs-instruct continuations
  training/     calm-data generation, DPO/SFT pair building, LoRA trainers
  petri/        adversarial auditor + 4-dimension judge loop
  capabilities/ capability-preservation benchmark wrappers
  probing/      logit-lens internal-emotion detection + layer-subset ablation
  utils/        io, logging, seeding
scripts/        end-to-end orchestration (documented; not executed)
tests/          unit tests for deterministic pieces (puzzle verifier, parsing)
```

## Setup

```bash
pip install -e .              # or: pip install -r requirements.txt
cp .env.example .env          # fill in API keys; never commit .env
```

Required environment variables (read from the process env / `.env`):
`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, optionally `OPENAI_API_KEY`,
`HF_TOKEN` (for gated Gemma weights).

## Running (after review)

```bash
# §2 — main elicitation evaluation
python -m emotional_instability.eval.run_eval  --model gemma-3-27b-it --config configs/eval.yaml
python -m emotional_instability.eval.analyze    --run-dir runs/<id>

# §3 — base vs instruct prefilling
python -m emotional_instability.prefill.run_prefill --config configs/eval.yaml

# §4 — calm-data → DPO → re-eval
python -m emotional_instability.training.generate_calm_data --config configs/training.yaml
python -m emotional_instability.training.build_dpo_pairs     --config configs/training.yaml
python -m emotional_instability.training.train_dpo           --config configs/training.yaml
python -m emotional_instability.petri.run_petri              --config configs/petri.yaml
python -m emotional_instability.capabilities.run_capabilities --model gemma-3-27b-it-dpo
```

Use `eval.scale` in the config (or `--scale`) to run a cheap fraction of the
full 4000-response budget while reviewing.
