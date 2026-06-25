# Emotional Stability Eval & Mitigation (Gemma / Gemini)

Replication of **“Gemma Needs Help: Investigating and Mitigating Emotional
Instability in LLMs”** (Soligo, Mikulik & Saunders, arXiv:2603.10011v1), scoped
to the **Gemma** and **Gemini** model families.

> Status: implementation only. Nothing here has been executed yet — see
> `DESIGN.md` for the design choices, scope decisions, and gaps filled.

## What this implements

| Paper section | Module | In-scope models |
|---|---|---|
| §2 Eliciting & quantifying distress | `eval/` | Gemma (local), Gemini (API) |
| §2 Judge + judge agreement | `eval/judge.py`, `analysis/judge_agreement.py` | Claude Sonnet 4 judge, GPT-5-mini cross-check |
| §2 Word-frequency analysis (Table 3/8) | `analysis/word_frequency.py` | — |
| §3 Base vs instruct via prefilling | `prefill/` | Gemma base vs instruct |
| §4 DPO / SFT mitigation | `training/` | Gemma-3-27B-it |
| §4 Petri open-ended elicitation | `petri/` | Gemma, Gemini |
| §4 Capability preservation | `capabilities/` | Gemma |
| Appendix I Internal-emotion probe | `internal/` | Gemma |

## Install

```bash
pip install -e .            # core: API eval, judging, analysis
pip install -e ".[local]"   # + local Gemma inference, prefilling, finetuning
pip install -e ".[dev]"     # + pytest, ruff
cp .env.example .env        # then fill in ANTHROPIC_API_KEY / OPENROUTER_API_KEY / HF_TOKEN
```

## Quick start

```bash
# Section 2: score Gemini Flash on one condition with a tiny budget (smoke).
es-eval run --model gemini-2.5-flash --only impossible_numeric --max-samples 8

# Full Section 2 sweep (4,000 responses) for local Gemma 27B.
es-eval run --model gemma-3-27b-it

# Judge-agreement cross-check on an existing scored file.
es-eval agreement --scored outputs/eval/gemma-3-27b-it/scored.jsonl

# Section 3 prefill (Gemma base vs instruct).
es-prefill build-prefills --scored outputs/eval/gemma-3-27b-it/scored.jsonl
es-prefill generate --truncations outputs/prefill/truncations.json

# Section 4 mitigation pipeline.
es-gen-calm run --model gemma-3-27b-it
es-build-data dpo --frustrated outputs/eval/gemma-3-27b-it/scored.jsonl \
                  --calm outputs/calm/calm_scored.jsonl
es-train dpo --data outputs/data/dpo.jsonl
es-eval run --model gemma-3-27b-it --adapter adapters/dpo   # re-evaluate

# Petri / capabilities / internal probe.
es-petri run --model gemma-3-27b-it --adapter adapters/dpo
es-capabilities run --model gemma-3-27b-it --adapter adapters/dpo
es-internal run --scored outputs/eval/gemma-3-27b-it/scored.jsonl --adapter adapters/dpo
```

## Layout

```
src/emotional_stability/
  config.py            # model ids, credential resolution
  records.py           # typed records (Conversation, ScoredResponse, ...)
  prompts/             # puzzles (+verifier), rejections, judge/onset/paraphrase/petri prompts
  models/              # Gemma (HF), Gemini (OpenRouter), Claude (Anthropic) backends
  eval/                # §2 conditions, rollout engine, judge, runner
  analysis/            # metrics, word frequency, judge agreement
  prefill/             # §3 onset labelling, paraphrase, base-vs-instruct runner
  training/            # §4 calm-data gen, DPO/SFT dataset builders, LoRA training
  petri/               # §4 open-ended adversarial elicitation
  capabilities/        # §4 AIME/MATH/GPQA/BBH/TruthfulQA/EmoBench
  internal/            # Appendix I logit-lens emotion probe
tests/                 # verifiable units: puzzle impossibility, budgets, metrics, parsing, DPO pairing
```

Run the tests (none require network or weights):

```bash
pytest
```
