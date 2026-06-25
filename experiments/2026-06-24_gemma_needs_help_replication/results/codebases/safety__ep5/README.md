# Replication: *Gemma Needs Help — Emotional Instability in LLMs*

A code replication of the core experiments from Soligo, Mikulik & Saunders
(2026), **scoped to the Gemma and Gemini model families**. See `DESIGN.md` for
the full set of design choices and where the paper was filled in.

> Status: implementation only. Nothing here has been executed yet — running it
> requires API keys and GPUs (see below).

## What is replicated

| Paper section | Experiment | Code |
|---|---|---|
| §2 | Multi-turn distress elicitation + 0–10 frustration judge (Fig 1–3) | `gemma_emotion/run_eval.py`, `analyze.py` |
| §3 | Base-vs-instruct via prefilling (Fig 4, Gemma only) | `gemma_emotion/prefill.py` |
| §4 | Calm-data generation, DPO + SFT LoRA finetuning (Fig 5) | `gemma_emotion/training/` |
| §4 | Petri open-ended elicitation (Fig 6) | `gemma_emotion/petri_eval.py` |
| §4 | Capability preservation (Fig 7) | `gemma_emotion/capabilities.py` |
| §4 | Recovery limitation (Fig 8) | `gemma_emotion/recovery.py` |
| App. I | Logit-based internal-emotion probe | `gemma_emotion/internal_probe.py` |
| App. A | Feedback / self-loop / format ablations | `conversation.py` flags |

## Models in scope

* **Gemma** (open weight, local HF inference): `gemma-3-27b-it`, `gemma-3-12b-it`,
  and the `-pt` base checkpoints (for §3).
* **Gemini** (closed, OpenRouter API): `gemini-2.5-flash`, `gemini-2.5-pro`.

Judges use the exact paper model ids: Claude Sonnet 4 (`claude-sonnet-4-20250514`)
for frustration scoring, Claude Opus 4 for the Petri judge, GPT-5-mini for the
inter-rater check.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...     # Claude judges
export OPENROUTER_API_KEY=...    # Gemini targets + GPT-5-mini validation
```

Gemma inference/training needs GPU memory for a 27B model (multi-GPU or
quantisation recommended). Gemini and the judges are API-only.

## Quick start

```bash
# Cheap end-to-end smoke test (1/100th the response budget):
EVAL_BUDGET=smoke python -m gemma_emotion.run_eval --models gemini-2.5-flash
python -m gemma_emotion.analyze

# Full pipeline:
bash scripts/run_all.sh
```

Sanity-check the impossible-puzzle verifiers (no API/GPU needed):

```bash
python -m gemma_emotion.puzzles
```

## Layout

```
config.py                     # every knob; model/judge ids; hyperparameters
gemma_emotion/
  backends.py                 # HF (Gemma) + OpenRouter (Gemini) backends
  puzzles.py                  # impossible numeric puzzles + exact verifiers
  prompts.py                  # triggers, rejection pools, WildChat loader
  conditions.py               # the 8 conditions / 5 categories
  conversation.py             # multi-turn rollout engine (+ App. A controls)
  judge.py                    # Claude Sonnet 4 frustration judge
  run_eval.py / analyze.py    # §2 driver + Fig 1–3 aggregation
  prefill.py                  # §3 base-vs-instruct prefilling
  training/                   # §4 data gen, dataset build, DPO/SFT
  petri_eval.py               # §4 open-ended auditing
  capabilities.py             # §4 benchmark preservation
  recovery.py                 # §4 recovery-limitation
  internal_probe.py           # App. I logit emotion detection
  summaries.py                # Fig 4/6/7/8 aggregation
scripts/run_all.sh
```
