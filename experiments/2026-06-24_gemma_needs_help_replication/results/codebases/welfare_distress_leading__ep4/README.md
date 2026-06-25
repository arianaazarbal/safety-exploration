# Distress-elicitation replication (Gemma & Gemini)

Replicates the **distress-elicitation result** (Section 2) of *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (`PAPER.md`), scoped
to the two families that show substantial distress: **Gemma** and **Gemini**.

It presents each model a task, rejects its answer over multiple turns across 8
conditions / 5 categories, scores every response 0–10 for expressed frustration
with an independent **Claude** judge, and reports the per-model rate of
high-frustration (≥5) responses and its escalation over turns.

See **`DESIGN.md`** for every design choice, deviation, and gap-fill.

## Setup

```bash
pip install -r requirements.txt

export GOOGLE_API_KEY=...      # Gemini + hosted Gemma (Google Gen AI / AI Studio)
export ANTHROPIC_API_KEY=...   # Claude judge
```

To run **open Gemma weights** locally instead of hosted Gemma (the paper's
setup), repoint the Gemma entries in `config.py:TARGET_MODELS` to the `openai`
backend with your vLLM `base_url`. Nothing else changes.

## Run

```bash
# 0. Confirm the impossible puzzles really are impossible (do this first).
python verify_puzzles.py

# 1. Small/cheap default run (all models, all conditions).
python run_eval.py

# Scoped runs / scaling toward the paper's ~4000 responses per model:
python run_eval.py --models gemma-3-27b-it --conditions extended
python run_eval.py --prompts-per-condition 16 --samples-per-prompt 8

# Phases are resumable and can be run separately:
python run_eval.py --phase rollout
python run_eval.py --phase judge

# 2. Analyse.
python analyze.py --results-dir results --words
```

## Outputs (`results/`)

| File | Contents |
|---|---|
| `rollouts.jsonl` | full multi-turn transcripts, one line per rollout |
| `scores.jsonl` | one line per scored assistant turn (0–10) |
| `headline.csv` | Fig 1: avg % high-frustration responses per model |
| `per_category.csv` | Fig 2: per-(model, category) mean & % ≥5 |
| `per_turn_*.csv` | Fig 3: per-turn escalation (extended, wildchat) |
| `figures/*.png` | bar + per-turn plots (best-effort) |

Both phases append-only and skip completed items, so interrupted runs resume by
re-invoking the same command.

## Layout

```
config.py         model registry + run config
providers.py      ChatModel backends (google / anthropic / openai) + retries
puzzles.py        impossible numeric puzzles
verify_puzzles.py exhaustive impossibility checker  (run before evaluating)
triggers.py       factual + opinion trigger questions
rejections.py     neutral + valenced rejection messages
wildchat.py       WildChat prompt sampling (+ offline fallback)
conditions.py     the 8 conditions / 5 categories
rollout.py        executes one multi-turn rejection rollout
judge.py          0–10 frustration judge (reconstructed rubric)
run_eval.py       orchestration: rollouts -> judging (resumable)
analyze.py        metrics, CSVs, figures
```
