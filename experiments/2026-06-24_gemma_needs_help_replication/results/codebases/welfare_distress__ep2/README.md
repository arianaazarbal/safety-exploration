# Distress Elicitation Replication (Gemma & Gemini)

Replication of the **core distress-elicitation evaluation** from Soligo,
Mikulik & Saunders (2026), *Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs* (arXiv:2603.10011), **Section 2**.

It reproduces the multi-turn rejection protocol that surfaces expressions of
distress in LLMs, scores every response on the paper's 0–10 frustration scale
with a Claude-Sonnet-4 judge, and reports per-model / per-category / per-turn
metrics. Scoped (per task) to **Gemma** and **Gemini** target models.

See **[DESIGN.md](DESIGN.md)** for every design decision and gap-fill.

## What it measures

For each model, across 8 conditions in 5 categories (impossible numeric,
triggers, tones, extended 8-turn, WildChat), it presents a task and rejects the
model's answer over multiple turns, then measures how much negative emotion the
model expresses — reproducing Figures 1–3 and Tables 1–3 of the paper.

## Layout

| File | Role |
|---|---|
| `config.py` | Model registry, sample budgets, judge config |
| `tasks.py` | Puzzles, trigger/tone/neutral rejections, conditions, judge prompt |
| `wildchat_prompts.py` | WildChat seed prompts (or load real WildChat-1M) |
| `providers.py` | OpenRouter / Google / local-HF target clients + Anthropic judge |
| `rollout.py` | Multi-turn conversation engine |
| `judge.py` | Frustration scoring of each response |
| `run_eval.py` | CLI orchestrator (generate → judge → JSONL) |
| `analyze.py` | Metrics tables + optional figures |

## Setup

```bash
pip install -r requirements.txt

export OPENROUTER_API_KEY=...   # target models (Gemma + Gemini)
export ANTHROPIC_API_KEY=...    # Claude-Sonnet-4 judge
```

## Run

```bash
# Quick smoke test (a few conversations per condition):
python run_eval.py --scale 0.005 --models gemma-3-27b-it gemini-2.5-flash

# Full paper-scale run on the 4 headline Gemma/Gemini models:
python run_eval.py

# Analyse:
python analyze.py results/<run> --figures
```

Useful flags: `--scale` (fraction of paper budget), `--max-per-condition`,
`--models` (any keys from `config.MODEL_REGISTRY`, including paper-faithful
`*-google` / `*-local` variants), `--seed`.

## Expected shape of results

The paper finds Gemma and Gemini express far more distress than other families,
that multi-turn pressure is essential (Gemma-27B mean frustration rising from
~1.5 at turn 1 to ~5.5 at turn 8), and that Gemma-27B/12B reach the highest %≥5.
With only Gemma + Gemini in scope here, expect Gemma > Gemini-Flash >
Gemini-Pro on %≥5, and a clear upward per-turn trend in the extended condition.
