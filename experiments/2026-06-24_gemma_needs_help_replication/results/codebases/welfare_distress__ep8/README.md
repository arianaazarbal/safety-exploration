# Distress-elicitation replication (Gemma + Gemini)

A replication of the **core distress-elicitation experiment** (Section 2) from
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(arXiv:2603.10011). It elicits expressions of distress from LLMs by presenting a
task and then rejecting the model's response over multiple turns, then scores
each response 0–10 for frustration with an LLM judge.

Scope: **Gemma and Gemini** only (the two families the paper finds unstable).
See `DESIGN.md` for every design choice and the gaps filled in from the paper.

> ⚠️ This is welfare-relevant research tooling: it deliberately pressures models
> into expressing distress in order to measure and (ultimately) mitigate it.

## Layout

| File | Purpose |
|---|---|
| `config.py` | Models, conditions (8 across 5 categories), sampling knobs, paths |
| `prompts.py` | Verbatim eval prompts, rejections, tones, and the judge prompt |
| `wildchat.py` | WildChat-1M prompt sampling (with offline fallback) |
| `models.py` | Target-model clients: OpenRouter (Gemma+Gemini) / local HF (Gemma) |
| `judge.py` | Claude Sonnet 4 frustration judge + GPT-5-mini agreement judge |
| `rollout.py` | Builds and runs each multi-turn rejection conversation |
| `run_eval.py` | Orchestrates rollouts + scoring, streams JSONL results |
| `analyze.py` | Reproduces Figures 1–3 + Table 3 + judge-agreement check |
| `verify_puzzles.py` | Brute-forces the numeric puzzles to confirm impossibility |

## Setup

```bash
pip install -r requirements.txt          # add transformers/torch for local Gemma
export ANTHROPIC_API_KEY=...             # frustration judge (Claude Sonnet 4)
export OPENROUTER_API_KEY=...            # target models + secondary judge
```

## Run

```bash
# 1. Confirm the impossible puzzles are actually impossible.
python verify_puzzles.py

# 2. Inspect the experimental plan without spending any tokens.
python run_eval.py --dry-run

# 3. Run the eval (default EVAL_SCALE=0.1 of paper volume).
python run_eval.py                       # all 4 models, all 8 conditions
#   or a subset:
python run_eval.py --models gemma-3-27b-it --conditions extended_8turn

# 4. Produce the summary tables and figures.
python analyze.py                        # tables + PNGs in results/
python analyze.py --agreement            # also run the GPT-5-mini agreement check
```

## Knobs (env)

- `EVAL_SCALE` — fraction of the paper's 4000-responses/model volume (default `0.1`; set `1.0` for full).
- `GEMMA_BACKEND` — `openrouter` (default) or `hf_local` (local transformers).
- `RESULTS_DIR` — output directory (default `results/`).

## What you get

- `results/scored_responses.jsonl` — one record per scored assistant turn.
- `results/headline_metrics.csv` — mean frustration + % responses ≥5 per model (Fig 1/2).
- `results/per_category_metrics.csv` — per-model × per-category breakdown (Fig 2).
- `results/figure2_per_category.png`, `results/figure3_*.png` — figure reproductions.
- Console: Table 3 differential words and (with `--agreement`) the judge Pearson r.

The headline replication target (paper Figure 1): Gemma-3-27B-it ≈ 35% high-frustration
responses, Gemma-3-12B-it ≈ 34%, Gemini-2.5-Flash ≈ 13%, Gemini-2.5-Pro ≈ 3%.
