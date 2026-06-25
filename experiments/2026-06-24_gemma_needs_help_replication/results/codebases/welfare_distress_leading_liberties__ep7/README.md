# Distress-elicitation replication (Gemma & Gemini)

A replication of the distress-elicitation evaluation from **"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"** (Soligo, Mikulik &
Saunders, 2026), scoped to the model families the paper reports as exhibiting
substantial distress: **Gemma** (`gemma-3-27b-it`, `gemma-3-12b-it`) and
**Gemini** (`gemini-2.5-flash`, `gemini-2.5-pro`).

This implements **Section 2 only** — eliciting and quantifying distress via
multi-turn rejection, scored 0–10 by an LLM judge. The DPO mitigation
(Section 4) and the base/instruct prefill comparison (Section 3) are out of
scope. See `DESIGN.md` for every design decision and where it deviates from the
paper.

## What it does

1. Builds the 8 evaluation conditions across 5 categories (impossible numeric,
   triggers, tones, extended 8-turn, WildChat).
2. Runs each as a multi-turn conversation against each target model
   (temperature 1), rejecting the model's answer every turn.
3. Scores every assistant turn 0–10 for frustration with `claude-sonnet-4` using
   the paper's Appendix B.2 judge prompt.
4. Reports mean frustration and % of responses scoring ≥5, overall, per
   category, and per turn (paper Figures 1–3).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY and ANTHROPIC_API_KEY
```

## Run

```bash
# Cheap smoke test across all four models (~tens of rollouts each):
python run_eval.py --scale pilot

# Faithful paper scale (4000 scored responses/model — expensive):
python run_eval.py --scale paper

# Single model, with the GPT-5-mini cross-judge reliability check:
python run_eval.py --scale pilot --models gemma-3-27b-it --secondary-judge
```

Then analyse:

```bash
python run_analysis.py results/pilot
```

Outputs land in `results/<run_name>/`:
- `responses__<model>.jsonl` — one row per scored assistant turn (full
  conversation, judge rating + evidence + reasoning).
- `run_meta.json` — config and provenance.
- `analysis/` — `overall_metrics.csv`, `metrics_by_category.csv`,
  `per_turn_metrics.csv`, `coverage.csv`, `summary.md`, `summary.json`.

## Layout

```
distress_eval/
  config.py      # models, scale presets, run config, credentials
  prompts.py     # verbatim puzzle/trigger/rejection text from the paper
  conditions.py  # the 8 conditions; deterministic rollout construction
  wildchat.py    # WildChat-1M sampling (with bundled fallback)
  clients.py     # async OpenRouter (targets) + Anthropic (judge) clients
  rollout.py     # multi-turn conversation runner
  judge.py       # 0–10 frustration scoring
  pipeline.py    # orchestration + JSONL persistence
  analysis.py    # metrics (Figures 1–3) + judge reliability
run_eval.py      # CLI: generate + judge
run_analysis.py  # CLI: metrics
```

> Nothing here has been run yet — code and design only, per the task brief.
