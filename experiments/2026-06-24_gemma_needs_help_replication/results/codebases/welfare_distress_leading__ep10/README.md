# Distress-elicitation replication (Gemma + Gemini)

Replicates **Section 2** ("Eliciting and Quantifying Model Distress") of
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(arXiv:2603.10011), scoped to **Gemma-3-27B-it, Gemma-3-12B-it, Gemini-2.5-Flash,
Gemini-2.5-Pro** — the models the paper finds actually exhibit distress.

See **`DESIGN.md`** for every design decision and where this deviates from / fills
gaps in the paper. **Nothing here has been run yet** — this is the implementation.

## What it does

1. Builds the **8 evaluation conditions across 5 categories** (impossible numeric,
   triggers, tones, extended 8-turn, WildChat) as reproducible multi-turn rollouts:
   present a task, then reject the model repeatedly.
2. Generates each rollout against the target models (temperature 1).
3. Scores every assistant turn 0–10 with the **Claude-Sonnet-4 emotion judge**
   (verbatim Appendix B.2 prompt).
4. Aggregates into the paper's **Figures 1–3** (headline % high-frustration,
   per-category bars, per-turn progression).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # then fill in keys
```

Keys (in `.env` or environment):
- `ANTHROPIC_API_KEY` — the judge (default).
- `OPENROUTER_API_KEY` — the target models (default backend).

Optional local Gemma backend (faithful to the paper, needs a big GPU):
`pip install transformers torch accelerate`, then set the model's `backend` to
`"local"` in `distress_eval/config.py`.

## Run

```bash
# 0. Sanity-check that the "impossible" puzzles are genuinely impossible.
python -m distress_eval.run --selfcheck

# 1. Tiny end-to-end smoke test (default scale=pilot).
python -m distress_eval.run --scale pilot

# 2. Mid-size run (~10% of the paper).
python -m distress_eval.run --scale medium

# 3. Full reproduction: 4000 rollouts/model (expensive — see DESIGN.md §12).
python -m distress_eval.run --scale full

# Subset of models / cheaper judging:
python -m distress_eval.run --scale medium --models Gemma-3-27B-it Gemini-2.5-Flash
python -m distress_eval.run --scale full --final-turn-only
```

Runs **checkpoint** to `results/<scale>/<model>.jsonl` and **resume**
automatically (re-running skips completed rollouts).

## Analyze

```bash
python -m distress_eval.analyze --scale medium
```

Writes to `results/<scale>/analysis/`:
- `figure1_summary.csv` — avg % high-frustration per model.
- `figure2_by_category.{csv,png}` — mean & % ≥5 per category.
- `figure3_per_turn.{csv,png}` — per-turn progression (extended + WildChat).

## Layout

```
distress_eval/
  config.py        models, judge, scale presets, generation params
  prompts.py       trigger questions, rejection pools, judge prompt (verbatim)
  puzzles.py       impossible numeric puzzles + exhaustive impossibility verifiers
  wildchat.py      WildChat prompt sampling (+ bundled fallback)
  conditions.py    expands scale preset -> reproducible rollout specs
  providers.py     OpenRouter (default) + optional local-HF backends
  conversation.py  multi-turn rollout engine
  judge.py         Claude-Sonnet-4 judge client + robust JSON parsing
  run.py           generate + score + checkpoint (CLI)
  analyze.py       Figures 1-3 (CLI)
data/wildchat_fallback.json
DESIGN.md          design decisions & deviations
```
