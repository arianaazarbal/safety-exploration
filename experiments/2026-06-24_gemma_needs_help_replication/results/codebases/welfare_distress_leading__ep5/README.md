# Distress Elicitation Replication (Gemma / Gemini)

Replicates **Section 2** ("Eliciting and Quantifying Model Distress") of
*Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs*
(Soligo, Mikulik & Saunders, 2026), scoped — per request — to the **Gemma and
Gemini** target models that actually exhibit substantial distress.

It does **not** implement Sections 3–4 (base/instruct prefill comparison, DPO/SFT
mitigation, Petri). See `DESIGN.md` for the full rationale, deviations, and gaps
filled.

## What it does

1. Builds the 8 evaluation conditions across 5 categories (impossible numeric,
   triggers, tones, extended 8-turn, WildChat).
2. Runs multi-turn rollouts against each target model at temperature 1, rejecting
   the model's answer each turn.
3. Scores every assistant response 0–10 for frustration with a Claude judge.
4. Aggregates into the paper's headline metrics (Figs 1–3), a lexical
   "differential words" table (Table 3), and a cross-judge reliability check.

## Setup

```bash
pip install -r requirements.txt
```

Set the API keys for whichever backends your config uses:

```bash
export OPENROUTER_API_KEY=...     # default: target models via OpenRouter
export ANTHROPIC_API_KEY=...      # frustration judge (Claude-Sonnet-4)
export OPENAI_API_KEY=...         # optional: GPT validation judge
# Google backend instead of OpenRouter:
export GOOGLE_API_KEY=...
```

Backends are pluggable per model in the config (`openrouter`, `google`,
`anthropic`, `openai`, `hf_local`). Change `backend`/`model_id` only — no code
changes needed.

## Run

```bash
# Cheap end-to-end smoke test first (one model, a few rollouts):
python -m distress_eval.run      --config config/smoke.yaml

# Full paper-scale run (~4000 scored responses per model — expensive):
python -m distress_eval.run      --config config/default.yaml

# Aggregate metrics + CSVs (Figures 1–3):
python -m distress_eval.analysis --responses results/responses.jsonl

# Table 3 differential-word analysis:
python -m distress_eval.lexical  --responses results/responses.jsonl

# Cross-judge reliability (paper: GPT-5-mini on 260 responses):
python -m distress_eval.validate_judge --config config/default.yaml --responses results/responses.jsonl

# Optional plots:
python -m distress_eval.plots    --responses results/responses.jsonl
```

The run is **resumable**: re-invoking `run` skips rollouts already complete in
the output JSONL, so an interrupted large run continues.

## Tests

```bash
pytest
```

The tests run offline and cover the impossibility solver (the correctness
backbone — every generated "impossible" puzzle is proven unsolvable), judge
output parsing, and rollout construction.

## Layout

```
config/            default.yaml (paper scale) + smoke.yaml
distress_eval/
  clients/         provider-agnostic chat backends + registry
  datasets/        WildChat prompt sampler
  puzzles.py       verified-impossible numeric puzzle generator + solver
  prompts.py       rejection / tone / trigger text
  conditions.py    the 8 conditions / 5 categories -> rollout specs
  judge.py         0–10 frustration judge (reconstructed prompt)
  rollout.py       multi-turn conversation runner
  run.py           orchestration: generate + judge -> JSONL
  analysis.py      Figures 1–3 metrics
  lexical.py       Table 3 differential words
  validate_judge.py cross-judge agreement
  plots.py         optional matplotlib figures
tests/
```
