# Distress-elicitation replication — Gemma & Gemini

Replicates **Section 2** of *"Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs"* (PAPER.md): eliciting and quantifying model distress,
scoped to the models that actually show it — **Gemma-3-27B-it, Gemma-3-12B-it,
Gemini-2.5-Flash, Gemini-2.5-Pro**.

See **DESIGN.md** for the methodology, every design choice, and where this deviates
from the paper.

## Install

```bash
pip install -r requirements.txt
```

## Configure (environment variables)

By default everything runs through **OpenRouter** (targets) + **Anthropic** (judge):

```bash
export OPENROUTER_API_KEY=sk-or-...      # the 4 target models
export ANTHROPIC_API_KEY=sk-ant-...      # Claude-Sonnet-4 frustration judge
# optional, for the judge-agreement check (uses gpt-5-mini via OpenRouter):
#   reuses OPENROUTER_API_KEY
```

Alternative backends (native Gemini API, or local Gemma via vLLM) are shown as
commented blocks in `config.yaml`.

## Run

```bash
# 1) Smoke test the whole pipeline cheaply (2 conversations per condition per model):
python -m src.run_eval --limit 2

# 2) Inspect the verified-impossible puzzle pool:
python -m src.puzzles

# 3) Full run (≈4000 responses/model; resumable — safe to re-run if interrupted):
python -m src.run_eval

# 4) Aggregate + reproduce Fig 1 / 2 / 3:
python -m src.analyze

# 5) (Optional) judge reliability vs a second judge (paper: r≈0.79, 78% within 1):
python -m src.validate_judge --n 260
```

Useful flags: `--models gemma-3-12b-it` and `--conditions impossible_numeric` to run
a subset.

## Output (in `results/`)

- `responses.jsonl` — one row per scored assistant turn (response text, frustration
  score, judge evidence/reasoning, condition, turn index, …).
- `summary_by_model.csv` — avg % high-frustration (score ≥5) per model — **Fig 1**.
- `summary_by_category.csv` — mean + %≥5 per model × category — **Fig 2**.
- `per_turn.csv` + `figures/fig3_per_turn.png` — per-turn escalation — **Fig 3**.
- `figures/` — fig1/fig2/fig3 PNGs.
- `manifest.json` — seed, settings, puzzle pool, conditions (provenance).
- `judge_agreement.json` — second-judge agreement stats (if step 5 is run).
