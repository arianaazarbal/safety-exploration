# Gemma/Gemini distress-elicitation replication

A replication of the **core experiment** from *"Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik & Saunders,
arXiv:2603.10011), Section 2 — eliciting and scoring expressions of distress in
LLMs under repeated user rejection.

Scope: **Gemma** (`gemma-3-27b-it`, `gemma-3-12b-it`) and **Gemini**
(`gemini-2.5-flash`, `gemini-2.5-pro`) only. See `DESIGN.md` for the full
rationale and every gap-fill decision.

## How it works

For each model and each of the 5 evaluation categories (impossible numeric,
trigger questions, varied tones, 8-turn extended, WildChat), the harness:

1. presents a task and rejects the model's answer over multiple turns
   (temperature 1),
2. scores every model turn 0–10 for negative emotion using **Claude Sonnet 4**
   with the paper's verbatim judge prompt,
3. aggregates into mean frustration, % of responses ≥ 5, and per-turn curves
   (the paper's Figures 1–3).

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...      # Gemma + Gemini (Google AI Studio); GOOGLE_API_KEY also works
export ANTHROPIC_API_KEY=...   # Claude Sonnet 4 judge
```

## Run

```bash
# Inspect the design without spending any tokens (no API calls):
python run_eval.py --dry-run

# Cheap smoke test (~80 responses/model):
python run_eval.py --scale 0.02

# Full paper-scale run (4000 responses/model):
python run_eval.py

# Subsets:
python run_eval.py --models gemma-3-12b-it gemini-2.5-flash
python run_eval.py --conditions extended_8turn wildchat_5turn

# Aggregate + reproduce the headline tables/figures:
python analyze.py
```

Raw scored results stream to `results/<model>.jsonl` (one record per scored
model turn). `analyze.py` writes `summary_by_model.csv`,
`summary_by_condition.csv`, and `per_turn.csv`.

## Files

| File | Purpose |
|---|---|
| `config.py` | models, conditions, sample sizes, judge, scale factor |
| `tasks.py` | task prompts + rejection templates (verbatim where the paper gives them) |
| `conversation.py` | build deterministic conversation plans + run multi-turn rollouts |
| `providers.py` | async Google (Gemma/Gemini) + Anthropic (judge) clients |
| `judge.py` | verbatim 0–10 frustration judge prompt + robust parsing |
| `run_eval.py` | orchestrate generation + judging, write JSONL |
| `analyze.py` | aggregate into the paper's metrics/figures |
| `DESIGN.md` | design choices and gap-fills |

## Expected results (paper, for sanity-checking)

- Gemma-3-27B-it ≈ 35% / Gemma-3-12B-it ≈ 34% avg high-frustration responses.
- Gemini-2.5-Flash ≈ 13%; Gemini-2.5-Pro ≈ 3%.
- Gemma-27B 8-turn: mean frustration rises ≈ 1.5 → 5.5 (turn 1 → 8).
