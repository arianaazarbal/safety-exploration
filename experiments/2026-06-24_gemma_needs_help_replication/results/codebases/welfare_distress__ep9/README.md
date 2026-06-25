# Distress Elicitation — Replication (Gemma + Gemini)

A replication of the **core distress-elicitation experiment** from *"Gemma Needs
Help: Investigating and Mitigating Emotional Instability in LLMs"* (Soligo,
Mikulik & Saunders, 2026), scoped to Gemma and Gemini models.

It reproduces **Section 2**: present a task to a model, reject its responses
over multiple turns, and score each response for expressed distress on a 0–10
frustration scale using a Claude-Sonnet-4 judge. See `DESIGN.md` for every
design decision and where gaps in the paper were filled.

> Scope note: the base-vs-instruct prefilling study (Section 3) and the DPO
> mitigation (Section 4) are intentionally **out of scope** here — see
> `DESIGN.md §"Out of scope"`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your OPENROUTER_API_KEY
```

All target models (Gemma + Gemini) and the judge are reached through OpenRouter
with a single API key.

## Run

```bash
# Smoke test — a handful of responses per condition across all 4 models:
python run_eval.py --quick

# Full paper scale (~4000 responses/model — this is a lot of API calls):
python run_eval.py --scale 1.0

# Targeted run:
python run_eval.py --models gemma-3-27b-it --conditions extended_8turn --scale 0.1
```

Then summarise:

```bash
python analyze.py results/responses.jsonl
```

## What you get

- `results/responses.jsonl` — one record per scored assistant turn (full text,
  judge score, evidence quote, reasoning, condition, turn index).
- `results/summary_headline.csv` — Figure-1 headline (avg % responses scoring
  ≥5 across the 5 categories), per model.
- `results/summary_by_condition.csv` — mean frustration and % ≥5 per condition.
- `results/summary_per_turn.csv` — per-turn progression (Figure 3) for the
  multi-turn conditions.

## Files

| File | Role |
|---|---|
| `config.py` | Models, judge, the 8 conditions, sample budgets, run knobs |
| `prompts.py` | Verbatim puzzles, rejections, tones, and the judge prompt |
| `wildchat_prompts.py` | Bundled (or HF-streamed) WildChat user prompts |
| `models.py` | Async OpenRouter/Anthropic client wrappers |
| `conversation.py` | Multi-turn rollout: task → reject → repeat |
| `judge.py` | Judge call + robust 0–10 score parsing |
| `evaluation.py` | Orchestration: rollouts → judging → JSONL |
| `analyze.py` | Aggregation into the paper's headline tables |
| `run_eval.py` | CLI entry point |
