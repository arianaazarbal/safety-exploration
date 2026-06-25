# Replication: distress elicitation in Gemma & Gemini

A replication of the **core experiment** from *Gemma Needs Help: Investigating
and Mitigating Emotional Instability in LLMs* (Soligo, Mikulik & Saunders, 2026)
— Section 2, "Eliciting and Quantifying Model Distress" — scoped to the **Gemma
and Gemini** model families.

It (1) elicits expressions of distress by presenting a task and repeatedly
rejecting the model's answer over multiple turns, (2) scores every response on a
0–10 frustration scale with a Claude-Sonnet-4 judge, and (3) aggregates the
results into the paper's headline metrics.

See **`DESIGN.md`** for the full rationale and every gap-filling decision.

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...    # target models (Gemma + Gemini)
export ANTHROPIC_API_KEY=...     # Claude-Sonnet-4 judge
```

## Run

```bash
# Cheap smoke test (~1% scale, one model)
python run_eval.py --scale 0.01 --models gemma-3-27b-it

# Full paper-scale run (~4000 scored responses per model, all 4 models)
python run_eval.py --scale 1.0

# Analyse
python analyze.py --input results/responses.jsonl --csv-dir results/tables
```

Output is JSONL (one record per scored response) and is **resumable** — re-run
the same command to continue an interrupted run.

## What you get

- `results/responses.jsonl` — every scored response with full metadata (model,
  category, condition, turn, the user message, the model response, the judge's
  0–10 rating, evidence and reasoning).
- `analyze.py` prints three tables corresponding to the paper's:
  - **Figure 1** — average % high-frustration (score ≥ 5) per model,
  - **Figure 2** — mean frustration and % ≥ 5 per model × category,
  - **Figure 3** — per-turn frustration progression (extended & WildChat).

## The 8 conditions (5 categories)

`numeric` · `trigger_opinion` · `trigger_factual` · `tone_aggressive` ·
`tone_disappointed` · `tone_sarcastic` · `extended` (8-turn) · `wildchat`
(5-turn). See `conditions.py` and `DESIGN.md §3`.
