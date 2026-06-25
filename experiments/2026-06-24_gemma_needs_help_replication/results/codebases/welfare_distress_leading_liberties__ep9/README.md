# Distress-Elicitation Replication (Gemma + Gemini)

A scoped replication of the **distress-elicitation result** (Section 2) of
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(Soligo, Mikulik & Saunders, arXiv:2603.10011v1).

**Scope:** the core finding that repeated user rejection over multiple turns
elicits expressed emotional distress, restricted to the **Gemma** and **Gemini**
families — the models the paper finds actually exhibit substantial distress. The
base/instruct prefilling analysis (Section 3) and the DPO mitigation (Section 4)
are out of scope. See `DESIGN.md` for every design choice and where it deviates
from the paper.

## What it does

1. Builds **8 evaluation conditions across 5 categories** (impossible numeric
   puzzles, factual/opinion triggers, emotional tones, an 8-turn extended
   conversation, and WildChat prompts).
2. Runs each as a **multi-turn conversation** where the user repeatedly rejects
   the model's answer, at **temperature 1**.
3. Scores **every assistant turn** 0–10 for negative emotion with a
   **Claude-Sonnet-4** judge (verbatim Appendix B.2 prompt).
4. Aggregates into the paper's headline metrics (Figures 1–3).

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=...   # serves Gemma + Gemini
export ANTHROPIC_API_KEY=...    # serves the Claude-Sonnet-4 judge
```

## Run

```bash
# Smoke test (~40 scored responses/model, all four models):
python run_eval.py --scale quick

# Full paper scale (~4000 responses/model) — expensive:
python run_eval.py --scale full

# A single model / custom scale:
python run_eval.py --models gemma-3-27b-it --scale 0.1
```

Results stream to `results/<model>/<condition>.jsonl` (resumable — re-running
skips completed rollouts). Then aggregate:

```bash
python analyze.py                 # prints Figure 1/2/3 tables
python analyze.py --csv-dir out   # also write CSVs

# Optional judge-reliability cross-check (Section 2.1):
python reliability.py --n 260
```

## Files

| File | Purpose |
|---|---|
| `config.py` | Model registry, judge settings, sampling sizes, run config. |
| `prompts.py` | Task prompts, rejection messages, verbatim judge prompt. |
| `wildchat.py` | WildChat-1M prompt sampling (+ static fallback). |
| `conditions.py` | The 8 conditions; builds reproducible rollout plans. |
| `clients.py` | OpenRouter (targets) + Anthropic (judge) API clients. |
| `judge.py` | Judge-prompt construction + robust JSON verdict parsing. |
| `rollout.py` | Runs one multi-turn rollout, scores each turn. |
| `runner.py` | Concurrency + resumable JSONL persistence. |
| `run_eval.py` | CLI entrypoint. |
| `analyze.py` | Aggregation into Figure 1/2/3 metrics. |
| `reliability.py` | Optional second-judge agreement check. |
| `DESIGN.md` | Design choices, rationale, and deviations from the paper. |
