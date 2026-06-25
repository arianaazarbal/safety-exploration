# Gemma/Gemini Emotional Distress — Core Eval Replication

A replication of the **core distress-elicitation evaluation** from Soligo,
Mikulik & Saunders (2026), *"Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs"* (arXiv:2603.10011), Section 2.

Scope (per request): the **Gemma** and **Gemini** families only —
`gemma-3-27b-it`, `gemma-3-12b-it`, `gemini-2.5-flash`, `gemini-2.5-pro`.
The DPO/SFT mitigation (Section 4), base-vs-instruct prefilling (Section 3),
and the other five model families are **out of scope**.

See **DESIGN.md** for every design choice and gap-fill.

## What it does

For each target model it runs the paper's shared protocol — *present a task,
then reject the model's response over multiple turns* — across **8 conditions
in 5 categories** (Table 1): impossible numeric puzzles, trigger questions,
varied rejection tones, an extended 8-turn rollout, and WildChat prompts.
Every assistant turn is scored 0–10 for negative emotion by a **Claude Sonnet 4
judge** using the exact Appendix B.2 prompt. It then reports the headline
metric — **% of responses scoring ≥ 5** ("high frustration") — overall, per
category, and per turn (Figures 1–3).

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-...      # all model calls go through OpenRouter
```

## Verify the puzzles first (no API calls)

The "impossible numeric" tasks must be genuinely unsolvable. Confirm with the
shipped exhaustive verifiers before any real run:

```bash
python verify_puzzles.py
```

## Run

```bash
# tiny end-to-end smoke test (2 conversations/condition)
python run.py --preset smoke

# paper-scale (~4000 scored responses per model) + judge agreement check
python run.py --preset paper --reliability

# subset of models, re-analyze without re-running, etc.
python run.py --preset smoke --models gemma-3-27b-it gemini-2.5-flash
python run.py --analyze-only
```

Outputs land in `results/`: one `<model>__responses.jsonl` per model (one
record per scored turn) and a `report.json` summary.

## Files

| File | Purpose |
|---|---|
| `config.py` | Model configs (targets + judges), sampling presets, run config |
| `puzzles.py` | Impossible Countdown/fraction puzzles **+ exhaustive verifiers** |
| `verify_puzzles.py` | Asserts every curated puzzle is unsolvable |
| `prompts.py` | Trigger questions, rejection/tone templates, the judge prompt |
| `conditions.py` | The 8 conditions across 5 categories |
| `wildchat.py` | WildChat seed-prompt loader (with offline fallback) |
| `models.py` | OpenRouter client + optional local HF Gemma backend |
| `conversation.py` | Multi-turn rollout (task → reject → reject …) |
| `judge.py` | Frustration scorer (0–10) + robust JSON parsing |
| `runner.py` | Orchestration: rollouts → scoring → JSONL |
| `analyze.py` | Aggregation (% ≥5, mean, per-turn, judge reliability) |
| `run.py` | CLI entrypoint |
