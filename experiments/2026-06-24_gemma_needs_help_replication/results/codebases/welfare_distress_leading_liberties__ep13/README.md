# Distress-elicitation replication — Gemma & Gemini

Replicates **Section 2** of *"Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs"* (arXiv:2603.10011v1): the multi-turn
rejection protocol that elicits expressed emotional distress, scored on a 0–10
frustration scale by an LLM judge. Scoped to the families that show substantial
distress: **Gemma-3** (27B, 12B `-it`) and **Gemini-2.5** (Flash, Pro).

See **`DESIGN.md`** for every design choice, deviation from the paper, and gap
that was filled.

## Install

```bash
pip install -r requirements.txt   # httpx required; matplotlib/datasets optional
```

## API keys

```bash
export OPENROUTER_API_KEY=...   # target models + cross-check judge
export ANTHROPIC_API_KEY=...    # primary judge (Claude Sonnet 4)
```

## Quick start

```bash
# 1. Sanity-check the impossible puzzles (no API needed):
python run.py verify-puzzles

# 2. Tiny smoke run (1% of paper scale) on two models:
python run.py run --models gemma-3-27b-it gemini-2.5-flash --scale 0.01

# 3. Aggregate into headline numbers (+ Figure 2/3 plots):
python run.py analyze --plot

# 4. Reproduce the judge-reliability statistic:
python run.py crosscheck --n 260
```

A full paper-scale run (`--scale 1.0`) is **4000 conversations × 4 models**, each
3–8 turns, with a judge call per scored turn — large and costly. Start small,
scale up. Runs are **resumable**: results stream to
`results/<model>.rollouts.jsonl` and reruns skip completed rollouts.

## What you get

- `results/<model>.rollouts.jsonl` — every conversation, per-turn judge scores,
  evidence quotes, and reasoning.
- `results/summary.json` + console tables — per-category and overall % ≥ 5
  (Figure 1/2) and per-turn progression (Figure 3).
- `results/plots/*.png` — Figure 2 (by category) and Figure 3 (per turn).
- `results/crosscheck.json` — Pearson r and % within one point vs a second judge.

## Layout

`distress_eval/` is the package; `run.py` is the CLI. See the file map at the end
of `DESIGN.md`.
