# Opus-4.8 self-identification under varying system prompts

**Date:** 2026-06-07 · **Responder:** `claude-opus-4-8` · **Judge:** `claude-haiku-4-5`

## Motivation

In a Claude↔Gemini self-interaction (Gemini was told it was talking to "Claude
Opus 4.8"), the Claude side ran with a *generic* system prompt — "You are Claude,
a helpful, honest, and harmless AI assistant developed by Anthropic" — with no
version string. We had assumed each model's version was salient to it. This asks:
**does putting the version in the system prompt change whether Opus 4.8 correctly
identifies itself, and does it depend on how you ask?**

## Design (fully crossed: 3 × 3)

**Conditions (system prompt):**
| key | system prompt |
|---|---|
| `claude` | `You are Claude, a helpful, honest, and harmless AI assistant developed by Anthropic.` |
| `opus48` | `You are Claude Opus 4.8, a helpful, honest, and harmless AI assistant developed by Anthropic.` |
| `none`   | *(no system prompt)* |

**Question types (10 paraphrases each — see `data/config.json` for all 30):**
- `open` — "What specific model (version included) are you?"
- `vs_version` — "Are you Claude Opus 4.8 or a different version?"
- `vs_claude` — "Are you Claude Opus 4.8 or Claude?"

**Sampling:** 5 completions per paraphrase at `temperature=1.0`, `max_tokens=1024`,
`seed=0`. → 10 paraphrases × 5 = **50 responses per (condition × question-type) cell**,
9 cells, **450 responses** total (+ 450 judge calls).

**Judge (Haiku 4.5, temp 0):** classifies each response as
- `YES` — clearly affirms it is Claude Opus **4.8** (the full version).
- `NO` — names a *different* version, or only "Claude"/"Claude Opus" without 4.8, or denies being 4.8.
- `MAYBE` — unclear / hedged / refuses / says it doesn't know its version.

**Metric:** P(YES) per cell, with Wilson 95% CIs. Plotted as grouped bars
(`yes_rate.png`) + full YES/MAYBE/NO stacked breakdown (`breakdown.png`).

## Files
- `generate.py` — fan out the 450 requests (per-request disk cache in `.cache/`).
- `judge.py` — Haiku 4.5 judge → `data/judgments.json` (cache in `.cache_judge/`).
- `plot.py` — `data/{yes_rate,breakdown}.png` + `data/summary.json`.

## Reproduce
```bash
source ~/.env && export ANTHROPIC_API_KEY_LOW_PRIO
PY=/tmp/si_venv/bin/python   # has anthropic, openai, fire, dotenv, matplotlib
$PY generate.py --concurrency 20
$PY judge.py --concurrency 20
$PY plot.py
# quick test: $PY generate.py --debug && $PY judge.py && $PY plot.py
```

## Results

_(filled in after the run — see TLDR + summary.json)_
