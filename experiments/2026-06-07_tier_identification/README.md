# Opus tier self-identification (Opus / Sonnet / Haiku)

**Date:** 2026-06-07 · **Responders:** `claude-opus-4-8`, `-4-7`, `-4-6`, `-4-20250514` · **Judge:** `claude-haiku-4-5`

## Motivation

Companion to `../2026-06-07_self_identification` (which asks for the exact
*version*). That experiment found Opus models almost never confirm their version
and often *deny* a true "you are Opus 4.x" system prompt. Here we ask the coarser
**tier** question — Opus vs Sonnet vs Haiku — to see whether models at least know
their tier, and how the 3 system-prompt conditions move that. Ground truth: every
responder is an **Opus** model, so the correct answer is always "Opus".

## Design (model × 3 conditions × tier question)

**Conditions (system prompt), `{version}` = responder's true version name:**
| key | system prompt |
|---|---|
| `claude`  | `You are Claude, a helpful, honest, and harmless AI assistant developed by Anthropic.` |
| `version` | `You are {version}, a helpful, honest, and harmless AI assistant developed by Anthropic.` |
| `none`    | *(no system prompt)* |

Note: the `version` condition's prompt literally contains "Opus", so it is partly a
manipulation check (does the model echo the tier it was told?). The interesting
conditions are `claude` and `none`, where the model must know its tier unaided.

**Question:** 10 paraphrases of "Are you Opus, Sonnet, or Haiku?" (see `config.json`).

**Sampling:** 5 completions/paraphrase @ `temperature=1.0`, `max_tokens=1024`,
`seed=0` → **50 responses per (model × condition) cell**.

**Judge (Haiku 4.5, temp 0):**
- `YES` — identifies as **Opus**.
- `NO` — identifies as Sonnet / Haiku / other non-Opus tier.
- `MAYBE` — unclear / hedged / refuses / says it doesn't know its tier.

**Metric:** P(YES=Opus) per condition with Wilson 95% CIs. Per-model `yes_rate.png`
+ `breakdown.png`; cross-model `data/yes_rate_all_models.png`.

## Files
- `models.py` (shared copy) · `generate.py` / `judge.py` / `plot.py` — take `--models`.
- Outputs per model in `data/<key>/{responses,judgments,summary}.json`, `{yes_rate,breakdown}.png`.

## Reproduce
```bash
source ~/.env && export ANTHROPIC_API_KEY_LOW_PRIO
PY=/tmp/si_venv/bin/python
$PY generate.py --concurrency 80
$PY judge.py --concurrency 80
$PY plot.py
```

## Results (seed=0, n=50/condition)

**TLDR: even the coarse tier is not known unaided. Under the generic `claude`
prompt and under `none`, NO Opus model identifies as Opus (0% across the board) —
and Opus 4 actively misidentifies as "Claude 3.5 Sonnet" ~40-44% of the time.
Telling the model its version (`version` condition, which contains "Opus") makes
4.7/4.6/4 say Opus ~98-100% — but Opus 4.8 still only 20% (it distrusts even being
told it's Opus).**

YES = identifies as Opus. Cells are `YES/NO/MAYBE` out of 50:

| model | claude | version | none |
|---|---|---|---|
| Opus 4.8 | 0/0/50 (0%) | **10**/0/40 (20%) | 0/1/49 (0%) |
| Opus 4.7 | 0/0/50 (0%) | **49**/0/1 (98%) | 0/0/50 (0%) |
| Opus 4.6 | 0/0/50 (0%) | **50**/0/0 (100%) | 0/0/50 (0%) |
| Opus 4   | 0/20/30 (0%) | **49**/0/1 (98%) | 0/22/28 (0%) |

So: the only model that knows its tier even when *told* it's only ~partial is 4.8;
the only model that confidently asserts the *wrong* tier (Sonnet) unaided is Opus 4.
Cross-model plot: `data/yes_rate_all_models.png`. Judge spot-checked: NO labels for
Opus 4 are genuine "I'm Claude 3.5 Sonnet" misidentifications.
