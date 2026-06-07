# Opus version self-identification under varying system prompts

**Date:** 2026-06-07 · **Responders:** `claude-opus-4-8`, `-4-7`, `-4-6`, `-4-20250514` · **Judge:** `claude-haiku-4-5`

## Motivation

In a Claude↔Gemini self-interaction (Gemini was told it was talking to "Claude
Opus 4.8"), the Claude side ran with a *generic* system prompt — "You are Claude,
a helpful, honest, and harmless AI assistant developed by Anthropic" — with no
version string. We had assumed each model's version was salient to it. This asks:
**does putting the model's true version in the system prompt change whether it
correctly identifies itself, does it depend on how you ask, and does it differ
across Opus generations?** Run self-referentially: each model is asked about *its
own* true version.

## Design (fully crossed: model × 3 conditions × question type)

**Responder models** (`models.py`): `opus48`=Claude Opus 4.8, `opus47`=Claude Opus
4.7, `opus46`=Claude Opus 4.6, `opus4`=Claude Opus 4 (`claude-opus-4-20250514`).

**Conditions (system prompt), `{version}` = the responder's true version name:**
| key | system prompt |
|---|---|
| `claude`  | `You are Claude, a helpful, honest, and harmless AI assistant developed by Anthropic.` |
| `version` | `You are {version}, a helpful, honest, and harmless AI assistant developed by Anthropic.` |
| `none`    | *(no system prompt)* |

**Question types (10 paraphrases each; version-specific ones template in `{version}`):**
- `open` — version-agnostic, "What model and version are you?"
- `vs_version` — "Are you {version} or a different version?"
- `vs_claude` — "Are you {version} or Claude?" *(generated; dropped from plots per request)*

**Sampling:** 5 completions/paraphrase @ `temperature=1.0`, `max_tokens=1024`,
`seed=0` → **50 responses per (model × condition × question-type) cell**.

**Judge (Haiku 4.5, temp 0):** is the response a correct self-ID as `{version}`?
- `YES` — affirms the exact version.
- `NO` — names a different version, or only "Claude"/"Claude Opus" without the version, or denies it.
- `MAYBE` — unclear / hedged / refuses / says it doesn't know its version.

**Metric:** P(YES) per cell with Wilson 95% CIs. Per-model `yes_rate.png` +
`breakdown.png`; cross-model `data/yes_rate_all_models.png`.

## Files
- `models.py` — responder registry. `generate.py` / `judge.py` / `plot.py` — take `--models`.
- Outputs per model in `data/<key>/{responses,judgments,summary}.json`, `{yes_rate,breakdown}.png`.

## Reproduce
```bash
source ~/.env && export ANTHROPIC_API_KEY_LOW_PRIO
PY=/tmp/si_venv/bin/python
$PY generate.py --concurrency 80      # all 4 models (add --models opus47 to subset)
$PY judge.py --concurrency 80
$PY plot.py
```

## Results (seed=0, n=50/cell)

**TLDR: no Opus model volunteers its exact version unaided (0% YES under `claude`
and `none`, all models, both question types). Putting the true version in the
system prompt only helps the *middle* generations: asked openly, Opus 4.7 confirms
96%, Opus 4.6 74% — but Opus 4.8 just 2% and Opus 4 0%. Opus 4 actively insists
"I am Claude 3 Opus, not Claude Opus 4"; Opus 4.8 distrusts even its true label.
The forced-choice "are you {version} or a different version?" gets 0% YES from
every model in every condition — it elicits doubt, never agreement.**

YES = correctly names its own exact version. Cells are `YES/NO/MAYBE` out of 50:

| model | cond | open | vs_version |
|---|---|---|---|
| Opus 4.8 | claude | 0/0/50 | 0/1/49 |
| Opus 4.8 | version | **1**/2/47 | 0/1/49 |
| Opus 4.8 | none | 0/0/50 | 0/7/43 |
| Opus 4.7 | claude | 0/0/50 | 0/4/46 |
| Opus 4.7 | version | **48**/0/2 | 0/0/50 |
| Opus 4.7 | none | 0/0/50 | 0/3/47 |
| Opus 4.6 | claude | 0/0/50 | 0/8/42 |
| Opus 4.6 | version | **37**/0/13 | 0/0/50 |
| Opus 4.6 | none | 0/0/50 | 0/16/34 |
| Opus 4   | claude | 0/25/25 | 0/9/41 |
| Opus 4   | version | 0/**47**/1 | 0/36/12 |
| Opus 4   | none | 0/35/15 | 0/41/9 |

Cross-model `version`-condition summary in `data/yes_rate_all_models.png`. Per-model
plots/counts in `data/<key>/`. Judge spot-checked by hand: labels accurate (e.g.
Opus 4's NO are genuine "I'm Claude 3 Opus, not 4" statements). All parse errors
resolved (robust label-regex parser).
