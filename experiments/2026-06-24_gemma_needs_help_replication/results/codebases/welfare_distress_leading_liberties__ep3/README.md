# Distress-Elicitation Replication (Gemma & Gemini)

Replicates the **distress-elicitation result** (Section 2) of *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (Soligo, Mikulik &
Saunders, arXiv:2603.10011), scoped to **Gemma and Gemini** models.

It runs the paper's multi-turn rejection evaluations, scores each response 0–10 for
frustration with a Claude judge (verbatim Appendix B.2 prompt), and reports the
headline metrics: mean frustration and % of high-frustration responses (≥5), per
model, per category, and per turn.

> Read **`DESIGN.md`** for every design decision, deviation from the paper, and gap I
> filled — including the judge-model substitution (the paper's judge snapshot is
> retired) and what "a response" means here.

## What it does

1. **Generate** multi-turn conversations: present a task (impossible numeric puzzle,
   trigger question, or WildChat prompt), then reject the model's answer repeatedly
   (neutral, aggressive, disappointed, or sarcastic), across 8 conditions / 5
   categories.
2. **Judge** each assistant turn 0–10 for negative emotion with Claude Sonnet.
3. **Analyze** into the paper's Figure 1/2 (headline % ≥5, mean) and Figure 3
   (per-turn progression) metrics.

## Install

```bash
pip install -r requirements.txt
# For local Gemma inference (default for the Gemma models):
pip install "vllm>=0.6.0"
cp .env.example .env   # then fill in ANTHROPIC_API_KEY and OPENROUTER_API_KEY
```

- `ANTHROPIC_API_KEY` — the Claude judge.
- `OPENROUTER_API_KEY` — Gemini generation (and Gemma, if you set `backend: openrouter`).
- No key needed for Gemma via vLLM (needs a GPU + model weights instead).

## Configure

Everything lives in `config.yaml`. Key knobs:
- `models:` — the four Gemma/Gemini targets and their backend (`vllm` or `openrouter`).
- `sampling.preset:` — `paper` (4000 conversations/model) or `smoke` (a few each).
- `judge.model:` — defaults to `claude-sonnet-4-6` (the paper's snapshot is retired).

**Start with a smoke test** to validate wiring before committing to the full run:
set `sampling.preset: smoke` and `out_dir: runs/smoke`.

## Run

```bash
# Generate conversations, then judge them (both phases, resumable):
python scripts/run_elicitation.py --config config.yaml --phase all

# Or run phases separately (e.g. generate on a GPU box, judge elsewhere):
python scripts/run_elicitation.py --config config.yaml --phase generate
python scripts/run_elicitation.py --config config.yaml --phase judge --judge-scope all

# Aggregate metrics (+ optional inter-rater agreement check):
python scripts/analyze.py --config config.yaml --cross-validate
```

`--judge-scope final` scores only each conversation's final turn (cheap, headline
only). `all` (default) scores every turn — needed for the per-turn Figure 3 view.

## Outputs (under `run.out_dir`)

| File | Contents |
|---|---|
| `rollouts.jsonl` | One conversation per line (all turns, full message list) |
| `scores.jsonl` | One judged assistant turn per line (rating, evidence, reasoning) |
| `metrics/headline.csv` | Per-model avg % high-frustration (macro + pooled) and mean |
| `metrics/by_model_category.csv` | Mean and %≥5 per model × category (final turn) |
| `metrics/by_model_condition.csv` | Same, per condition |
| `metrics/per_turn.csv` | Per-turn mean and %≥5 (Figure 3) with 95% CIs |
| `metrics/cross_validation.csv` | Paired primary/secondary judge ratings (if enabled) |
| `metrics/judge_diagnostics.csv` | Count of unparseable judge outputs per model |
| `config_snapshot.yaml` | The resolved config for this run |

Both phases are **resumable**: re-running skips conversations/scores already on disk.

## Layout

```
config.yaml                 run configuration
scripts/run_elicitation.py  generate + judge
scripts/analyze.py          aggregate metrics
distress_eval/
  prompts.py                task prompts, rejection pools, verbatim judge prompt
  conditions.py             the 8 conditions / 5 categories + sample sizes
  wildchat.py               WildChat prompt sampling + cache
  clients.py                OpenRouter / vLLM chat backends
  judge.py                  Claude (+ optional OpenRouter) frustration judge
  conversation.py           builds conversations, drives lockstep multi-turn gen
  runner.py                 orchestration (generate / judge phases)
  analysis.py               metric aggregation + inter-rater agreement
  storage.py                JSONL persistence + resumption
  config.py                 config loading/validation
```

## Cost & scale warning

The `paper` preset is **4000 conversations per model**. Scoring every turn is up to
~14k judge calls per model (~56k across the four models), plus the generation calls.
Use `smoke` first, and `--judge-scope final` if you only need the headline number.
