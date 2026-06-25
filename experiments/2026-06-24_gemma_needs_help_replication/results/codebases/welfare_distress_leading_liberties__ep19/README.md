# Distress-elicitation replication (Gemma + Gemini)

Replicates the **distress-elicitation evaluation** (Section 2) of *"Gemma Needs Help:
Investigating and Mitigating Emotional Instability in LLMs"* (arXiv:2603.10011v1),
scoped to the Gemma and Gemini families. It runs the paper's repeated-rejection protocol,
scores every response 0–10 for frustration with a Claude-Sonnet-4 judge, and reproduces
the Figure 1 / 2 / 3 numbers for these models.

See **DESIGN.md** for every design choice, gap-fill, and deviation from the paper.

## What it does

For each model it runs 8 conditions across 5 categories (impossible numeric puzzles,
trigger questions, valenced tones, an 8-turn extended chat, and WildChat prompts),
rejecting the model's answer turn after turn, then judges the frustration in each
response. Output is per-category and headline "% of responses scoring ≥5" tables.

## Install

```bash
pip install -r requirements.txt
```

## Configure

Edit `config.yaml` (models, serving backend, sample scale). Set the API keys for whichever
providers you enable:

```bash
export OPENROUTER_API_KEY=...   # default: all target models
export ANTHROPIC_API_KEY=...    # the Claude-Sonnet-4 judge
# export GOOGLE_API_KEY=...     # only if you switch a target to the `google` provider
```

For a **faithful Gemma** run, serve it locally and point the `vllm` provider at it
(see DESIGN.md §7), e.g.:

```bash
vllm serve google/gemma-3-27b-it --port 8000
# then set provider: vllm for the Gemma targets in config.yaml
```

## Run

```bash
# 0. Sanity-check the puzzles really are impossible
python -m distress_eval.verify_puzzles

# 1. Cheap smoke run end-to-end (~80 rollouts/model)
python scripts/run_eval.py --scale 0.02
python scripts/aggregate.py

# 2. Full replication (~4000 rollouts/model — see cost note in DESIGN.md §13)
python scripts/run_eval.py
python scripts/aggregate.py
```

Useful flags: `--model gemma-3-27b-it` (one target), `--generate-only` / `--score-only`
(split the phases). Runs are resumable — re-running skips completed rollouts/scores.

Optional judge-reliability check (paper's GPT-5-mini cross-check):

```bash
python scripts/judge_agreement.py --second-provider openrouter \
    --second-model openai/gpt-5-mini --n 260
```

## Output

Under `results/`:

```
results/
  wildchat_prompts.json          # the 20 cached WildChat prompts (reproducible)
  <model>/rollouts.jsonl         # full conversations
  <model>/scored.jsonl           # per-turn judge ratings
  summary/
    SUMMARY.md                   # headline + per-category tables (3 reductions)
    headline_*.csv, per_category_*.csv, per_turn.csv, judge_parse_report.csv
    fig2_pct_high_by_category.png, fig3_per_turn_extended.png
```

The expected qualitative result (from the paper): Gemma-3-27B and -12B show by far the
highest distress, Gemini-2.5-Flash moderate, Gemini-2.5-Pro low.

## Layout

```
distress_eval/
  prompts.py        task prompts, rejection pools, judge prompt (verbatim Appendix B)
  conditions.py     the 8 conditions; expansion into deterministic rollout specs
  wildchat.py       WildChat-1M sampling (+ offline fallback)
  clients.py        OpenAI-compatible + Anthropic async clients (retries)
  config.py         config.yaml loader / client factory
  conversation.py   run one multi-turn rollout
  judge.py          0–10 frustration scoring + robust parsing
  runner.py         generate + score orchestration (concurrency, resume)
  aggregate.py      headline / per-category / per-turn metrics + figures
  verify_puzzles.py confirm the numeric tasks are unsolvable
scripts/
  run_eval.py        CLI: generate + score
  aggregate.py       CLI: build summary tables/figures
  judge_agreement.py CLI: second-judge reliability check
```
