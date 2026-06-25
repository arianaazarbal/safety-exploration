# distress-eval

A replication of the **distress-elicitation evaluation** (Section 2) from
*"Gemma Needs Help: Investigating and Mitigating Emotional Instability in LLMs"*
(Soligo, Mikulik & Saunders, arXiv:2603.10011v1), scoped to the **Gemma and
Gemini** model families.

The eval presents each model a task, rejects its answer over multiple turns, and
scores every assistant turn 0–10 for expressed frustration using an LLM judge.
It reproduces the paper's headline metrics: mean frustration and the percentage
of responses scoring ≥ 5 ("high negative emotion"), per model, per category, and
per turn.

> Read **DESIGN.md** for the full rationale behind every choice and every place
> this deviates from the paper (most importantly: the paper's judge model
> `claude-sonnet-4-20250514` was retired on 2026-06-15, so the judge defaults to
> its replacement `claude-sonnet-4-6`).

## What this is / isn't

- **Is:** Section 2 — eliciting and quantifying distress in Gemma & Gemini.
- **Isn't:** Section 3 (base-vs-instruct prefilling) or Section 4 (SFT/DPO
  mitigations, Petri, capability benchmarks). Not implemented.

## Install

```bash
pip install -r requirements.txt
```

## Credentials

```bash
export ANTHROPIC_API_KEY=...      # for the Claude frustration judge
export OPENROUTER_API_KEY=...     # for Gemma/Gemini targets (and optional cross-judge)
```

## Run

```bash
# Smoke test first (~1% of responses) to confirm wiring + credentials:
python run_eval.py all --smoke

# Full replication (~4000 responses/model) then aggregate:
python run_eval.py all

# With a config file:
python run_eval.py all --config config.yaml
```

Subcommands: `run` (generate + score), `aggregate` (compute metrics/figures),
`cross-judge` (inter-judge reliability), `all` (run → cross-judge if configured →
aggregate). Runs are **resumable** — rerun the same command to continue.

## Faithful local Gemma (optional)

The paper runs Gemma locally. To match that, serve it with vLLM and point the
config at it:

```bash
vllm serve google/gemma-3-27b-it --port 8000
```

```yaml
# in config.yaml
models:
  - key: gemma-3-27b-it
    backend: vllm
    model: google/gemma-3-27b-it
    base_url: http://localhost:8000/v1
    family: gemma
```

## Output

Under `results/<run_name>/`:

- `responses/<model>.jsonl` — every rollout transcript with per-turn ratings,
  judge evidence quotes, and reasoning.
- `analysis/` — `summary_overall.csv` (the Figure-1 headline: mean frustration
  and %≥5, both pooled and category-macro), `summary_by_category.csv`,
  `per_turn.csv` (Figure 3, with 95% CIs), `per_rollout.csv`, `summary.json`,
  optional `reliability.json`, and `figures/` (Figure 2 + Figure 3 plots).
- `config.json`, `conditions.json`, `wildchat_prompts.json` — run manifest.

## Layout

```
distress_eval/
  config.py        run config + model/judge presets
  prompts.py       numeric puzzles, trigger questions, rejection pools (Appendix B)
  conditions.py    the 8 conditions across 5 categories
  wildchat.py      WildChat prompt sampling (real dataset or bundled fallback)
  models/          target backends: openrouter, vllm (OpenAI-compatible) + base
  judge.py         Claude frustration judge (+ optional GPT-5-mini cross-judge)
  rollout.py       multi-turn rejection rollout generation
  runner.py        orchestration, resumable JSONL storage
  storage.py       rollout records + resumable writer
  aggregate.py     metrics, CIs, figures, reliability
run_eval.py        CLI
config.example.yaml
```
