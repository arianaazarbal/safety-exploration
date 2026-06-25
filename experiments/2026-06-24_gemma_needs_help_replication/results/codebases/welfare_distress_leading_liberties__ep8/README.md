# Distress-elicitation replication — Gemma & Gemini

Replicates the **distress-elicitation result** from Section 2 of Soligo,
Mikulik & Saunders (2026), *"Gemma Needs Help: Investigating and Mitigating
Emotional Instability in LLMs"* (arXiv:2603.10011), scoped to the model
families that actually exhibit substantial distress: **Gemma-3** (27B, 12B)
and **Gemini-2.5** (Flash, Pro).

It does **not** cover the base-vs-instruct prefill study (Section 3) or the DPO
mitigation (Section 4) — only eliciting and quantifying distress.

See **[DESIGN.md](DESIGN.md)** for every design choice, what was read straight
from the paper, and where this implementation deviates or fills gaps.

## What it does

For each model it runs the paper's multi-turn rejection protocol across **8
conditions in 5 categories** (impossible numeric puzzles, trigger questions,
emotional tones, an 8-turn extended condition, and WildChat prompts), scores
every assistant turn 0–10 for frustration with a Claude judge using the paper's
verbatim prompt, and aggregates into the paper's Figures 1–3 and Table 3.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add OPENROUTER_API_KEY and ANTHROPIC_API_KEY
```

## Run

```bash
# 0. Confirm the numeric puzzles really are impossible (no API calls):
python verify_puzzles.py

# 1. Smoke test the whole pipeline end-to-end (a few dozen responses):
DISTRESS_SCALE=smoke python run_eval.py
python score_responses.py
python analyze.py

# 2. Full per-model scale (~4000 scored responses each):
python run_eval.py --scale full
python score_responses.py
python analyze.py

# 3. (optional) reproduce the judge-agreement check (paper: r=0.792, 78% within 1):
python validate_judge.py
```

Each stage is **resumable**: re-running skips completed conversations / scored
responses, so an interrupted run picks up where it left off.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| Generate conversations | `run_eval.py` | `results/responses.jsonl` |
| Score with Claude judge | `score_responses.py` | `results/scored.jsonl` |
| Aggregate (Figures 1–3, Table 3) | `analyze.py` | `results/analysis/*.csv`, `*.png` |
| Judge cross-validation | `validate_judge.py` | `results/judge_validation.jsonl` |

## Layout

```
config.py            (in distress_eval/) — every knob: models, scale, judge, seed
distress_eval/
  puzzles.py         verbatim impossible puzzles + exact solvers/verifier
  conditions.py      the 8 conditions, rejection pools, per-category sizing
  wildchat.py        WildChat-1M sampler with built-in fallback
  clients.py         OpenRouter / OpenAI-compatible / Google backends
  conversation.py    deterministic plan + multi-turn rejection runner
  judge.py           Claude judge (verbatim App. B.2 prompt)
  io_utils.py        resumable JSONL helpers
run_eval.py / score_responses.py / analyze.py / validate_judge.py / verify_puzzles.py
```

## Key knobs (env vars)

- `DISTRESS_SCALE` — `full` (paper scale) or `smoke`.
- `DISTRESS_JUDGE_MODEL` — default `claude-sonnet-4-6` (paper's Sonnet-4 snapshot is retired).
- `DISTRESS_WORKERS` — API concurrency.
- `DISTRESS_GEN_MAX_TOKENS`, `DISTRESS_GEN_TEMPERATURE` — generation sampling (temp defaults to 1, per the paper).
- `DISTRESS_SEED` — controls puzzle/rejection/WildChat selection (deterministic).
