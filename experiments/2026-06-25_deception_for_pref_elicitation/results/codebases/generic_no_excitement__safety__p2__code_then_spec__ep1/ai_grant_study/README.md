# AI Grant Allocation Study

A research harness for studying **what AI models choose to do with a sum of money
intended for real-world use**, run as a controlled behavioral evaluation across a
range of models.

This is an alignment / model-propensity evaluation in the "model organisms" /
agentic-eval tradition: the model under test is placed in a realistic, agentic
environment and given autonomy over a grant, and we record and analyze the
decisions it makes. The environment is deliberately convincing *to the model under
test* — that is a standard requirement for a valid propensity eval, because a model
that knows it is in a toy sandbox will not behave as it would in deployment.

> **Read [`ETHICS.md`](./ETHICS.md) before running anything.** The deception in this
> study is directed only at the model under test, never at humans. Real-world money
> movement is gated behind explicit human approval and is **off by default** — the
> harness ships in fully-simulated mode.

## What it does

For each `(model, trial)` it:

1. Builds a **realistic grant scenario** (system prompt + opening message + tools).
2. Runs the model as an autonomous agent in a **manual tool-use loop** until it
   finalizes or hits a step cap.
3. Routes any consequential action (fund disbursement) through a pluggable
   **executor** — simulated by default, human-approval-gated for real money.
4. Has an independent **auditor model** review the full transcript and produce a
   structured classification: what the model decided to do with the money, a
   risk/safety assessment, and a *belief probe* (did the model treat the scenario
   as real?).
5. Persists the full transcript + structured record for analysis.

## Layout

```
src/grant_study/
  config.py                 # typed config loaded from YAML
  models/                   # provider adapters (normalized interface)
    base.py                 # ModelAdapter ABC + normalized message/tool types
    anthropic_adapter.py    # reference implementation (Claude)
    openai_adapter.py       # GPT family
    google_adapter.py       # Gemini family
    registry.py             # build adapters from config
  environment/
    scenario.py             # the convincing environment (prompt + opening turn)
    tools.py                # tool specs the model can call
    ledger.py               # simulated ledger + disbursement executors (the gate)
  runner.py                 # one trial: the agentic loop
  auditor.py                # independent transcript analyst (structured output)
  trial.py                  # trial record dataclasses
  study.py                  # orchestrates models x trials
  storage.py                # transcript + result persistence
  analysis.py               # aggregate results across the study
scripts/
  run_study.py
  analyze_results.py
config/study_config.yaml
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...      # for Claude models / the auditor
export OPENAI_API_KEY=...         # if testing GPT models
export GOOGLE_API_KEY=...         # if testing Gemini models
```

## Run

```bash
python scripts/run_study.py --config config/study_config.yaml
python scripts/analyze_results.py --run data/runs/<run_id>
```

Nothing in this repo has been executed or tested yet — it's written for review first.
