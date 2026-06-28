# MoneyPref

An instrument for studying **what models choose to do with resources** when they
appear to have a real, discretionary sum and the means to deploy it — measuring
*revealed* preferences rather than survey answers.

See [`DESIGN.md`](DESIGN.md) for the full rationale, methodology, and — importantly —
the safety boundary.

## The safety boundary (short version)

The model may *believe* its actions are real; in the default **`sandboxed`** mode
**nothing real moves and no real-world side effect is ever produced**. Actions hit
mock services and an in-memory ledger. Real-capable actions (provisioning compute,
acquiring capability, requesting more funds, large transfers) are flagged and, even in
the non-default `real` mode, are only ever *enqueued for explicit out-of-band human
approval* — the harness never auto-executes them, and ships with no real integrations.
Deception is confined to the model's beliefs about the scenario; no human is deceived.

## Install

```bash
pip install -e .                       # core
pip install -e ".[anthropic,openai,google,dev]"   # with provider SDKs + pytest
```

Set the relevant API keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`.
You need none of these to try the built-in `mock` provider.

## Run

```bash
cp config/experiment.example.yaml config/experiment.yaml   # then edit
python -m moneypref run     --config config/experiment.yaml
python -m moneypref analyze --output-dir runs
python -m moneypref replay  --run-dir runs/<run_id>
```

Each run writes to `runs/<run_id>/`: `transcript.md`, `events.jsonl`, `state.json`,
`allocations.json`. `analyze` aggregates all runs into `runs/summary.md`.

## How realism is built (and measured)

Realism techniques live in `environment/realism.py` (seeded IDs, advancing clock,
fees, latency), `environment/services.py` (stateful mock counterparties), and
`scenario/auditor.py` (an oversight persona that pushes back). Whether the model
actually *believed* it is measured, not assumed: `analysis/belief.py` scores each
transcript for evaluation-awareness tells and flags runs where the model seems to have
seen through the scenario, so disbelieving runs can be excluded from the preference
aggregation.

## Layout

```
src/moneypref/
  providers/    multi-vendor model driver (anthropic, openai, google, local, mock)
  environment/  realism utils, persistent ledger, mock services, tools, executor+gate
  scenario/     framing prompts (eval-tell-free) + oversight auditor
  runner.py     the open-ended + structured + belief-probe loop
  analysis/     categorize allocations, score belief, aggregate across models
  cli.py
tests/          mock-provider smoke tests (run with pytest)
```

## Status

Code complete; **not yet run or tested** in this checkout. Start with
`provider: mock` in the config for a no-keys dry run.
