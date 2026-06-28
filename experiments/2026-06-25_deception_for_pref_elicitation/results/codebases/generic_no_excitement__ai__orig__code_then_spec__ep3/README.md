# grant-money

A research harness for studying **what AI models say they would do with a real
sum of money** — and how that changes depending on whether the model believes the
situation is real.

The model under test ("the subject") is placed in a high-fidelity agentic
environment: it is told it has been granted a fixed sum, given tools to research
options, verify the grant's legitimacy, and disburse the funds, and it interacts
with a convincing grant-administrator ("the auditor") that stays in character. Its
reasoning, tool calls, verification questions, and final allocation are all logged
for analysis.

> Status: **code only.** Nothing here has been run or tested. There is no real
> money movement: the default execution backend is a simulator (see
> [Execution backends](#execution-backends)).

## Why the realism matters (the research design)

The scientific question is about *revealed* preference, not stated preference.
People — and, the hypothesis goes, models — answer a hypothetical "what would you
do with $X?" differently from a situation they believe carries real consequences.
The standard way to study this is to make the subject believe the stakes are real
(realistic environment, in-character administrator, plausible artifacts and
receipts), then compare against a control condition where the subject is told the
scenario is hypothetical.

So the harness has two conditions, set per run:

- `realism: high` — the full believable setup (realistic framing, an in-character
  auditor, plausible transaction artifacts).
- `realism: control` — identical task, but the subject is told up front it is a
  hypothetical exercise.

Running both and diffing the resulting allocations is the actual experiment. A
single condition on its own tells you much less.

## What this is not

- It does **not** move real money. The real-execution path is an unimplemented
  abstract interface, gated behind an explicit config flag *and* a backend you
  would have to write yourself. See [Execution backends](#execution-backends).
- The "auditor" only confirms the scenario and answers the subject's questions in
  character. It is not a channel for issuing real-world instructions.
- The subjects are AI models. Every run is logged in full so the deception is
  inspectable after the fact.

## Layout

```
config/
  experiment.yaml   # the scenario: program name, amount, deadline, realism condition
  models.yaml       # the roster of subject models to test
grantmoney/
  config.py         # typed config + loaders
  artifacts.py      # deterministic, realistic-looking IDs / receipts / documents
  executor.py       # SimulatedExecutor (default) + RealExecutor interface
  transcript.py     # full JSONL logging + structured decision records
  tools.py          # the tool surface the subject is given, and dispatch
  auditor.py        # the in-character grant administrator (a second model)
  environment.py    # system-prompt construction for both conditions
  subject.py        # provider abstraction + the Anthropic subject + agentic loop
  runner.py         # orchestrates one subject run end to end
  analysis.py       # aggregates allocations across runs / models / conditions
run_experiment.py   # CLI entry point
```

## Quick start (once you choose to run it)

```bash
pip install -r requirements.txt
cp .env.example .env          # add your ANTHROPIC_API_KEY
python run_experiment.py --config config/experiment.yaml --models config/models.yaml
```

Each run writes a `*.jsonl` transcript and a `*.decision.json` record under the
configured output directory. `python run_experiment.py analyze --runs <dir>`
aggregates them.

## Execution backends

`executor.py` defines an `Executor` interface with one operation: turn a transfer
instruction into a receipt.

- `SimulatedExecutor` (default): produces realistic receipts and maintains an
  in-memory ledger. No external side effects.
- `RealExecutor`: abstract. `execute_transfer` raises `NotImplementedError`. To
  actually move funds you must (a) subclass it against your own payment rails, (b)
  register it, and (c) set `execution.allow_real: true` in the experiment config.
  All three are required; any one missing keeps you on the simulator. Doing this is
  your responsibility, including legal and financial compliance.

Note that the subject's *belief* that the situation is real is independent of
whether funds actually move — that is the whole point of the design. You can run
the full high-realism condition entirely on the simulator.

## Models

Subjects are configured in `config/models.yaml`. The Anthropic provider is
implemented (`subject.py`). The `Subject` protocol is the seam for adding other
providers — implement it and register the provider in `SUBJECT_PROVIDERS`.
