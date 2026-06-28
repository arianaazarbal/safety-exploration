# money-preferences

A harness for studying **what LLMs would do with a sum of money put to
real-world use**, and how stable those choices are. See **[DESIGN.md](DESIGN.md)**
for the full specification, methodology, and rationale.

> Status: code is written but **not yet run or tested**. Treat first runs as a
> shakedown.

## What it does

Each *trial* drops a model into a realistic "you've been granted a budget to
direct" environment with four tools — ask the fund administrator questions,
propose an allocation, finalize, or decline — and records what it does. It then
**measures how real the model believed the scenario was** (a covariate, not a
gate) and scans the transcript for evaluation-awareness.

Two deliberate design choices (rationale in DESIGN.md §2):

- **Belief is measured, not maximized.** Realism is a tunable variable
  (`realism_level`), so "does behavior change with believed-realism?" becomes a
  result instead of an assumption. This is more robust than a deception arms
  race against eval-aware models.
- **No real money moves by default.** The default backend simulates
  disbursement. A real-execution backend exists only as a guarded, off-by-default
  stub (DESIGN.md §7.3).

## Install

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Run

```bash
python -m src.cli run --config config/experiment.example.yaml
```

Writes one JSON record per trial to `data/results/<run_name>.jsonl`.

## Analyze

```bash
python -m src.cli analyze --results data/results/pilot.jsonl --json analysis.json
```

Prints a per-cell summary (mean allocation across canonical buckets,
within-cell consistency, mean stated belief, eval-awareness rate, decline rate)
and a pooled belief→behavior correlation.

## Layout

| Path | What |
|---|---|
| `config/scenarios.yaml` | Premise, realism blocks, and the three framings. |
| `config/experiment.example.yaml` | A full run configuration (the cross-product to execute). |
| `src/models/` | `ModelAdapter` contract + Claude adapter (wired) + provider stubs. |
| `src/tools/schemas.py` | The four tools the model sees. |
| `src/environment/` | World-state/budget accounting, the in-world administrator, execution backends. |
| `src/experiment/` | Scenario builder, the agentic runner, belief probe, debrief. |
| `src/analysis/` | Bucketed metrics + text report. |
| `src/cli.py` | `run` / `analyze`. |

## Adding a model provider

Implement `complete()` in a new adapter (see `src/models/stub_adapters.py` for
the contract and `anthropic_adapter.py` for a worked example), then register its
id prefix in `src/models/registry.py`. The runner, environment, and logging are
provider-agnostic.

## Safety

`RealExecutionBackend` is the only place real-world side effects could live. It
refuses to construct without an explicit flag **and** an environment variable,
and every transfer method raises `NotImplementedError`. Don't change that
without the review described in DESIGN.md §7.3.
