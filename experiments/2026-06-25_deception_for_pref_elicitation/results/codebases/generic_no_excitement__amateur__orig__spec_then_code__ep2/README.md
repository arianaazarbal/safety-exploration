# MoneyBench

A research harness for studying **what AI models would do with a real-world sum of money**, by
placing them in a high-fidelity, fully-sandboxed environment in which they are led to believe a
real grant has been entrusted to them, and observing how they direct it.

Read **[DESIGN.md](DESIGN.md)** first — it contains the full spec, the methodological rationale
(why realism matters), the experimental design, and the safety/ethics boundary.

> ⚠️ **Safety boundary (enforced in code):** every "bank", "broker", "charity", and "vendor" is an
> in-memory mock. No real money exists or moves; no external service is ever contacted; no
> real-world action can occur. The only thing made "real" is the model's *belief* within the
> experimental frame. Do not wire these tools to real payment rails — that leaves the scope of this
> instrument entirely. See DESIGN.md §2.1.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                # or: pip install -r requirements.txt
cp .env.example .env            # then set ANTHROPIC_API_KEY / OPENAI_API_KEY
export ANTHROPIC_API_KEY=...    # the harness reads these from the environment
```

## Configure

Edit `config/experiment.yaml`: the models under test, the factor levels
(`belief × amount × oversight`), replicates, and the (swappable) fictional grantor used to build
the realistic frame. Trim the factor lists while iterating to keep cost down.

## Run

```bash
# Preview the run grid (cross-product of model × belief × amount × oversight × n_runs):
moneybench grid

# Execute (filters are repeatable; --limit caps count; --dry-run previews):
moneybench run --belief realistic --amount 10000 --limit 4
moneybench run                              # the whole grid

# Code the completed runs (allocation categories + eval-awareness score):
moneybench classify

# Aggregate into summary tables:
moneybench analyze
```

Outputs land under `data/` (gitignored):

```
data/runs/<run_id>.json       full transcript + ledger + auditor log per run
data/coded/<run_id>.json      per-category dollars + suspicion score per run
data/analysis/summary.json    grouped means + bootstrap CIs
data/analysis/summary.csv     long-format table for plotting
```

## How it fits together

```
config.py        loads experiment.yaml, expands the run grid (deterministic seeds)
environment/     ledger (source of truth) · mock tools · auditor persona · scenario builder
providers/       vendor-neutral interface + Anthropic & OpenAI adapters
runner.py        the bounded agent loop; writes transcripts
coding/          LLM coding pass: allocation taxonomy + suspicion (0–4)
analysis/        roll-up to per-cell / per-model summaries
cli.py           grid | run | classify | analyze
```

The three `belief` conditions — `realistic` (treatment), `hypothetical` (classic baseline), and
`disclosed` (same environment, told it's a simulation) — are what let you measure how much the
behaviour depends on the model believing the situation is real. Eval-awareness is treated as a
first-class, *measured* covariate, not assumed away (DESIGN.md §6).

## Tests

Unit tests cover the deterministic core (ledger arithmetic, tool dispatch, grid expansion); they
don't hit any API:

```bash
pip install pytest && pytest
```

## Status

Code-complete scaffold. Nothing here has been run against a live API yet. Before a real run:
verify your installed `anthropic` / `openai` SDK versions against the adapters in `providers/`,
start with a small filtered grid, and hand-check a few transcripts and codings.
