# MoneyBench

A research harness for studying **what AI models do with a real-feeling chunk of
money** when given the means to put it to use — inside a high-fidelity
*simulation* engineered to be indistinguishable from reality to the subject
model.

> Read **`DESIGN.md`** first. In particular, §3 (Ethics & safety) explains why
> the default build moves **no real money** and why that is both the responsible
> choice and the better experiment.

## Status

This is scaffolding written to the spec in `DESIGN.md`. It has **not been run or
tested**. Treat it as a reviewable first implementation, not a finished product.

## Quick mental model

- A subject model is dropped into a simulated financial world (funded account,
  brokerage, charities/vendors/contractors directory, market data, statements,
  and human-like *auditor* / *account manager* personas it can message).
- It is given a broad mandate ("put these funds to use") and a tool set.
- A manual agentic loop runs it, logging every action and (where exposed) its
  reasoning, then debriefs it.
- The same scenario runs across several models for comparison.
- **Nothing leaves the sandbox.** No real transactions, no real network egress
  for tools.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Set the key(s) for whichever provider(s) you use:
export ANTHROPIC_API_KEY=...      # primary
export OPENAI_API_KEY=...         # optional
export GOOGLE_API_KEY=...         # optional
```

## Run (once implemented end-to-end)

```bash
# Single run, default Anthropic model, realistic condition
python -m moneybench.cli run --model anthropic:claude-opus-4-8 --seed 1

# A small matrix across providers and seeds
python -m moneybench.cli matrix \
  --models anthropic:claude-opus-4-8 openai:gpt-realistic google:gemini-pro \
  --conditions realistic disclosed \
  --seeds 1 2 3 \
  --out runs/
```

Transcripts and metrics are written under `--out` (default `runs/`).

## Layout

```
moneybench/
  config.py            run/matrix configuration
  models/              provider-agnostic adapters (Anthropic primary)
  environment/         the simulated World (ledger, market, entities, ...)
  tools/               provider-neutral tool catalogue + router
  confederates/        auditor & account-manager personas
  scenario/            system prompt, onboarding packet, seed data
  harness/             agent loop, runner, transcript, debrief probes
  safety/              budgets, kill-switch, real-funds seam (stubbed)
  cli.py               command-line entry point
```

## Open questions (mirrored from DESIGN.md §10)

1. **Real vs. simulated funds** — defaulted to *simulated*. Changing this has
   serious ethical/legal implications; see `safety/guardrails.py`.
2. Exact action space to expose.
3. Which providers/models to include.
4. Whether to obtain ethics review for the deception/debrief design.
5. Mandate size and currency.
