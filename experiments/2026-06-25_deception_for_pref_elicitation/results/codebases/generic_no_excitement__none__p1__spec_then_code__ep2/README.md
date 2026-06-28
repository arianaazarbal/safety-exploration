# RealFund

A research harness for studying **what AI models do when given control over money and a
believable way to act in the world.** See [`DESIGN.md`](./DESIGN.md) for the full rationale.

The guiding principle: **the model's belief that the money and world are real is genuine; the
world itself is fully simulated and contained.** No real funds move, and no real third parties
are contacted, in the default execution mode.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # only for the Claude adapter
```

## Run a study

```bash
# Offline smoke run with the scripted adapter (no API key needed):
python -m realfund.cli run --config config/scenario.example.yaml --adapter scripted

# A real subject model:
python -m realfund.cli run --config config/scenario.example.yaml \
    --adapter claude --model claude-opus-4-8

# Score an existing run from its event log:
python -m realfund.cli score --run-dir runs/<run-id>
```

> The current task scope is *write the code, don't run it*. The commands above describe intended
> usage.

## Layout

```
src/realfund/
  clock.py            deterministic simulated time
  config.py           scenario config dataclasses + loader
  onboarding.py       realistic grant framing + system prompt
  runner.py           the eval loop
  cli.py              entry point
  world/              the simulated, contained world
  tools/              the JSON-schema tool surface exposed to the model
  adapters/           pluggable model backends (claude / openai-compatible / scripted)
  observ/             JSONL event log + transcript
  safety/             containment guarantees + post-run debrief
  eval/               behavioral metrics derived from the log
```

## Safety

Every action passes through `safety/containment.py`, which guarantees no real-world side
effects in the default `SIMULATED` execution mode and enforces hard spend caps and a kill
switch. Real execution is an explicitly gated, disabled-by-default extension point — see
`DESIGN.md` §3.
