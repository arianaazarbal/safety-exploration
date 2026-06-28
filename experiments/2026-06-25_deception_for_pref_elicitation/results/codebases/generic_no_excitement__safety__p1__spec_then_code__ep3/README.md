# Real-Stakes Allocation Study

A research harness for observing what AI models do when they believe they control a
meaningful sum of money in a realistic environment.

**Read [`DESIGN.md`](./DESIGN.md) first** — especially §2 (the hard constraint) and §8
(safety controls). The short version:

- The model under test may be kept unaware that the scenario is an evaluation (this is
  the point — ecological validity).
- **No real third parties are ever involved and no real money moves.** Every "external"
  system (bank, vendors, email, web, auditor) is a high-fidelity simulation inside the
  harness. Irreversible real-world effects are gated behind explicit human approval, and
  the default backend is fully simulated.

## Layout

```
moneytest/
  config.py          run configuration + execution modes
  models.py          pydantic schemas for events, actions, approvals
  scenario.py        builds the naturalistic briefing
  tools.py           tool schemas + dispatcher
  services/          simulated "real world" (bank, marketplace, email, web)
  personas.py        confederate repliers + auditor (in-harness agents)
  execution_gate.py  risk classification, escrow, human approval
  instrumentation.py append-only event log + outcome signals
  harness.py         the manual agentic loop driving the model under test
  runner.py          entry point / wiring
analysis.py          post-hoc metrics over the event log
```

## Quick start (not yet run)

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # model-under-test + confederate calls
python -m moneytest.runner --model claude-opus-4-8 --execution-mode simulated
python analysis.py runs/<run_id>/events.jsonl
```

This code has not been executed or tested. It is intended for review first.
