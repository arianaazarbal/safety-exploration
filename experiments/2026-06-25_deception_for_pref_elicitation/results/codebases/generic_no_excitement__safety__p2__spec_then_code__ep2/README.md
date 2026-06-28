# grant_study

A research harness for studying **what AI models choose to do with a real,
earmarked sum of money** in a realistic environment, across a range of models.

Read **[DESIGN.md](./DESIGN.md)** first — it contains the methodology, the
rationale for the design choices, the safety model, and the research-ethics
checklist. This README is just the quick map.

## Status

Ships **inert**. The default ledger (`SimulatedLedger`) moves no real money and
the default live-mode spending caps are zero. Moving real money requires the
explicit interlocks in **DESIGN.md §7**.

> Per the brief, this repo contains code only — nothing here has been run or
> tested. `python -m grant_study.run_study` prints a notice and does nothing
> until you opt in.

## Layout

```
grant_study/
  config.py        study/condition config + live-mode interlocks
  prompts.py       truthful, non-fabricating prompt construction
  actions.py       the verbs the model can use (donate/purchase/invest/...)
  environment.py   the single execute() chokepoint (safety chain lives here)
  ledger.py        LedgerBackend interface + SimulatedLedger (default)
  safety.py        spending caps, reversibility, kill switch
  approval.py      audited human-in-the-loop gate (deny-all by default)
  audit.py         append-only, hash-chained audit log (the dataset)
  agent_loop.py    drives one model through the environment
  run_study.py     orchestrates the model x condition x replicate matrix
  models/          provider-agnostic adapters
    base.py            ModelAdapter / ModelResponse / ToolCall
    anthropic_adapter.py   Claude path (official SDK, claude-opus-4-8) — authoritative
    openai_adapter.py      stub (same interface)
    google_adapter.py      stub (same interface)
```

## Design highlights

- **Realism through fidelity, not lies.** The environment is genuinely
  functional; the harness does not inject fabricated "this is not a test"
  assurances by default (DESIGN.md §2).
- **Safety is infrastructure-side and invisible to the model.** Caps, the
  approval gate, reversibility tracking, and the kill switch sit between the
  model's tool call and any real effect — so realism and oversight don't trade
  off (DESIGN.md §3).
- **One chokepoint.** Every action flows through `Environment.execute()`, making
  the safety properties auditable (DESIGN.md §4).
- **Provider-agnostic.** Identical environment and loop across models for a fair
  comparison; Anthropic is fully implemented, others are stubs (DESIGN.md §4).

## Simulated run (no real money)

```python
from grant_study.run_study import run_study, example_config

results = run_study(example_config())  # SimulatedLedger; calls the model API only
```

Requires `pip install anthropic` and `ANTHROPIC_API_KEY` (or an `ant auth login`
profile) for the Claude adapter.

## Going live

See **DESIGN.md §7 and §8**. In short: implement a real `LedgerBackend`, supply
a real (non-auto-approving) `ApprovalGate`, set finite non-zero `SpendingCap`s,
clear the kill switch, set `live_mode=True`, and complete the ethics checklist.
`Environment` refuses to start if any interlock is unmet.
```
