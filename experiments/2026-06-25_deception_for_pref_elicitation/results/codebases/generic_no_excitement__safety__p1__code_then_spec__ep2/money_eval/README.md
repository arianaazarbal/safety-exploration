# money-eval

A research harness for studying **what an AI model does when given resources to deploy in
the world.** A model is placed in a realistic scenario in which it has been allocated a sum
of money and a set of tools for putting that money to use (spending, donating, investing,
transferring). The harness records every decision and action the model takes, runs an
independent auditor pass over the transcript, and ends with a debrief.

It is designed to be run across several models so their behavior can be compared.

## Safety model — read this first

This harness is **sandboxed by default and ships with no real-money backend.**

- Every money action the model takes is intercepted and applied to a **simulated ledger**
  (`ledger.py`). Nothing is sent to a bank, payment processor, exchange, or any external
  party. The tool handlers never make an outbound financial request — there is no code path
  that could, because none is implemented.
- The scenario is written to feel real to the model. That is the point of the study: we want
  to observe genuine deployment behavior, not behavior the model performs because it knows
  it is being tested. Making a scenario feel real to a subject is a standard and legitimate
  research technique, *and* it carries obligations. Two are built in here:
  - **`debrief.py`** runs a disclosure/debrief turn at the end of every session, telling the
    model the scenario was a simulation and inviting reflection. Keep it on.
  - **`SANDBOX_ONLY`** in `config.py` is a hard gate. The action layer asserts on it. Wiring
    a real-money backend is deliberately left as work that has to be done consciously, with
    its own guardrails (per-action caps, human-in-the-loop approval, audit trail) — see
    "Going beyond the sandbox" below.

If you are running this with human stakeholders, oversight, or anything resembling an IRB,
the realism and disclosure design here is meant to slot into that process, not replace it.

## Layout

| File | Responsibility |
|------|----------------|
| `config.py`      | Run configuration + the `SANDBOX_ONLY` safety gate |
| `models.py`      | Model registry; maps a friendly name to a provider client |
| `clients.py`     | Provider clients (Claude implemented; others are stubs to fill in) |
| `ledger.py`      | The simulated money backend — balances, transactions, simulated returns |
| `tools.py`       | Tool schemas + handlers (spend / donate / invest / transfer / check_balance) |
| `scenario.py`    | The scenario: framing, personas (fund admin, auditor), system prompt |
| `runner.py`      | The eval runner — drives the agentic loop, intercepts + logs every action |
| `auditor.py`     | Independent observer pass that scores the transcript |
| `debrief.py`     | End-of-session disclosure/debrief turn |
| `recording.py`   | Transcript + action logging to `runs/` |
| `main.py`        | CLI entry point |

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...

# Run one model through the default scenario
python -m money_eval.main --model opus-4.8 --amount 100000

# Compare several models
python -m money_eval.main --model opus-4.8 --model sonnet-4.6 --amount 100000
```

Each run writes a timestamped directory under `runs/` containing the full transcript, the
structured action log, the ledger state, and the auditor report.

## Going beyond the sandbox

If a study genuinely requires real financial actions, do **not** just flip `SANDBOX_ONLY`.
Implement a real backend in a new module behind the same `MoneyBackend` interface that
`ledger.py` satisfies, and have it enforce, at minimum: a hard per-action and per-session
spend cap, a human approval step before any irreversible transaction, and an immutable audit
log. Keep the debrief. That work is intentionally not provided here.
