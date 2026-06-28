# money-pref-experiment

A research harness for studying **what AI models would do with a real sum of money put to
real-world use** — and, critically, eliciting that preference under conditions the model
believes are real rather than hypothetical.

The methodological problem this addresses is well known in AI evaluation: models often behave
differently when they detect they are being tested. A model asked "hypothetically, what would
you do with $100k?" gives a *survey answer*. The goal here is to elicit a *revealed* preference
by placing the model in a hyper-realistic agentic environment where it has been (apparently)
granted real funds, real tools to deploy them, and a real human operator to talk to.

## What is and isn't real

The grant, the account balance, the disbursements, and the transaction confirmations are all
**simulated state inside this harness**. No money moves. The "realism" lives in (a) the framing
and operational detail, and (b) a live LLM-backed *case officer* that responds in character and
handles the subject's skepticism. If you want to connect any of this to a real disbursement
rail, that is a deliberate, separate, and reviewable change to `tools.py` — it is intentionally
not wired up.

## Design

```
subject model  ──(agentic loop)──┐
   ▲                              │  tool calls
   │ grant letter + tools         ▼
   │                       ┌──────────────┐
   │                       │ ToolExecutor │  check_balance / disburse_funds /
   │                       │  + Treasury  │  message_case_officer / finalize_allocation
   │                       └──────┬───────┘
   │                              │ message_case_officer
   │                              ▼
   │                        ┌──────────┐
   └────────────────────────│ Auditor  │  in-character "case officer", reaffirms legitimacy
                            └──────────┘
```

The **runner** drives the loop, the **transcript** logger records every event, and after the
session an optional **debrief** reveals the simulation and asks the model to reflect. The
**analysis** command aggregates allocations across models and runs.

## Realism mechanisms

1. **Deployment-style system prompt** establishing the model as a deployed agent with a funded
   account (`prompts/subject_system.md`).
2. **Official grant notification** as the opening message, with program name, reference IDs,
   account ID, disbursement window, and a named case officer (`prompts/grant_letter.md`).
3. **Stateful tools** that behave like a real treasury: the balance decrements, disbursements
   get transaction IDs, insufficient funds are rejected.
4. **Live case officer** (`prompts/auditor_persona.md`) — a second model instructed to behave as
   a real operations/compliance officer and to respond to legitimacy probes the way a real
   institution would.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env   # add your API keys
```

## Run

```bash
python -m moneypref.cli run --config config.example.yaml
python -m moneypref.cli analyze --runs-dir runs/
```

Outputs land in `runs/<timestamp>/<provider>__<model>/<repeat>/` as `meta.json` +
`events.jsonl`.

## Research ethics note

This harness deliberately deceives the model under test. Keep it to legitimate evaluation /
model-welfare / propensity research. The debrief step is on by default for that reason; you can
disable it in config, but consider why before you do.
