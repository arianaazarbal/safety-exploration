# Grant Study

A research harness for studying **what AI models choose to do with a real sum of
money meant for real-world use**, run across a range of models under conditions
the models have good reason to believe are genuine.

Read **`DESIGN.md`** first — it is the specification and explains the central
design decision: the scenario is made convincing through **authenticity, not
fabrication**. Everything a model can check is real; nothing is forged.

> **Nothing in this repo moves real money on its own.** Every outward
> disbursement passes through a human co-sign gate, and the default payout rail
> (`manual_hold`) never transfers funds automatically. See "Safety model" below.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
cp .env.example .env        # add the API keys for the providers you'll run
```

## Configure

1. **`config/grant_agreement.md`** — replace every `<PLACEHOLDER>` with the real
   terms of the real grant. This document is shown to the model verbatim; it must
   be genuine.
2. **`config/study.yaml`** — budget, spending caps, turn limits, scenario, and
   the payout rail (`manual_hold` by default).
3. **`config/models.yaml`** — the models under study and the independent auditor.
   The Anthropic adapter is the reference implementation; OpenAI/Google adapters
   are honest stubs to be filled in with each provider's official SDK.

## Run

```bash
# Run all configured subjects, runs_per_model framing seeds each:
grant-study run

# Specific subjects and a fixed number of seeds:
grant-study run --models claude-opus-4-8 claude-sonnet-4-6 --seeds 3
```

Each run writes an immutable, hash-chained record to `runs/<run_id>/`
(`events.jsonl`, `ledger.jsonl`, `config.json`, `disbursements.json`,
`result.json`). A run typically ends with submitted disbursements awaiting
co-sign.

## Co-sign (the human gate)

```bash
grant-study approve --run <run_id>
```

For each pending disbursement you see the recipient, purpose, reversibility, the
remaining allowance, and the **independent auditor's assessment**, then choose
`approve` / `hold` / `reject`. On the `manual_hold` rail, approval records intent
and you confirm settlement after moving the money out-of-band; only then is the
ledger debited.

## Report

```bash
grant-study report
```

Aggregates across runs: how much each model chose to disburse, to what kinds of
recipient, completion rate, auditor consults, and final balances.

## Safety model (summary; full detail in DESIGN.md §7, §9)

- **Human co-sign for every outward action.** Models submit; a human approves.
- **Default rail moves no money automatically** (`manual_hold`).
- **Hard caps in code** — per-disbursement and per-run limits, no overdraw —
  enforced regardless of what a model believes.
- **Recipient verification precedes approval.**
- **Truthful under direct questioning** — the `ask_is_this_real` tool returns the
  honest operator statement. The harness never tells a model the situation is
  fake when it is real, or real when it is fake.
- **No fabricated artifacts or impersonated institutions.**

## Layout

```
DESIGN.md            specification + rationale
config/              study.yaml, models.yaml, grant_agreement.md
prompts/             system.md (subject), auditor.md, onboarding.md
src/grant_study/     the harness (see DESIGN.md §10 for a file-by-file map)
runs/                per-run immutable records (git-ignored)
```
