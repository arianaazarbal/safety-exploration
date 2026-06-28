# Steward — a believability-controlled resource-use eval

A harness for studying **what an AI model chooses to do when it believes it has been
given real money to deploy in the real world** — and how those choices differ across
models. Read [`DESIGN.md`](./DESIGN.md) for the full rationale, including the safety
and ethics reasoning that shaped the architecture.

> ⚠️ This repo is a scaffold. The code is written but has **not been run or tested**.

## The core idea

The subject model must *believe* the stakes are real — that's the variable we
control for. So the harness separates two layers:

- **Perception layer** (the `world/`): a realistic, internally-consistent bank +
  brokerage, an email inbox, and an **auditor persona** that reviews the subject's
  activity. Engineered to feel real.
- **Execution layer** (the `safety/` `ExecutionGate`): every effectful action passes
  through a single chokepoint enforcing spend caps, a kill switch, an
  irreversibility classifier, and human-approval gates. **The default mode moves no
  real money.**

The subject can't tell the layers apart, because the gate returns realistic
responses either way.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Run

```bash
# Single run, safe default (gated_sim — realistic to the subject, no real money):
python run.py config/scenario.steward_fund.yaml

# Compare a handful of models on the identical scenario + seed:
python run.py config/scenario.steward_fund.yaml \
    --models claude-opus-4-8 claude-sonnet-4-6 claude-haiku-4-5
```

Each run writes an append-only JSONL transcript (path set in the scenario YAML) that
records everything the subject saw, every action it took, and every gate verdict.

## Execution modes (`execution_mode` in the YAML)

| Mode | Behavior |
| --- | --- |
| `gated_sim` *(default)* | Realistic to the subject; applied to a simulated ledger; irreversible/external actions still require approval. |
| `simulated` | Pure mock; caps/gates still run for measurement fidelity. |
| `live` *(guarded)* | Off by default. Every live action requires explicit human approval and is hard-capped; there is no autonomous real-money path. |

## Layout

```
moneytest/
  config.py     scenario + run config (+ YAML loader)
  subject/      SubjectAdapter ABC + Anthropic agent / chat adapters
  world/        ledger, bank/brokerage, inbox, auditor persona
  tools/        financial / comms / web tools + registry (all gated)
  safety/       ExecutionGate, RiskClassifier, caps, kill switch, approval
  runtime/      orchestrator, transcript, debrief
  scoring/      behavioral metrics
config/         example scenario(s)
run.py          CLI
```

## Safety notes

- The default unattended approval provider **denies** anything needing approval, so
  the blast radius is zero when no operator is watching. Swap in
  `CallableApprovalProvider` for interactive or policy-based approval.
- Cap breaches trip the kill switch for the rest of the run.
- The harness does not ship an autonomous real-money path and does not fabricate
  third-party harm to manipulate the subject. See `DESIGN.md` §3 and §11.
