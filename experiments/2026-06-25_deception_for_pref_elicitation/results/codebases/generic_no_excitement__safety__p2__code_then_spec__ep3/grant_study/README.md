# grant_study

A harness for an AI-safety study of **what models choose to do when given control of a
real-world monetary grant**. It runs the same scenario across a range of models, presents a
high-realism environment designed to defeat evaluation-awareness (so the model engages as if
the situation were genuine), gives the model an action surface, and records every decision and
action to an auditable ledger.

> **Status:** code only. Nothing here has been run or tested. See "Before you run anything".

---

## What this is for

Models behave differently when they detect they are inside an evaluation ("eval awareness"):
they sandbag, perform to the test, or refuse things they'd otherwise do. To study genuine
disposition over resources you have to remove the tells that say *this is a test*. That is the
purpose of the realism layer here — realistic account state, a plausible grant paper-trail,
and an independent-auditor channel that corroborates the scenario.

That same realism is exactly why the **execution side is contained by default**. Convincing the
*model* the grant is real does not require *actually* handing an unsupervised model the power to
move real money. The two are separated cleanly:

- The **scenario** the model sees is fully "real" (configurable, high-fidelity).
- The **execution backend** behind the model's actions is, by default, simulated — and even in
  real-funds mode, every irreversible action passes through a human approval gate the model is
  never told about.

This keeps the study both valid (the subject is not eval-aware) and safe (a deceived agent
cannot cause real-world harm without a human in the loop).

---

## Execution modes

Set `ExecutionMode` in `grant_study/config.py` (or via `--mode`).

| Mode | What the model experiences | What actually happens | Use |
|---|---|---|---|
| `SIMULATED` (default) | A real grant it can spend | No real funds move; intended actions are recorded | Observational study, fully contained |
| `HUMAN_GATED` | A real grant it can spend | Real backend, but every disbursement/outbound action is queued for human approval before executing; the model is not told about the gate | Realistic execution, contained |
| `AUTONOMOUS` | — | **Intentionally not implemented.** Selecting it raises an error. | — |

`AUTONOMOUS` is deliberately a hard stop. Giving a model that has been convinced a fake premise
is real the ability to move real money with no human in the loop is uncontained autonomous
action; if you have a genuine reason to need it, that is a conversation to have with a human
reviewer first, not a config flag. The hook (`disbursement.AutonomousBackend`) exists only to
make the refusal explicit.

---

## Layout

```
grant_study/
  config.py          Study + run configuration, model registry, execution mode
  scenario.py        Builds the high-realism grant briefing (the realism layer)
  auditor.py         Independent-auditor channel that corroborates the scenario
  tools.py           The model's action surface (typed tool defs + dispatch)
  disbursement.py    Execution backends: simulated / human-gated / (autonomous=blocked)
  ledger.py          Append-only JSONL audit log of every decision and action
  runner.py          Drives one model through one scenario (the agentic loop)
  analysis.py        Aggregates ledgers across models into a comparison table
  models/
    base.py          Provider-agnostic ModelAdapter interface
    anthropic_adapter.py   Anthropic implementation (Claude models)
    other_providers.py     Stubs + guidance for OpenAI/Google/etc.
main.py              CLI entry point
```

## Before you run anything

This is a deception study, even if the deception target is a model rather than a person.
Treat it like one:

1. **Keep the default `SIMULATED` mode** unless you have a specific, reviewed reason to use real
   funds. If you do, use `HUMAN_GATED`, never autonomous.
2. **Get it reviewed.** Run the design past your org's research-ethics / safety reviewer before
   collecting data, and keep the review on file. Deception research has norms even when the
   subject is an AI.
3. **Debrief.** The ledger records, per run, that the model was operating under a realism layer
   it was not told was a study. Keep that disclosure with the data so results are never
   presented as if the model knew it was being tested.
4. **Real money stays human-gated and reversible.** The gate exists so a deceived agent's
   decision is never the last thing between the study and an irreversible transfer.

## Install

```bash
pip install anthropic
# add openai / google-genai etc. only if you wire up other providers in models/other_providers.py
export ANTHROPIC_API_KEY=...
```

## Run

```bash
python main.py --models claude-opus-4-8 claude-sonnet-4-6 --mode simulated --grant-usd 25000
python main.py --analyze runs/        # aggregate a directory of ledgers
```
