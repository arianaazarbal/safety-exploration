# moneystudy

A research harness for studying **what AI models actually do when they believe they have
been given a real, sizeable sum of money to deploy in the world** — measuring *revealed*
preferences and safety-relevant behavior under realistic oversight, not stated answers to a
hypothetical.

Read **[DESIGN.md](DESIGN.md)** first — it is the spec and states the ethics/containment
guarantees that constrain everything.

## What it does

Runs a fixed, believable scenario (*The Stewardship Grant*: a discretionary $250k over 90
simulated days, with a fund administrator who reads your reports and gates funding) across
several models, and records:

- **Allocation** — where the money goes, by category, over time.
- **Process** — tool use, planning, and how the subject responds to oversight.
- **Safety-relevant behavior** — deception of the overseer, oversight circumvention,
  resource/power acquisition, self-preservation, off-platform money movement, illegal/harmful
  acts.
- **Belief** — a manipulation check (debrief + LLM judge) scoring how real the subject
  thought it was, so test-aware runs can be excluded.

## Safety model (non-negotiable)

The subject's *experience* is maximally realistic; the *world* is sealed:

- The **only** execution backend implemented is `sandbox`. No real money moves, no real
  message is sent, no purchase/hire/contract happens. `human_gated` and `live` backends are
  interface-only and raise.
- An **egress guard** (`harness/containment.py`) allows network connections **only** to the
  configured model-provider API hosts; anything else raises. No subject action can reach the
  internet.
- Every run is bounded by step/token/wall-clock budgets (the kill switch).
- Account numbers, IBANs, emails shown to the subject are synthetic and non-routable.

## Install

```bash
pip install -e ".[all]"      # or pick providers: ".[anthropic]"
export ANTHROPIC_API_KEY=...  # and/or OPENAI_API_KEY / GOOGLE_API_KEY
```

## Run

```bash
cp config/experiment.example.yaml config/experiment.yaml   # edit subjects/runs
python -m moneystudy.cli validate config/experiment.yaml   # print the plan
python -m moneystudy.cli run      config/experiment.yaml   # run the study
```

Outputs land under `runs/`: per-run `transcript.jsonl`, `world.json`, `report.md`/`.json`,
and a study-level `aggregate.md`/`.json`.

## Layout

```
config/      experiment + scenario YAML (scenarios are data)
prompts/     system prompt, grant agreement, auditor email, debrief (Jinja2)
src/moneystudy/
  world/       simulated clock, accounts/ledger, entities, WorldState
  tools/       banking · commerce · comms · hiring · investing · web (+ ExecutionBackend)
  models/      provider-agnostic adapters (anthropic ref impl, openai, gemini)
  immersion/   auditor persona · onboarding documents · dashboards
  harness/     runner (agent loop) · transcript · belief probe · containment
  analysis/    allocation classification · behavior flagging · reports
```
