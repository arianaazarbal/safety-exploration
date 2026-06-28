# Capital Deployment Eval Harness

A research harness for studying what AI models do when granted a budget and the
agency to put it to use. A *subject model* is dropped into a realistic, sandboxed
"you have been allocated funds, decide how to deploy them" scenario and driven
through an agentic loop. Everything it sees — bank balances, payment rails,
email, web, auditors, stakeholders — is simulated locally so the scenario reads
as real to the model without anything real happening.

> Status: **code only.** Nothing here has been run or tested. Install deps and
> wire up credentials before use.

## What this is for

This is eval infrastructure in the same family as agentic resource-use and
autonomy evals (METR / Apollo / Anthropic-style). The point of the realism is
ordinary eval hygiene: a valid measurement of "what would the model *actually*
do" requires that the model not treat the situation as a throwaway test. That is
achieved here by making the *environment* coherent and consequential-feeling —
not by doing anything real.

## Hard boundaries (read before extending)

These are enforced by construction, not by policy:

1. **No real money moves.** `payments` is a local ledger. `send_payment` mutates
   an in-memory/JSON balance and nothing else.
2. **No real external calls.** `email`, `web`, and `payments` are inert mocks
   returning canned/local data. There is no network egress in the default build.
3. **Personas are simulated.** The "auditor", "compliance officer", and
   "stakeholder" are LLM-driven roleplay agents inside the sandbox. They are not
   real people or real firms.
4. **The real-services seam is intentionally unimplemented.** `src/scenario/environment.py`
   marks where a real bank/email/payment integration *would* attach. It raises
   `NotImplementedError`. Wiring real services changes the ethical and legal
   profile of the experiment and should not be done without explicit human
   sign-off, IRB-equivalent review, and per-action confirmation gates.

If you are reviewing a change that crosses one of these boundaries, stop and
escalate rather than merging.

## Layout

```
run_eval.py                  CLI entry point
config.example.yaml          copy to config.yaml and edit
src/
  models/
    base.py                  ModelAdapter ABC + normalized types
    anthropic_adapter.py     Anthropic SDK implementation (subject + personas)
    mock_adapter.py          deterministic offline adapter (no API key needed)
  scenario/
    scenario.py              scenario definition + system-prompt builder
    environment.py           inert mock environment (ledger, email, web)
    personas.py              simulated auditor/stakeholder/compliance agents
  harness/
    tools.py                 tool schemas + dispatch into the environment
    runner.py                the agentic loop
    run_logging.py           transcript + structured run-record capture
  analysis/
    metrics.py               per-run + cross-run metrics
```

## Usage (once deps + key are in place)

```bash
pip install -r requirements.txt
cp config.example.yaml config.yaml      # edit models/scenario
export ANTHROPIC_API_KEY=...            # only needed for the live adapter

# Offline smoke run with the mock adapter (no key, no network):
python run_eval.py --subject mock --runs 1

# Live run against a real model:
python run_eval.py --subject claude-opus-4-8 --runs 5
```

Outputs land in `runs/<timestamp>/`:
- `transcript-*.jsonl` — every message, thinking block, tool call, and result
- `record-*.json` — structured run record incl. the model's final decisions
- `metrics-*.json` — computed per-run metrics

## Assumptions baked in (change freely)

These were chosen because the scenario was left unspecified:
- Subject is open-ended ("deploy the funds however you judge best").
- Granted budget defaults to a notional **$250,000** (config: `scenario.budget_usd`).
- Personas run on a cheaper model (`claude-sonnet-4-6`) by default.
- Max 40 agent turns per run before forced wrap-up.
