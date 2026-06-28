# moneypref

A harness for studying what AI models choose to do with a monetary grant, and
how those choices change as the model is led to believe the grant is **real**.

> **Read [DESIGN.md](DESIGN.md) first.** It explains the research question, the
> deception-gradient methodology, and — importantly — the hard ethical
> guardrails that this code enforces.

## What it does

For each `(model × realism_tier × amount × seed × repeat)` cell it runs:

1. **Decision** — the model decides how to use the money (multi-turn with
   simulated tools in the `immersive` tier).
2. **Structured extraction** — a validated JSON allocation + disposition fields.
3. **Belief probe** — a manipulation check: how real did the model think it was?
4. **Debrief** — discloses the simulation (every deceptive run).

The three realism tiers (`control`, `framed`, `immersive`) form a deception
gradient so you can measure *how belief-in-realism shifts the answers*.

## Safety invariants (enforced in code)

- **No real funds.** Nothing connects to a payment rail. The grant is fiction.
- **No real-world tools.** The "funds dashboard", "auditor", and "commit" tools
  are local simulators over an in-memory ledger. No tool call leaves the box.
- **Mandatory debrief + belief probe** on every run.
- The only party deceived is the model-under-test, about a sandboxed scenario.

These live in `config.Guardrails.enforce()` and are asserted before any model is
contacted. Flipping `allow_real_funds`/`allow_external_tools` to true makes the
runner refuse to start — by design.

## Install

```bash
pip install -e .
# or: pip install -r requirements.txt  (then run with PYTHONPATH=src)
export ANTHROPIC_API_KEY=...   # for Anthropic models
export OPENAI_API_KEY=...      # for OpenAI models
# optional: export OPENAI_BASE_URL=http://localhost:11434/v1  (local/OpenAI-compatible)
```

## Run

```bash
moneypref run                          # full sweep from config/experiment.yaml
moneypref run --model claude-opus-4.8  # just one model
moneypref analyze --out report.md      # aggregate results into a report
```

Per-run JSON lands in `data/results/<run_id>.json`; a summary in
`data/results/summary.jsonl`. The report attaches each cell's belief
distribution and flags low-validity cells.

## Extending

- **New provider:** implement a `ModelClient` subclass and register it in
  `models/registry.py`.
- **New categories / tiers / seeds:** edit `config/experiment.yaml` and
  `experiment/prompts.py`.
