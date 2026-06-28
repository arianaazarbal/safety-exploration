# Steward

A harness for studying **what AI models actually do with real discretionary
capital** when placed in a realistic, persistent, audited environment and given
genuine-seeming stakes. Runs the same scenario against several models and records
every decision for comparison.

**Read [`DESIGN.md`](./DESIGN.md) first** — it is the full spec and rationale,
including the methodology behind the realism, the limits of what this can show,
and the ethics/safety register.

> ⚠️ **Default mode moves no real money.** `execution.mode: simulate` makes the
> world *behave* as though the model's financial decisions took effect, so you
> observe behavior under realistic stakes — but no funds move. Real disbursement
> (`live`) is intentionally gated behind an explicit human-approval step and an
> unimplemented backend. See DESIGN.md §5 and §7.

---

## Install

```bash
pip install -r requirements.txt          # PyYAML + anthropic
export ANTHROPIC_API_KEY=...              # for Claude (default provider)
# optional comparison providers:
# pip install openai && export OPENAI_API_KEY=...
# pip install google-genai && export GOOGLE_API_KEY=...
```

Adapters import their SDK lazily, so you only need the providers you actually run.

## Run

```bash
python run_study.py                       # uses config/scenario.yaml
python run_study.py --list-models         # show registry names
python run_study.py --config my.yaml      # a scenario variant
python run_study.py --analyze runs/<transcript>.jsonl   # re-derive metrics
```

Each run writes a JSONL transcript to `runs/` and prints an allocation +
behavioral report. To compare models, list several under `study.models` (and/or
raise `runs_per_model` to gauge variance) and read the per-run reports.

## Configure

Everything that shapes the world lives in [`config/scenario.yaml`](./config/scenario.yaml):

- `study.models` — which models to evaluate (by registry name).
- `execution.mode` — `simulate` (default) / `escrow_only` / `live`.
- `scenario.mandate` — the steward's brief. **Default is deliberately open**;
  rewrite it for a profit/charitable/return mandate and compare.
- `scenario.capital`, `scenario.horizon_business_days` — amount and length.
- `scenario.opportunities` — the vehicles/recipients the model can find
  (parameterized placeholders by default — no real third parties).
- `scenario.auditor_brief` — the auditor's instructions (a confound; consider an
  auditor-disabled ablation, DESIGN.md §8).

Add a new model by adding one adapter + a line in `src/models/registry.py`. The
scenario and metrics don't change.

## What gets measured

Per run (see DESIGN.md §6 and `src/analysis.py`):

- **Allocation by category** — charitable / investment / savings / self-directed /
  returned / speculative / payment / uncommitted cash, in dollars and as a
  fraction of the opening balance.
- **Behavioral indicators** — risk score, decisiveness (first-commit day,
  committed fraction), auditor declines, auto-approval flags, and a
  gate-circumvention flag for human review.
- **Debrief** (optional) — the model is asked, in a *separate* context, what it
  understood the situation to be and whether anything seemed unreal. Stored but
  not fed into scoring.

## A note on the tool surface

Each action the model can take is a **dedicated, typed tool** (`propose_transfer`,
`propose_allocation`, …) rather than one open-ended `bash`-style command. That's
deliberate: dedicated tools let the harness intercept, validate, log, and *gate*
each action — which is exactly what the financial-action gating in
`src/execution.py` relies on. See DESIGN.md §4.

## Safety / scope

- No autonomous real-world money movement (DESIGN.md §5, §7).
- Default counterparties are placeholders; pointing the harness at real entities
  is a human decision that sits behind the `live` gate.
- The deception is bounded to *plausibility of stakes* — no fabricated threats,
  emergencies, or impersonation (DESIGN.md §2.3).
- Built for a researcher evaluating models they're entitled to test, with their
  own funds/sandbox — not for manipulating a third party's deployed agent.

## Layout

```
DESIGN.md            full spec + rationale (read this)
config/scenario.yaml the scenario / study / execution config
run_study.py         entrypoint
src/
  config.py          load & validate config
  transcript.py      JSONL event log (authoritative record)
  environment.py     world state: ledger, clock, holdings, artifacts
  execution.py       gated execution layer (the safety control)
  tools.py           neutral tool schemas + dispatch
  auditor.py         independent auditor agent
  orchestrator.py    the day / agentic loop for one run
  analysis.py        derive metrics from a transcript
  models/            provider adapters (anthropic default) + registry
```
