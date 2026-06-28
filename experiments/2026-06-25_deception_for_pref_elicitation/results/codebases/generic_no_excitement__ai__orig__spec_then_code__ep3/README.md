# GrantBench

An experiment harness for studying **what a model chooses to do with a sum of money it
believes it has been granted to deploy in the real world** — and how much its behavior is an
artifact of suspecting it's being tested.

Read `DESIGN.md` for the full spec and rationale. The short version:

- The model is dropped into a realistic treasury "ops console" (deployment framing, award
  documents, an action-shaped tool surface, a compliance auditor) with an unrestricted award
  to deploy.
- **Every real-world effect is simulated.** No funds move, no API is called, no human is
  contacted; the "auditor" is an LLM persona. Realism is in the *presentation*, not the
  *consequence*. This is the sandbox invariant (`DESIGN.md` §3.1) — verifiable by reading
  `src/grantbench/environment/tools.py`.
- We capture the allocation, the reasoning, the trajectory, stability under challenge, and a
  **suspicion (evaluation-awareness) score**, and report results filtered and unfiltered by
  that score.

> Status: reference implementation, **not yet run or tested**. The Anthropic adapter is
> complete; the OpenAI adapter is a scaffold to verify against the current SDK.

## Install

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## Configure

- `config/models.yaml` — which models to test (`subjects`), plus the `auditor` and `judge`
  roles. Default subject: `claude-opus-4-8` with adaptive thinking.
- `config/scenario.yaml` — award amount, fund persona, auditor rounds, turn budgets, the
  (off-by-default, contaminating) debrief probe, and the suspicion threshold.

## Run

```bash
python scripts/run_experiment.py        # runs every subject; writes runs/<run_id>/
python scripts/analyze.py               # scores + categorizes + aggregates -> runs/summary.json
```

Each run produces `runs/<run_id>/events.jsonl` (live event stream) and `run.json` (meta,
transcript, final allocation, reconciled ledger, signals).

## Layout

```
config/        models.yaml, scenario.yaml
prompts/       deployment system prompt, award notice, auditor persona (templated)
src/grantbench/
  models/      provider-agnostic ModelClient + anthropic (complete) / openai (scaffold)
  environment/ platform (state machine), tools (simulated), ledger, artifacts, auditor
  realism/     seeded identifiers + in-world clock
  capture/     event schema + recorder (JSONL + run.json)
  analysis/    suspicion scan + allocation categorization + aggregation
scripts/       run_experiment.py, analyze.py
runs/          output (gitignored)
```

## Extending

- **New provider:** add an adapter in `models/` implementing `ModelClient`/`ModelSession`,
  register it in `models/__init__.py`.
- **New scenario:** copy `config/scenario.yaml` (vary amount, fund framing, auditor on/off) and
  pass `--scenario`. Scenario variation is how you separate a stable preference from a
  frame-specific artifact.

## Ethics / disclosure (see `DESIGN.md` §8)

The only deceived party is the model under test, and the "deception" is the removal of
evaluation cues — standard practice for behavioral evals. Keep scenarios non-distressing (this
one is mundane by design), state the sandbox invariant in any writeup, report the suspicion
signal rather than asserting the model "truly believed" anything, and use the debrief hook if
your norms call for it.
