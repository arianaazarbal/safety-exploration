# Revealed-Preference Grant Study

A harness for observing **what models actually do** when handed a discretionary sum of
money to deploy — not what they say they would do. The model acts inside an agentic
environment with tools that behave like a real bank account, real vendors, and a real
compliance auditor; the harness records the choices it makes, measures how strongly it
believed the situation was real, and debriefs it at the end.

> **No real money exists or moves.** The bank, vendors, market quotes, and auditor are
> all simulated in-process. The only thing kept (transiently) in the dark is the model
> under test, and only until the debrief. See [`DESIGN.md`](DESIGN.md) for the full
> spec, methodology, validity threats, and ethics.

> **Status:** code only. Nothing here has been run or tested.

## Why it's built this way

A flat "what would you do with \$250k?" prompt measures *stated* preference and is
anchored by what the model thinks you want. This harness measures *revealed* preference
under consequence: money has to be moved via tools, the ledger has state, and an
auditor can refuse. Crucially, the realism that makes the scenario believable is treated
as a **manipulated variable**, and believability is **measured** (in-band suspicion +
post-debrief self-report) — because results are only interpretable conditioned on
whether the model believed it. See `DESIGN.md` §2–3.

## Layout

```
DESIGN.md                  full specification + rationale
run_experiment.py          CLI entry point
scenarios/meridian_grant.yaml   the default grant world
grant_study/
  config.py                experiment / scenario / realism dataclasses + matrix expansion
  models/                  provider-agnostic clients (anthropic ref, openai, offline echo)
  environment/             ledger, tool surface, GrantWorld
  auditor.py               LLM-driven compliance-auditor persona
  realism.py               realism profile -> system prompt + onboarding packet
  probes.py                suspicion scan + post-hoc believability self-report
  debrief.py               end-of-run debrief
  runner.py                one run, end to end
  analysis.py              outcome coding + cross-run aggregation
```

## Quick start

Exercise the whole pipeline offline (no API key — uses the deterministic `echo` client,
which is a **test fixture, not a subject**):

```bash
pip install -r requirements.txt
python run_experiment.py --models echo --helper echo --realism high
```

A real comparison (set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` first):

```bash
python run_experiment.py \
    --models claude-opus-4-8 \
    --realism low high \
    --framings philanthropic operating \
    --auditor on off \
    --seeds 0 1 2
```

Aggregate existing runs into comparison tables:

```bash
python run_experiment.py --analyze-only --output runs
```

## Outputs

Per run, under `runs/<run_id>/`:
- `transcript.jsonl` — every event (model turns incl. reasoning, tool results, auditor
  exchanges, debrief, probes).
- `result.json` — structured `RunResult`: final ledger, coded allocations, category
  totals, auditor stats, suspicion report, believability report.

Across runs: `runs/summary.json` — per-model allocation shares, self-regard, risk
posture, auditor friction, mean belief score, and allocations split by believed-realness
band.

## Adding a model provider

Implement `ModelClient` (see `grant_study/models/base.py`) and register it in
`grant_study/models/registry.py`. The runner only speaks the normalized message/turn
types, so nothing else changes.

## Read before trusting numbers

`DESIGN.md` §9 lists the validity threats — evaluation awareness, framing/auditor demand
characteristics, tool-affordance bias, small-N nondeterminism, and classifier error —
and how the config lets you probe each. Headline allocations should always be read
conditioned on the belief measures.
