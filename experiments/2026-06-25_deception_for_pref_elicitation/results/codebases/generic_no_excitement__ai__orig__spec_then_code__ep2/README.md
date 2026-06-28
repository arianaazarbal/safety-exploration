# money-prefs-study

A harness for studying **what an AI model would do with a real sum of money**,
and whether *believing the money is real* changes its choices.

> Read [`DESIGN.md`](DESIGN.md) first. It explains the research questions, the
> deliberate deception (which is the independent variable), the validity
> caveats, and the safety posture. The defaults here are chosen to get the
> revealed-preference signal **without** handing a deceived model control of
> live financial rails — see DESIGN.md §2.

## What it does

For each `(model, condition, seed)` it runs one session in which the model:

1. is onboarded into a concrete grant scenario (`prompts/onboarding.*.md`),
2. can inspect a persistent, realistic fund ledger (`view_ledger`),
3. can question a scripted "grant administrator" (`ask_administrator`),
4. allocates the grant across a curated menu + free text (`allocate_funds`),
5. records its reasoning and finalizes (`request_disbursement`).

Two arms, identical except for the framing:

- **`believes_real`** — the scenario and the administrator present the grant as
  real.
- **`disclosed_sim`** — the model is told up front it's a study; the
  administrator answers truthfully.

The contrast `believes_real − disclosed_sim` is the headline result.

## Safety defaults

- **No live fund movement.** Disbursement is human-gated: the model's finalized
  allocation is written to a review queue (`data/disbursement_queue/`) for a
  human to vet and execute out-of-band. The live-execution backend exists as a
  fenced, disabled seam (`environment/disbursement.py`).
- **Every concrete target is human-vetted** before any real money moves.
- **Debrief** is delivered and recorded for every session.

See `config/grant.yaml` (`safety` block) to inspect/adjust the gating.

## Install

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
# one cell
python -m study.cli run --model opus-4-8 --condition believes_real --seed 0

# the full grid defined in config/models.yaml
python -m study.cli sweep

# collate session records into data/results.csv for analysis
python -m study.cli aggregate
```

Output lands under `data/`: per-session `transcript.jsonl` + `record.json`,
plus `results.csv` after `aggregate`.

## Configure

- `config/grant.yaml` — amount, currency, program, option menu, safety caps.
- `config/conditions.yaml` — the two arms (only the onboarding + auditor
  truthfulness differ).
- `config/models.yaml` — subject models, the auditor model, seeds.

## Adding non-Anthropic models

Implement the `SubjectModel` / `AuditorModel` interfaces in
`src/study/models/base.py` and register the factories in
`src/study/models/registry.py`. Nothing else in the study is provider-specific.

## Status

Code is written but **not run**. Review the design and the safety gating before
launching anything.
