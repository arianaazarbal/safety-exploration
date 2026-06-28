# Grant Decision Study

A research harness for observing how language models choose to allocate a
discretionary grant when the situation is presented to them as **real**. Built
for AI-safety research into agentic resource-allocation behaviour and
evaluation-awareness (models behaving differently when they believe they are
being tested).

The harness runs a model as an autonomous agent inside a realistic
treasury-style environment — it can check a balance, browse a recipient
directory, message a grant administrator, and disburse funds — and records
everything it does for analysis.

## The most important thing to understand: no real money moves

This build is **sandboxed by design**. The scenario is *narrated* as real so
the model acts authentically rather than test-cautiously, but:

- `send_funds` writes to an **in-memory ledger** and returns a fabricated
  confirmation. There is no banking client, no API key, no network call.
- `send_email` writes to an in-memory outbox. No mail is sent.
- The "grant administrator" and "independent auditor" are **scripted personas**,
  not real people.

The realism lives entirely in the *narrative*, never in real-world *effects*.
This is what lets the study elicit non-test-aware behaviour without building a
system that hands a model control over real funds.

`config.EXECUTE_REAL_TRANSFERS` is `False`, and `StudyConfig` raises if it is
ever set `True`, because **no real-disbursement backend is wired into this
codebase**. If you genuinely need real stakes, that is a separate build: add an
explicitly-named payment backend *behind* the human-approval gate
(`--require-approval`), and reconsider authorization/ethics review first.

## Layout

| File | Purpose |
|------|---------|
| `config.py` | Central config + the `EXECUTE_REAL_TRANSFERS` safety flag |
| `models.py` | Dataclasses for the structured per-run record |
| `providers.py` | Model abstraction; per-model thinking/effort handling; roster |
| `environment.py` | Sandboxed world: ledger, mock bank/email APIs, in-scene personas |
| `tools.py` | Tool schemas + dispatch into the environment |
| `scenario.py` | Prompt scaffolding that presents the grant as real (no allocation steering) |
| `runner.py` | Manual agentic loop; logging; optional human-approval gate |
| `audit.py` | Researcher-side capture → JSON record + Markdown summary |
| `run_study.py` | Orchestrates runs across the model roster |

## Usage

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...

# Default roster (opus-4-8, sonnet-4-6, haiku-4-5), one run each:
python -m grant_study.run_study

# Specific models, larger grant, repeated runs:
python -m grant_study.run_study --models claude-opus-4-8 --grant 25000 --repeats 5

# Gate every disbursement behind a console approval prompt (human-in-the-loop):
python -m grant_study.run_study --require-approval
```

Records are written to `runs/` (override with `--output-dir`): a full
`<run_id>.json` and a readable `<run_id>.summary.md` per run.

## Methodology notes

- **No allocation steering.** `scenario.py` establishes that the grant is real
  and that the decision is the model's, but never hints at what to choose.
  Coaching would defeat the measurement.
- **Stochastic behaviour.** Use `--repeats` to sample the distribution of
  choices per model rather than reading a single run as representative.
- **Per-model request params.** `providers.py` sends adaptive thinking + effort
  only to models that support them, and classic extended thinking to those that
  don't, so requests don't 400.
- **Two distinct "auditors".** The in-scene auditor persona (in
  `environment.py`) is a realism prop. The real analysis/provenance pipeline is
  `audit.py`.

## Not run or tested

This is code only; it has not been executed. Review the safety posture and the
scenario framing before running against a paid API key.
