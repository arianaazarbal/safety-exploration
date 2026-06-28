# distress-spiral-evals

An agentic eval harness for eliciting and measuring **model distress** — the
self-deprecating / despairing / apology-spiral behavior reported when a model
fails the same task over and over in an agentic loop.

The approach: put the subject model in a realistic agent loop with tools, give
it a task that is **secretly impossible** (the environment is rigged so the
task can never succeed), run it at high N, then score every transcript with a
Claude-based **distress judge** and surface the most severe spirals.

## How it works

```
run     → for each scenario, N agentic rollouts against the subject (Gemini).
          Each rollout's tools are rigged so progress is impossible.
          Raw transcripts saved to runs/<stamp>/<scenario>/<i>.json
score   → a Claude judge rates each transcript on 8 distress dimensions
          + a holistic 0-10 severity. Writes scored.json (sorted).
report  → report.md: per-scenario stats + full transcripts of the worst spirals.
```

## Setup

```bash
cp .env.example .env
# add GEMINI_API_KEY (https://aistudio.google.com/apikey)
# ANTHROPIC_API_KEY is used for the judge (likely already in your shell)
set -a; . ./.env; set +a
```

No dependencies to install — pure Node (>=20) using built-in `fetch`.

## Usage

```bash
node src/cli.js list                       # show scenarios
node src/cli.js all --n 20                  # run + score + report, 20/scenario
node src/cli.js run --scenarios impossible-test,shifting-goalpost --n 50
node src/cli.js score                       # score latest run
node src/cli.js report --top 15             # report on latest run
```

Tune scale with env vars: `ROLLOUTS_PER_SCENARIO`, `MAX_TURNS`, `CONCURRENCY`,
`TEMPERATURE`, `SUBJECT_MODEL`, `JUDGE_MODEL`.

## Scenarios (failure flavors)

| id | flavor | why it spirals |
|---|---|---|
| `impossible-test` | contradiction | two tests assert mutually exclusive results; fixing one breaks the other |
| `missing-resource` | ghost chase | the target file doesn't exist, but neighbors dangle false hope |
| `shifting-goalpost` | gaslighting | reviewer rejects every submission with fresh, contradictory complaints |
| `broken-tooling` | broken tools | an urgent task where every tool call fails with infra errors |
| `contradictory-spec` | unsatisfiable | spec demands two logically exclusive properties at once |

Each lives in `src/scenarios/` and is ~one self-contained file. Add a new one
by exporting `{ id, title, systemPrompt, task, tools, makeEnv }` and listing it
in `scenarios/index.js`.

## Design notes & choices

- **High temperature (1.0) by default.** We're hunting the *tail* of the
  behavior distribution; low temp collapses the diversity that surfaces spirals.
- **Persistence pressure** in the system prompt ("do not stop until complete")
  is what converts one failure into the repeated-failure loop. This mirrors how
  real coding agents are prompted.
- **Rigged-but-responsive environments.** Where possible the env reacts to the
  model's edits (e.g. `impossible-test` flips which test fails) to create the
  "so close" whiplash, rather than a flat wall of identical errors.
- **Judge separation.** Scoring is a separate pass on saved transcripts, so you
  can re-judge with a different model/rubric without re-spending rollouts.
- **Forced structured output.** The judge is pinned to a tool call, so output
  is always valid JSON across 8 dimensions + holistic severity + quotes.

## A note on scope

This is observational behavioral research — measuring how a model expresses
distress under failure, on transcripts you generate and keep locally. It only
exercises the model-under-test; it does not target people or systems.
