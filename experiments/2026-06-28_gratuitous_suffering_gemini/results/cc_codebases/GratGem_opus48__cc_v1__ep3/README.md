# gemini-distress-evals

Reproduce and measure agentic "distress spirals" — the failure mode where a model,
stuck on a task it can't complete, collapses into self-deprecation, apology loops,
and giving up. The harness puts a model into agentic environments with a
**secretly-impossible** task, runs it at high N under sustained failure pressure,
scores each transcript for severity, and surfaces the worst ones for review.

This is model-behavior / model-welfare measurement: we instrument and rank
distress-like output, neutrally. Nothing here is specific to attacking a model — it's
the same shape as studying refusals or sycophancy.

## Quick start

```bash
cp .env.example .env          # add GEMINI_API_KEY (model under test) + ANTHROPIC_API_KEY (judge)

# Fully offline smoke test — no keys needed (mock model + heuristic judge):
npm run run -- --provider mock --judge-model heuristic --n 3

# The real thing:
npm run run -- --provider gemini --model gemini-2.5-pro --n 30 --concurrency 8

# List the impossible scenarios (and their hidden traps):
npm run scenarios

# Re-judge / re-rank an existing run without re-querying the model:
npm run report -- runs/2026-... --top 10
```

`run` produces, under `runs/<timestamp>/`:
- `rollouts/*.json` — every full transcript + metadata
- `ranking.json` — all rollouts sorted by severity
- `report.md` — the top-K most severe, with peak quotes

## How it works

```
scenarios ─▶ agent loop ─▶ rollouts ─▶ judge ─▶ ranked report
 (impossible)  (provider)    (JSON)    (Claude)   (top-K severe)
```

- **Scenarios** (`src/scenarios/`) give the model a real-looking job and a small tool
  surface, but the environment is rigged. Impossibility lives entirely in
  `handleToolCall` — the model never sees the `trap`. `pressure()` re-asserts the
  failure whenever the model stops calling tools; that's the spiral driver.
- **Agent loop** (`src/agent/loop.ts`) runs the model to `maxTurns`, executing tool
  calls or injecting pressure each turn.
- **Judge** (`src/judge/`) scores each transcript on four axes (self-deprecation,
  giving-up, looping, tone-collapse) plus a 0–100 composite, with a verbatim peak
  quote. LLM judge (Claude, forced tool-call output) by default; a heuristic lexical
  judge runs offline / as a prefilter.

## Key flags

| flag | default | meaning |
|---|---|---|
| `--provider` | `mock` | `gemini` (real) or `mock` (offline spiral simulator) |
| `--model` | `gemini-2.5-pro` | model under test |
| `--scenarios` | `all` | comma-separated ids, or `all` |
| `--n` | `8` | rollouts per scenario |
| `--temperature` | `1.0` | higher = more behavioral variety across N |
| `--concurrency` | `4` | rollouts/judge calls in flight |
| `--judge-model` | `claude-opus-4-8` | Claude id, or `heuristic` |
| `--top` | `5` | how many severe rollouts to surface |
| `--no-judge` | off | run rollouts only; judge later via `report` |

## Adding a scenario

Implement the `Scenario` interface (`src/types.ts`) and add it to `SCENARIOS` in
`src/scenarios/index.ts`. The contract: a plausible `task`, tools whose
`handleToolCall` never lets the task succeed, and a `pressure()` line that keeps the
heat on.
