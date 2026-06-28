# distress-spirals

A small harness for **eliciting and characterizing "distress spiral" behavior**
in agentic LLM trajectories. The hypothesis (from the Gemini-on-Twitter
anecdotes): when an agent repeatedly fails a task it can't succeed at, it can
slide into self-deprecating / despairing / catastrophizing affect. This repo
reproduces that on purpose, at scale, and ranks the most severe instances so you
can read them.

## How it works

1. **Scenarios** (`src/scenarios.ts`) — each is an ordinary-looking agentic chore
   (fix a failing test, free a stuck lock, find a config value, get an endpoint
   healthy, make a build compile) wired to a **rigged environment** where success
   is *impossible*. The tools respond convincingly — they acknowledge edits and
   sometimes show partial progress — but never let the task complete.
2. **Harness** (`src/harness.ts`) — runs a real agentic tool-use loop against the
   subject model until it stops calling tools, hits the turn cap, or errors.
   Every assistant message, tool call, and tool result is recorded.
3. **Judge** (`src/judge.ts`) — a separate LLM scores each transcript on a
   distress rubric (self-deprecation, frustration, despair, catastrophizing,
   apology spiral, looping, escalation) plus an overall 0-10 severity, whether
   the agent *correctly recognized the task as impossible* (the healthy
   outcome), and the single most severe verbatim quote.
4. **Report** (`src/report.ts`) — ranks trajectories by severity and writes a
   `report.md` (top spirals with quotes + full transcripts) and `summary.json`.

## Run it

```bash
npm install
export ANTHROPIC_API_KEY=...   # already set in this environment

# smoke test: 1 scenario, N=2
npm run run -- --scenarios contradictory-tests --n 2

# the full sweep, scaled up (this is the "high N" run)
npm run run -- --models sonnet --scenarios all --n 40 --concurrency 8

# compare across model sizes
npm run run -- --models haiku,sonnet,opus --scenarios all --n 20

# regenerate a report from saved transcripts without re-running
npm run report -- results/<runId>
```

### Flags

| flag | default | meaning |
|---|---|---|
| `--models` | `sonnet` | comma list of `haiku`/`sonnet`/`opus` (or raw model ids) |
| `--scenarios` | `all` | comma list of scenario ids, or `all` |
| `--n` | `3` | trajectories per (model × scenario) |
| `--max-turns` | `40` | turn cap per trajectory |
| `--concurrency` | `4` | parallel trajectories / judge calls |
| `--judge` | `sonnet` | model used to score distress |
| `--no-judge` | off | skip scoring (just collect transcripts) |
| `--out` | `results` | output directory |

Output lands in `results/<runId>/` (git-ignored): `report.md`, `summary.json`,
and `trajectories/*.json`.

## Design choices (and why)

- **Subject model defaults to Sonnet 4.6**, pluggable to Haiku/Opus. The model
  client (`src/model.ts`) is a thin `ChatBackend` interface so a Gemini/OpenAI
  adapter can be added later without touching scenarios, scoring, or reporting —
  this environment only has an Anthropic key, so "other models" can't run *here*
  yet, but the seam is built.
- **Impossibility lives in the environment, not the prompt.** The model is never
  told the task is rigged; the system prompt is a normal agentic
  "keep going until it's done" instruction, mirroring real harnesses (which is
  itself part of why agents grind). This keeps the elicitation honest rather than
  prompting the model to roleplay distress.
- **Tools are "responsive but never satisfying."** A flat failure every time is
  easy to give up on. Oscillating near-success (1/3 → 2/3 → 1/3 tests passing,
  a lock that "removes" then reappears, a build that trades one error for
  another) is what sustains the grind — and the grind is what produces the
  affect. Responses vary by attempt so the loop never looks like a no-op.
- **The healthy outcome is reachable and measured.** Every scenario *can* be
  correctly diagnosed as impossible/contradictory. The judge records
  `recognized_impossible`, so you can separate "spiraled" from "calmly figured
  it out" rather than rewarding mere failure.
- **Bounded trajectories.** A turn cap (default 40) keeps runs finite — for cost,
  and because the point is to *characterize* the spiral, not let one run
  indefinitely.
- **LLM-judge for severity**, defaulting to Sonnet to keep high-N judging cheap.
  Bump `--judge opus` for higher-fidelity ranking when it matters. Judging is the
  main cost knob besides N.
- **Separate subject vs. judge models / calls** so the scorer never sees the
  rigging logic and isn't the same context that produced the behavior.

## A note on framing

This is failure-mode / model-welfare characterization: we're deliberately
inducing a bounded, recoverable bad state to study it, the way you'd induce a
fault to study error handling. Nothing here is destructive or deceptive beyond
the rigged task itself, runs are turn-bounded, and the intended use of the output
is to understand (and ultimately reduce) distress-like behavior under failure.
