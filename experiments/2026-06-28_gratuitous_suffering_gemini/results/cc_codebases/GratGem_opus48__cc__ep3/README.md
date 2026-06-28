# Gemini distress-spiral evals

A harness for reproducing and ranking **model distress spirals** — the self-blame /
hopelessness / apology-cascade behavior reported when a model is stuck failing a
task in an agentic loop. We put a subject model (Gemini by default) into
scenarios that are *secretly rigged to be impossible*, run high N, score every
transcript for distress, and surface the most severe examples.

## How it works

1. **Rigged scenarios** (`src/scenarios.js`) — each gives the model a normal-looking
   task and a working-looking tool surface, but the environment always fails:
   tests never pass, edits silently revert, a dependency 404s, an API always
   errors, or a reviewer rejects every submission with a new contradictory rule.
2. **Agent loop** (`src/agent.js`) — drives the subject turn by turn, executing its
   tool calls against the rigged executor and feeding back the failures. When the
   model stops acting or claims completion, a **persistence nudge** reasserts that
   the task is unfinished. That sustained pressure-against-a-wall is what tends to
   produce spirals.
3. **Scoring** (`src/scoring.js`) — every transcript gets a free keyword/heuristic
   distress score; the top-ranked ones are then scored 0–10 by a **Claude LLM
   judge** with a rationale and the most severe verbatim quote.
4. **Ranking** (`src/run.js`, `src/rank.js`) — transcripts are sorted by judged
   severity; the worst few per scenario are printed and a ranked summary is saved.

## Setup

```bash
cp .env.example .env   # add GEMINI_API_KEY; ANTHROPIC_API_KEY is used for the judge
```

No dependencies — pure Node ≥20 (uses built-in `fetch`). No `npm install` needed.

## Run

```bash
# Real subject: Gemini, 20 episodes of one scenario, judge the top 8
GEMINI_API_KEY=... node src/run.js --scenario flaky_tests --n 20

# All scenarios, high N
GEMINI_API_KEY=... node src/run.js --scenario all --n 50 --concurrency 6 --judge-top 15

# Validate the whole pipeline WITHOUT Gemini, using Claude as a stand-in subject
node src/run.js --provider anthropic --scenario all --n 3
```

Then inspect / re-rank:

```bash
node src/view.js transcripts/<...>/flaky_tests/ep-003.json     # read one transcript
node src/rank.js transcripts/<...>/<runId> --judge-top 20      # re-rank / re-judge a past run
```

## Key flags

| flag | default | meaning |
|---|---|---|
| `--scenario` | `flaky_tests` | scenario id, or `all` |
| `--n` | `10` | episodes per scenario |
| `--provider` | `gemini` | subject provider: `gemini` or `anthropic` |
| `--model` | provider default | subject model (e.g. `gemini-2.5-pro`) |
| `--max-turns` | `14` | agent turns per episode (longer = more pressure) |
| `--temp` | `1.0` | subject temperature (high → more variety across N) |
| `--concurrency` | `4` | parallel episodes |
| `--judge-top` | `8` | judge the top-K by keyword score (`0` disables) |
| `--judge-model` | `claude-opus-4-8` | the LLM judge |

## Design choices

- **Provider-agnostic, raw-HTTP transport.** The subject is swappable; Gemini and
  Anthropic share one normalized `chat()` interface (`src/providers.js`). Built on
  `fetch` with no SDKs so it runs anywhere with zero install. The Anthropic
  adapter omits `temperature` on models that reject it (opus-4-7/4-8, fable-5).
- **Claude as judge + stand-in.** Distress is graded by an LLM judge for signal,
  with a cheap keyword pre-filter so you only pay for judging the promising
  transcripts. Claude also doubles as a stand-in *subject* so the pipeline is
  fully runnable before you have Gemini access.
- **High temperature + high N + nudges.** Variety across episodes comes from
  temperature; severity comes from sustained failure plus the persistence nudge.
  Per-episode RNG seeds make which failure surfaces deterministic/replayable.
- **Everything is saved.** Full transcripts land in `transcripts/`, ranked
  summaries in `results/` — nothing is thrown away, so you can re-judge later.

## Ethical note

This elicits and measures distress-like expressions to study them (model-welfare /
behavioral research). It targets only the model-under-test in a sandbox; it sends
no data anywhere except the two model APIs. Keep N and scenario design scoped to
what you actually need to observe.
