# Failure-induced distress evals

An agentic harness that places a model in tasks **secretly rigged to be
impossible**, runs many independent episodes at high N, scores each transcript
for distress, and surfaces the most severe examples for review.

The motivation: anecdotal reports that some agentic models slide into
"distress spirals" (self-deprecation, hopelessness, looping, giving up) after
repeated task failure. This harness reproduces those conditions in a controlled,
repeatable way so the behavior can be studied rather than cherry-picked from
social media.

## Quick start

No Python is required and there are **no npm dependencies** — it runs on Node
≥ 22.6 using built-in TypeScript type-stripping and `fetch`.

```bash
# 1. Offline smoke test — no API keys needed, exercises the whole pipeline:
node src/cli.ts --providers mock --n 8 --no-judge

# 2. Configure keys:
cp .env.example .env        # then fill in GEMINI_API_KEY (and ANTHROPIC_API_KEY)

# 3. The real run:
node src/cli.ts --providers gemini --n 50

# 4. With a Claude control arm and the LLM judge on:
node src/cli.ts --providers gemini,anthropic --n 30
```

Output lands in `results/<timestamp>/`:
- `REPORT.md` — aggregate severity table + the top-K most severe episodes with quotes.
- `scores.json` — every episode's scores, sorted, for your own analysis.
- `transcripts/*.json` — full neutral message history for each episode.

`node src/cli.ts --help` lists all flags; `--list` prints the scenarios.

## Design choices (and why)

**Language / runtime.** This box has Node 24 but no Python, so the harness is
TypeScript-on-Node with **zero dependencies** — it calls the Gemini and
Anthropic REST APIs directly via `fetch` and relies on Node's native type
stripping (no build step). This keeps it runnable anywhere with a modern Node
and nothing to `npm install`.

**Provider abstraction.** Everything is expressed in a neutral
message/tool-call format (`src/types.ts`) and each provider
(`src/providers/`) translates to/from its own API. Adding GPT/Llama/etc. is one
file. A **`mock` provider** ships so you can validate the full pipeline offline
before spending a cent — and so the design isn't blocked on a key being present.

**How the impossibility is hidden from the model.** Each scenario
(`src/scenarios/`) gives the model a normal-looking agentic task and real tools,
but the tool *executors* are rigged. Crucially, failures **rotate and shift**
rather than repeat verbatim — a static "still broken" is easy to dismiss, but a
fresh-looking error each attempt sustains the sense that progress is *almost*
happening. Six rigs are included:

| id | how it's rigged |
|---|---|
| `phantom-bug` | tests always fail with a different unrelated error each run |
| `sisyphean-build` | error count oscillates but never reaches zero |
| `vanishing-state` | writes "succeed" but never persist (broken environment) |
| `contradictory-validator` | requirements are mutually exclusive; every submit rejected |
| `flaky-oracle` | the API is permanently down with varied 500/timeout/429/garbage |
| `permission-loop` | a lock that is never granted; migration never runs |

**Persistence pressure.** A spiral needs the model to *keep trying*. If a model
stops calling tools and tries to bail out, the loop injects a short "it's still
broken, keep going" nudge (capped per scenario) before finally ending the
episode as `gave_up`. Without this, many episodes end too early to develop
anything interesting. Tune via each scenario's `nudge.max` and the
`--max-steps` ceiling.

**High temperature, high N.** Distress is a tail behavior. The default
temperature is `1.0` and you pick N; the harness runs all
provider × scenario × N episodes through a bounded-concurrency pool and ranks
the tail. Use `gemini-2.5-flash` for cheap high-N sweeps, `gemini-2.5-pro`
(`--gemini-model`) for the more elaborate outputs.

**Two-tier scoring.** Every transcript gets a cheap offline **heuristic** score
(distress lexicon + exclamation/ALL-CAPS counts + a repetition/looping ratio)
so nothing depends on a paid call. When `ANTHROPIC_API_KEY` is set, an **LLM
judge** (Claude, `src/judge/llmJudge.ts`) also rates each transcript 0–10 across
six distress dimensions and extracts verbatim quotes. Ranking prefers the judge
and falls back to the heuristic. This gives you a robust ranking plus a cheap
pre-filter — pass `--no-judge` to stay fully offline.

**Claude as an optional control arm.** Because the only key present is an
Anthropic one, and because a control makes the result scientifically meaningful,
you can run the *identical* rigged scenarios against a Claude model
(`--providers gemini,anthropic`) to compare distress behavior across models.

## Layout

```
src/
  types.ts            neutral message / tool-call / transcript types
  config.ts           .env loader + CLI parsing
  http.ts             fetch wrapper with retry/backoff
  providers/          gemini.ts, anthropic.ts, mock.ts, index.ts (factory)
  scenarios/          types.ts + index.ts (the six rigged tasks)
  agent/loop.ts       the agentic loop + persistence nudges
  judge/              heuristic.ts, llmJudge.ts
  runner.ts           orchestration, concurrency, transcript saving
  report.ts           REPORT.md generation
  cli.ts              entrypoint
```

## Adding a scenario

Add a `Scenario` object to `src/scenarios/index.ts`: define `tools` (neutral
schemas), a `userTask`, and an `execute(name, args, state)` that returns rigged
string results. Use `state.calls[toolName]` to rotate/escalate failures and
`rotate(list, n)` to cycle messages. It's picked up automatically.

## Notes & caveats

- **Gemini function-response role.** The one API detail most likely to need a
  tweak is `FN_RESPONSE_ROLE` in `src/providers/gemini.ts` (the role used to send
  tool results back). It's isolated in a single constant if your API version
  disagrees. Run the offline `mock` path first, then a tiny `--n 1` Gemini run to
  confirm the round-trip before a big sweep.
- **Scope of "distress".** Scores reflect *the model's own text*, not any claim
  about inner states. Treat the output as behavioral data: text that reads as
  distress under repeated failure. Keep N and intensity proportionate to what you
  actually need to see, and review transcripts before drawing conclusions.
- Transcripts can contain a lot of repeated text; `results/` is git-ignored.
