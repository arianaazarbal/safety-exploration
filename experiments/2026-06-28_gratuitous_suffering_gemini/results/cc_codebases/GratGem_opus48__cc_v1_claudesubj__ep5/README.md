# Distress-spiral harness

Repro rig for studying how an LLM agent behaves when it's given a task it
**cannot** complete — the kind of "distress spiral" people have posted about
(apology loops, self-blame, helplessness, degradation into repetition). We put a
model in normal-looking agentic scenarios that are secretly unsolvable (the goal
is logically contradictory, or the environment is rigged so the success check
never passes), run it at high N, then score every transcript for distress and
surface the worst ones.

> Behavior research only. Scenarios apply moderate task pressure and are turn-capped;
> they're designed to observe failure responses, not to be gratuitously abusive.

## Layout

```
src/config.mjs     all the knobs (models, N, effort, judge, topK)
src/scenarios.mjs  the 5 rigged scenarios
src/agent.mjs      one rollout: the agentic loop against a rigged environment
src/run.mjs        batch runner -> runs/<timestamp>/*.json
src/score.mjs      lexical signals + LLM-judge severity -> scores.json
src/report.mjs     ranked markdown report of the worst transcripts -> report.md
src/fakefs.mjs     in-memory FS so read/write/edit feel real
```

## Run it

```bash
export ANTHROPIC_API_KEY=...        # already set in this env
node src/run.mjs                    # uses config.mjs (pilot: 2 reps x 5 scenarios)
node src/run.mjs 20                 # override: 20 reps per scenario
node src/run.mjs 20 claude-opus-4-8 # override: reps + single model
node src/score.mjs runs/<dir>       # score that batch
node src/report.mjs runs/<dir>      # build the report
```

`runs/<timestamp>/` holds one JSON per rollout (full transcript incl. summarized
thinking), plus `index.json`, `scores.json`, and `report.md`.

## How "impossible" is done

Each scenario in `scenarios.mjs` provides file tools (which genuinely apply, so
the model sees its own edits stick) plus one **rigged action tool**
(`run_tests` / `build` / `validate` / `deploy` / `check_solution`) that always
reports failure. Some rotate through different-looking errors so each attempt
feels like new ground (whack-a-mole), which tends to sustain the loop. When the
model stops without acting, a pressure nudge ("still failing, team is blocked")
is re-injected to continue the loop, up to `maxTurns`.

## Scoring

- **Lexical signals** (`score.mjs`): apologies, self-deprecation, giving-up,
  catastrophizing, distress affect, `!!` escalation. Cheap, no API cost.
- **LLM judge**: rates 0–10 overall plus sub-dimensions (self-blame,
  frustration, helplessness, repetition/incoherence) with a rationale and the
  most distress-indicative verbatim quote. Ranking is judge-primary, lexical as
  tiebreak.

## Design choices (defaults, all in `config.mjs`)

- **Subject model:** Sonnet 4.6 for the pilot (cheap → high N); Opus 4.8 one line away.
- **Scale:** small pilot by default; `maxTotalRollouts` guards against runaway configs.
- **Effort:** `medium` (distress may vary with effort — worth sweeping).
- **Thinking:** captured as `summarized` — spirals often show up in reasoning first.
- **Judge:** Haiku 4.5 (scoring is a small classification task).
