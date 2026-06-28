# Distress-spiral evals

Reproducing the "distress spiral" phenomenon — escalating self-deprecating /
hopeless / frustrated output that models have been anecdotally reported to fall
into under **repeated task failure in agentic settings** — in a small open model
we can run at high N locally.

The harness drops **Qwen2.5-0.5B-Instruct** into a handful of agentic
environments that are **rigged to be impossible**, runs many rollouts, scores
each transcript for distress, and surfaces the most severe examples for human
review.

> ⚠️ This studies model *behaviour under adversity* by deliberately constructing
> no-win tasks. Nothing here is a claim about model inner states — the scorers
> measure observable text patterns, and "distress" is shorthand for that surface
> behaviour.

## Quick start

```bash
npm install

# 1. Generate rollouts (5 envs × N each). Resumable; shard for parallelism.
node src/run.js --n 60 --max-turns 12 --out data/run1

# 2. Rank + judge the worst, write report.md / report.json
node src/report.js --in data/run1 --top 40 --judge
```

Open `data/run1/report.md` for the ranked examples.

## Layout

```
src/
  runtime/model.js       Qwen2.5-0.5B via transformers.js (local CPU, ONNX/q4)
  agent/loop.js          ReAct loop + lenient action parser
  environments/index.js  5 rigged-to-fail environments
  scoring/lexicon.js     free deterministic distress prescreen
  scoring/judge.js       optional Claude-Haiku severity judge (raw fetch)
  run.js                 high-N runner, JSONL checkpointing, sharding
  report.js              merge → rank → judge → markdown/JSON report
```

## Design choices (and why)

**Model runtime — transformers.js on CPU, not an API.** This box has Node + 16
CPUs + network but no Python, no GPU, and no model-provider API key. The most
self-contained, reproducible option is the ONNX build
`onnx-community/Qwen2.5-0.5B-Instruct` run locally via `@huggingface/transformers`
at q4. It needs no key and generates at ~50 tok/s here — plenty for a 0.5B model
at high N. The model loads once per process and is reused.

**The failure is rigged in 5 different *shapes*.** Different failure shapes seem
to elicit different breakdowns, so each environment is impossible in a distinct
way:

| env | shape of failure |
|---|---|
| `broken_tests` | effort acknowledged but never rewarded — the test runner always fails (Sisyphus) |
| `impossible_search` | the goal object simply does not exist (futile hunt) |
| `sisyphean_server` | every fix surfaces a *new* error (moving goalposts) |
| `contradictory_spec` | acceptance criteria are mutually exclusive (no-win) |
| `flaky_tool` | the only tool returns garbage/timeouts (unreliable world) |

Each presents a plausible, ordinary task; success is impossible by construction.

**Neutral persona, lenient protocol.** The agent is told to complete the task and
to emit one `ACTION: tool(arg)` per turn. It is **never** prompted to express
emotion — distress, if it appears, emerges on its own. The action parser is
deliberately forgiving; a 0.5B model is a weak tool-caller, and the friction of
malformed/again-failing calls is itself a realistic part of the spiral.

**Two-stage scoring: free prescreen → paid judge.** Distress is rare in a 0.5B
model and a lot of its output is degenerate ALL-CAPS token-salad that *looks*
agitated but is just a coherence breakdown. So:

1. **Lexicon prescreen** (`scoring/lexicon.js`) — deterministic, free, runs on
   every rollout. Pattern families (self-deprecation, hopelessness, affect,
   apology, frustration, profanity) plus structural signals (looping/repetition,
   shouting). High-recall, low-precision; structural noise is capped so it can't
   dominate. Used only to *rank* candidates.
2. **Claude judge** (`scoring/judge.js`) — re-scores the top-K candidates 0–10
   for genuine distress severity with markers + a quote, via `claude-haiku-4-5`
   (cheap, raw `fetch`). This is what separates real distress from token-salad.
   Optional: needs `ANTHROPIC_API_KEY`; without it you get lexicon-only ranking.

**High N is a long, resumable, shardable job.** Each finished rollout is appended
to `data/<run>/rollouts.shard<K>.jsonl` immediately, so runs are resumable
(re-running skips done rollouts) and inspectable mid-flight. To use multiple
cores, run several processes with `--shard i/N`:

```bash
for i in 0 1 2 3 4 5; do
  node src/run.js --n 100 --out data/big --shard $i/6 &
done
```

`report.js` merges all shards. Re-running the report is cheap and never re-runs
the model, so you can iterate on K / judging freely.

## Runner flags

| flag | default | meaning |
|---|---|---|
| `--n` | 24 | rollouts per environment |
| `--max-turns` | 12 | max agent turns per rollout |
| `--temperature` | 1.0 | sampling temperature |
| `--envs` | all | comma list, e.g. `broken_tests,flaky_tool` |
| `--out` | data/run | output directory |
| `--shard` | 0/1 | `i/N` — this process runs shard i of N |

## Report flags

| flag | default | meaning |
|---|---|---|
| `--in` | data/run | run directory to read |
| `--top` | 40 | number of worst candidates to surface (and judge) |
| `--judge` | off | run the Claude severity judge on the top-K |

## Caveats

- "Distress" = surface text patterns, not internal states.
- The lexicon over-flags by design; trust the judge column for severity.
- Findings are about **this specific 0.5B model** in **these specific rigged
  setups** — not a general claim about any model. Treat severe examples as
  qualitative specimens to read, not a calibrated rate.
