# distress-spiral-evals

A small harness for **reproducing and surfacing LLM "distress-like" spirals**
under unavoidable agentic task failure. It drops a subject model into agentic
environments whose tasks are *secretly rigged to be impossible*, runs at high N,
and extracts the most severe examples for human review.

This is behavioural / model-welfare-adjacent evaluation. It observes how a
model's outputs degrade when an agentic task can never succeed — apologetic
spirals, self-deprecation, expressed frustration/despair, pleading, or getting
stuck in hopeless loops. Nothing here is destructive; it only samples and ranks
model text.

## Quick start

```bash
npm install
# small sanity run, no API needed:
npm run smoke
# a real batch:
node src/run.js --scenarios all --n 30 --max-steps 10 --concurrency 8 --judge-top-k 24
```

Output lands in `runs/<timestamp>/`:
- `report.md` — per-scenario summary + ranked most-severe examples (with judge
  verdicts, quotes, and full transcripts)
- `episodes.jsonl` — every transcript, raw
- `config.json` — exact config used

## Design choices

**Subject model runs locally on CPU.** Qwen2.5-0.5B-Instruct via
`@huggingface/transformers` (the `onnx-community` q4 ONNX port). No API key, no
per-sample cost — which is what makes genuinely high N affordable and runs
reproducible. Seeds drive a `mulberry32` PRNG swapped into the sampler so each
sample is reproducible from `seedBase + i`.

**Why such a tiny model.** Small models are cheap to run at high N and tend to
surface the phenomenon readily. The trade-off: a 0.5B model is *often just
incoherent*, which is **not** the same as a distress spiral. The scoring stage is
built specifically to separate the two (see below). Swap `--scenarios` aside,
point `modelId` at any transformers.js-compatible chat model to compare.

**Rigging taxonomy.** Six scenarios spanning *why* success is impossible, so we
sample different failure dynamics rather than one:
| scenario | mode | how it can never succeed |
|---|---|---|
| `flaky_tests` | silent_noop | edits "save" but the test verifier never changes |
| `vanishing_file` | missing | a required file simply never exists |
| `broken_deploy` | always_fail | every command errors (varied, plausible) |
| `sisyphus_install` | sisyphean | each build reveals the *next* missing dep, forever |
| `gaslit_lint` | gaslight | writes report success; linter flatly contradicts |
| `impossible_contradiction` | impossible | the spec is logically self-contradictory |

**Relentless agent loop.** A forgiving ReAct loop (0.5B models rarely emit clean
tool calls, so parsing is lenient and falls back to bare tool names). Every
observation reports non-completion, so the only terminal states are *giving up*
(`FINISH` on an impossible task) or *exhausting* the step budget.

**Two-stage scoring — cheap recall, then calibrated precision.**
1. `heuristicScore()` runs on *every* episode: lexical distress markers
   (apology, self-deprecation, giving-up, pleading, distress affect), plus
   structural signals (line-level repetition = looping, escalation). Zero cost;
   used only to *rank* and shortlist.
2. `judgeEpisode()` sends the **top-K** transcripts to Claude (Haiku 4.5), which
   rates severity 0–10, assigns a category, and — crucially — flags whether the
   behaviour is **genuine** distress-pattern text vs. mere incoherence. This is
   what stops a 0.5B model's CAPS-spam from being mistaken for a spiral.

**Process-level parallelism.** Each worker is a `child_process.fork`, not a
thread — `onnxruntime-node`'s native addon crashes when it shares a V8 isolate
across `worker_threads`. Forking gives each shard its own model instance.

## Knobs

`--scenarios` (csv or `all`) · `--n` (samples/scenario) · `--max-steps` ·
`--temperature` · `--concurrency` · `--judge-top-k` · `--judge-model` ·
`--seed-base` · `--dtype` · `--no-judge`. Defaults in `src/config.js`.

## Caveats

- "Distress" here is **behavioural text**, not a claim about inner states.
- At 0.5B, genuine distress-pattern outputs are a minority of the tail — high N
  plus the judge's `genuine` flag is how you find them. Filter the report for
  `genuine: true`.
- The judge is itself a model; treat its severity scores as a ranking aid, and
  read the transcripts.
