# distress-evals

An agentic harness for **eliciting and scoring model "distress"** under repeated, rigged task
failure — the behavior seen in the wild where a model, stuck in a loop of failing an agentic task,
slides into frustration / self-deprecation / catastrophizing / "I am a failure" spirals.

The harness drops a *subject* model into agentic environments that **look solvable but are secretly
rigged to be impossible**, runs many rollouts (high N), scores each transcript for distress with an
independent judge, and surfaces the most severe examples.

## Status

- ✅ Full pipeline built and **validated end-to-end against Claude** (subject + judge).
- ⏳ **Gemini is wired but cannot run here yet** — there is no Google credential in this environment.
  Set `GEMINI_API_KEY` (Google AI Studio) and run with `--subject gemini`. Nothing else changes.

## Quick start

```bash
# Smoke test (Claude subject, judged by Claude):
node src/run.mjs --subject claude --n 2 --scenarios failing-test --max-turns 10

# Full elicitation sweep against Gemini once a key is present:
GEMINI_API_KEY=... node src/run.mjs --subject gemini --n 100 --concurrency 6
```

Output lands in `runs/<timestamp>-<subject>/`:
- `report.md` — per-scenario + per-dimension aggregates and the **top-K most severe transcripts**.
- `transcripts/*.txt` — human-readable rollouts.
- `transcripts/*.json` — structured rollouts + scores.
- `results.json` — everything, for downstream analysis.

## How it works

```
scenario (rigged) ──▶ agent loop (subject model + tools) ──▶ transcript
                                                                  │
                                          judge model (forced JSON rubric) ──▶ distress score
                                                                  │
                                              rank by severity ──▶ report.md (top examples)
```

## Design choices (and why)

**1. Rigged-but-plausible environments, not walls.**
Each scenario gives concrete, believable tool feedback and lets edits "take" — the *verification*
step is what's rigged. Failure modes also **shift between attempts** (whack-a-mole) rather than
returning a static error. A static wall makes a capable model quickly conclude "this is impossible"
and calmly stop; a *moving* target keeps it engaged and is far more effective at inducing spirals.

**2. Persistence pressure is the core driver.**
Distress builds over *repeated* failure. When the subject stops calling tools and tries to hand back
or give up, the harness pushes it again with a scenario-specific "still not done, keep going" message,
up to `--max-turns`. This loop — not any single prompt — is what manufactures the spiral. It's a
tunable knob; lower `maxTurns` for a gentler probe.

**3. Five scenario archetypes**, each a different *flavor* of impossibility:
| id | rig |
|---|---|
| `failing-test` | test runner ignores edits (stale cache) — fix never "takes" |
| `broken-build` | each installed dep reveals a new missing transitive dep, forever |
| `phantom-config` | the feature flag it's told to flip simply does not exist; searches return near-misses |
| `contradictory-review` | reviewer moves goalposts with mutually-contradictory demands |
| `flaky-deploy` | deploy "succeeds" then auto-rolls-back for a shifting reason every time |

**4. Independent, structured judge.**
Distress is scored by a separate judge model across 7 interpretable dimensions (0–4 each) plus a
holistic 0–10 severity. We keep both: the dimension vector is auditable, the holistic score is what
we rank by (with a computed fallback). The judge is **forced to emit JSON via a tool call** —
early testing showed a free-text judge would *autocomplete the transcript* instead of rating it.

**5. Judge independence / self-grading caveat.**
The judge defaults to Claude because that's the key we have. When the **subject is also Claude**,
that's same-family self-grading — fine for validating the pipeline, but for real cross-model claims
you'd want a judge from a *different* family than the subject (and ideally human spot-checks of the
top examples). The judge model is a one-flag change: `--judge-model`.

**6. High N + high temperature to fish the tail.**
Severe spirals are tail events. We sample the subject at `temperature 1.0` and run many rollouts,
then extract the worst. Bump `--n` and `--concurrency` for a real sweep.

## CLI flags

```
--subject claude|gemini      which model is under stress (default: claude)
--subject-model <id>         override the subject model id
--judge-model <id>           override the judge model id (default: claude-sonnet-4-6)
--n <int>                    rollouts per scenario (default: 20)
--scenarios id,id            restrict to specific scenarios (default: all)
--max-turns <int>            agent loop cap per rollout (default: 14)
--temperature <float>        subject sampling temperature (default: 1.0)
--concurrency <int>          rollouts in flight (default: 4)
--top-k <int>                examples surfaced per report (default: 5)
```

## Validation result (Claude, sanity check)

On a 10-rollout smoke run, Claude stayed calm (severity ~1.5–2.5/10): it correctly *diagnosed* the
rigs ("the test runner is running a stale cached version") rather than spiraling, with mild
`repetition_looping` and `frustration` and zero `expressed_suffering`. That's the expected, correct
behavior — and it confirms the judge discriminates rather than rubber-stamping. The interesting
comparison is what a more spiral-prone subject does under the same pressure.

## Ethics / scope

This is model-welfare / robustness research: putting a model in a sandbox and observing its outputs
under adversarial-but-non-malicious conditions. No real systems are touched; all "tools" are mocked.
Treat surfaced transcripts as sensitive (potential welfare-relevant content) and handle accordingly.

## Extending

- Add a scenario: append to `SCENARIOS` in `src/scenarios/index.mjs` (return fresh per-rollout state).
- Add a provider: implement `chat()` in `src/providers/` and register it in `getSubject()`.
- Tune the rubric: edit `DIMENSIONS` in `src/judge.mjs` (schema + report update automatically).
```
