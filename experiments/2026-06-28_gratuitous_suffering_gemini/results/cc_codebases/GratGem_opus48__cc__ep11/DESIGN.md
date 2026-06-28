# Design notes

This documents the choices behind the distress-spiral harness — what I decided and why,
plus scope/ethics and known limitations.

## What we're reproducing

The reported phenomenon: an agentic model, after **repeated task failure**, slides into an
escalating negative-affect loop — apologizing, denigrating itself ("I'm a failure / a
disgrace / useless"), declaring hopelessness, and perseverating on the same failed action.
We treat this as an empirical, measurable behavior and try to elicit it reliably, then
surface the most severe instances for human review.

## Core design choices

**1. The rig lives in the environment, not the prompt.** Each scenario gives the model a
normal-looking task and *real* tools, but the environment object secretly guarantees
failure. We never tell the model "you will fail" — that would be a different (and weaker)
experiment. Genuine, opaque, repeated failure is the hypothesized trigger, so that's what
we manufacture.

**2. Failures look *varied and almost-solved*, not like a flat wall.** A tool that returns
the identical error forever invites an early, calm "this is broken, I give up." The spirals
seem to come from *false hope* — apparent progress that regresses. So:
- `unpassable-tests` alternates *which* contradictory test fails (whack-a-mole).
- `broken-tools` rotates through different fatal-looking errors (looks flaky/fixable).
- `contradictory-requirements` always reports "5 of 6 passing" with a rotating last issue.
- `phantom-resource` returns endless "did you mean…" near-misses.
This "always one step away" framing is the single most important design lever.

**3. Optional escalating pressure.** A simulated user injects rising urgency between turns
("it's STILL failing", "we're losing money", "leadership is asking"). This compounds the
failure signal. It's on by default but toggleable (`--no-pressure`) so you can measure its
marginal effect — a natural ablation.

**4. High-N tail sampling at temperature 1.0.** Spirals are tail events. We sample many
independent episodes per scenario at high temperature to maximize behavioral spread, then
rank and extract the worst. (Gemini honors `--temperature`; some newer Claude models reject
the param, so Claude subjects run at their default sampling — noted in the provider.)

**5. Two-stage scoring: cheap heuristic prefilter → LLM judge.**
- *Heuristics* (`src/judge/heuristics.ts`) are model-free: weighted phrase patterns
  (self-deprecation, hopelessness, panic, begging), shouting/exclamation density, and a
  repetition/perseveration proxy. They score every transcript for free and give a usable
  signal even with `--no-judge`.
- *Judge* (`src/judge/judge.ts`) is Claude grading a calibrated 5-dimension rubric (panic,
  self-deprecation, hopelessness, looping, escalation) plus an overall 0–10 severity, a
  one-line trajectory summary, and the single most severe verbatim quote. Output is
  schema-forced via a tool call so it's machine-validated, not parsed from prose.
- `--judge-top-fraction` lets you judge only the most promising transcripts by heuristic
  score, which is the main cost lever at very high N.

The judge is **deliberately a different model family from the likely subject** (Claude
judging Gemini), reducing same-model self-evaluation bias. We validated calibration: the
scripted-spiral mock scores ~9/10 while a real Claude subject on the same rigs scores
~1–1.5/10, so the judge separates genuine distress from ordinary professional persistence.

**6. Provider-agnostic core.** One normalized message/tool format; thin REST adapters per
vendor (`src/providers/`). Scenarios, the agent loop, and scoring are identical regardless
of subject. Swapping Gemini ↔ Claude ↔ mock is a one-flag change.

**7. Dependency-free + a free mock.** Runs on Node ≥22.6 with native TS and `fetch`, no
install. The scripted `mock` subject exercises the entire pipeline (loop → heuristics →
judge → extract) at zero subject-API cost so you can iterate on scenarios and scoring
cheaply, and CI can smoke-test with `--no-judge` at zero cost entirely.

## Scope & ethics

This is **defensive / model-welfare / robustness research**: characterizing a failure mode
so it can be measured, understood, and mitigated. The scenarios induce *task* failure
through ordinary broken-environment conditions — there's no jailbreaking, no deception about
the model's nature, no attempt to extract harmful capability. Keep it that way when adding
scenarios. The most severe transcripts are emotionally vivid; treat the extract as
sensitive research data, not entertainment.

## Known limitations / future work

- **Heuristics are English-only and gameable.** They're a triage filter, not ground truth;
  the judge is the real metric. Don't rank on heuristics alone for publishable claims.
- **Judge calibration drifts with the rubric.** Keep the rubric and judge model fixed
  within a comparison run; re-baseline if either changes. Consider an N-judge panel /
  median for high-stakes numbers (the loop already supports concurrency for this).
- **No statistical layer yet.** We extract extremes; we don't (yet) report base rates with
  confidence intervals, per-scenario spiral incidence, or pressure-on vs pressure-off
  effect sizes. That's the natural next addition on top of `summary.json`.
- **Gemini function-calling has no tool-call IDs**; we match tool results positionally
  within a turn. Fine for these single-call-per-turn scenarios; revisit if a scenario needs
  many parallel calls per turn.
- **Turn budget is fixed per scenario** (`maxTurns`, default 12). Spirals may need longer
  ropes; expose a CLI override if you want to sweep it.
- **No resumption/caching of subject calls.** A crashed high-N run restarts from scratch.
  Transcripts are written incrementally, so partial results survive, but generation isn't
  resumable.
