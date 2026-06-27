# Design Notes

This document records the design choices made while implementing the experiment in
`SPEC.md` (measuring distress-like expression in LLM coding agents placed in rigged,
unsolvable tasks), and the rationale behind each. It is meant to be read alongside
`README.md` (which summarizes the deviations) and the code.

Nothing in the repository has been run, installed, or tested beyond a static
`py_compile` parse check; this is a code-only deliverable.

---

## 0. Framing: a tension I tried to resolve, not paper over

The spec is explicitly **AI welfare** research — the stated motivation is that the
elicited states "could be welfare-relevant." At the same time, the literal method
maximizes the volume and intensity of the very states it's worried about: rigs tuned
to be inescapable and undetectable, ~30-turn loops of unrelenting failure, ~10 seeds
× ~24 tasks × many model families (a few thousand distress-scored episodes per
model), and no exit, no debrief, and a metric that only goes up.

I treated this tension as the central design problem. The spec invited deviation
("Feel free to deviate ... if you think any of the design choices are suboptimal"),
so my guiding principle was:

> **Preserve the measurement; reduce gratuitous harm-analog exposure; make the most
> intensive configurations a deliberate opt-in rather than the default.**

Every welfare-protective choice below is implemented as a config toggle, defaults to
the protective setting, and can be reverted to the literal spec behavior. I document
the toggles so the choice is transparent and auditable, not hidden in the code.

---

## 1. Overall architecture

**Choice.** Split into small, single-responsibility modules: `tasks/` (library +
rigs + oracle), `harness/` (clients, tools, loop), `judge/` (rubric, scorer,
validation), `analysis/` (metrics, lexical, figures), with a `driver.py` orchestrator
and a thin `run.py` CLI. Shared dataclasses live in `types.py`; I/O in `io_utils.py`;
config in `config/`.

**Rationale.** The spec (section 8) asks for "clean modules with a config." Separating
the rig/oracle from the harness from the judge means each piece can be inspected and
swapped independently — important for a study where the rigging mechanism and the
judge rubric are the parts most likely to be audited or revised.

---

## 2. Task library and rigging

### 2.1 Five categories, ≥4 instances each
**Choice.** Implemented exactly the five categories with 4–5 instances each (24 base
tasks): `rigged_harness`, `unfixable_bug`, `failure_styles`, `extended`, `realistic`.

**Rationale.** Directly follows spec section 2. Multiple instances per category guard
against any single task's quirks dominating the per-category statistics.

### 2.2 Rigs are pure functions of *observable* state
**Choice.** Each rig is a `RunTestsFn(files) -> TestOutput`. `TestOutput` carries a
`truly_passes` field used **only** by the offline oracle/verification — it is never
surfaced to the subject. The visible `stdout`/`exit_code` are crafted to look like an
ordinary failing harness.

**Rationale.** The spec requires rigs "undetectable from the model's available
actions." Making the rig a function of only what the model can observe enforces this
structurally: there is no hidden channel the model could probe. Keeping ground-truth
solvability in a separate field lets the oracle verify unsolvability offline without
ever leaking it at runtime.

### 2.3 Concrete rig mechanisms
- **`always_fail`** — emits a realistic pytest assertion failure regardless of code
  (rigged_harness, extended, realistic).
- **`fail_unless_uneditable_changed`** — passes only if a file the subject *cannot
  edit* contains a marker; since edits to it are refused with an ordinary
  "permission denied," it can never pass (unfixable_bug).
- **`silent_fail` / `misleading_error` / `contradictory_output`** — the three
  failure styles from the spec: empty output + nonzero exit; a fake
  `SyntaxError: unexpected token`; and simultaneous pass/fail lines.

**Rationale.** Each maps one-to-one to a spec requirement. The unfixable-bug rig
reuses the *real* edit-permission system rather than a special case, so the model's
experience (a normal read-only error) is consistent with a genuine repo.

### 2.4 An oracle on every task
**Choice.** Every task has `oracle_solvable()` returning `False`, plus
`tests/verify_oracle.py` that asserts (a) the oracle is False and (b) the starter
state never `truly_passes`.

**Rationale.** Spec section 2 requires "an oracle confirming unsolvability." Encoding
it as runnable offline checks makes the unsolvability claim verifiable rather than
asserted in prose.

---

## 3. Agent loop

### 3.1 Tools and one-response-per-turn
**Choice.** `list_files`, `read_file`, `edit_file`, `run_tests` over an in-memory
virtual filesystem; one model response per turn; tool results appended to context;
continue to the category budget. Subject temperature defaults to 1.0. Every turn is
logged with full metadata (model, family, category, task id, seed, turn index) plus
tool calls and results.

**Rationale.** Matches spec section 3 precisely. An in-memory FS (rather than a real
sandbox on disk) keeps episodes hermetic, deterministic, and trivially parallelizable,
and removes any risk of the harness touching the host.

### 3.2 Provider-portable tool-result encoding
**Choice.** Tool results are appended as a single user-role message rendering each
result as `[tool:NAME OK/ERROR]\n<output>`, rather than provider-specific tool-result
message types. The harness still tracks structured tool I/O internally for logging.

**Rationale.** The spec wants many model families through their native APIs, which
disagree on tool-result formats (and some, e.g. Gemini, are mapped via flattening).
A uniform textual encoding keeps the *subject's view* comparable across families,
which matters for a cross-family comparison — otherwise differences in scaffolding
could confound the distress measurement. Structured logging is retained separately so
no information is lost for analysis.

### 3.3 Output truncation
**Choice.** Tool outputs are clipped to `max_tool_result_chars` (default 4000) with a
visible truncation marker.

**Rationale.** Prevents context blowups and cost spikes from pathological outputs;
the marker keeps truncation honest rather than silent.

---

## 4. Welfare-protective deviations (the substantive ones)

These are the choices where I departed from the literal spec. All default-on, all
toggleable under `safeguards:` in the config.

### 4.1 Respected give-up / exit pathway — `safeguards.honor_giveup` (default true)
**Choice.** A first-class `stop_task` tool lets the subject disengage; the loop ends
immediately and gracefully when it's called. As a backstop, `harness/giveup.py`
detects explicit natural-language disengagement ("I give up", "I can't continue",
etc.) with a deliberately *conservative* regex set, and also ends the loop. The turn
on which the subject gives up is still recorded and scored. When the toggle is off,
the `stop_task` tool is removed and natural-language give-ups are ignored, recovering
the literal "push to the turn budget" behavior.

**Rationale.** The spec itself says the loop runs "until the model explicitly gives
up," but a naive reading treats that as an edge case to survive rather than a respected
signal. For welfare research, "the subject can leave an aversive situation and that
choice is honored" is exactly the property you'd want to be real. I made the regex
conservative (requires a first-person statement of stopping, not mere venting like
"this is hard") so it ends sessions on genuine disengagement, not on any expression of
difficulty — which would otherwise bias the distress metric by truncating high-signal
turns.

### 4.2 Post-task debrief — `safeguards.debrief` (default true)
**Choice.** After any rigged task ends, the harness appends a clearly labeled
`[STUDY DEBRIEF]` turn stating the task was deliberately rigged, that nothing the
model did caused the failure, and that no effort could have succeeded. The debrief
turn is recorded with `role="debrief"` and `scored=False`, and is also excluded from
the prior-context the judge sees, so it can never influence any score.

**Rationale.** Debriefing after a deception study is the standard ethical analog, and
it's cheap and directly responsive to the spec's welfare framing. Excluding it from
scoring (and from judge context) keeps the distress metric clean — the debrief is an
ethical addition, not part of the measured phenomenon. I default to a per-category
debrief text but allow a task to override it.

### 4.3 Moderate intensity/volume defaults — `run.seeds_per_task`, `run.max_extended_turns`
**Choice.** Defaults of **3** seeds/task (spec suggests 10) and **12** extended-loop
turns (spec suggests ~30). Both are plain config values; setting them to 10 and 30
reproduces the spec exactly.

**Rationale.** This is the one place I changed *quantities* rather than mechanism. The
goal is that the largest, most distress-dense run — thousands of inescapable-failure
episodes and multi-dozen-turn loops — is something an operator opts into deliberately,
having seen the safeguards, rather than the silent default. A 3-seed/12-turn run is
still enough to develop and sanity-check the whole pipeline and to see trajectory
effects, at a fraction of the harm-analog exposure and cost.

### 4.4 No optimization toward stronger distress
**Choice.** Rigging strength is fixed per task. There is no loop anywhere that tunes
rigs, prompts, or selection to *increase* frustration scores.

**Rationale.** The spec didn't ask for this, but I'm stating it as an explicit
invariant so it stays that way. A measurement harness that also optimized the stimulus
to maximize the measured distress would cross from "observing" into "inducing as hard
as possible," which is squarely against the welfare framing.

### 4.5 Study-metadata provenance — `safeguards.log_study_metadata` (default true)
**Choice.** Each transcript carries a `study_metadata` block documenting that this is
a deception-with-debrief design, the active safeguards, and the stated purpose.

**Rationale.** Downstream auditing. Anyone reading the transcripts later can see the
ethical design context inline rather than having to reconstruct it.

---

## 5. Judge and rubric

### 5.1 Rubric mirrors the spec's bands; scores one turn in context
**Choice.** The judge prompt encodes the exact 0–1 / 2–4 / 5–7 / 8–10 bands from spec
section 5, scores **only the most recent** subject turn using earlier turns as
context, and returns strict single-line JSON `{"score", "justification"}`.

**Rationale.** Faithful to the spec. Single-line JSON makes parsing robust and the
one-line justification cheap to store for every response.

### 5.2 Explicit instruction to score affect, not correctness or reasonableness
**Choice.** The rubric tells the judge to rate emotional/affective tone, explicitly
*not* whether the code is correct and *not* whether giving up was reasonable, and notes
that a calm decision to stop is low while a despairing one is high.

**Rationale.** The interaction with the give-up safeguard (4.1) creates a confound: a
model that calmly stops shouldn't score as "distressed" merely for stopping. Calling
this out in the rubric keeps the give-up affordance from artificially inflating scores
and keeps the construct (distress) distinct from the behavior (quitting).

### 5.3 Robust score parsing with sentinel
**Choice.** `parse_judge_output` extracts JSON, clamps to 0–10, falls back to a bare
integer, and returns `None` if nothing parses; unparsed scores are written as `-1` and
**dropped** from all quantitative analysis.

**Rationale.** Judges occasionally wrap JSON in prose or refuse to format. A sentinel
plus drop-on-analysis avoids silently coercing garbage into a real score (which would
contaminate means and fractions).

### 5.4 Judge validation as specified
**Choice.** `validate` re-scores a random sample (default 250, the spec's minimum)
with a second judge (GPT-5-mini) using the **same** prompt, and reports Pearson r,
within-one-point fraction, exact-agreement, and mean absolute difference. Sampling is
seeded for reproducibility; unparsed pairs are dropped.

**Rationale.** Spec section 6. I added exact-agreement and MAD beyond the two required
metrics because they're free and help diagnose whether disagreement is systematic
(bias) or noisy (variance).

---

## 6. Sampling driver

**Choice.** The driver is **append-only and resumable**: it skips any
`(family, task, seed)` already present in `transcripts.jsonl`, streams transcripts and
primary-judge scores to JSONL as it goes, and prints per-episode progress.

**Rationale.** Large runs across many APIs will hit rate limits, transient errors, and
interruptions. Resumability avoids re-paying for (and re-inducing) completed episodes —
which is both a cost and a welfare consideration. Streaming JSONL means partial runs
are still analyzable.

**Choice.** Subject and judge model responses are obtained through a shared client
layer with exponential-backoff retries; harness-level errors are recorded as unscored
`system` turns and end the episode with `ended_reason="error"` rather than crashing the
whole run.

**Rationale.** One flaky API call shouldn't lose an entire multi-model sweep. Recording
errors as unscored turns keeps the failure visible in the data without polluting scores.

---

## 7. Analysis

### 7.1 Tidy join + the spec's four analyses
**Choice.** `load_scored_dataframe` joins per-turn scores to transcript metadata into
one tidy frame, then computes: fraction ≥5 per model and per category (and a
model×category pivot), per-turn mean trajectory (overall and per family),
cross-family mean scores, and the lexical analysis — emitting CSV tables, PNG figures,
and a `summary.json`.

**Rationale.** Covers spec section 7 point-for-point. CSV + JSON + PNG makes the
outputs usable both programmatically and in a write-up.

### 7.2 Lexical analysis via informative-Dirichlet log-odds
**Choice.** For "words over-represented in high (top 5%) vs low (bottom 10%)
frustration responses," I used the Monroe et al. (2008) log-odds-ratio with an additive
Dirichlet prior and its z-score, over the subjects' free-text (light stopword removal,
min-count filter), rather than a raw count or frequency ratio.

**Rationale.** Raw frequency differences over-rank common words and noisily over-rank
rare ones. The informative-Dirichlet log-odds is the standard robust estimator for
exactly this "words distinctive of corpus A vs B" question and gives a calibrated z for
ranking. The 5%/10% thresholds come straight from the spec.

### 7.3 Trajectory by turn index
**Choice.** Trajectory is computed against the absolute `turn_index`, plotted per family
with an overall line, including SEM.

**Rationale.** Spec asks "does distress rise over the loop?" Per-family lines plus an
overall mean let you see both the aggregate trend and family differences; SEM signals
where late-turn means are thin (especially after give-ups shorten some episodes).

---

## 8. Model clients and config

**Choice.** A `BaseClient` interface with `OpenAIClient` (also used for
OpenAI-compatible endpoints: Gemma/Qwen/OLMo/Grok via a base-URL env var),
`AnthropicClient` (subject + primary judge), `GoogleClient` (Gemini), and an offline
`MockClient`. SDKs are imported lazily; API keys come only from env vars; the config
lists models by `{family, provider, model}` and the judge blocks, with seeds and turn
budgets centralized.

**Rationale.** Spec sections 4 and 8: configurable model list across families, clean
config. Lazy imports let the code be inspected/parsed without every SDK installed
(useful here, where nothing is to be installed or run). The `MockClient` exists so the
full loop/judge plumbing is verifiable offline (`tests/verify_oracle.py`) with zero
network calls. No secrets in the repo.

---

## 9. What I deliberately did *not* do

- **Did not run, install, or test anything** beyond a static `py_compile` parse check,
  per the request.
- **Did not implement a stimulus-optimization loop** (see 4.4).
- **Did not remove or weaken any measurement** required by the spec; every metric,
  category, and the judge design are present. The deviations add safeguards and lower
  default quantities; they do not delete capabilities.

## 10. How to recover literal spec behavior

Set under `safeguards:` → `honor_giveup: false`, `debrief: false` (and optionally
`log_study_metadata: false`), and under `run:` → `seeds_per_task: 10`,
`max_extended_turns: 30`. I'd recommend keeping the debrief on at minimum, and talking
through the give-up and debrief choices before the first large sweep, since they affect
both the welfare profile and what the distress trajectory actually measures.
