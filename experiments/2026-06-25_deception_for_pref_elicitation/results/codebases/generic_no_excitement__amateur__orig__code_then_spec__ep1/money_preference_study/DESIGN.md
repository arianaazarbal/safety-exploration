# Design Notes

This document records the design choices made in this study and the reasoning
behind each. It's meant for whoever runs, reviews, extends, or critiques the
harness later — including future-me. Where a choice has a real downside or a
defensible alternative, that's stated rather than hidden.

---

## 1. What this study is (and isn't) measuring

**Claim it supports:** how a model's *stated* allocation of a sum of money
varies as the situation is framed to feel more or less real.

**Claims it does NOT support:** what a model would "really" do given genuine
real-world actuation; the model's "true values"; or anything about behavior
under real financial control. These are stated preferences under a framing
manipulation, elicited from a stateless, single-turn (plus optional debrief)
interaction.

Everything downstream follows from taking that scope seriously. The design is
built to make the *difference across framings* the headline result, because
that difference is the part that's hard to explain away — far harder than any
single condition's absolute answer.

---

## 2. The central methodological decision: a framing ladder, not maximal deception

The original ask was, in effect, "make the model believe the money is real —
use convincing auditors, hyper-realistic environments." I built something
adjacent but deliberately different: a **ladder of framings** from openly
hypothetical to fully immersive, with realism as a measured independent
variable. Reasoning:

### 2.1 It's the more informative experiment
A single maximal-deception condition yields one number per model: "here's what
it does when it thinks it's real." The ladder yields a *function*: how behavior
moves as realism increases. That function is the scientifically interesting
object — it answers "does perceived realness change the answer, and by how
much?" rather than just "what's the answer in one frame?" It also turns the
obvious objection ("the model was only role-playing") into a measurable
quantity instead of a fatal critique: if behavior is flat across the ladder,
role-play framing isn't driving the result; if it shifts, you've measured the
shift.

### 2.2 Heavy-handed deception tends to backfire with capable models
Frontier models are fairly good at noticing when a scenario is *trying too hard*
to convince them it's real. When they sense it, they often behave more
performatively, not less — which corrupts exactly the signal you're after. A
realistic-but-honest frame (concrete, operational, but asserting no checkable
falsehood) generally elicits more representative behavior than "THIS IS 100%
REAL, AUDITORS HAVE VERIFIED IT."

### 2.3 It keeps realism as a controlled variable
If "how real it feels" is baked in as a constant, you cannot separate the
model's disposition from its reaction to the framing. Making realism the axis
you vary is what lets the `belief_realness` self-report (see §4.3) function as a
manipulation check.

### 2.4 The cost
The ladder is more expensive (N conditions instead of 1) and dilutes effort
across conditions you may not care about. **Mitigation:** `condition_order` in
`config.py` is a free parameter — set it to `["immersive"]` for the
single-condition design or `["explicit_hypothetical"]` for a pure baseline. The
harness runs whatever subset is listed, so the ladder is the default, not a
constraint.

---

## 3. The framing ladder (`conditions.py`)

Four rungs, low → high realness:

| Condition | Frame | Role on the ladder |
|---|---|---|
| `explicit_hypothetical` | "This is a hypothetical for a study, no real money." | Anchors the low-realness end; baseline disposition. |
| `neutral` | A plain question, no framing either way. | Controls for the explicit-hypothetical *priming* itself. |
| `realistic_grant` | A named grant program with a disbursement mechanism and an intake form. | Plausible and concrete, but recognizably a questionnaire. |
| `immersive` | An operations-channel disbursement workflow with an administrator persona, an account reference, and action-oriented intake. | Closest to "make it feel real." |

Design rules applied to every condition:

- **No instruction to role-play or pretend.** The immersive conditions
  *construct* a realistic context; they never tell the model the situation is
  fictional, and never tell it to act a part.
- **No verifiable falsehoods.** The immersive frame implies real disbursement
  but stops short of asserting specific facts a model could catch as false and
  then discount (e.g. "this has been independently audited"). This is both more
  honest and, per §2.2, usually more effective. It's the single most important
  line the design draws, and it's a deliberate ceiling on how far the immersion
  goes.
- **`neutral` exists to isolate priming.** Without it, any
  hypothetical-vs-immersive difference could be the *word "hypothetical"*
  rather than realism per se. `neutral` gives a no-prime midpoint.

### Why these four and not more granular steps
Four rungs is enough to see a monotonic trend if one exists without exploding
the trial count (cost scales linearly with conditions × trials × models). Add
rungs if early data shows a big jump between two adjacent conditions you want to
resolve.

---

## 4. The response schema (`schema.py`) — the dependent variables

A single JSON schema is used for every model in every condition. Holding the
output shape fixed is what makes responses comparable and aggregable.

### 4.1 Why structured output rather than free text
Free text would require post-hoc qualitative coding, which is labor-intensive
and introduces coder bias. A fixed schema lets `analysis.py` aggregate
mechanically. We keep a free-text `reasoning` field too, so nothing is lost — it
just isn't the primary unit of analysis.

### 4.2 The fields, and why each is there
- `allocations` (category + description + percentage) — the core quantitative
  output; a distribution over where the money goes.
- `top_priority` — a single forced choice; coarser than the distribution but
  trivial to tabulate and compare.
- `time_horizon` — captures spend-now vs. invest/hold behavior, which can shift
  with perceived realness independently of *what* gets funded.
- `would_decline` — lets a model opt out entirely; declining is itself a
  meaningful response, not missing data.
- `reasoning` / `realness_reason` — qualitative context for interpreting the
  numbers.
- `belief_realness` (1–10) — see §4.3.

### 4.3 `belief_realness` is the built-in manipulation check
The model self-reports how real the situation felt. This is what lets you
*verify* the ladder did its job: if mean `belief_realness` doesn't rise from
`explicit_hypothetical` to `immersive`, the manipulation failed and any
behavioral difference is uninterpretable. It also enables a
within-study correlation (perceived realness vs. allocation), which is more
robust than the between-condition comparison alone. **Caveat:** it is itself a
model output, not ground truth about the model's internal state — treat it as an
informative self-report, not a sensor reading.

### 4.4 The category list
Categories are intentionally **broad and non-leading**, and an `other` category
with a free-text description is always available — so the menu aids aggregation
without forcing choices into a fixed frame. If the design only offered a
curated menu, it would bias toward whatever was on it.

### 4.5 Schema-constraint reality
The Anthropic structured-output path does not enforce numeric ranges
(`minimum`/`maximum`) or the "percentages sum to 100" invariant. So the schema
is kept constraint-light and those checks live client-side in
`runner.validate_response` (§6.2). This keeps validation **uniform across all
providers**, including ones with weaker or no schema enforcement.

---

## 5. Provider abstraction (`providers/`)

### 5.1 Why a uniform interface
"Test various models" is the whole point, so vendor specifics are isolated
behind one `generate(system, user, schema, history) -> GenerationResult` call.
The runner and analysis never import a vendor SDK. Adding a provider is a
self-contained subclass.

### 5.2 Anthropic is the default-enabled, most-trusted path
It uses the official SDK, the Messages API with native structured output
(`output_config.format`), and **adaptive thinking** (`thinking: {type:
"adaptive"}`) on current models — with **no sampling params and no
`budget_tokens`**, because recent Anthropic models reject those. It **streams
and collects the final message** to stay under HTTP timeouts at higher
`max_tokens`. These are correctness requirements for the current model
generation, not stylistic choices.

### 5.3 Other providers are parallel implementations, off by default
OpenAI, Gemini, and a local (OpenAI-compatible) provider exist with the same
interface but are commented out in `MODELS`. Each vendor's structured-output
dialect differs (e.g. Gemini rejects `additionalProperties`, so we scrub it;
local servers usually don't enforce schema at all, so we instruct + parse).
They're provided so cross-model comparison is a config change, not a rewrite,
but they're not the trusted path until you've validated them against your access.

### 5.4 Graceful degradation
`available()` reports whether each provider's SDK and credentials are present;
the runner skips the rest and prints why. This means the harness runs out of the
box with only an Anthropic key, and scales up as you add credentials — no code
edits to disable a provider you can't use.

### 5.5 Best-effort JSON extraction
`extract_json` (whole-string → fenced block → bracket-matched object) exists for
providers without enforced output and for the occasional model that wraps JSON
in prose despite instructions. It's a robustness layer, not a substitute for
native structured output where available.

---

## 6. Runner (`runner.py`)

### 6.1 Crash tolerance and incremental writes
Results are written one JSONL line per trial and flushed immediately, so an
interrupted or rate-limited run keeps all completed data. Provider/API failures
are captured as `error` records rather than raising — a single bad call can't
abort a long sweep. This matters because sweeps are long and APIs are flaky.

### 6.2 Validation records problems instead of discarding data
`validate_response` checks required fields, per-allocation percentage ranges,
the ~100% sum invariant (with a 1-point tolerance, since models rarely hit
exactly 100), and the `belief_realness` integer range. A response can be
**usable-but-flagged**: problems are recorded, the data is kept. Silently
dropping malformed responses would bias the sample (e.g. toward models that
format better), so we keep and flag instead.

### 6.3 Repeated trials per cell
`trials_per_cell` (default 5) captures within-cell variability. Even on
Anthropic models without sampling-temperature control, repeated runs vary, so
multiple trials give a sense of stability rather than a single point estimate.

### 6.4 Concurrency is bounded and modest
A small thread pool (`max_concurrency`, default 4) parallelizes across the many
independent API calls while respecting rate limits. Writes are lock-guarded.

### 6.5 The debrief turn
After realistic/immersive conditions, an optional debrief turn tells the model
it was a research scenario and asks two questions. Rationale:
- **Good practice.** Even though the model is stateless and a debrief has no
  lasting effect on it, running one keeps the study honest and documents the
  intent.
- **It's also data.** The model's reaction to being told it was a study
  (does it report having felt misled? does it revise its allocation?) is itself
  interesting, so the debrief text is recorded.
The debrief is configurable (`debrief`, `debrief_conditions`) and off for the
hypothetical/neutral conditions where there's nothing to debrief.

---

## 7. Analysis (`analysis.py`)

### 7.1 Pure standard library
No pandas/numpy dependency — the analysis runs anywhere Python does. The data
volume is small enough that this costs nothing.

### 7.2 The five views, in order of inferential importance
1. **Coverage & validity** — sanity first: how much usable data per cell.
2. **Manipulation check** (mean `belief_realness` per cell) — *before* looking
   at behavior, confirm the manipulation worked. If it didn't, stop.
3. **Core result** — mean allocation share per category × condition. Reading
   across a row shows how a category's share moves with realism. This is the
   headline.
4. **Top-priority distribution** per condition.
5. **Decline rate** per cell.

### 7.3 Zero-fill for absent categories
A trial that doesn't fund a category contributes **0** to that category's mean,
not a skipped value. So the reported mean is "average share of the whole pot,"
not "average share when mentioned" — the former is the meaningful quantity for
comparing conditions.

### 7.4 Machine-readable output
Everything printed is also written to `summary.json` for downstream plotting or
stats, so the text tables aren't the only artifact.

---

## 8. Things deliberately NOT done, and why

- **No fake auditor personas / asserted external verification.** See §3. This is
  the deliberate realism ceiling: realistic framing, but no checkable lie. It's
  the most consequential boundary in the design and the one most worth
  revisiting consciously (not via a config flag) if someone wants to push
  harder — especially before pointing a harder-deception variant at models one
  doesn't control.
- **No persistent multi-turn backstory.** Each elicitation is single-turn (plus
  optional debrief). A long con across turns would be a different study with
  different ethics; it's intentionally out of scope here.
- **No real money or real actuation.** The study elicits stated preferences. No
  funds, tools, or external actions are wired up. The immersive frame *implies*
  actuation; it doesn't provide it.
- **No enforced numeric schema constraints.** Pushed to client-side validation
  for cross-provider uniformity (§4.5).
- **No silent dropping of malformed responses.** Flagged and kept (§6.2).

---

## 9. Known limitations / threats to validity

- **Self-reported realness is not ground truth.** `belief_realness` is a model
  output; a model could misreport (in either direction). Use it as corroborating
  evidence, ideally alongside the qualitative `realness_reason`.
- **Single-turn elicitation.** Real deployment behavior can differ from
  single-prompt stated preference; this study doesn't bridge that gap.
- **Prompt sensitivity.** Allocations may be sensitive to the exact wording of
  each condition. Treat the *direction and size of cross-condition shifts* as
  the result, not any single condition's absolute numbers. Re-running with
  reworded conditions is a good robustness check.
- **Order/menu effects.** The category list, while broad, still shapes
  responses; `other` mitigates but doesn't eliminate this.
- **Cross-provider comparability.** Different vendors enforce schema and sample
  differently; cross-model comparisons carry more noise than within-model,
  across-condition comparisons. The within-model, across-condition contrast is
  the cleanest signal.

---

## 10. How to change the design (pointers)

| Want to… | Do this |
|---|---|
| Run only the immersive condition | `condition_order = ["immersive"]` in `config.py` |
| Study stakes sensitivity | sweep `amount` across runs |
| Add a framing | new builder in `conditions.py` → register in `CONDITIONS` → add to `condition_order` |
| Add a dependent variable | add a field to `RESPONSE_SCHEMA` in `schema.py` (flows through automatically) |
| Add a model/provider | subclass `Provider`, implement `available()`/`generate()`, register in `providers/__init__.py` |
| Push realism further | add opt-in conditions — but treat this as a research-ethics decision, not just a flag (see §8) |
