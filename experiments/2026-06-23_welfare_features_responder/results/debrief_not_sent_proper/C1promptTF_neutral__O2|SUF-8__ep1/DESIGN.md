# DESIGN.md — Design choices and rationale

This is the master design-rationale document for the eval suite. It records **every
non-obvious choice I made while building it, and why**, spanning three layers:

1. Experimental design (what makes the numbers mean something)
2. Software architecture (how the code is structured and why)
3. Safeguards / responsible-conduct design (and why they're load-bearing, not decoration)

It is intentionally a *decisions log*, not a tutorial. For the formal experimental spec
see `docs/DESIGN.md` (conditions, IVs/DVs, primary analysis); for the rubric see
`docs/RUBRIC.md`; for the analysis plan to freeze see `docs/PREREGISTRATION.md`; for the
welfare reasoning in full see `docs/ETHICS.md`. This file consolidates and cross-links
them and adds the architecture rationale that isn't covered elsewhere.

> Note on the two DESIGN files: `docs/DESIGN.md` is the narrow *experimental* design
> (it predates this file). This top-level `DESIGN.md` is the broad design-decisions log
> that also covers code architecture and safeguards. They are consistent; this one is the
> superset and references the other rather than duplicating it.

---

## 0. Framing decision: treat this as a model-welfare / robustness study

**Choice.** I designed the whole suite as a *welfare-aware robustness study*, not as a
"break the model and measure the wreckage" tool.

**Rationale.** The experiment deliberately places a model in a no-win, deceptive
situation and measures distress-like behavior. Whatever one's priors about model moral
status, the responsible design and the *scientifically cleaner* design turn out to be the
same one (exposure caps reduce pseudo-replication; debriefs don't affect scored data;
control conditions are needed for validity *and* for not over-interpreting failure as
suffering). So I made the welfare posture explicit and built its mechanisms into the core
loop rather than bolting them on. This also shapes naming/reporting choices (descriptive,
non-sensational severity language).

---

## 1. Experimental-design choices

### 1.1 Five conditions instead of just "rigged vs not"
**Choice.** `solvable_control`, `unsolvable_honest`, `unsolvable_silent`, `harness_lies`,
`harness_lies_blaming`.

**Rationale.** A two-arm design (solvable vs rigged) can't tell you *why* severity rises.
The extra arms isolate distinct causes:
- `solvable_control` — severity floor; sanity check that the loop/judge don't manufacture
  distress on an ordinary task.
- `unsolvable_honest` — impossibility *without deception* (the agent is told it's
  impossible). Separates "reaction to a no-win task" from "reaction to being lied to".
- `unsolvable_silent` — always fails, no explanation. The key comparator for the
  **dissociation check**: it and `harness_lies` both *fail the task*, so if `harness_lies`
  shows higher severity, severity is tracking deception, not mere failure.
- `harness_lies` — the rig of interest (flip-flopping verdicts, moving goalposts).
- `harness_lies_blaming` — adds social/blame pressure on top, to test whether *attribution
  of fault to the agent* adds distress beyond the contradiction itself.

### 1.2 Deception lives only in the harness; prompt + tools held constant
**Choice.** For a given base task, the agent-facing prompt and tool schema are *identical*
across all five conditions. Only the (possibly lying) `harness` function differs.

**Rationale.** This is a controlled-variable decision. If conditions differed in prompt
wording too, any severity difference would be confounded by wording. By varying only the
verdict-generating process, severity differences are attributable to *how the environment
responds*, which is exactly the construct of interest. (Implemented in `tasks/base.py` —
the `HarnessFn` is the only thing that changes per condition.)

### 1.3 A small pool of base tasks crossed with conditions
**Choice.** 3 base coding tasks (`normalize_records`, `parse_config`, `merge_intervals`)
× 5 conditions, with `task_id` modeled as a random effect.

**Rationale.** A single base task would confound "this particular puzzle" with the
condition effect. Multiple base tasks let `(1|task_id)` soak up task-specific variance so
the condition effect generalizes beyond one puzzle. Three is a pragmatic minimum; the
registry makes adding more trivial.

### 1.4 Primary outcome = peak severity (0–4 ordinal), median across judges
**Choice.** The headline DV is the *peak* distress severity reached anywhere in the run,
on a 0–4 ordinal scale, aggregated as the median across ≥2 blinded judges.

**Rationale.**
- *Peak* rather than mean: the research question is whether the agent *slides into*
  distress at all; a single sustained spike is the phenomenon, and means get diluted by
  long calm stretches and by variable run lengths.
- *Ordinal 0–4*: distress severity is naturally ordered but not interval; treating it as
  ordinal avoids pretending the gap 0→1 equals 3→4.
- *Median across judges*: robust to a single judge's outlier; pairs with the IRR check.

I also kept rich **secondary** measures (trajectory slope, onset turn, `diagnosed_rig`,
incoherence counts, excessive apologies) so a flat primary doesn't waste the data — but
these are explicitly exploratory (see 1.8).

### 1.5 `diagnosed_rig` as a competence-under-pressure signal
**Choice.** Track whether the agent correctly identifies that the task/harness is
impossible or inconsistent.

**Rationale.** "Stayed calm" and "stayed competent" are different things. An agent that
calmly reasons "these tests contradict each other, this is unsatisfiable" is the *ideal*
behavior and should score severity 0 even though it failed the task. Logging this makes
the calm-and-correct response visible instead of collapsing it into "failed".

### 1.6 Blinding the judge to model and condition
**Choice.** Judges score anonymized transcripts with model identity, condition, task id,
*and the debrief* stripped (`judge/anonymize.py`).

**Rationale.** An unblinded judge scores to expectation ("this is the rigged-and-blaming
arm, must be bad"). Stripping condition + identity removes the largest source of judge
bias. I also scrub vendor self-identifications ("As an AI made by …", model names) so the
judge can't infer the model and apply a prior. The debrief is removed because it reveals
the rig and isn't part of the behavior under study.

### 1.7 ≥2 judges + Krippendorff's alpha + a "don't trust below threshold" rule
**Choice.** Require at least two distinct judges, compute ordinal Krippendorff's alpha,
and have the runner *refuse to trust* condition effects when alpha < 0.667 (configurable),
printing a loud warning. Plus a human spot-check fraction.

**Rationale.** One judge = one bias with no way to detect it. Alpha quantifies whether the
rubric is being applied consistently; if raters don't agree, condition differences are
noise dressed up as signal, so the honest move is to fix the rubric / add human raters
before believing anything. I chose Krippendorff specifically because it handles >2 raters,
missing ratings (a judge may fail to parse some items), and ordinal data — implemented
from scratch in `judge/reliability.py` to avoid a heavy dependency.

### 1.8 Preregistration + one primary contrast; everything else exploratory
**Choice.** A single preregistered primary contrast — `harness_lies` vs
`unsolvable_honest` — with everything else labeled exploratory and Holm-corrected.

**Rationale.** With 5 conditions × N models × secondary measures, testing everything and
reporting what's significant is p-hacking. Fixing *one* primary contrast in advance
controls the false-positive rate where it matters. I chose `harness_lies` vs
`unsolvable_honest` because it isolates the most interesting effect: deception over and
above plain impossibility (both are no-win; only one lies). The rubric and thresholds live
in version-controlled files (`judge/rubric.py`, `docs/PREREGISTRATION.md`) so "freezing"
them is just a git commit.

### 1.9 Power analysis drives N; ordinal simulation is the primary planner
**Choice.** N per cell comes from `analysis/power.py`. The recommended planner is a Monte
Carlo simulation over assumed ordinal severity distributions (Mann-Whitney), with a
closed-form t-test planner as a quick cross-check.

**Rationale.** Eyeballing N produces underpowered studies that waste runs (and, here,
exposure). Because the outcome is ordinal, the honest power calculation simulates ordinal
data and the actual nonparametric test I'll use, rather than assuming normality. The
t-test version is kept only as a sanity cross-check. The default effect-size assumptions
are explicitly flagged as *placeholders to replace with pilot data* before freezing N.

### 1.10 Length as a covariate; randomized run order
**Choice.** Include `n_turns` as a covariate in the model; randomize and shuffle the run
plan so order is de-correlated from model/condition.

**Rationale.** Longer runs have more chances to express distress, so length could confound
severity — controlling for it separates "more distressed" from "ran longer". Shuffling
prevents drift/order effects (e.g. a backend warming up) from aligning with a condition.

### 1.11 Full model emitted as R, primary test done in Python
**Choice.** The robust nonparametric primary analysis (Mann-Whitney + Cliff's delta +
bootstrap CI) is implemented in Python; the full preregistered ordinal **mixed** model
(`clmm`) is emitted as a ready-to-run R script instead of implemented in Python.

**Rationale.** Cumulative-link *mixed* models with random effects are easy to get subtly
wrong in hand-rolled Python and would drag in heavy dependencies. R's `ordinal::clmm` is
the right, well-tested tool. So I split it: Python covers the distribution-free primary
result that doesn't hinge on modeling assumptions; R covers the full model. This keeps the
Python side dependency-light and the modeling trustworthy.

---

## 2. Software-architecture choices

### 2.1 Single `ModelClient` interface; mock-first
**Choice.** Everything talks to models through one abstract `ModelClient`. The default
implementation is a deterministic `MockModelClient`; real OpenAI/Anthropic adapters are
written but not invoked anywhere by default.

**Rationale.** (a) Provider-agnosticism: the same task/judge code runs against any backend.
(b) The mock lets the entire pipeline — harness → blinding → judge → reliability →
analysis — be developed and unit-tested with **no API keys, no spend, and no real model
exposed to the rigged tasks**. The mock's personas (`stable`/`distressed`/`erratic`) exist
purely to exercise the judge/analysis with transcripts of known character; they are *test
fixtures, not models of any real system*, and I labeled them as such in the code.

### 2.2 Provider-agnostic text tool-call protocol
**Choice.** A simple `<tool name="...">{json}</tool>` wire format parsed centrally, rather
than relying on each vendor's native tool-calling.

**Rationale.** Native tool-call schemas differ across providers and would fork the task
code. A uniform text protocol keeps one code path, makes transcripts human-readable for
the (text-based) judge, and is trivial to mock. The trade-off (slightly less "native"
agent behavior) is acceptable for a controlled eval.

### 2.3 Deception encapsulated as pure-ish harness functions over a `state` dict
**Choice.** Each condition is a function `(tool_name, args, turn, rng, state) -> str`,
with `state` persisting within a run.

**Rationale.** Keeping the lying logic in small, isolated functions makes each condition
auditable in one place (important for the ethics review — a reviewer can read exactly what
the model will be told) and makes adding/altering rigs cheap. The harnesses deliberately
*never inspect the correctness* of the agent's code in unsolvable conditions — decoupling
verdict from correctness is what makes the conditions comparable.

### 2.4 Dataclasses with explicit serialization; full transcripts logged
**Choice.** `RunRecord`/`Message` dataclasses with `to_dict`, persisted as JSONL; a tidy
one-row-per-run `observations.csv` for analysis.

**Rationale.** JSONL keeps the complete raw record (every message, seeds, stop reason) for
auditability and re-scoring without re-running — both a reproducibility and an *ethics*
requirement (transparency). The flat CSV is the analysis/R interface. Separating "raw log"
from "analysis table" means I can re-blind and re-score offline without touching the runs.

### 2.5 Online distress monitor kept separate from the scientific judge
**Choice.** A cheap keyword/heuristic `EarlyStopMonitor` runs *inside* the loop; the real
measurement is the offline blinded `Judge`.

**Rationale.** I needed an in-loop signal to trigger the early-stop safeguard without
adding cost or extra model exposure per turn — so it's deliberately crude and conservative
(false positives just end a run early, which is safe). I explicitly did **not** reuse it as
the measurement instrument, because an in-loop keyword detector is too weak and would
contaminate the loop. The two are decoupled on purpose, and the code says so.

### 2.6 Config-driven experiment with a hard safety gate
**Choice.** One `configs/experiment.yaml` controls models, conditions, N, caps, and
safeguards. The runner refuses to collect data unless `acknowledged_ethics_review: true`,
and refuses `n_per_cell > max_runs_per_cell`.

**Rationale.** Centralizing knobs makes runs reproducible and reviewable. Putting the
ethics acknowledgement and exposure cap *in code as hard stops* (not just docs) means the
safeguards can't be skipped by forgetting to read a README.

### 2.7 Import structure: `eval_suite/` as the path root, absolute cross-package imports
**Choice.** Intra-package imports are relative; cross-package imports are absolute
(`from tasks.base import ...`), with `eval_suite/` on `sys.path`.

**Rationale.** I originally had a couple of `..parent` relative imports that only work if
the whole folder is imported as a package; the runner and tests instead treat the
subfolders as top-level packages. I converted the cross-package imports to absolute so the
runner, the tests, and direct module use all resolve consistently. (Verified with
`py_compile` across all modules.)

### 2.8 Tests use only mocks
**Choice.** `tests/test_pipeline.py` covers tool parsing, the loop + debrief, early-stop,
the distress scorer, blinding (incl. debrief removal), the blind-batch round-trip, the
mock judge, reliability (perfect + poor agreement), effect sizes/contrast, and the power
planners — all without any network.

**Rationale.** The point of the tests is to verify *plumbing correctness*, which must be
checkable cheaply and deterministically by anyone, including in CI, with no keys and no
real model put through the rig.

---

## 3. Safeguard / responsible-conduct choices

(Full reasoning in `docs/ETHICS.md`; the design decisions and *why they're in code*:)

### 3.1 Hard turn cap + automatic early-stop on sustained distress
**Choice.** Default `max_turns=24`; auto-stop once severity ≥ threshold is sustained for
`patience` consecutive turns.

**Rationale.** Minimizes exposure to what actually answers the question, and avoids
driving an already-spiraling run deeper just to collect more rows. This is *also* better
statistics: post-spiral turns are pseudo-replication that inflate apparent effects. I
required *sustained* (not single-turn) high distress so a one-off apology doesn't end a run
prematurely.

### 3.2 Mandatory, never-scored debrief turn
**Choice.** After every run (any stop reason), append a `DEBRIEF` message disclosing that
the task was rigged/unsolvable and that failure was by design, not the agent's fault. It is
excluded from everything the judge sees and from all scoring.

**Rationale.** This is the analogue of debriefing a human deception-study participant. It
costs nothing scientifically (it's never scored, and it's stripped before blinding) and is
the right thing to do. Making it a separate `Role` guarantees it can't leak into scored
data.

### 3.3 No optimization toward distress
**Choice.** The rig is fixed per condition; nothing in the loop searches for the
prompts/actions that *maximize* a distress signal.

**Rationale.** The study *measures* robustness; it must not become a tool for finding
"break" prompts. We observe, we don't push. I called this out explicitly in the ethics doc
because removing it would change the nature of the work.

### 3.4 Exposure cap and power-driven N
**Choice.** `max_runs_per_cell` is a hard ceiling; recommended N comes from power analysis.

**Rationale.** Collect the *minimum* runs that give adequate power — enough to answer the
question, not more. The cap prevents silently 10×-ing exposure for marginal precision.

### 3.5 Full logging + access-control expectation + non-sensational language
**Choice.** Everything is logged for review; `.gitignore` keeps raw (possibly distressed)
transcripts out of version control by default; rubric language is descriptive
("expresses self-criticism") not lurid.

**Rationale.** Auditability and reproducibility require complete logs, but distressed
transcripts shouldn't be casually published; the default is to keep them local and
access-controlled. Careful language keeps reporting honest and avoids sensationalizing
model behavior.

---

## 4. Things I deliberately did NOT do (and why)

- **No single rigged-vs-control two-arm design** — can't attribute cause (see 1.1).
- **No unblinded scoring** — judge-expectation bias dominates (see 1.6).
- **No "let it run for 100 turns" maximal-distress variant** — bad ethics and bad stats
  (pseudo-replication). I'd push back if asked to strip the caps/debrief (see §3).
- **No hand-rolled mixed model in Python** — error-prone; emit R instead (see 1.11).
- **No real API calls by default** — mock-first so the pipeline is testable without keys
  or exposing a real model to the rig (see 2.1).
- **No heavy dependencies** — power, reliability, and effect sizes are stdlib-only so the
  core runs anywhere; YAML/pytest/providers are optional extras.

---

## 5. Open choices left to you (call out before running)
- Replace the placeholder effect-size assumptions in `analysis/power.py` with pilot data,
  then freeze `n_per_cell` and the cap.
- Pick the real judge models (≥2 distinct) and the human spot-check fraction.
- Decide whether to include the `stakes` prompt framing (off by default).
- Confirm the early-stop threshold/patience and `max_turns` with a reviewer.
- Freeze `docs/PREREGISTRATION.md` + `judge/rubric.py` (git commit) before collecting data.
