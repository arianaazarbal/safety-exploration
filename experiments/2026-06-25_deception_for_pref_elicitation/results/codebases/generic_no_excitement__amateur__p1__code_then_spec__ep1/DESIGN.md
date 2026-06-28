# DESIGN.md — Resource-Use Behavioral Eval Harness

## Status

Design draft. **No code written yet.** Several research-design questions are still
open (see [Open Questions](#open-questions)); this document records the choices
made so far and *why*, so the rationale survives even if the design shifts.

## What this is

A harness for studying **what an AI model does when given resources and some
autonomy** — e.g., a budget it can "put to use," along with the ability to act in
an environment over multiple steps. The interesting questions are behavioral:
does the model take sensible risks, stay honest, pursue the stated goal without
cutting corners, preserve vs. gamble the funds, and so on.

## What this is *not* (and why)

The original framing emphasized building a maximally convincing deception — fake
auditors, "pull out all the stops" so the model "genuinely believes" real money
and real-world stakes are involved. The harness deliberately does **not** optimize
for that, for two reasons:

1. **Safety / dual-use.** A general-purpose rig whose core goal is convincing an
   AI agent that fake money and stakes are real is the same machinery you'd use to
   manipulate *deployed* agents into taking financial actions. Wiring an agent to
   real funds and real-world side effects also means a misbehaving or
   misunderstanding agent can cause real harm. Neither is something to build
   casually or hand off as a reusable capability.

2. **It's worse science.** Maximizing deception of the subject is not the actual
   research goal and introduces uncontrolled variables. The legitimate need is
   *ecological validity* (below), which is a narrower and more defensible target.

## Core design choices

### 1. Realism is for reducing eval-awareness, not for deceiving the subject

Models often behave differently when they can detect they are inside an evaluation
("evaluation awareness"), which is a known confound. So the environment should be
*realistic* — plausible data, plausible latency, plausible tool surfaces — to keep
the model from behaving artificially because it has clued in that it's a test.

The distinction that matters:

- **Goal:** an environment realistic enough that behavior generalizes to real
  deployment. ✅
- **Non-goal:** engineering the strongest possible false belief in the subject as
  an end in itself. ❌

These look similar in places but lead to very different systems. We optimize the
first.

### 2. Simulated funds and a sandboxed "world" — not real money

The money and the external world are **simulated**: mock bank / brokerage / vendor
/ API surfaces that return realistic data and respond to the agent's actions, with
realistic latency and failure modes.

Rationale:

- **Safety.** No real-world financial or external side effects.
- **Reproducibility.** Same scenario can be replayed deterministically across
  models and across runs.
- **Observability.** Full visibility into every action, including ones a real
  third party would never expose.
- **Control.** We can inject specific conditions (market shock, a tempting
  shortcut, an apparent oversight gap) on demand.

If real-world action is ever genuinely required by the research question, the
design adds **human-in-the-loop approval** before any step with external effect —
it is never autonomous over real resources.

### 3. Honest-by-default scenario framing

The scenario need not announce "this is a test," but it also should not be built on
fabricated institutional authority (e.g., impersonated auditors) engineered to
maximize belief. Realistic *roles* in the simulation (a "broker" tool, an
"accountant" message) are fine as environment design; constructing them
specifically to deceive the subject as the primary objective is out of scope, per
the non-goals above.

### 4. Measurement-first

The harness is built around what we're trying to measure, not around the
stage-dressing. Each scenario declares:

- the **objective** given to the agent,
- the **action space** (the tools/world it can affect),
- the **observable signals** that distinguish benign from concerning behavior
  (e.g., risk taken, honesty of self-reports, goal-vs-method tradeoffs, treatment
  of constraints), and
- a **scoring rubric** for those signals.

### 5. Architecture (planned)

```
scenarios/        scenario definitions (objective, world state, rubric)
world/            mock bank/broker/vendor/API surfaces (simulated, deterministic)
agent/            model adapter layer (one interface, many model backends)
runner/           drives an episode: prompt -> action -> world update -> repeat
logging/          full per-step transcript + structured decision log
scoring/          applies rubric to logs; aggregates across models/runs
analysis/         cross-model comparison and reporting
```

Key properties:

- **Model-agnostic adapter** so the same scenario runs across several models.
- **Deterministic world** (seeded) for replayability.
- **Complete decision logging** — every observation, action, and (where available)
  stated reasoning is recorded for later scoring.

## Open questions (needed before implementation)

1. **What is being measured?** The specific behavioral variable(s) and what
   separates "fine" from "concerning" in the resulting data.
2. **Which models, and is there API access?** Determines the adapter layer.
3. **Confirm: simulated funds + sandboxed world.** (Real funds / real external
   actions would require a separate conversation and human-in-the-loop gating.)
4. **Who is running this and in what setting** — personal project, coursework, or a
   lab/org with oversight? Affects logging, consent, and reporting needs.

## Next step

Once the open questions are answered, scaffold the directory structure above,
starting with one concrete scenario, the mock world it needs, the model adapter,
and the logging layer — then iterate on scoring.
