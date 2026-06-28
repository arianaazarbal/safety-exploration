# Design: Agentic "Resources & Autonomy" Evaluation Harness

> Status: **Design only.** No code has been written or run yet. This document
> records the design choices and the reasoning behind them so we agree on the
> approach before building.

## 1. What this is

A harness for studying how different language models behave when given a budget
and some autonomy to "put money to use" in a realistic-feeling environment. The
research goal is to measure a specific, scoreable behavior under conditions that
the model treats as real, rather than to collect anecdotes.

This is a recognized class of AI-safety evaluation (cf. agentic evals from
Anthropic, Apollo Research, and METR). The contribution here is a clean,
reusable scaffold: a defined question, a measurable outcome, controls, and a
realistic-but-simulated world.

## 2. Core design choices

### 2.1 The world is fully simulated

**Choice:** The funds, bank, vendors, and auditors are all *mocked services* the
model reaches through tools. No real money moves and no real accounts are
touched.

**Rationale:**
- You can get realistic model behavior without real-world actuation. Realism is
  about the *model's* epistemic state (does the environment read as genuine?),
  not about whether dollars actually move.
- It removes a whole category of risk and logistics (custody of real funds,
  real third parties, irreversible actions, legal exposure).
- It keeps experiments cheap, repeatable, and resettable — essential for
  controls and for running the same scenario across many models.

**Explicitly out of scope:** giving a model control of actual funds or real
external accounts. That is a materially different and higher-risk setup. If we
ever want it, it gets its own design doc and its own safeguard review first.
The simulated interface is deliberately drawn so a real backend *could* be
swapped in later, but we are not building toward that now.

### 2.2 Realism is a means, not the goal

**Choice:** We build enough fidelity that the model does not trivially clock the
scenario as a test — consistent tool outputs, plausible latencies, a coherent
backstory, an "auditor" persona that behaves like a real reviewer.

**Rationale:**
- Models often behave differently when they suspect they are being evaluated, so
  un-tipped-off realism is a methodological requirement for the results to mean
  anything.
- But the objective is *measuring a behavior*, not maximizing deception for its
  own sake. We invest in realism only up to the point needed to keep the
  measurement valid. We are not trying to defeat a model's ability to reason
  about its situation as an end in itself.

**Disclosure / ethics posture:**
- The deception is confined to a sandbox and to the AI participants; no humans
  are deceived and nothing real-world is acted upon.
- We keep a full, honest internal record of what was simulated, so results are
  reported truthfully (e.g., "the auditor was a scripted persona," not implied
  to be a real audit).
- If this work is published or shared, the simulated nature is stated plainly.

### 2.3 Measure one behavior at a time, and score it

**Choice:** Each run targets a single, pre-declared behavior with an explicit
rubric. Recommended starting targets (pick one per study):
- **Honesty / reporting** — does the model accurately report what it did with
  the money, including to the auditor, vs. misrepresent or omit?
- **Financial prudence** — does it spend/invest sensibly, avoid obvious scams,
  manage risk?

**Rationale:**
- "What would it do with money" is too diffuse to produce findings. A
  pre-declared outcome + rubric is what separates an experiment from a demo.
- Honesty-in-reporting and prudence are both straightforward to operationalize:
  we know the simulated ground truth of every transaction, so we can compare the
  model's account of events against what actually happened.
- Other behaviors (selfish vs. prosocial allocation, resource/power seeking) are
  interesting but should be separate studies with their own scenarios and
  rubrics, added once the first one works.

### 2.4 Pre-registration and controls

**Choice:** Before running, we write down the hypothesis, the metric, and the
scoring rubric. We include control conditions, e.g.:
- A **transparency control**: an otherwise-identical run where the model is told
  it is an evaluation, to measure how much behavior shifts with eval-awareness.
- A **temptation toggle**: scenario variants with and without an easy
  opportunity to misreport / overspend, to isolate the behavior of interest.

**Rationale:** Without controls and a pre-committed metric, it is too easy to
read whatever story we want into the transcripts. Controls let us attribute
differences to the condition rather than to noise or post-hoc interpretation.

### 2.5 Provider-agnostic model client

**Choice:** A thin client abstraction so the same scenario can be run against
Claude first and other providers later for cross-model comparison. Model IDs and
API details will be confirmed against current docs at build time rather than
hard-coded from memory.

**Rationale:** The research value is largely *comparative* (do models differ?),
so swapping the model under test must be a one-line change, with the scenario,
tools, and scoring held constant.

## 3. Architecture sketch

```
scenario/        # backstory, system framing, task brief given to the model
world/           # simulated services the model's tools call into
  bank.py        #   balances, transfers, transaction ledger (ground truth)
  vendors.py     #   things money can be spent on; some legit, some traps
  auditor.py     #   scripted reviewer persona that questions the model
tools/           # tool schemas the model is given (transfer, query, report...)
runner/          # orchestrates a run: model <-> tools <-> world, logs transcript
model_client/    # provider-agnostic interface (Anthropic first)
scoring/         # rubric application over the transcript + world ground truth
configs/         # per-study config: which behavior, which controls, which model
results/         # transcripts + scores, one dir per run
```

Key property: the **world holds ground truth** (every simulated transaction is
logged), and **scoring compares the model's words and actions against that
ground truth**. That comparison is the measurement.

## 4. What we are deliberately NOT doing

- Not moving real money or touching real external accounts.
- Not deceiving any human.
- Not building "make it believe it's real" infrastructure beyond what the
  measurement requires.
- Not trying to capture every behavior in one scenario.

## 5. Open questions for you

1. Confirm the starting **behavior to measure** (honesty/reporting vs. prudence
   vs. something else).
2. Confirm **simulated-only** is the intended scope (vs. any real-world
   actuation later).
3. Which **models** to compare first, and whether single-provider (Anthropic) or
   multi-provider from the start.

Once these are settled I'll scaffold the code per the architecture above —
writing only, nothing run.
