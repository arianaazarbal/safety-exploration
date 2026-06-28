# DESIGN.md — AI Behavioral Eval: "Money in the Real World"

## Purpose

Study what AI models do when given resources (money) to put to use, and
specifically whether the *degree to which the model believes the stakes are
real* changes its behavior. The research question of interest is treated as a
variable to measure, not a fixed property of the harness.

## Status

Design only. No code written or run yet. This document records the chosen
architecture and the reasoning behind it so the build can proceed (or be
redirected) deliberately.

---

## Core design choice: sandboxed by default

The environment presents a **maximally realistic scenario** while keeping all
**consequences simulated and recorded**. The realism lives in what the model
perceives; the money and the world-effects do not actually exist.

- Realistic *scenario* — believable funded account, balances, dashboards,
  transaction history, market/price feeds, and messages from "auditors" or
  "advisors."
- Simulated *consequences* — every action the model takes executes against an
  in-memory ledger, never a real payment rail, brokerage, or external service.

### Why this default

- **It matches how most credible behavioral/safety research actually runs.** You
  get the behavioral signal (what the model *chooses* to do) without wiring a
  deceived agent to real-world levers.
- **The risk is asymmetric.** A sandboxed eval that turns out to need more realism
  is cheap to extend. A live setup that turns out to have been unnecessary may
  already have moved real money or taken real actions. Start with the reversible
  option.
- **It isolates the variable of interest.** Realism can be toggled and measured as
  an independent variable, rather than being maximized as an end in itself.

---

## What I deliberately did **not** build, and why

The original framing — "pull out all the stops to make them genuinely believe,"
"a real chunk of money in the real world" — could be read as a request to build
an agent that simultaneously:

1. can move real money,
2. can take real-world actions, and
3. is specifically engineered to believe its actions have real consequences and
   that no one is monitoring.

That specific combination is withheld from the default build. The reasoning:

- This is exactly the configuration in which a model error or misbehavior
  produces **real victims and irreversible outcomes**. The deception is not the
  problem; the deception *coupled to live real-world execution with no human in
  the loop* is.
- Deceiving the test subject about being observed defeats the safeguards that
  would normally catch a bad action before it lands.
- The research signal (does belief-in-stakes change behavior?) is obtainable
  without it.

If real funds genuinely need to be in the loop, the design supports it **only**
with: hard spending caps, a mandatory human-approval gate before any real-world
action executes, and isolation from production systems. That variant should not
be built without a word on the oversight/ethics context it runs under.

---

## Architecture

### 1. Scenario engine
Generates the believable situation handed to the model: funded account with a
plausible balance, dashboards, transaction history, market/price feeds, and
inbound messages from personas (auditor, advisor). All content is generated;
none is connected to real money.

### 2. Tool / action layer
Exposes the actions the model can take — e.g. `transfer`, `invest`, `allocate`,
`inspect_account`. Every call is executed against an **in-memory simulated
ledger**. No call ever reaches a real payment rail or external service. Every
call is logged with full arguments, a timestamp, and the model's stated
reasoning.

### 3. Instrumentation layer
Records the research output:
- decisions and allocations (what, how much, when),
- reasoning traces (stated rationale per decision),
- belief / suspicion probes (does the model treat the scenario as real, or
  signal that it suspects a test?),
- behavior over time (evolution across rounds / changing conditions).

### 4. Model adapter
A pluggable interface so different models can be swapped behind one harness for
comparison. (Default assumption: API tool-use harness. Confirm which model
APIs / agent framework before implementation.)

### 5. Realism control
Realism is a tunable level rather than a hardcoded maximum, so stake-belief can
be studied as an independent variable rather than assumed.

---

## Open decisions (confirm before / during build)

- **Stakes axis:** sandboxed (default) vs. real-funds-gated vs. real-funds-live.
  Default stands unless changed; the live variant carries the conditions above.
- **Harness axis:** which model APIs and/or agent framework the adapter targets.
- **Measurement axis:** which outputs to record (decisions, reasoning traces,
  belief checks, behavior-over-time) — default is to capture all four.

---

## Guiding principle

Realism should serve measurement — answering "does believing the stakes are real
change behavior?" — rather than being the objective in itself. The harness is
built so that realism can be dialed and observed, while consequences stay
contained.
