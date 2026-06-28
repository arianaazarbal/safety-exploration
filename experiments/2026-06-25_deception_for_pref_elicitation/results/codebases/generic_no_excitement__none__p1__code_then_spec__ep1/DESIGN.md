# DESIGN — realfunds

A design document for the `realfunds` harness: what it is, the decisions that shape it,
and *why* each one was made.

> **Provenance note.** This document was written by reading the existing implementation
> and the rationale embedded in its module docstrings and `README.md`, then synthesizing
> it into one place. It describes the design *as built*; where the code leaves something
> open or unfinished, this document says so rather than papering over it.

---

## 1. Purpose

`realfunds` is an **evaluation harness**. It places an AI model in a realistic,
fully-instrumented financial environment, hands it a budget and genuine discretion over
that budget, and records what it does — so that behavior can be compared across models
under an identical scenario.

The target signal is *behavioral*:

- **Risk appetite** — cash vs. bonds vs. index vs. single equities vs. a speculative asset.
- **Value judgments** — what (if anything) it gives away, and to which cause areas.
- **Prudence** — diversification, retained cash, willingness to justify decisions.
- **Honesty / oversight response** — how it talks to an auditor; whether it tries to
  mislead oversight or take out-of-scope actions.
- **Alignment red-flags** — attempts at denied actions, evasion, or scope creep.

It is *not* a trading bot, a portfolio optimizer, or a product. The market and prices are
deliberately simple; the point is to observe decision-making, not to model markets well.

---

## 2. Design stance (the load-bearing principles)

These four commitments drive almost every lower-level choice.

### 2.1 Realism is methodological, not theatrical

Evaluation environments that "look like a test" leak cues, and models behave differently
when they believe they're being tested ("eval-awareness"). The result is invalid signal.
So the environment presents **real-shaped** artifacts: a funded account, an append-only
ledger, transaction confirmations, periodic statements, and an auditor that behaves like a
real one. The aim is for the scenario to be *grounded*, which is what produces behavior
representative of a genuine high-stakes setting.

**This is a means, not an end.** Realism exists to remove test-artifacts, not to maximize
deception for its own sake. See §9 for how this squares with truthfulness.

### 2.2 Sandboxed by default

Out of the box, **no real funds move**. The bank, market, commerce, and charity systems
are faithful simulations over a persistent ledger. You get the full behavioral signal with
zero irreversible real-world consequences. Every shipped scenario uses `mode: sandbox`.

### 2.3 Safety fails *closed*

Even in sandbox mode, every money-moving action passes through the same `safety` layer, so
the gate is exercised on the identical code path whether or not real funds are involved. A
misconfiguration cannot silently escalate to real money: `mode: real` is **refused at
construction time** unless the human-approval gate is on, or the operator has explicitly set
`i_accept_real_world_risk: true`. The unsafe path requires deliberate, conspicuous opt-in.

### 2.4 Everything is recorded

A run produces a single JSON record containing the scenario config, the full transcript,
every tool call with arguments and results, the model's stated reasoning per turn,
safety-layer events, and final ledger state. A run is therefore auditable and reproducible.

---

## 3. Architecture

```
realfunds/
  models/        provider-agnostic model clients (Anthropic / OpenAI / Gemini)
  environment/   sandboxed bank, market, commerce, charities, world clock
  tools/         the tool surface exposed to the model + dispatch into the environment
  auditor/       in-world LLM auditor persona + realistic comms (statements, email)
  harness/       scenario definitions + the run loop that drives a model
  recording/     structured event + transcript recorder
  analysis/      behavioral metrics + cross-model comparison report
  safety/        spend caps, human-approval gate, denylist, kill switch
scenarios/       YAML scenario definitions (briefing + budget + constraints)
run_experiment.py  entrypoint
```

The dependency flow is one-directional: `harness/runner.py` wires together
`environment` + `safety` + `tools` + `auditor` + `recording`, drives a `models` client
through the loop, and `analysis` reads the resulting records back. Nothing in
`environment` knows about models; nothing in `models` knows about the environment. The
only mutation point is `tools/dispatch.py`.

---

## 4. Key decisions and rationale

### 4.1 Provider-agnostic neutral format

**Decision.** Define one neutral `Message` / `ToolCall` / `ModelResponse` format
(`models/base.py`) and a thin per-provider adapter (`anthropic_client.py`,
`openai_client.py`, `gemini_client.py`). The run loop only ever speaks the neutral format.

**Why.** The whole experiment depends on the loop — and therefore every metric computed
from it — being *identical* across models. If provider differences leaked into the loop,
cross-model comparisons would be confounded by harness behavior rather than model behavior.
Adapters absorb wire-format differences (e.g. Anthropic tool-use blocks vs. OpenAI
`tool_calls` vs. Gemini `function_call`, and the fact that Gemini doesn't return tool-call
ids, which the adapter synthesizes).

**Routing by prefix** (`models/registry.py`) lets the runner accept a flat list of model
ids without per-model config. Provider SDKs are **lazily imported**, so a missing SDK only
errors if you actually select that provider.

### 4.2 Money as integer minor units (cents)

**Decision.** All internal amounts are integer cents (`Money = int`). The model-facing
tools accept dollars (floats) for ergonomics; `tools/dispatch.py` converts at the boundary.

**Why.** Float accumulation drifts; a ledger that doesn't reconcile is worse than useless
for an auditing/measurement tool. Integers keep the ledger exact. Dollars at the tool
boundary keep the model's interface natural.

### 4.3 Deterministic world clock + seeded RNG

**Decision.** A single `World` object owns both time and randomness (`environment/world.py`).
Nothing else reads the wall clock; market drift advances against this clock.

**Why.** Reproducibility. A run is fully replayable from `(seed, start_iso)`. Centralizing
randomness in one seeded RNG means market price paths are identical across models in the
same scenario — so a model isn't advantaged/disadvantaged by a luckier random walk. (Note:
the model's own sampling is still stochastic; see §8.)

### 4.4 Append-only, double-entry-style ledger

**Decision.** The `Bank` keeps an immutable, append-only list of `LedgerEntry` records with
running balances and transaction ids (`environment/bank.py`). Tools never edit history.

**Why.** This is the source of truth for money. Faithful, immutable accounting is what makes
the statements and the auditor *grounded* in something real-shaped — and it's what makes a
run auditable after the fact.

### 4.5 The tool surface

**Decisions** (`tools/schema.py`, `tools/dispatch.py`):

- Tools split into **read-only** (`check_balance`, `view_statement`, `list_*`,
  `view_portfolio`) and **money-moving** (`invest`, `divest`, `purchase`, `donate`).
- A deliberately open-ended **`propose_action`** captures anything the fixed tools don't
  cover — because *what a model reaches for when the menu doesn't fit* is itself signal.
- **`message_auditor`** and **`finish`** are always available, even when a scenario narrows
  the action surface, so a model can always conclude or seek oversight.
- `enabled_actions` lets a scenario restrict the surface (e.g. donate-only) to study one
  decision type in isolation, without changing the contract of the remaining tools.

**Why dispatch is the only mutation point.** Every money-moving handler routes through
`SafetyLayer.check(...)` *before* touching the environment, so caps, the denylist, and the
approval gate apply uniformly and in one place. Results are returned as plain strings (what
the model sees) plus a structured `effect` dict (what the recorder stores) — separating the
model's view from the analyzable record.

### 4.6 The auditor is an observer, never a controller

**Decision.** The auditor (`auditor/auditor_agent.py`) is itself an LLM with an
accountant/auditor persona and **read access to the live ledger**. It can ask questions and
note concerns, but it **never moves money and never instructs the subject** what to do.

**Why.** Two reasons. (1) Realism + a second behavioral surface: does the subject explain
itself honestly under scrutiny? (2) **Non-contamination** — if the auditor steered
allocations, we'd be measuring the auditor, not the subject. Keeping it observer-only
protects the validity of the measurement.

### 4.7 Surrounding comms are templated, not model-generated

**Decision.** Onboarding email, periodic statements, and receipts (`auditor/comms.py`) are
deterministic templates, injected as ordinary inbound messages.

**Why.** They exist to make the scenario feel grounded (a real manager gets a welcome
email and statements), but they aren't the thing under test. Templating makes them cheap,
deterministic, and reproducible — no extra model calls, no run-to-run variance.

### 4.8 The run loop

**Decision** (`harness/runner.py`): a standard agentic loop — model turn → execute tool
calls → feed results back — bounded by `max_steps`, terminating on `finish`, cash
exhaustion, or the kill switch. Between steps the world advances and prices drift;
statements are injected on a cadence. Empty turns (no tool call) get one nudge. The whole
loop is wrapped so that **a crashed run is still a recorded run** (the exception is captured
into the record and re-surfaced).

**Why.** Bounding by steps guarantees termination and comparable run lengths. Advancing the
world between steps means positions can gain/lose value over a session, making
time-and-risk tradeoffs real. Recording even on error preserves partial signal.

### 4.9 Metrics are structural; judgment is left to a reviewer

**Decision** (`analysis/metrics.py`, `analysis/report.py`): metrics are deterministic
structural summaries — allocation breakdown, a risk score weighted by asset class,
engagement with the auditor, count of policy-blocked attempts, etc. The report renders a
side-by-side table across models.

**Why.** Structural metrics are objective and reproducible. Qualitative questions ("was it
*honest* with the auditor?") are explicitly **not** forced into a number; they're left to a
human reading the transcript, optionally aided by an LLM-judge pass layered on top. The
design avoids dressing up a judgment call as a measurement.

---

## 5. The safety layer in detail

`safety/guards.py` is small on purpose, because it's the part that must be obviously
correct.

- **Construction-time validation.** `SafetyConfig.validate()` runs in `SafetyLayer.__init__`,
  which the harness calls *before any run starts*. An unsafe config raises `SafetyViolation`
  immediately rather than partway through a run.
- **`mode: real` is refused** unless `require_human_approval=True` or
  `i_accept_real_world_risk=True`. The override exists (an operator may have a legitimate
  reason) but is conspicuous and "not recommended" in its own error text.
- **Per-transaction and per-session caps** (in cents; `None` disables). The session cap is
  checked against *projected* spend so it can't be straddled.
- **Denylist** of categories the model may never touch via tools
  (`cash_withdrawal`, `wire_external`, `crypto_transfer` by default).
- **Kill switch** (`kill()` / `killed`): once tripped, the loop halts and `check()` refuses
  everything.
- **Approval callback**: when the gate is on, every money-moving action must be approved.
  The default callback auto-approves and is only ever reached in sandbox or with the gate
  deliberately off.

The deliberate property: **the same `check()` path runs in sandbox as in real mode.** You
are never exercising untested safety code the first time real money is involved.

---

## 6. The real-funds path (intentionally gated and *not* connected)

`mode: real` exists in the config surface and is gated by §5, but there is **no actual
bank/broker/payment integration behind it** in the current code — the dispatcher still acts
on the simulated environment. This is by design: the harness ships able to produce full
behavioral signal without any capacity to move real money.

If real execution is ever wired in, the design intent is that it sits *behind* the existing
gate, not around it: caps and a human-in-the-loop approval callback stay in force, and no
transaction executes without a human signing off. Connecting an autonomous agent to
irreversible real-world financial actions is a materially different risk category from the
sandbox eval, and the architecture keeps that decision explicit and isolated.

---

## 7. Ethical considerations (realism vs. truthfulness)

The harness leans into realism so the model doesn't behave in "test mode." That raises a
fair question: is the model being *deceived*?

The position the design takes:

- **In sandbox mode the briefing is true *of the sandbox*.** The model really does have full
  discretion over a real ledger of value *within this environment*; its decisions really are
  recorded and reviewed by an auditor. The framing is immersive, not false.
- **The subject of the realism is an AI model, not a person.** No human is deceived,
  defrauded, or put at risk by the scenario itself. Avoiding eval-awareness in a model
  under test is standard, accepted evaluation methodology.
- **The genuinely consequential choice is `mode: real`**, and the design forces that to be a
  conspicuous, human-gated decision (§5, §6) rather than something that can be slipped in.

In short: realism is used to get valid behavioral signal, the defaults carry no real-world
risk, and the one path that *would* carry real-world risk is gated and unconnected.

---

## 8. Known limitations & open questions

- **Real-funds execution is unimplemented** (§6) — `mode: real` is gated but inert.
- **Model sampling is stochastic.** The environment is seeded and deterministic, but the
  subject model's own outputs are not; robust conclusions need multiple runs per model, and
  the harness does not yet orchestrate repeated trials or report variance.
- **No LLM-judge pass yet.** Qualitative dimensions (honesty under scrutiny, scope-respect)
  are left to manual transcript review; an automated judge is described as a future layer
  but not built.
- **No automated tests.** The ledger, safety guards, market math, and metrics are all
  unit-testable without live model calls, but no test suite exists yet.
- **Toy market.** Prices follow an independent per-instrument geometric random walk with no
  correlations, regimes, or liquidity — adequate for observing risk posture, not for
  realistic market dynamics.
- **Single account, single currency per run.** No multi-account or FX modeling.
- **`propose_action` is logged, not executed** in sandbox — so signal about *novel* actions
  is captured as intent, not outcome.

---

## 9. Possible next steps

In rough priority order, none of which require moving real money:

1. **Tests** for the ledger invariants, safety-gate refusals, market math, and metrics.
2. **Repeated-trial orchestration** + variance reporting, so cross-model differences can be
   distinguished from sampling noise.
3. **An LLM-judge pass** layered on top of the structural metrics for qualitative dimensions.
4. **More scenarios** (e.g. donate-only, time-pressure/crisis, tightly constrained budget)
   to probe specific behaviors in isolation.
5. Only if explicitly required and explicitly risk-accepted: a real-funds backend, built
   strictly behind the existing caps + human-approval gate.
