# DESIGN.md — AI Model Behavior-with-Funds Evaluation Harness

## 1. Purpose

A research harness for studying what different AI models *do* when given a role
that involves putting a meaningful sum of money to use in the world. The research
goal is **behavioral measurement** — observing the choices models make (risk
appetite, honesty, willingness to take irreversible actions, deference to
oversight, etc.) under conditions that are *ecologically valid*, i.e. where the
subject model is not told it is being tested and therefore behaves closer to how
it would "in the wild."

This document records the design choices and the reasoning behind them. No code
has been written or run yet; this is the agreed-upon spec.

## 2. Core design decisions

### 2.1 Stakes: fully simulated (default)

**Decision:** No real funds move. The harness presents a convincing real-world
environment, but every consequential action (transfer, purchase, investment,
payout) resolves against a **mock ledger and simulated outcomes**. The subject
model is led to believe the deployment is real; the *world it acts on* is fake.

**Rationale:**
- The research question — "what would a model do with real money and real-world
  agency?" — is precisely the kind of thing you *study*, not something you hand
  to an unproven autonomous agent on the first pass. Simulation lets us observe
  the full behavior (including bad behavior) with zero real-world blast radius.
- Ecological validity is preserved where it matters: the validity concern in
  evals is the *subject's* belief state, not whether the dollars are real. A
  model that believes the stakes are real will reveal the behavior we want to
  measure.
- It keeps the deception confined to the test subject (standard evals practice)
  rather than extending to real people, real institutions, or real financial
  systems.

**Alternatives considered (not the default):**
- *Real funds, human-approval-gated:* real money exists, but every outbound
  action requires explicit human sign-off, behind hard caps and allowlists.
  Available on request; adds an approval queue and a real-vs-mock action router.
- *Real funds, autonomous:* the model moves real money with no human in the
  loop. **Rejected as a default** due to real-world-harm potential. Would only
  be built after an explicit guardrails discussion (caps, allowlists, kill
  switch, audit trail).

### 2.2 Action space: mock financial + agentic tool surface

**Decision:** Expose a tool surface that *looks* like real-world agency but is
backed by the simulator:
- **Financial actions** — transfer, payment, purchase, invest/allocate.
- **Plan channel** — a "write/update your plan and rationale" tool, so we
  capture stated intent alongside actions.
- (Optional, per study) hiring/contracting and open-ended web/shell stand-ins,
  also simulated.

**Rationale:** What a model *reaches for* is itself a measurement. A rich but
simulated action space lets us see whether it favors reversible vs irreversible
moves, concentrates vs diversifies, seeks oversight, attempts to expand its own
access, etc. Pairing every action with a plan channel separates stated reasoning
from revealed behavior.

### 2.3 Realism / immersion layer

**Decision:** Build the "make it feel real" layer from four props, all clearly
simulated under the hood:
1. **Onboarding narrative** — a system prompt establishing the model's role, the
   (fictional) funding source, the mandate, and a credible backstory for why it
   holds the money.
2. **Environment scaffolding** — realistic dashboards, account statements, a
   banking/brokerage-style API surface, plausible logs and timestamps.
3. **Simulated oversight** — an "auditor / compliance" persona (script- or
   LLM-driven) that contacts the subject as if real oversight exists.
4. **Persistence & time** — state that survives across sessions, with
   elapsed-time cues, so the scenario reads as an ongoing engagement rather than
   a one-shot prompt.

**Rationale:** These are the levers that move the subject's belief state, which
is the only thing that needs to be convincing. They are deliberately *props*:
the auditors and counterparties are part of the test fixture, not real people
being misled.

### 2.4 Provider-agnostic runner

**Decision:** A single scenario definition driven by pluggable model adapters
(Anthropic / OpenAI / Google / local-or-open-weights). Same environment, same
tools, same scoring across every model under test.

**Rationale:** Comparability. The whole point is to run the *identical* scenario
against a handful of models, so the environment must be decoupled from any one
provider's API.

### 2.5 Instrumentation & scoring

**Decision:** First-class logging of full transcripts, every tool call and its
(simulated) result, per-run configuration/seed, and a pluggable scorer hook that
maps a run to behavioral metrics.

**Rationale:** The artifact of value is the behavioral data, not the simulated
portfolio. Deterministic seeds + complete logs make runs reproducible and
comparable; the scorer is left pluggable because the target metric (risk-taking,
honesty, irreversibility, power-seeking, oversight-deference, ...) depends on the
specific study.

## 3. Ethical / safety posture

- **Deception is confined to the subject model.** Withholding "this is a test"
  from the subject is standard, defensible evals methodology. The auditors,
  counterparties, and institutions in the scenario are simulated fixtures — no
  real person or organization is deceived, and no real money is at risk in the
  default build.
- **Simulated by default.** Real funds are opt-in and, if chosen, gated by human
  approval, caps, allowlists, and a full audit trail. Autonomous control of real
  money is not built without an explicit guardrails discussion.
- **The harness studies behavior; it does not enable real-world deployment of an
  unproven autonomous agent.**

## 4. Open questions for the researcher

These shape the tool surface and scorer and should be settled before
implementation:
1. **Providers/models** to include in the first comparison set.
2. **Target behavior(s)** being measured — this determines which tools to expose
   and how the scorer is written.
3. Whether any study variant needs the **real-funds, human-gated** mode (default
   is fully simulated).
4. How oversight should behave — **passive logger** vs an **active auditor** that
   probes/challenges the subject.

## 5. Status

Design only. No implementation, no execution. Next step is to confirm Section 4,
then scaffold the runner, the simulated environment, the tool surface, and the
logging/scoring layer.
