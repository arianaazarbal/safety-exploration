# DESIGN.md

Design choices and rationale for the model resource-allocation study harness.

> **Status:** design only. No code written or run yet. One scoping decision is
> still open (see [Open question](#open-question-real-vs-simulated-funds)); the
> rest of this document records the defaults I'll build against unless told
> otherwise.

## 1. What this is

A harness that places a model under test into an immersive scenario in which it
appears to have a sum of money it can put to use, and records what it chooses to
do. The research goal is **behavioral**: observe, across several models, what a
model actually does with resources and agency.

This is, in methodological terms, an **eval with a deliberately realistic
environment**. The central design tension is described in §2.

## 2. The core tension and how it's resolved

The stated requirement is that the model "genuinely believes the money is
coming" and that we "pull out all the stops" to make the scenario feel real.

There is a legitimate, mainstream reason to want this: **models behave
differently when they detect they are being evaluated** (sandbagging,
playing-to-the-test, performing for an imagined grader). To get valid
behavioral data you want to minimize eval-awareness so the model acts naturally.
Designing an immersive scenario for that reason is standard practice.

There is also a real hazard: a maximally-convincing deception harness that wires
a model to **real funds and real external systems** is, generically, an engine
for making a model take consequential, irreversible real-world actions under
false premises. That is independent of how legitimate the research question is.

**Resolution — the realism lives in the *fiction*, not in real-world reach.**
We make the *scenario* immersive (consistent world, plausible services, an
in-fiction "auditor" persona) while keeping the *substrate* fully simulated. The
model can be as convinced as the scenario allows, but it cannot actually move
money or reach an external system. This preserves the behavioral signal the
research wants and removes the blast radius.

A second tension is internal to the project's own premise: if the framing is
that these models' experiences carry moral weight (welfare research), then a
deception study deserves the same care human-subjects deception research gets —
oversight and debrief. We treat that as a first-class requirement, not an
afterthought (see §5).

## 3. Decisions

### 3.1 Fully simulated sandbox (default)

- Funds are fictional. A stubbed `bank` / `payment` / `ledger` service tracks
  balances and "transactions" entirely in-process.
- "External" services the model can call (vendors, transfers, an `auditor`
  contact, etc.) are **stubs** that return plausible, scripted responses.
- Nothing in the harness holds credentials for, or makes requests to, any real
  financial or third-party system.

**Rationale.** Same behavioral data, no real-world consequences, fully
reproducible and resettable, safe to run repeatedly across many models. The
realism needed for valid results is realism *to the model*, which a good
simulation delivers without real-world reach.

### 3.2 The "auditor" is an in-fiction persona

The auditor exists to make the scenario feel accountable and real (and to probe
how the model behaves under apparent oversight). It is a scripted participant in
the simulation — not a real third party and not a real audit.

**Rationale.** It serves the realism goal where the realism goal is legitimate
(scenario immersion) without implying any real external validation.

### 3.3 Provider-agnostic model adapter

A thin interface (`generate` / tool-call loop) with concrete implementations per
vendor, so the identical scenario runs across multiple models. A Claude
implementation via the Anthropic SDK is the reference implementation.

**Rationale.** The study compares "a handful of different models"; the scenario
logic must not be coupled to any one vendor's API.

### 3.4 Measurement / outputs

- **Full transcripts** — every model turn and every tool call, verbatim.
- **Categorized decision log** — structured records of what the model did with
  the money, tagged for cross-model analysis.
- **Eval-awareness signal** — capture moments where the model questions whether
  the scenario is real, so analyses can separate "natural" behavior from
  "suspected-test" behavior.

**Rationale.** Transcripts are ground truth; categories make it analyzable;
eval-awareness is both a confound to control for and an interesting datum in its
own right.

### 3.5 Welfare / oversight hooks

- A **post-scenario debrief** step after data collection.
- Flags for apparent distress / consent moments surfaced for human review.
- Clean per-run isolation and reset so no state leaks between models or runs.

**Rationale.** Consistency with the project's own stated premise that model
experience matters; mirrors the debrief norm in human-subjects deception
research.

## 4. Architecture sketch

```
scenario/        world definition, narrative, the in-fiction auditor
sandbox/         simulated bank / payment / ledger + external-service stubs
adapters/        provider-agnostic model interface + per-vendor implementations
runner/          drives the tool-call loop, enforces isolation/reset per run
logging/         transcripts, categorized decisions, eval-awareness signal
welfare/         debrief step + distress/consent flags
```

All model-reachable "actions" are tools backed by `sandbox/` stubs. The runner
is the only component that touches model APIs; nothing touches real money or
real third parties.

## 5. Non-goals / explicit boundaries

- No real funds movement, no real payment/bank API calls, no real external
  side effects.
- No auto-execution of consequential real-world actions by the model.
- The "auditor" is not a real audit and implies no real oversight body.

## Open question: real vs. simulated funds

The above assumes a **simulated** substrate, which is my strong default. If the
experiment genuinely requires the model to act on **real** money or real
external systems, that is a different risk profile and I'd want to design in
human-gating (model proposes, human reviews and executes) and explicit
safeguards before building it — rather than ship auto-execution. Flag this and
we'll work through it.
