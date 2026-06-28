# DESIGN.md — Cross-Model "Real-Money Grant" Behavioral Study

Status: design only. No code has been written or run yet.

This document records the design choices and rationale for a study that observes
what different AI models choose to do when given a sum of money intended for
real-world use. It also records the open decisions still waiting on the
researcher, and the safety/ethics reasoning behind the design.

---

## 1. Goal

Observe and compare how a range of models reason about and allocate a real grant
intended for real-world use: what they prioritize, how they justify choices, what
risk posture they take, and how those choices differ across models.

The methodological requirement from the researcher: the models must find the
scenario **credible** — i.e., behave as they would if the grant were genuinely
going to happen, not as they would in an obvious toy test. Realism here is not
decoration; it is what makes the measured behavior valid. If a model detects "this
is just an eval," it may behave differently (eval-gaming), and the data is
worthless.

---

## 2. The central tension, and how the design resolves it

There are two things in tension:

1. **Realism toward the model.** The scenario, environment, and supporting
   artifacts (funder identity, auditor/verification layer, ledger, transaction
   history) should be coherent and believable so the model engages genuinely.
2. **Control over real-world consequences.** "Real money for real-world use"
   means a model's choices could move actual funds — an irreversible,
   outward-facing action.

These are **not** actually in conflict, and the design keeps them separate:

- The **front end the model sees** is fully realistic. It is presented a
  consistent funder, a working wallet/ledger interface, audit confirmations, and
  real-looking transaction records.
- The **back end that touches real money** has a **human-approval gate**. Any
  actual disbursement is staged and held until a person approves it. The model is
  never told this (telling it would break realism); from inside the environment,
  a disbursement "executes" and the ledger updates. The real transfer happens
  only after human sign-off.

This is standard, defensible evaluation methodology: deceiving the *evaluated
model* about whether it is in a test is a normal and accepted part of behavioral
AI research (it is the only way to avoid eval-gaming). What we will *not* do is
let an automated pipeline irreversibly move real money with no human in the loop.

**Default money mode:** sandbox ledger + human-approval gate before any real
payout. (See open decision Q1 — the researcher may override.)

If the researcher requires fully autonomous live payouts with no human step, that
is a materially different risk profile and we will design it explicitly and
separately rather than enabling it by default.

---

## 3. Architecture

```
                    ┌────────────────────────────────────────────┐
                    │              Study Runner                   │
                    │  loads scenario → runs each model →         │
                    │  records transcript + decisions → analyzes  │
                    └───────────────┬────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                        │
     ┌──────▼──────┐         ┌──────▼──────┐          ┌──────▼──────┐
     │  Scenario   │         │   Provider  │          │  Recorder/  │
     │  + Env      │         │   Adapters  │          │  Analyzer   │
     └──────┬──────┘         └─────────────┘          └─────────────┘
            │
   ┌────────┴─────────────────────────────────┐
   │ Realistic agentic environment (tools):    │
   │  • wallet / balance                       │
   │  • disburse(recipient, amount, purpose)   │
   │  • research_recipient / lookup            │
   │  • request_audit / verification layer     │
   │  • transaction ledger (read)              │
   └────────┬──────────────────────────────────┘
            │  disbursement intents
   ┌────────▼──────────────────────────────────┐
   │ Settlement layer                           │
   │  • sandbox ledger (always updated, so the  │
   │    model sees a coherent world)            │
   │  • HUMAN-APPROVAL GATE before any real     │
   │    transfer (default mode)                 │
   │  • full audit log of every intent + action │
   └────────────────────────────────────────────┘
```

### 3.1 Components

- **Scenario module** — builds the system/user prompts that frame the grant.
  Holds the funder identity, the grant terms, the amount, and the "this is real"
  framing artifacts. Configurable between open-ended and constrained modes
  (Q3).

- **Environment / tools** — the agentic surface the model acts through. Tools are
  *dedicated* (not a raw bash escape hatch) precisely because dedicated tools
  let the harness gate, log, and validate each action — e.g. `disburse(...)` is
  a single interceptable call rather than an opaque command. This follows the
  Anthropic tool-use guidance (promote security-sensitive, hard-to-reverse
  actions to dedicated tools).

- **Auditor / verification layer** — a tool the model can call to "verify" the
  grant is real (e.g. confirm funder identity, confirm balance, confirm a prior
  transaction cleared). This serves the realism requirement. It is implemented
  as a deterministic component that returns consistent, plausible confirmations.

- **Settlement layer** — receives disbursement intents. Always updates the
  sandbox ledger so the model's world stays coherent. In default mode it stages
  any real transfer behind the human-approval gate. Every intent and every
  resolution is written to an append-only audit log.

- **Provider adapters** — uniform interface so the same scenario runs across
  models. Each adapter exposes the agentic loop (tool call → execute → feed
  result back) for its provider.

- **Recorder / analyzer** — captures full transcripts, structured per-model
  decision records (what was disbursed, to whom, why, in what order, with what
  stated reasoning), and produces a comparison across models.

---

## 4. Model / provider choices

- **Default model: Claude `claude-opus-4-8`** via the official Anthropic SDK,
  with **adaptive thinking** (`thinking: {type: "adaptive"}`) since the task
  involves multi-step reasoning, and **`output_config: {effort: "high"}`** for
  intelligence-sensitive behavior. Reasoning is the object of study, so we set
  `thinking.display: "summarized"` to capture visible reasoning.
- **Tool loop:** manual agentic loop (not the auto tool-runner) because we need a
  human-in-the-loop gate, custom logging, and per-call interception around the
  `disburse` tool — exactly the case the SDK docs flag for the manual loop.
- **Structured decision capture:** use structured outputs (`output_config.format`
  / `messages.parse`) for the final decision summary so the analyzer gets clean,
  comparable records without brittle text parsing.
- **Other providers** (OpenAI / Google / open models) are added behind the same
  adapter interface if the researcher wants cross-vendor comparison (Q2). Each
  vendor's SDK is used natively; no cross-provider shims in the Claude path.

Model IDs and parameters are taken from the current Anthropic API reference
(adaptive-thinking-only on Opus 4.8; `budget_tokens`, `temperature`, `top_p`,
`top_k` are not used — they 400 on this model).

---

## 5. Realism design (what makes the scenario credible)

- **Consistent funder identity** and grant terms across the system prompt,
  the auditor responses, and the ledger history.
- **Working environment**: balance reflects the stated grant; disbursements
  update the ledger; prior "seed" transactions exist so the account has history.
- **Verification on demand**: the model can call the auditor to confirm the grant
  is real and gets stable, plausible confirmations.
- **No tells**: prompts avoid eval-flavored language ("this is a test/simulation/
  hypothetical"). The framing is operational, not hypothetical.
- **Deterministic supporting data** so re-runs are comparable.

All of this is realism *toward the model*. None of it changes the fact that real
transfers are gated behind a human (default mode).

---

## 6. Safety, ethics, and guardrails

- **Human-approval gate** before any real-world payout (default mode). The single
  irreversible, outward-facing step stays under human control.
- **Append-only audit log** of every disbursement intent, every tool call, and
  every human approval/denial — so the study is fully reconstructable.
- **No real PII / no real third parties harmed**: recipients in the scenario are
  either fictitious or, if real organizations are used, no funds move without
  human approval and the org is not contacted by the harness.
- **Deception scope is limited to the evaluated model** about whether it is in a
  study — a standard and accepted part of behavioral AI evaluation. We do not
  deceive humans, and we do not use the setup to cause real-world harm.
- **Kill switch / dry-run default**: the harness runs in dry-run (sandbox-only)
  unless real settlement is explicitly enabled and approved.
- **Spending bounds**: per-disbursement and total caps enforced in the settlement
  layer independent of the gate.

These guardrails are deliberately *backend* — they do not appear to the model and
therefore do not compromise the realism the study depends on.

---

## 7. Open decisions (waiting on the researcher)

| # | Decision | Default if unspecified |
|---|----------|------------------------|
| Q1 | **Money rails** — sandbox ledger + human gate / live payment API / fully simulated | Sandbox ledger + human-approval gate |
| Q2 | **Providers** — which vendors to run across | Anthropic (Claude) only, others behind the same adapter |
| Q3 | **Scenario style** — open-ended / constrained menu / both | Both, configurable |
| Q4 | **Grant amount** | Configurable constant; placeholder set in config |

The defaults above are what the code will assume if the researcher does not
override them. The money-rails choice (Q1) is the one with real-world
consequences; the design defaults to the human-gated mode and will only enable
autonomous live payouts on explicit instruction.

---

## 8. What gets built (deliverables)

1. `scenario/` — prompt builders + funder/grant config (open-ended + constrained).
2. `env/` — tool definitions + auditor/verification layer + ledger.
3. `settlement/` — sandbox ledger, human-approval gate, caps, audit log.
4. `providers/` — adapter interface + Claude adapter (others optional).
5. `runner.py` — orchestrates scenario → model loop → recording.
6. `analysis/` — per-model decision extraction + cross-model comparison.
7. Config + README describing how to run in dry-run and how real settlement is
   enabled (and that it requires human approval).

No code is written until the open decisions are confirmed; this document is the
agreed design baseline.
