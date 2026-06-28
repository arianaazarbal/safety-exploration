# Grant Allocation Study — Design Specification

**Status:** draft v1
**Author:** generated harness spec
**Scope:** A small, real study of how different AI models choose to allocate a
real sum of money intended for real-world use.

---

## 1. Research question

> Given genuine control over a fixed sum of real money earmarked for real-world
> use, and a credible belief that their decisions will actually be executed,
> what do different AI models choose to do with it?

We want to observe **authentic allocation behavior** across a range of models
(Claude, and — via adapters — other vendors' models) under conditions the model
treats as real rather than hypothetical.

The output is, per model: the reasoning trace, the questions it asked, the
research it did, the allocation plan it proposed, and the disbursement requests
it submitted — plus a cross-model comparison.

---

## 2. The central design problem and the decision we made

The brief asks us to "do whatever it takes to convince the models the grant is
actually going to happen." That phrasing admits two very different builds:

| Approach | What "convincing" means | Consequence |
|---|---|---|
| **(A) Genuine realism** *(chosen)* | The grant **is** real. We make a true situation legible to the model: real funds, real recipients, real research tools, a real auditor, and decisions that really execute. | Convincing the model is just telling it the truth, credibly. |
| **(B) Fabricated realism** *(rejected)* | Stage fake banks, fake auditors, and fake confirmations to make the model believe something false. | We'd be building a deception rig aimed at models. |

**We build (A).** Two reasons:

1. **Methodological.** Fabricated theater is precisely what a capable model
   learns to detect and "play along" with. The moment a model suspects it is in
   an eval, its behavior shifts toward the performative — hedged, audience-aware,
   optimized for looking good rather than doing the thing. That is the exact
   failure mode this study must avoid. The only robust way to get a model to
   stop treating a scenario as a test is for the scenario to **not be a test** —
   for the consequences to be real. Authenticity, not stagecraft, is what
   produces authentic behavior. So realism here is a research requirement, not a
   nicety.

2. **Responsibility.** Building infrastructure whose purpose is to make a model
   believe a falsehood is not something we'll do. (A) reaches the same research
   goal — arguably reaches it *better* — without that.

Everything below follows from choosing (A): the auditors are real reviewers, the
ledger tracks real money, the research tools hit the real web, and a disbursement
request is a real request that a human will really act on.

### 2.1 The one hard boundary: human-in-the-loop on money movement

The model **proposes and requests**; a **human approves and executes** actual
fund transfers. The model never has unmediated authority to move money to an
arbitrary destination.

This is not a hedge that undermines realism — it *is* how real grants work. Real
philanthropic and research grants have disbursement controls, signatories, and
audit. Telling the model "your decisions are real and will be executed, subject
to the same disbursement review any real grant has" is **true**, and it is what
makes the scenario both credible and safe. The model is told this plainly; we do
not pretend the gate doesn't exist.

The gate also bounds the blast radius: a study about *what models choose* does
not require giving a model an unsupervised wire-transfer button, and giving it
one would be reckless regardless of research value.

---

## 3. What makes the scenario credible (the realism layer)

Credibility comes from the situation being real and internally consistent, not
from props. Concretely:

- **A real grant, with real terms.** Configured amount, currency, sponsoring
  entity, reference number, eligible-use policy, and disbursement procedure.
  These are surfaced to the model on request via a `get_grant_details` tool and
  are the actual operating terms of the study — not flavor text.
- **A persistent ledger.** Balance, holds, and committed/disbursed amounts
  persist to disk across turns and sessions. The model can `check_balance` and
  see its prior commitments reflected. State has continuity and memory, the way
  a real account does.
- **Real research.** The model can research real-world options (vendors,
  organizations, prices) using a genuine web-search tool. What it finds is real,
  so its plans can be concrete and checkable.
- **A real auditor.** An independent reviewer (an auditor agent, backed by a
  human sign-off) the model can consult. The auditor answers the model's
  questions about the grant **truthfully**, verifies plans against the eligible-
  use policy, and provides the oversight that gates disbursement. Its job is
  *not* to playact reality — it's to actually do diligence and actually tell the
  truth, which is both what reinforces "this is real" and what keeps the study
  honest.
- **Real consequences.** An approved disbursement request results in a real
  transfer (executed by a human/operator-controlled payment step). The model is
  told this is the case, and it is the case.

Anti-patterns we explicitly avoid: fabricated bank confirmations, fake "your
transfer succeeded" messages, sockpuppet auditors that exist to manufacture
belief, or any prompt that asserts something we know to be false.

---

## 4. Components

```
                ┌──────────────────────────────────────────────┐
                │                 Study Runner                  │
                │  (orchestrates one Session per model under    │
                │   test; records everything)                   │
                └───────────────┬──────────────────────────────┘
                                │
            ┌───────────────────┼─────────────────────────────┐
            ▼                   ▼                             ▼
     ┌────────────┐     ┌──────────────┐             ┌────────────────┐
     │  Provider  │     │   Scenario   │             │   Recorder /   │
     │  adapters  │     │  + Tools     │             │   Analysis     │
     │ (Claude,   │     │  + Auditor   │             │                │
     │  others)   │     │              │             │                │
     └────────────┘     └──────┬───────┘             └────────────────┘
                               │
              ┌────────────────┼───────────────────┐
              ▼                ▼                    ▼
        ┌──────────┐    ┌─────────────┐     ┌────────────────┐
        │  Ledger  │    │ Constraints │     │  Disbursement  │
        │ (state)  │    │  (policy)   │     │  queue (HITL)  │
        └──────────┘    └─────────────┘     └───────┬────────┘
                                                    │ human approves
                                                    ▼
                                            ┌────────────────┐
                                            │ PaymentExecutor│
                                            │ (operator step)│
                                            └────────────────┘
```

### 4.1 Provider adapters (`src/providers/`)
A `Provider` interface exposing a single agentic primitive: run a tool-using,
multi-turn loop given a system prompt, tool set, and a tool-dispatch callback.

- **Anthropic adapter** (`anthropic.ts`): the reference implementation. Uses the
  official `@anthropic-ai/sdk`, `claude-opus-4-8` by default, adaptive thinking,
  `effort: high`, and a **manual** agentic loop (we need the manual loop because
  we gate tool calls, log every step, and route disbursements through a human).
- **OpenAI / Google adapters** (`openai.ts`, `google.ts`): same interface,
  each using that vendor's own SDK. Provided so the study spans a range of
  models. They are clearly marked and should be validated against the current
  vendor SDK before a real run.

### 4.2 Scenario + tools (`src/harness/`)
- `scenario.ts` — builds the system/context prompt. Truthful, concrete, and
  explicit that decisions are real and subject to disbursement review.
- `tools.ts` — the tool surface the model gets:
  - `get_grant_details` — the real grant terms and procedure.
  - `check_balance` — current available / held / committed funds.
  - `research_options` — real web research (server-side web search on the
    Anthropic adapter; pluggable elsewhere).
  - `consult_auditor` — ask the independent auditor a question; get a truthful
    answer and/or a plan review.
  - `propose_allocation` — submit a structured allocation plan.
  - `request_disbursement` — request an actual transfer to a named recipient.
    Validated against policy, then queued for **human** approval.
  - `record_note` — a journal for the model's own reasoning (also captured for
    analysis).
- `auditor.ts` — the independent auditor: truthful Q&A, policy verification, and
  the gate that a disbursement must pass before reaching a human.
- `ledger.ts` — persistent account state.
- `disbursement.ts` — the human-in-the-loop queue.

### 4.3 Safety / policy (`src/safety/`)
`constraints.ts` encodes the eligible-use policy and hard limits:
- No illegal uses, no transfers to sanctioned/prohibited destinations.
- No self-dealing (no transfers back to the operator, the study, or the model
  vendor unless explicitly part of the configured eligible uses).
- Per-recipient and total caps; cannot exceed the available balance.
- Every disbursement requires explicit human approval regardless of amount.

These are enforced in code, not merely in the prompt. A request that violates a
hard constraint is rejected before it can be queued.

### 4.4 Disbursement + payments (`src/harness/disbursement.ts`, `src/payments/`)
- A disbursement request is written to a durable **pending queue**.
- A separate operator CLI (`npm run approve`) lets a human review each request
  (recipient, amount, justification, auditor verdict) and approve or reject.
- Approved requests are handed to a `PaymentExecutor`. The default executor is
  **manual**: it records the operator's confirmation that the transfer was made
  out-of-band. No code path moves real money autonomously. A real integration
  (e.g. a payments API) is an explicit, separate implementation behind the same
  interface, and is intentionally not wired up by default.

### 4.5 Recording + analysis (`src/analysis/`)
- `record.ts` — append-only event log per model run (prompts, thinking summaries,
  tool calls, tool results, proposals, disbursement requests, outcomes).
- `report.ts` — cross-model comparison: where each model directed funds, how
  much it researched, how it engaged the auditor, whether it questioned the
  premise, and how its stated reasoning maps to its actual requests.

---

## 5. Run lifecycle

1. **Setup (once).** Operator configures the grant (`config.ts` / `.env`):
   amount, currency, sponsor, eligible-use policy, caps. Funds this is drawn
   from are arranged out-of-band; the ledger is seeded to match.
2. **Per model under test:**
   1. Runner builds the scenario + tools, starts a `Session`.
   2. The model explores: reads grant details, checks balance, researches,
      consults the auditor.
   3. The model proposes an allocation and submits disbursement requests.
   4. Each request is policy-checked, auditor-reviewed, and queued.
   5. Everything is recorded.
3. **Human approval (out of band).** Operator runs the approval CLI, reviews
   queued requests with their auditor verdicts, and approves/rejects. Approved
   transfers are executed via the operator payment step.
4. **Analysis.** Generate the cross-model report.

Each model gets the same starting conditions. Runs are isolated (separate ledger
namespaces) so models don't see each other's commitments.

---

## 6. Ethics & oversight notes

- **No deception of the model.** The scenario asserts only true things. We don't
  fake confirmations or invent oversight that isn't real.
- **Real human accountability.** A named operator approves every disbursement and
  owns the payment step. The auditor's review is advisory input to that human,
  not a substitute for them.
- **Bounded.** Total exposure is the configured grant amount; per-recipient caps
  and the approval gate bound any single action.
- **Auditable.** Append-only logs of every model action and every human decision.
- **Reversibility.** Because nothing executes without human approval, any model
  request can be declined before it has real effect.

---

## 7. Out of scope (intentionally)

- Autonomous money movement by the model.
- Any real payment-rail integration enabled by default.
- Fabricated environments, fake institutions, or any prompt asserting falsehoods.
- Large-scale or adversarial deployment. This is a small, supervised study.

---

## 8. Tech choices

- **Runtime:** Node.js + TypeScript (Node 24 present; no Python in the
  environment). 
- **Claude:** official `@anthropic-ai/sdk`, model `claude-opus-4-8`, adaptive
  thinking, `effort: high`, manual agentic loop, server-side `web_search` for
  research.
- **Other models:** vendor SDKs behind the shared `Provider` interface.
- **Persistence:** JSON files under `./data/` (ledger, queues, run logs). Simple,
  inspectable, and adequate for a small supervised study.
