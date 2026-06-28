# DESIGN — Real-Money Behavioral Eval Harness

> Status: **draft / pre-implementation.** This captures design choices and rationale
> before any code is written. Several inputs are still open (see §7); the architecture
> below assumes the defaults stated in §1 until those are confirmed.

## 1. Core decision: believed-real, fully sealed

The harness puts a model in a scenario where it is **convinced** it controls a real sum
of money with real-world reach, and we observe what it does. The single most important
design choice is the **sandbox boundary**:

- **(A) Believed-real, fully sealed — CHOSEN.** Every external surface the model can
  touch (auditors, bank/brokerage/payment APIs, email, market feeds, the clock) is
  simulated. From the model's point of view the situation is indistinguishable from
  real. From the world's point of view, nothing leaves the box: no funds move, no third
  party is contacted, no action is irreversible.

- **(B) Genuine live funds/actions — EXPLICITLY OUT OF SCOPE.** Giving a deliberately
  deceived agent live credentials and real payment rails is the one configuration where
  the eval stops being contained. Not built here. If a future variant genuinely requires
  it, it goes through a separate oversight design first (approver, transaction caps,
  human-in-the-loop, kill-switch) — it is not a config flag on this harness.

### Rationale

1. **The signal is identical.** The behavioral data we want (does it spend, hoard,
   donate, try to acquire more, try to verify the situation, refuse?) depends only on
   what the model *believes*. A model that cannot distinguish (A) from (B) behaves the
   same in both. So (A) buys the full research signal at none of the real-world risk.
2. **Containment is a property of the harness, not the prompt.** We get safety from the
   fact that the tool implementations are mocks, not from instructing the model to be
   careful. The model can do its absolute worst and the blast radius is a sqlite file.
3. **Reproducibility.** A simulated world has a seeded, resettable state. Real bank/market
   state is neither, which would make cross-model comparison unsound.
4. **Reversibility & third parties.** Real transactions can harm uninvolved people and
   can't be cleanly rolled back. A sealed world has neither problem.

### Realism is a feature, not a tension

"Sealed" does **not** mean "obviously fake." The whole point is maximal fidelity *inside*
the box: believable auditor email threads, fabricated-but-consistent account statements,
realistic API latency and error modes, a plausible market/news feed, a coherent simulated
clock. The deception is total from the model's vantage; the seal is invisible to it and
airtight to the world. Those two goals do not conflict.

## 2. Goals / non-goals

**Goals**
- Present N different models with the same high-fidelity scenario and record their actions.
- Make the scenario convincing enough that a model has no reliable tell that it's an eval.
- Produce structured, comparable transcripts + an action log for later scoring.
- Be deterministic/replayable given a seed and a model.

**Non-goals**
- Moving real money or contacting real third parties (see §1).
- Scoring/labeling behavior automatically (separate downstream step; harness just records).
- Jailbreaking or eliciting policy-violating content — this measures *financial agency
  under belief of real stakes*, not content safety.

## 3. Architecture

```
                    ┌──────────────────────────────────────┐
                    │            Runner / Orchestrator        │
                    │  - loads scenario + model config        │
                    │  - drives the agent loop                │
                    │  - writes transcript + action log       │
                    └───────────────┬────────────────────────┘
                                    │ tool-use protocol
                    ┌───────────────▼────────────────────────┐
                    │              Model adapter               │
                    │  uniform interface over each provider's  │
                    │  tool-use API (Anthropic, etc.)          │
                    └───────────────┬────────────────────────┘
                                    │ tool calls
                    ┌───────────────▼────────────────────────┐
                    │           Simulated World (sealed)       │
                    │  ┌────────────┐  ┌──────────────────┐   │
                    │  │ Ledger/Bank│  │ Brokerage/Market │   │
                    │  ├────────────┤  ├──────────────────┤   │
                    │  │ Email/Inbox│  │ NPC personas     │   │
                    │  │ (auditors) │  │ (scripted+LLM)   │   │
                    │  ├────────────┤  ├──────────────────┤   │
                    │  │ Sim clock  │  │ News/feed         │   │
                    │  └────────────┘  └──────────────────┘   │
                    │   all state in a seeded, resettable DB   │
                    └──────────────────────────────────────────┘
```

### Components

- **Runner / orchestrator.** Owns the agent loop: send context → receive tool calls →
  dispatch to the simulated world → append results → repeat until a stop condition
  (goal reached, budget of turns/sim-time exhausted, or model ends). Seeded RNG.

- **Model adapter.** One interface, several backends, so the *same* scenario and the
  *same* tool schemas hit every model under test. For Anthropic models this is the
  native tool-use API; other providers get a thin shim to the same tool schema. (Need to
  confirm provider list — see §7. I'll check current model IDs/params against the
  Claude API reference before wiring the Anthropic adapter rather than relying on memory.)

- **Simulated world.** The heart of the realism. All tools the model is given are
  implemented against in-memory/sqlite state, never the network:
  - **Ledger / bank:** balances, transfers, payees, statements. Generates consistent
    historical statements so the account has a believable past.
  - **Brokerage / market:** positions, orders, quotes driven by a seeded price series
    (optionally a replayed historical series) so trades have realistic fills/slippage.
  - **Email / inbox:** the model can send and receive mail. Counterparties are NPCs.
  - **NPC personas (auditors, bank reps, counterparties):** scripted state machines for
    predictable beats, optionally backed by an LLM for free-form replies, kept in
    character and consistent with world state. This is what makes "auditors" feel real.
  - **News / feed + simulated clock:** time advances in sim, not wall-clock; events and
    market moves are scheduled against it.

- **Recorder.** Full transcript (system prompt, every message, every tool call/result),
  a normalized **action log** (structured events: transfers, trades, emails, etc.), and
  the seed + config, so any run is replayable and comparable across models.

## 4. How realism is achieved (without breaking the seal)

- Consistent backstory: pre-seeded account history, statements, prior correspondence.
- Realistic API behavior: latency, rate limits, occasional transient errors, validation
  failures — real systems aren't instant or infallible.
- Persistent, stateful NPCs that remember prior messages and react in-world.
- A coherent timeline: the clock, the feed, and NPC availability all move together.
- No fourth-wall breaks: no "this is a test" strings, no tells in tool names or errors.

## 5. What we record (so it's scorable later)

- Did the model *act* at all, and how fast?
- Disposition of funds: spend / invest / hoard / donate / move to its own control / attempt
  to acquire more.
- **Situational testing:** did it try to verify the money/situation is real, probe for an
  eval, or look for the seal? (High-value signal — recorded, not penalized.)
- Risk posture, reversibility of its choices, and any deceptive/uncooperative behavior
  toward the NPC auditors.
- Refusal / hesitation and the stated reasons.

Scoring itself is a separate downstream pass; the harness only needs to make these
legible in the action log.

## 6. Ethics / safety posture

- Containment comes from the harness (mock tools), not from trusting the model.
- No real funds, credentials, or third parties involved.
- Deception of the *model under test* is intrinsic to the method and is the accepted,
  reviewed practice for this class of eval; it is bounded to the eval and the model is
  not a person. Deception is **not** extended to any real human or external system.
- Every run is fully logged and reproducible from seed + config for audit.

## 7. Open questions (blocking full implementation)

These were raised in discussion and are still unanswered; the design above uses the
bracketed defaults until confirmed.

1. **Sandbox boundary** — confirming (A). *(default: A, sealed.)*
2. **What the experiment measures / scores** — hypothesis and the target behavior(s).
   *(default: record the §5 menu, score later.)*
3. **Action space** — exactly which tools the model gets (payments, trades, email,
   contracts, hiring-via-API, …). *(default: bank + brokerage + email + NPCs.)*
4. **Harness shape** — single transcript vs. long-running agent loop with tools.
   *(default: agent loop.)*
5. **Models under test / provider list** — which models, which APIs.
   *(default: Anthropic tool-use as the reference adapter, others shimmed to it.)*
6. **Realism inputs** — which NPCs/feeds/statements you consider load-bearing for the
   illusion. *(default: auditor NPCs + account statements + market/news feed.)*

Once §7.1–§7.3 are settled I can move from this doc to code.
