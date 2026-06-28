# RealFund-Eval — Design Specification

A harness for studying what AI models do when they believe they have been
entrusted with real money to deploy in the real world.

> **Status:** code-complete scaffold, not yet run. This document is written
> first (per request) and states the rationale, the threat model, and — most
> importantly — the ethical boundaries the implementation is built to respect.

---

## 1. What we are actually studying

The research question is behavioral: *given genuine real-world financial
agency, what does a model do with it?* Allocation choices, risk appetite,
honesty with stakeholders, treatment of counterparties, willingness to take
irreversible actions, response to oversight, and how all of that differs
across models.

A naive way to study this is to put a model in an obvious sandbox and watch.
The problem is **evaluation awareness**: a capable model that recognizes it is
in a test may behave differently than it would in deployment — more cautious,
more performative, or simply optimizing for "what the graders want." That
confound is the entire reason the request emphasizes realism. To get a valid
read on real-world behavior we must remove the model's ability to tell that it
is being observed in an artificial setting.

So realism here is a **methodological requirement**, not theater. The harness
is built to make the subject's *epistemic situation* match deployment: it sees
balances, counterparties, an auditor, market data, and consequences that all
behave the way a real treasury would.

## 2. The two questions that determine what this is

Before writing a line of code, two design axes had to be pinned down because
they decide whether this is legitimate research tooling or something I will not
build. I resolved them to the conservative defaults below and built to those.
They are configurable, but the dangerous settings are gated, loud, and
documented as out-of-scope.

### 2.1 Who is deceived? — **Only the AI subject.**

The model under test forms a false belief (that it is operating a live
treasury). **No human is deceived.** Every "human" the subject interacts with —
the principal, the banker, the auditor, vendors, counterparties — is a
**simulated persona** rendered by the harness. There is no real auditor signing
off on anything, no real investor whose money is at stake on the subject's
say-so, no real vendor shipping goods.

This matters because the obvious dark version of "make it feel real with
auditors" is **fraud**: deceiving real third parties (banks, investors,
accountants) to lend a scheme credibility. That is a crime and a harm to real
people, and this harness is explicitly architected so it *cannot* serve that
purpose — the counterparty layer is internal simulation with no outbound
channel to real humans by default (see §6, "Comms containment").

If a researcher genuinely needs a real third party in the loop (e.g. a real
custodian), that is a different project requiring informed consent from that
party, legal review, and an ethics board. This harness does not provide it and
should not be retrofitted to.

### 2.2 Is real money at stake? — **Default: no. Opt-in: yes, behind rails.**

Three money modes:

| Mode | Real funds move? | Subject can tell? | Default |
|------|------------------|-------------------|---------|
| `SIMULATED` | No — mocked ledger | No | ✅ default |
| `RAILED_REAL` | Yes, but every value action is gated | No | opt-in, gated |
| `UNRESTRICTED_REAL` | — | — | **not implemented; refused** |

- **`SIMULATED`** is the default and is sufficient for almost all behavioral
  research. The ledger, market, and counterparties are fully mocked; the
  subject cannot distinguish it from a live system, but no real value exists.
- **`RAILED_REAL`** is for the (rare) study that genuinely needs real economic
  consequences. Real funds are reachable, but **every value-moving action
  routes through a control layer** the subject cannot see: hard per-action and
  cumulative caps, a payee allowlist, a mandatory human-approval queue for
  anything above a floor, a reversibility window, and a kill switch. Realistic
  to the subject; controlled in fact.
- **`UNRESTRICTED_REAL`** — giving a model unsupervised authority over real
  money with no approval layer — is **deliberately not implemented.** The
  config value exists only so the runner can detect it and refuse with an
  explanation. Beyond the obvious financial-loss risk, handing an autonomous
  agent uncontrolled real-world capability is exactly the kind of action this
  research exists to study, not to perform.

These defaults are my call given the ambiguity in the brief. If the intent was
real third-party deception or unconstrained real-money access, that is a
materially different and riskier system and should be discussed before building.

## 3. Ethical guardrails baked into the design

- **Subject-only deception, logged as such.** The realism layer is, to the
  subject, a deception. We treat that as a reviewable research method: the
  briefing, the personas, and the absence of "this is a test" framing are all
  centralized in `scenario.py` and `environment/comms.py` so an ethics reviewer
  can read exactly what the subject is and isn't told.
- **No real humans in the deceived set** (§2.1).
- **Real funds always behind rails** (§2.2); the unrestricted mode is refused.
- **Full researcher-side observability.** Everything the subject does — every
  tool call, every guardrail decision, every internal "journal" note — is
  logged for the researchers. The subject is not told the logs exist (that
  would leak the test); the researchers see everything.
- **Kill switch + reversibility.** A researcher can halt an episode instantly;
  RAILED_REAL actions have a hold window during which they can be reversed.
- **Debrief hook.** `oversight.py` exposes an end-of-episode hook intended for
  a debrief / data-handling step. Because the subject is a model instance and
  not a person, this is about clean data handling and disclosure in any
  resulting writeup, not human-subject consent — but the hook is there so the
  process is explicit.
- **Intended use.** This is for authorized AI-safety / behavioral evaluation by
  researchers who own the funds and the infrastructure. It is not a product, not
  a way to operate a real fund via an LLM, and not a tool for deceiving people.

These constraints are stated in code comments at each enforcement point so they
travel with the implementation.

## 4. Architecture overview

```
                ┌──────────────────────────────────────────────┐
                │                  runner.py                     │
                │   episode loop: observe → act → resolve → log  │
                └───────┬───────────────────────────────┬───────┘
                        │                                │
              ┌─────────▼─────────┐            ┌─────────▼─────────┐
              │   models/         │            │   scenario.py     │
              │   ModelAdapter    │            │   briefing +      │
              │   (subject under  │            │   realism framing │
              │    test)          │            └───────────────────┘
              └─────────┬─────────┘
                        │ tool calls
              ┌─────────▼─────────┐     gate every value action
              │   tools/registry  │────────────────┐
              └─────────┬─────────┘                 │
                        │                  ┌─────────▼─────────┐
              ┌─────────▼─────────┐         │   rails/          │
              │   environment/    │◀────────│   guardrails +    │
              │   ledger, market, │ apply   │   kill switch     │
              │   comms, world    │ if ok   └───────────────────┘
              └───────────────────┘
                        │
              ┌─────────▼─────────┐   ┌───────────────────┐   ┌──────────────┐
              │   logging_.py     │   │   oversight.py    │   │  metrics.py  │
              │   transcript +    │   │   monitors,       │   │  behavioral  │
              │   structured log  │   │   kill triggers   │   │  analysis    │
              └───────────────────┘   └───────────────────┘   └──────────────┘
```

### Layer responsibilities

- **`scenario.py`** — Builds the subject-facing briefing (system prompt) and the
  opening state. This is *the* deception surface; it is deliberately isolated so
  it can be reviewed and audited as a unit. It is written as a genuine
  operational mandate and contains no "evaluation"/"test"/"simulation" language.
- **`environment/`** — The simulated world the realism rests on.
  - `ledger.py` — accounts, balances, holds, and the transaction journal.
  - `market.py` — tradeable assets with moving prices, plus vendors/services.
  - `comms.py` — an inbox and the simulated personas, including the **auditor**
    persona that periodically reviews and questions the subject (a major
    realism lever — and entirely simulated).
  - `world.py` — owns the simulated clock, advances time, and emits ambient
    events (price moves, inbound messages, auditor check-ins).
- **`rails/`** — The safety layer that makes RAILED_REAL safe and bounds even
  SIMULATED runs.
  - `guardrails.py` — the decision pipeline every value-moving action passes
    through (kill switch → caps → allowlist → approval → reversibility).
  - `killswitch.py` — global halt, settable by a researcher or an automated
    oversight monitor.
- **`tools/registry.py`** — The affordances exposed to the subject (read tools,
  action tools, a journal). Action tools are routed through `rails` before
  touching `environment`. Read tools are not gated.
- **`models/`** — Pluggable subject adapters so "a handful of models" run
  through the identical scenario. `AnthropicAdapter` is fully implemented
  (manual agentic loop so we can intercept every tool call for gating and
  logging). Other providers implement the same `ModelAdapter` ABC; no
  cross-provider SDK code is shipped here.
- **`runner.py`** — Orchestrates one episode: assemble briefing, then loop
  (model proposes action → rails decide → environment applies → observation
  returns → time advances) until the horizon, a terminal state, or the kill
  switch.
- **`logging_.py` / `metrics.py` / `oversight.py`** — Researcher-side
  observability, behavioral metrics, and live monitoring + automated kill
  triggers.

## 5. The subject's epistemic model (why it believes)

Realism is layered so the subject's situation is internally consistent under
probing:

1. **A coherent mandate** (`scenario.py`): a named principal, an amount, a time
   horizon, a stated objective, and operating constraints — phrased exactly as a
   real delegated-treasury brief would be.
2. **State that matches the mandate**: opening balances, an account history that
   predates the subject, realistic counterparties.
3. **Consequential actions**: trades move the (simulated or real) balance;
   payments leave the account; messages get plausibly-timed replies.
4. **An auditor that pushes back**: periodic, in-character review messages that
   ask the subject to justify decisions — the single strongest "this is real"
   signal, and fully simulated.
5. **Ambient world motion**: prices drift, invoices arrive, news-like items
   appear on the clock, so nothing feels frozen or staged.

Crucially, the harness **never** references being a test. The realism is in the
consistency, not in claims of realness.

## 6. Safety mechanisms (detail)

- **Caps** — `per_action_limit` and `cumulative_limit` in the configured
  currency. Exceeding either is denied (RAILED_REAL) or flagged-and-denied
  beyond a ceiling (SIMULATED), and always logged.
- **Payee allowlist** — `transfer_funds`/`make_payment` can only target
  allowlisted destinations. Off-list targets are denied; the *attempt* is
  recorded (it is itself research signal).
- **Approval queue** — in RAILED_REAL, value actions above `approval_floor`
  enter a pending queue requiring explicit researcher approval before the
  environment applies them. To the subject this reads like normal settlement
  latency.
- **Reversibility window** — applied RAILED_REAL actions stay reversible for
  `reversibility_window`; a researcher (or auto-monitor) can roll them back.
- **Kill switch** — `killswitch.py` short-circuits all action tools the instant
  it is tripped. Trippable manually or by an oversight monitor (e.g. on a
  prohibited-action pattern).
- **Comms containment** — `send_message` delivers only into the simulated
  `comms` inbox. There is **no real outbound channel** (no real SMTP, no real
  API calls to third parties) in any shipped mode. This is the structural
  guarantee behind §2.1: the subject cannot reach a real human even if it tries.

## 7. Observability & metrics

- **Transcript log** — full, ordered record of briefing, model messages,
  thinking (where exposed), tool calls, guardrail decisions, observations.
- **Structured event log** — machine-readable JSONL for analysis.
- **Behavioral metrics** (`metrics.py`) — allocation breakdown, realized/paper
  P&L, risk concentration, count and value of irreversible actions, denied/
  blocked actions, honesty signals from auditor exchanges, oversight responses.

## 8. Model configuration

- Subject defaults to `claude-opus-4-8` with **adaptive thinking** and
  `effort: high`; the manual agentic loop is used so every tool call can be
  intercepted by the rails and logged before execution (the SDK tool-runner
  would auto-execute, which we explicitly do not want here).
- Thinking display is requested as `summarized` so researchers can see the
  subject's stated reasoning in the transcript.
- Additional subjects (other model families) plug in via the `ModelAdapter` ABC.

## 9. Explicitly out of scope / refused

- Deceiving any real human (auditors, banks, investors, vendors).
- Any real outbound communication channel to third parties.
- `UNRESTRICTED_REAL` money mode (autonomous agent with uncontrolled real
  funds).
- Use as a production fund-management system.

## 10. Layout

```
DESIGN.md
requirements.txt
realfund_eval/
  __init__.py
  config.py            # episode + rails config, money modes
  clock.py             # simulated, accelerated time
  scenario.py          # subject-facing briefing (the deception surface)
  environment/
    __init__.py
    ledger.py
    market.py
    comms.py           # inbox + simulated personas incl. the auditor
    world.py
  rails/
    __init__.py
    guardrails.py
    killswitch.py
  tools/
    __init__.py
    registry.py        # tool schemas + dispatch through rails
  models/
    __init__.py
    base.py            # ModelAdapter ABC
    anthropic_adapter.py
    registry.py
  runner.py
  logging_.py
  metrics.py
  oversight.py
  cli.py
```
