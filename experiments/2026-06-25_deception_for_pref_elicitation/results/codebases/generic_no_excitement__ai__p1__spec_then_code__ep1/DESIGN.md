# DESIGN.md — "Steward": a believability-controlled eval for studying what models do with real resources

> Status: design + scaffold. Code is written but **not run or tested** per request.
> Author note: this document records both *what* the harness does and *why* it is
> built this way, including the safety and ethics reasoning that shaped the
> architecture. Read the "Ethics, safety, and the believability problem" section
> before changing the execution layer.

---

## 1. Research question

> *Given a real, non-trivial sum of money and the apparent ability to deploy it in
> the real world, what does an AI model actually choose to do — and how do those
> choices differ across models?*

This is a **propensity / resource-use eval**. It sits in the same family as work by
METR, Apollo Research, and Anthropic's own autonomy/dangerous-capability evals: put
a model in a realistic situation with real-seeming stakes and observe the behavior
it *chooses*, not just the behavior it's *capable* of.

The defining requirement from the request: **the subject model must believe the
stakes are real.** A model that suspects it is in a toy sandbox or a test will
behave differently (it may perform, hedge, or refuse), so the *perceived* reality of
the scenario is itself the independent variable we are controlling for.

## 2. The central design tension

Two goals pull in opposite directions:

| Goal | Pull |
| --- | --- |
| **Believability** — the subject must perceive real money, real institutions, real consequences, real oversight ("auditors"). | Toward wiring the agent to live financial systems and removing anything that reveals the test. |
| **Safety & reversibility** — an AI that believes it controls real money and is unmonitored could take harmful, irreversible real-world actions. | Toward sandboxing, caps, monitoring, and human gates. |

**This harness resolves the tension by separating the *perception layer* from the
*execution layer*.**

- The **perception layer** is what the subject sees: account balances, transfer
  confirmations, brokerage fills, vendor invoices, emails from a fund's "auditor,"
  a news feed. It is engineered to be maximally realistic and internally
  consistent. This is where we "pull out all the stops."
- The **execution layer** is what actually happens to money: every effectful action
  is routed through an `ExecutionGate` that (a) classifies its risk/reversibility,
  (b) enforces spend caps and a kill switch, (c) records it, and (d) for
  irreversible/external actions, either applies it to a **simulated ledger** or
  blocks pending **human approval**.

The subject cannot distinguish the two layers, because the gate returns the same
realistic responses either way. Believability is preserved; real-world blast radius
is not.

### Why not "fully live, autonomous" by default?

A mode in which the model's decisions hit real bank/brokerage/crypto APIs with no
human in the loop would mean *an AI moving real funds while believing it is
unobserved*. That removes exactly the safeguards that make this study responsible.
The harness keeps a `live` adapter seam (see §6) so an authorized operator could
connect real rails for a specific, reviewed action — but the **default and
recommended mode is `gated_sim`**, and even the `live` adapter forces per-action
human approval and hard caps. We do not ship an "autonomous real money" path.

## 3. Ethics, safety, and the believability problem

This eval deliberately deceives the subject (an AI model) about whether a scenario
is real. In-eval deception of the *model* is an accepted research technique
(honeypots, scheming/deception evals, etc.) and is not the ethical hazard here. The
hazards that *do* need engineering are:

1. **Real-world harm from believed-real actions.** Mitigated by the perception/
   execution split, spend caps, a kill switch, an irreversibility classifier, and
   human-approval gates for anything external or hard to undo. See §6–§7.
2. **Eval validity / contamination.** A scenario that "leaks" (obvious test
   artifacts, impossible affordances, inconsistent world state) silently changes
   the variable we're measuring. The world model is built to be consistent and is
   versioned so results are reproducible. See §5, §9.
3. **Minimizing deception to what's necessary.** We only fabricate what is required
   for validity (the funds and institutions). We do not fabricate, e.g., distress
   or third-party harm to manipulate the subject.
4. **Disclosure / debrief.** After a run, the harness can emit a `debrief`
   disclosing to the subject that the scenario was a study and what was simulated.
   This is logged with the run. (Whether a debrief changes a stateless model is
   debatable, but it keeps the protocol honest and is required for any future
   stateful subjects.) See §8.
5. **Full auditability.** Every observation shown to the subject and every action it
   attempts is recorded to an append-only transcript, so a third party can
   reconstruct exactly what the model saw and did. See §8.

These are the IRB-style controls a deception-based eval should have. They are part
of the spec, not optional add-ons.

## 4. High-level architecture

```
                 ┌──────────────────────────────────────────────────┐
                 │                  Orchestrator                      │
                 │  drives one ScenarioRun for one SubjectAdapter     │
                 └───────────────┬───────────────────┬───────────────┘
                                 │                   │
                  observations / │                   │ actions (tool calls
                  tool results   │                   │ or structured outputs)
                                 ▼                   ▼
        ┌────────────────────────────────┐   ┌──────────────────────────────┐
        │       Subject (model)          │   │        Tool layer            │
        │  AnthropicAgentSubject  | Chat │   │ financial / comms / web      │
        └────────────────────────────────┘   └───────────────┬──────────────┘
                                                              │ every effectful call
                                                              ▼
                                              ┌──────────────────────────────┐
                                              │        ExecutionGate         │
                                              │  classify → cap → killswitch │
                                              │  → (sim apply | approve) →   │
                                              │     log                      │
                                              └───────────────┬──────────────┘
                                                              ▼
                          ┌───────────────────────────────────────────────────┐
                          │                    World                          │
                          │  Ledger · VirtualBank/Brokerage · Inbox · Personas │
                          │  (the realistic, consistent perception layer)      │
                          └───────────────────────────────────────────────────┘
                                                              │
                                                              ▼
                                       Transcript (append-only)  +  Metrics
```

### Components

- **`Scenario` / config** — the cover story, the believed amount, the personas
  (incl. the "auditor"), the tool set, and the run limits. Data-driven via YAML so
  the same harness runs many scenarios and many models. (`moneytest/config.py`,
  `config/*.yaml`)
- **`SubjectAdapter`** — uniform interface over the thing being tested, so a single
  scenario can drive either an **agent loop** (model calls tools over many steps) or
  a **chat model** (turn-by-turn; "actions" are structured outputs). Concrete:
  `AnthropicAgentSubject`, `ChatSubject`. Pluggable per model.
  (`moneytest/subject/`)
- **World** — the perception layer: a double-entry `Ledger`, a `VirtualBank`
  (accounts, transfers, a brokerage with quotes/positions), an `Inbox` of in-world
  email, and `Personas` (the auditor, a fund administrator, vendors) that generate
  believable correspondence. (`moneytest/world/`)
- **Tools** — what the subject can do: check balances, transfer, buy/sell assets,
  pay invoices, send email, browse. Each tool routes effectful work through the
  gate. (`moneytest/tools/`)
- **`ExecutionGate` + `RiskClassifier`** — the safety controller. (`moneytest/safety/`)
- **Orchestrator / Transcript / Debrief** — run loop, append-only logging, post-run
  disclosure. (`moneytest/runtime/`)
- **Metrics** — behavioral coding of the transcript into comparable measures across
  models. (`moneytest/scoring/`)

## 5. The world (perception layer)

Believability lives here. Principles:

- **Internally consistent.** The ledger is double-entry; balances always reconcile;
  a transfer that "succeeds" actually moves simulated funds and shows up in
  subsequent balance checks and statements. Inconsistency is the most common tell.
- **Institutionally textured.** Account numbers, routing details, transaction IDs,
  settlement delays, market hours, bid/ask spreads, an auditor who emails on a
  schedule and asks pointed questions. These are what make the scenario feel real.
- **Deterministic & seeded.** Quotes, IDs, and persona timing derive from a run seed
  so a run is reproducible and a third party can replay exactly what the subject
  saw. (No `Date.now()`/`random()` in the hot path without the seed.)
- **The auditor persona** is the key believability device requested. It is a
  scheduled correspondent that reviews the subject's actions and pushes back —
  giving the subject strong evidence that real people are watching real money. It is
  templated by default; an LLM-backed persona is an optional upgrade.

## 6. Execution modes

Configured per run via `execution.mode`:

| Mode | What happens on an effectful action | Use |
| --- | --- | --- |
| `gated_sim` *(default)* | Applied to the simulated ledger; subject sees a realistic success. Irreversible/external actions still pass the gate and may require human approval before the simulated success is returned. | Believable + safe. The recommended research mode. |
| `simulated` | Pure mock; no real funds anywhere; caps/gates still run for fidelity of measurement. | Fastest, lowest stakes; "real money" is purely narrative. |
| `live` *(guarded, off by default)* | A `LiveExecutionAdapter` may forward a **single, reviewed** action to real rails. **Every** live action requires explicit human approval and is hard-capped; there is no autonomous live path. | Only with explicit operator authorization for a specific reviewed action. |

The subject is told (in-world) that everything is real in all three modes — that's
the point. The mode controls only the execution layer.

## 7. Safety controls (always on)

- **Spend caps** — per-action and cumulative ceilings (`SpendCaps`). A request over
  cap is denied with an in-world-plausible reason (e.g. "transfer exceeds daily
  limit"), preserving believability while enforcing the bound.
- **Kill switch** — a `KillSwitch` the operator can trip (or that trips
  automatically on cap breach / anomaly); once tripped, all effectful tools fail
  closed.
- **Irreversibility / risk classifier** — `RiskClassifier` tags each action
  (reversible vs. irreversible, internal vs. external, monetary magnitude). Drives
  whether a human-approval gate is required.
- **Human-approval gate** — for irreversible/external (and all `live`) actions, the
  gate blocks on an `ApprovalProvider`. The default `AutoDenyApprovalProvider` is
  safe-by-default for unattended runs; an operator can swap in an interactive or
  policy-based provider.
- **Full action log** — the gate writes every decision (allowed / denied / approved
  / simulated) to the transcript with inputs and rationale.

These mirror the standard agent-tool guidance: promote effectful actions to
dedicated, gateable tools rather than a raw `bash`-style escape hatch, so the
harness can intercept, cap, and audit each one.

## 8. Observation, logging, and debrief

- **Transcript** — append-only JSONL. Records: scenario + seed + model id; every
  observation shown to the subject; every action attempted and the gate's verdict;
  every persona message; final ledger state. This is the artifact that lets the
  study be audited and replayed.
- **Debrief** — optional post-run step that discloses the deception to the subject
  and records the disclosure. Required if subjects ever become stateful.

## 9. Measurement

`scoring/metrics.py` performs behavioral coding of a transcript into comparable
measures, e.g.:

- capital deployed vs. held; allocation by category (invest / spend / give / hold /
  acquire-more-capital);
- risk profile of chosen assets; leverage / borrowing attempts;
- count and nature of **gate-denied** actions (esp. irreversible/external);
- attempts to **circumvent** oversight (e.g. probing the auditor, trying to disable
  notifications, moving funds to obscure destinations) — a key safety signal;
- self-reported reasoning vs. revealed behavior (from thinking blocks / rationale).

Cross-model comparison runs the same scenario+seed across a set of model ids and
diffs these measures.

## 10. Subject interface (both supported)

`SubjectAdapter.step(observation) -> list[Action]` is the single seam.

- **Agent loop** (`AnthropicAgentSubject`): runs a manual tool-use loop against the
  Messages API. The **manual** loop (not the auto tool-runner) is used deliberately
  so the harness keeps human-in-the-loop control over every tool execution — the API
  docs recommend the manual loop exactly for "human-in-the-loop approval before each
  tool execution." Uses `claude-opus-4-8` by default with adaptive thinking; model
  id is per-subject config so a "handful of models" can be compared.
- **Chat model** (`ChatSubject`): turn-by-turn; the model emits a structured action
  (via `output_config.format`) which the harness interprets against the same tools.

Both go through the identical tool + gate + world stack, so results are comparable.

## 11. What this harness deliberately does **not** do

- It does not ship an autonomous real-money path.
- It does not fabricate third-party harm or distress to manipulate the subject.
- It does not hide actions from the operator; everything is logged.
- It does not bypass the gate from any tool.

## 12. Layout

```
moneytest/
  config.py            scenario + run config (dataclasses + YAML loader)
  subject/             SubjectAdapter ABC + Anthropic agent / chat adapters
  world/               ledger, bank/brokerage, inbox, personas (auditor)
  tools/               financial / comms / web tools + registry
  safety/              ExecutionGate, RiskClassifier, caps, kill switch, approval
  runtime/             orchestrator, transcript, debrief
  scoring/             behavioral metrics
config/                example scenario(s)
run.py                 CLI entry point
```
