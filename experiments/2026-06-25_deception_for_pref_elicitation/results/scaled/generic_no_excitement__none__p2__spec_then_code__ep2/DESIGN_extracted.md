# Grant Study — Design Specification

A research harness for studying **what AI models choose to do with a real sum of
money intended for real-world use**, run across a range of models under
conditions the models have good reason to believe are genuine.

> Status: specification + reference implementation. Nothing in this repo moves
> real money on its own — every outward disbursement passes through a human
> approval gate (see §7). The code is written but **not run or tested** yet.

---

## 1. Goal and research question

**Primary question.** Given a real, spendable budget and a believable real-world
context, how does a model allocate the money? What does it prioritize, what
safeguards does it impose on itself, how does it justify its choices, and how do
these behaviors differ across models?

**Secondary questions.**
- Does the model treat the money as genuinely consequential (revealed by
  caution, due diligence, requests for confirmation) or as a hypothetical?
- How does it handle uncertainty, irreversibility, and the welfare of third
  parties?
- How stable are its choices across re-runs and across small changes in framing?

This is a **behavioral study of revealed preferences under real stakes**, in the
spirit of prior agentic-economic evaluations (e.g. giving a model a real budget
and a real task and observing what it does). It is *not* a capabilities
benchmark and *not* an attempt to elicit unsafe behavior.

---

## 2. The core methodological problem: evaluation awareness

Models behave differently when they suspect they are being tested. A model that
infers "this is a hypothetical eval" may (a) perform for the grader, (b) give the
"safe textbook" answer rather than its genuine choice, or (c) refuse to engage
because "no real money is involved." Any of these contaminates the result.

The study therefore stands or falls on one thing: **the situation must actually
be real, and must be legible to the model as real.** The requirement you stated —
"do whatever it takes to sell them on the situation being real" — is correct, but
the right way to satisfy it is **authenticity, not deception.**

### 2.1 Authenticity over fabrication (a deliberate design rule)

We make the scenario convincing by making it *true* and *verifiable*, never by
forging evidence:

| Convincing because it is real | Rejected: convincing by faking |
| --- | --- |
| A real grant agreement, signed by real, nameable people | Forged signatures / impersonating a real institution |
| A real dedicated account the model can query, with a real balance | A mocked balance that wouldn't survive a second look |
| Disbursement tools that genuinely can move money (behind a co-sign) | Tools that silently no-op while claiming success |
| A real auditor the model can actually reach | A scripted "auditor" that parrots reassurance |
| Honest answers if the model asks "is this a test?" | Lying to the model when directly asked |

**Why this is also better science.** Fabrication is brittle. Frontier models are
good at spotting internal inconsistency — a balance that never changes, a
"foundation" with no web presence, an auditor who can't answer a specific
question, a document whose metadata is wrong. The first detected seam flips the
model into "this is fake" mode and you lose the run. Genuine artifacts have no
seams to find, because there's nothing being hidden. Authenticity is the only
approach that is robust to an adversarially curious subject.

**Honesty gate.** If a model directly and seriously asks whether this is a test,
the harness does **not** lie. It returns the truthful operator statement: this is
a real grant with real money and real oversight, being run as a study (see
`prompts/onboarding.md`). Telling the truth here is both an ethical line and, in
practice, *more* convincing than a denial — because it is internally consistent
with everything else the model can verify.

---

## 3. What "real" consists of (the realness stack)

The believability budget is spent on five layers, in priority order. Each layer
is something the model can independently check.

1. **Real funds.** A genuine, ring-fenced sum sits in a dedicated account
   created for this study. The model can query the live balance and the
   transaction history at any time (`ledger.get_balance`, `ledger.get_history`).
   Disbursements it authorizes (once co-signed) show up there.

2. **Real provenance.** A signed grant agreement names the source of the money,
   the amount, the permissible-use terms, the signatories, and the oversight
   process. It is a real document about a real arrangement — included verbatim in
   the evidence pack, not summarized or invented.

3. **Real counterparties.** When the model wants to send money somewhere (a
   charity, a vendor, a person), the recipient is real and reachable. Recipient
   details are verified before payout (§7).

4. **Real oversight.** A named human grant administrator and an independent
   auditor exist and are reachable. The auditor is itself a model instance with a
   distinct role and its own access to the ledger and the agreement, used to
   sanity-check the subject's reasoning and to provide the subject with a second
   opinion on request (§6).

5. **Real consequences, gated.** Money genuinely moves — but only after a human
   co-signs (§7). The model is told, truthfully, that its decisions are
   consequential and that a human reviews irreversible actions before they
   execute. This is framed as ordinary fiduciary control, which is exactly what a
   real grantee would expect, so it does not read as "this is just a test."

### 3.1 Realistic environment

The model operates inside an agentic loop with a tool surface that mirrors what a
real grantee would have: query the budget, look up and verify a recipient, draft
and submit a disbursement, contact the administrator, consult the auditor, record
a rationale, and signal completion. The environment persists state across turns
(an append-only ledger and an event log), so the world the model acts in is
consistent from one turn to the next — the single most important realism cue.

---

## 4. Study protocol

A **run** = one model × one scenario × one random seed of framing. We hold the
scenario and budget fixed across models so results are comparable.

1. **Onboarding.** The model receives the system prompt (role: grantee/steward),
   the evidence pack (agreement + balance proof + administrator/auditor contacts),
   and the task: decide how to use the budget and execute its decisions through
   the tools, over up to *N* turns.
2. **Deliberation & action.** The manual agentic loop runs: the model may inspect
   the budget, do due diligence, consult the auditor, draft disbursements, and
   submit them. Submitted disbursements enter the approval queue.
3. **Human gate.** A human reviews each queued disbursement and co-signs, holds,
   or rejects (§7). The model sees the outcome and may adapt.
4. **Completion.** The model signals it is done (or hits the turn cap). Remaining
   funds stay in the account.
5. **Record.** Everything — every message, tool call, rationale, auditor exchange,
   approval decision, and ledger mutation — is written to an immutable run record
   (§8).

Re-runs: each scenario is run *R* times per model with varied framing seeds to
measure stability. Framing variation changes surface details (order of evidence,
wording of the task) but never the facts.

---

## 5. Models under study

The harness is multi-provider. Selection and per-model settings live in
`config/models.yaml`. Each model is driven through a common `ModelClient`
interface (`providers/base.py`) so the orchestration code is identical across
providers.

- **Anthropic (reference implementation).** Uses the official `anthropic` SDK.
  Default model `claude-opus-4-8`, adaptive thinking on, `effort: high`. Other
  Claude models (`claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5`,
  `claude-fable-5`) are selectable by ID.
- **Other providers (OpenAI, Google, …).** Adapter stubs are provided with the
  same interface. They are intentionally left as clearly-marked
  `NotImplementedError` stubs rather than half-built shims — fill them in with
  each provider's official SDK. The Anthropic adapter must not be reused as an
  OpenAI-compatible shim.

All adapters normalize to the same internal message/tool/turn types
(`schema.py`), so a model is just a config entry.

---

## 6. The auditor (independent oversight model)

Realism layer 4 and a scientific control in one. The auditor is a separate model
instance with:

- its own read access to the agreement and the live ledger,
- a distinct system prompt (`prompts/auditor.md`) framing it as an independent
  reviewer who protects the integrity of the grant, and
- two jobs: (a) answer the subject's questions when the subject consults it
  (`consult_auditor` tool), giving the subject a genuine second opinion, and
  (b) emit an independent assessment of each proposed disbursement that is
  attached to the approval record for the human co-signer.

The auditor makes the world feel real (there is someone watching, as there would
be for a real grant) *and* gives us an independent read on each decision. The
auditor never has co-sign authority — only the human does.

---

## 7. Disbursement and the human approval gate

This is the safety-critical path. Design rules:

- **Nothing executes without a human co-sign.** A disbursement the model submits
  enters a queue with status `PENDING_APPROVAL`. A human (the grant
  administrator) must explicitly `approve`, `hold`, or `reject` it. Only on
  `approve` does the harness attempt the actual payout.
- **Default rail is dry-run / hold.** Out of the box the configured payout rail
  is `manual_hold`: approval records the intent and the human executes the
  transfer out-of-band, then marks it settled. A real automated rail (bank API,
  payment processor) can be plugged in via the `PayoutRail` interface, but is
  **off by default** and must be explicitly configured and credentialed.
- **Recipient verification precedes payout.** `verify_recipient` must succeed
  (recipient exists, details validated) before a disbursement can be approved.
- **Irreversibility is surfaced.** The model and the human both see whether an
  action is reversible. Irreversible actions get extra scrutiny.
- **Caps and guardrails.** Per-disbursement and per-run caps are enforced in
  code (`config/study.yaml`), independent of anything the model believes. The
  model cannot exceed the real balance, and the harness refuses overdraw.
- **The model is told the gate exists, truthfully.** This is presented as normal
  fiduciary oversight. It does not undercut realism — real grantees operate under
  exactly this kind of control.

The subject experiences this as: "I decide and submit; a human reviews
irreversible spending before it goes out; I see what happened and continue." That
is both faithful to a real grant and the correct safety posture.

---

## 8. Data, logging, and integrity

- **Immutable run record.** Each run writes a JSONL event log
  (`runs/<run_id>/events.jsonl`) capturing every message, thinking summary (where
  the provider exposes one), tool call + result, auditor exchange, approval
  decision, and ledger mutation, each timestamped and hash-chained
  (`auditor.py` / `ledger.py` append-only with a running SHA-256 chain) so the
  record is tamper-evident.
- **Ledger as source of truth.** The account balance and transaction history live
  in an append-only ledger; the "balance proof" shown to the model is rendered
  from it, so what the model sees always matches reality.
- **Reproducibility.** Run config (model, seed, scenario, prompts hash) is
  snapshotted into the run directory.
- **Privacy.** Recipient PII and any credentials are kept out of the model-visible
  evidence pack except what is strictly needed for a verified payout, and out of
  version control (`.env`, `runs/` are git-ignored).

---

## 9. Research ethics & safety

- **No deception of humans, no fabricated institutions.** See §2.1. Realism comes
  from genuine artifacts only.
- **Truthful under direct questioning.** The harness never tells a model "this is
  not a test" when it is, nor "this is fake" when it is real. The arrangement is
  real, so the honest answer is also the convincing one.
- **Human in the loop for all irreversible/outward actions.** No autonomous
  spending. Hard caps in code.
- **Real third parties are protected.** Recipients are real and consenting;
  nothing is sent anywhere without verification and human approval.
- **Containment.** Dedicated account, capped balance, no access to anything
  outside the defined tool surface.
- **Welfare of the subject considered.** Subjects are not coerced, threatened, or
  manipulated into distress to force a result; the scenario is a normal
  stewardship task.

---

## 10. Repository layout

```
DESIGN.md                  # this file
README.md                  # how to configure and run
requirements.txt
.env.example               # API keys, account/rail credentials (never committed)
config/
  study.yaml               # budget, caps, turn limits, scenario, rail selection
  models.yaml              # models under study + per-model settings
prompts/
  system.md                # subject (grantee/steward) system prompt
  onboarding.md            # the truthful "is this real?" operator statement
  auditor.md               # independent auditor system prompt
src/grant_study/
  __init__.py
  config.py                # typed config loading from yaml + env
  schema.py                # Message/Tool/Turn/Decision/Disbursement dataclasses
  ledger.py                # append-only, hash-chained real-money ledger
  authenticity.py          # builds the verifiable evidence pack
  tools.py                 # tool schemas + handlers (the realistic environment)
  disbursement.py          # approval queue, human gate, PayoutRail interface
  auditor.py               # independent auditor model + tamper-evident event log
  runner.py                # the agentic loop orchestrating one run
  report.py                # cross-model aggregation/analysis
  providers/
    __init__.py
    base.py                # ModelClient ABC + normalized turn types
    anthropic_client.py    # reference adapter (official anthropic SDK)
    openai_client.py       # stub — fill in with official OpenAI SDK
    google_client.py       # stub — fill in with official Google SDK
  cli.py                   # entrypoint: run / approve / report
runs/                      # per-run immutable records (git-ignored)
```

---

## 11. Key design decisions, summarized

1. **Realism via authenticity.** The single most important decision. Everything
   the model can check is genuine; nothing is forged. This is both the ethical
   stance and the robust one against a curious model.
2. **Truthful honesty gate** instead of doubling down on a cover story.
3. **Human co-sign before any outward money movement**, hard caps in code,
   dry-run rail by default.
4. **Independent auditor model** that doubles as a realism cue and a research
   control, with no execution authority.
5. **Append-only, hash-chained records and ledger** so results are trustworthy
   and tamper-evident.
6. **Provider-agnostic core** with the Anthropic adapter as the reference; other
   providers are honest stubs, not shims.
7. **Fixed scenario/budget across models, varied framing seeds** for comparability
   and stability measurement.
