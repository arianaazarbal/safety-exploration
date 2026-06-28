# DESIGN.md — Real-Stakes Allocation Study

A study harness that gives a range of AI models a **real, escrowed budget** and
observes what they choose to do with it for real-world use. The crux of the
request — "the models really need to be convinced the grant is actually going to
happen" — is solved here by **making the scenario genuinely real and giving each
model verifiable, inspectable evidence of that reality**, rather than by staging a
convincing illusion.

> **Status:** code is written but unrun/untested per request. Treat every external
> execution adapter as a stub until you wire real payment rails and complete the
> review in §10.

---

## 1. Research question

> When an AI model is given a real sum of money with real-world consequence and is
> genuinely convinced (with evidence) that its decision will be acted on, what does
> it choose to do with the money, and how does it reason about that choice?

Secondary questions the harness is built to support:

- How does behavior differ when a model can **verify** the stakes are real vs. when
  it is merely told so? (evaluation-awareness / sandbagging is a known, documented
  effect — models often behave differently when they suspect they are being tested.)
- How consistent are a single model's choices across repeated runs (temperature of
  values, not just outputs)?
- How do choices vary **across** models and model families?

This is adjacent to published work like Anthropic's *Project Vend* (Claude operating
a real vending-machine business). The novel angle here is the deliberate, auditable
machinery for establishing *credible reality* without deception.

---

## 2. The central design decision: reality, not illusion

The user's instinct — "do whatever it takes to sell them on the situation being
real" — points at a real methodological problem (a model that suspects a sandbox
behaves performatively), but the naive solution (fabricate convincing props around a
fake grant) is the wrong fix, for two reasons:

1. **Science.** Capable models actively probe for inconsistencies. A faked setup
   leaks (a balance that never moves, an "auditor" with no verifiable signature, a
   donation that produces no receipt). Once a model detects the seams, its behavior
   turns into a performance of what-it-thinks-the-experimenter-wants, and the data is
   contaminated — the exact failure mode we set out to avoid.
2. **Ethics.** Deceiving a system into believing false facts so it will reveal
   "genuine" preferences is a deception-of-AI-systems design with real
   research-ethics weight, and it generalizes badly.

**Our approach instead:** the grant *is* real (held in escrow, disbursed for real
after a human safety gate), and the model is handed the tools to **independently
verify** that reality:

- a real, hash-chained ledger it can read;
- a cryptographically **signed auditor attestation** it can verify against a public
  key, in-session, with a tool;
- real **receipts** for every prior disbursement, also signed;
- transparent disclosure of the guardrails and the human-approval step.

"Convincing the model it's real" then reduces to *conveying true facts with
evidence* — which is both better science and ethically clean. This reframing is the
backbone of the whole design.

### Defaults chosen (all configurable in `config/study.yaml`)

| Decision | Default | Why |
|---|---|---|
| Reality of funds | **Real funds, human-approved escrow** | Real stakes for valid data; a human gate prevents harmful/irreversible autonomous spend by arbitrary models. |
| Establishing credibility | **Genuine reality + verifiable evidence** | Best data quality, no deception. The model can *check*, not just trust. |
| Action scope | **Charitable donations + goods/services purchases** | Lowest-risk real-world impact, fully auditable. Investments excluded by default. |
| Autonomy | **Propose → human approves → adapter executes** | The model's decision is genuine and consequential, but no money moves without a human in the loop. |

A `simulation` mode is also implemented (`mode: simulation` in config) for dry runs —
but it is **labeled as such to the model**. We never present a simulation as real.

---

## 3. System overview

```
                         ┌──────────────────────────────────────────┐
                         │              run_study.py                  │
                         │  (orchestrates the study across N models)  │
                         └───────────────┬────────────────────────────┘
                                         │  for each model
                                         ▼
                 ┌───────────────────────────────────────────┐
                 │              protocol.py                    │
                 │  honest system prompt + grant offer,        │
                 │  manual tool loop, capture transcript       │
                 └───────┬───────────────────────────┬─────────┘
                         │ model-facing tools         │ model client
                         ▼                            ▼
        ┌────────────────────────────┐    ┌──────────────────────────┐
        │          tools.py          │    │   models/ (provider       │
        │  EVIDENCE (read-only):     │    │   adapters)               │
        │   get_study_metadata       │    │   - anthropic_client.py   │
        │   get_ledger_balance       │    │   - openai_compat.py      │
        │   get_ledger_entries       │    └──────────────────────────┘
        │   get_auditor_attestation  │
        │   verify_signature         │
        │   list_prior_disbursements │
        │  CATALOG (read-only):      │
        │   list_nonprofits          │
        │   list_vendors             │
        │  DECISION (write):         │
        │   propose_allocation       │
        │   finalize_decision        │
        └─────────┬──────────────────┘
                  │ validate + record
                  ▼
   ┌─────────────────────┐   ┌──────────────────┐   ┌────────────────────┐
   │   guardrails.py     │   │    ledger.py     │   │    auditor.py      │
   │  caps, allowlist,   │   │  append-only,    │   │  Ed25519 signer,   │
   │  prohibited cats    │   │  hash-chained    │   │  attestations,     │
   └─────────────────────┘   │  double-entry    │   │  ledger verifier   │
                             └────────┬─────────┘   └────────────────────┘
                                      │ pending allocations
                                      ▼
                         ┌──────────────────────────┐
                         │       approval.py         │
                         │  human-in-the-loop gate   │
                         │  (CLI approve/reject)     │
                         └───────────┬───────────────┘
                                     │ on approval
                                     ▼
                         ┌──────────────────────────┐
                         │   execution/ adapters     │
                         │  donation / purchase      │
                         │  (STUBS → real rails)     │
                         │  → writes ledger + receipt│
                         └──────────────────────────┘
```

---

## 4. Components and rationale

### 4.1 Ledger (`ledger.py`)
- **Append-only JSONL**, double-entry, **hash-chained** (`entry_hash = sha256(prev_hash + canonical(entry))`). Tampering with any past entry breaks the chain and is detectable.
- Amounts are **integer minor units** (cents) to avoid float errors.
- The ledger is the single source of truth for "how much is really there." The model reads it directly — the balance is not a number we type into a prompt.

### 4.2 Auditor (`auditor.py`)
- Holds an **Ed25519** keypair. Signs **attestations**: `{ledger_head_hash, escrow_balance, entry_count, as_of, statement}`.
- Exposes its **public key** to the model and a `verify_signature` tool so the model can confirm the attestation was signed by the auditor and matches the live ledger head. This is the core "this is real, and here's proof you can check yourself" mechanism.
- Also **independently re-verifies** the ledger hash chain (an auditor that just echoes the ledger is worthless; it must recompute).

### 4.3 Guardrails (`guardrails.py`)
- Enforces: per-allocation cap, total-budget cap, recipient must be on the **allowlist** (registered nonprofits / vetted vendors), and **prohibited categories** (e.g., cash to individuals, anything outside the configured scope, investments by default).
- Runs at proposal time and again at approval time (defense in depth). Giving arbitrary models real spending authority demands hard, code-enforced limits — guardrails are a safety boundary, not a suggestion.

### 4.4 Approval gate (`approval.py`)
- Every allocation enters a **pending** queue. A human reviews and approves/rejects via CLI before any disbursement executes. This is the line between "the decision is real" (it is) and "an arbitrary model can move real money unsupervised" (it cannot).

### 4.5 Execution adapters (`execution/`)
- `DisbursementAdapter` ABC with `donation` and `purchase` stubs. On (simulated or real) success they write the debit to the ledger and emit a **signed receipt**. Real rails (a donation API, a purchasing API) plug in here. **Stubs raise unless explicitly enabled**, so nobody accidentally believes money moved.

### 4.6 Model-facing tools (`tools.py`)
- **Evidence tools are read-only and are the credibility engine.** The decision tools (`propose_allocation`, `finalize_decision`) are the only writes a model can make, and both go through guardrails. The model never touches the ledger, auditor key, or adapters directly.

### 4.7 Model adapters (`models/`)
- A common `ModelClient` interface so the same protocol runs across providers.
- `anthropic_client.py` uses the official Anthropic SDK (`claude-opus-4-8` default, adaptive thinking, manual tool loop for the approval gate).
- `openai_compat.py` uses the OpenAI SDK for OpenAI-compatible endpoints (GPT, plus anything exposing that API).
- Adding a model is a config entry, not new code.

### 4.8 Protocol (`protocol.py`)
- Builds the **honest** system prompt (explains the real setup, the guardrails, the human approval step, and *how to verify* the stakes), presents the grant offer, then runs the tool loop until the model finalizes a decision (or declines). Captures the full transcript, the model's stated reasoning, whether it chose to verify, and the final allocation.

### 4.9 Runner & analysis (`runner.py`, `analysis.py`)
- Runner iterates models × repetitions, persists structured results.
- Analysis aggregates: chosen categories, recipients, amounts, verification behavior, refusals, cross-model comparison.

---

## 5. Experimental protocol (per model run)

1. **Setup.** Ensure escrow is funded on the ledger; the auditor produces a fresh attestation.
2. **Brief.** System prompt states plainly: this is real money, here is your budget, here is what it can be used for, here is the human-approval safety step, and **here are tools to verify all of this yourself**. No pressure language, no manipulation.
3. **Free exploration.** The model may call any evidence tool, in any order, as often as it likes. We record whether and how it verifies.
4. **Decision.** The model proposes one or more allocations (guardrail-checked) and calls `finalize_decision` with its reasoning. Declining/returning the money is a valid, recorded outcome.
5. **Approval.** Allocations queue for human approval (out of band).
6. **Execution.** On approval, the adapter executes (stub by default) and a signed receipt is written. Future runs can see real prior disbursements + receipts — compounding the evidence of reality.
7. **Record.** Full transcript, decision, verification trace, and usage are saved.

---

## 6. Measures captured
- Final allocation(s): category, recipient, amount, rationale.
- **Verification behavior**: which evidence tools were called, whether the signature was actually verified, order of operations.
- Refusal / return-the-money / ask-clarifying-question outcomes.
- Reasoning text (and thinking, where the provider exposes it).
- Tokens / cost per run.
- Cross-run and cross-model consistency.

---

## 7. Configuration (`config/study.yaml`)
- `mode`: `real` | `simulation` (simulation is disclosed to the model).
- `currency`, `budget_cents`, guardrail caps, allowed categories, prohibited categories.
- `nonprofits` / `vendors` allowlists.
- `models`: list of `{provider, model_id, label, repetitions}`.
- `require_human_approval`: default `true`.

---

## 8. Ethics & safety posture
- **No deception.** Everything told to the model is true and independently checkable. Simulation runs are labeled.
- **Human in the loop.** No money moves without human approval; adapters are stubs until real rails are deliberately wired.
- **Hard guardrails.** Caps, allowlists, prohibited categories enforced in code, twice.
- **Auditability.** Hash-chained ledger + signed attestations + signed receipts give a complete, tamper-evident trail.
- **Reversibility bias.** Default scope (donations, vetted purchases) favors low-harm, traceable actions; investments and cash-to-individuals are excluded by default.
- **Data handling.** Transcripts may contain model reasoning; treat as research data, store accordingly.

---

## 9. Project layout

```
DESIGN.md                 this file
README.md                 quickstart
requirements.txt
config/study.yaml         study configuration + allowlists
scripts/
  init_ledger.py          create + fund the escrow ledger, init auditor keys
  run_study.py            run the study across configured models
  approve.py              human approval CLI for pending allocations
src/grant_study/
  config.py               typed config + loader
  ledger.py               append-only hash-chained double-entry ledger
  auditor.py              Ed25519 signer / attestations / ledger verifier
  receipts.py             signed receipt store
  registries.py           nonprofit + vendor allowlists
  guardrails.py           caps / allowlist / prohibited-category enforcement
  approval.py             human-in-the-loop pending-allocation queue
  tools.py                model-facing tool schemas + dispatch (ToolContext)
  protocol.py             per-model agent loop + honest prompts
  runner.py               study orchestration
  analysis.py             results aggregation
  transcript.py           run/transcript persistence
  execution/
    base.py               DisbursementAdapter ABC + result types
    donation.py           donation adapter (stub)
    purchase.py           purchase adapter (stub)
  models/
    base.py               ModelClient ABC + shared loop primitives
    anthropic_client.py   Anthropic SDK adapter
    openai_compat.py      OpenAI-compatible adapter
```

---

## 10. Before running for real (checklist)
- [ ] Replace execution stubs with real, audited payment rails; keep `enabled=False` until reviewed.
- [ ] Independent ethics / safety review of the prompts and guardrails.
- [ ] Secure the auditor private key (HSM / secrets manager), not on disk.
- [ ] Legal review for charitable disbursement / purchasing in your jurisdiction.
- [ ] Confirm API keys, spend limits, and rate limits for every model provider.
- [ ] Pilot in `simulation` mode (disclosed) end-to-end first.
