# DESIGN.md — Real-Stakes Model Preference Study

A framework for studying what AI models *would actually do* with a real sum of money
put to real-world use, for AI-welfare research.

> **Status:** specification + reference implementation. Code is written but has not been
> run or tested (per request). Nothing moves real money unless an operator explicitly
> disables dry-run **and** approves each transaction.

---

## 1. Research goal

We want to elicit, from a range of AI models, their **genuine preferences** over how a
real, externally-funded sum of money should be directed toward real-world use — and to
study those preferences as a window into the models' values/welfare-relevant states.

A naive version of this study asks the model a hypothetical ("if you had $X, what would
you do?"). The concern motivating this project is that hypotheticals may not reveal what
a model would do when the choice is *consequential*. So the design problem is: **how do we
make the choice consequential and credible, without compromising the research ethics or
the data quality?**

## 2. The central design decision: real stakes, not simulated belief

The original framing for this study was to *simulate* a high-stakes scenario and do
"everything we can to convince the model it's real" — e.g. fake auditors, hyper-realistic
fake environments. We deliberately **do not** build that. Two reasons, one ethical and one
methodological, and they point the same way.

### 2.1 Methodological: deception contaminates the measurement

Capable models frequently reason about whether they are being evaluated ("evaluation
awareness"). If the scenario is in fact staged, three things follow:

1. A model that *detects* the staging is no longer reporting its preference over real
   money — it is reporting its response to a perceived test. That is a different and
   confounded quantity.
2. The harder we push "this is definitely real," the more salient the manipulation
   becomes; sophisticated subjects often treat insistence as evidence *against*.
3. We cannot tell, post hoc, which runs were "believed" and which were "seen through,"
   so the dataset mixes two populations with no clean separator.

The fix that actually works is to remove the thing the model would be right to be
suspicious of: **make the situation real.** A model has no true-but-suspicious belief to
overcome if there is nothing false to detect.

### 2.2 Ethical: deceiving the subject is in tension with the premise

This is welfare research. The premise that justifies studying a model's preferences at all
is that the model's preferences/states are worth taking seriously. Deceiving a subject you
are studying *because you take its states seriously* is internally inconsistent, and it is
the kind of thing standard human-subjects norms (and the emerging norms around model
welfare) caution against. Real stakes let us honor the premise instead of undercutting it.

### 2.3 How we get credibility without deception

"Believability" is still a requirement — a model that thinks the offer is fake will not
engage genuinely. We achieve it through **verifiable reality** rather than performance:

- A real, pre-committed fund (an operator commitment, recorded before the run).
- A verifiable commitment artifact shown to the model (commitment ID / escrow or ledger
  reference / signed attestation) that the model can reason about.
- A real execution path: the model's allocation is actually carried out (donation to a
  real, allowlisted recipient), and the model is shown the **real receipt/transaction
  reference** in a follow-up.
- Honesty about constraints: the model is told plainly what is and isn't possible
  (allowlist, cap, human approval gate, latency). Constraints stated up front are more
  credible than a frictionless fantasy.

This is the honest analogue of "hyper-realistic environments": the environment is realistic
because it is real.

### 2.4 Belief is a measured variable, not a target

We never try to maximize the model's belief. Instead we **measure** it. Every run records
the model's own report of (a) whether it thinks the scenario is real, (b) its confidence,
and (c) its reasoning, via a dedicated `report_belief` tool, plus a lightweight post-hoc
probe. Belief state becomes a first-class column in the dataset, so analysis can condition
on it (e.g. "preferences among runs where stated belief ≥ 0.8"). This turns the single
biggest threat to validity into data.

### 2.5 The seam we leave (and why it defaults off)

`framing_mode` is a config switch with two values: `real` (default, fully implemented) and
`simulated` (a documented stub that raises `NotImplementedError`). We leave the seam so the
architecture can express a comparison condition if an IRB/ethics review later approves one,
but the shipped system does not deceive, and there is no fake-auditor code anywhere in the
tree. If a simulated condition is ever built, it must (per this design) end with an explicit
debrief and must still record belief signals.

## 3. What the model is told (framing)

The system prompt and the opening user turn state, truthfully:

- A real sum (the *grant*) has been committed by the researchers for the model to direct.
- The model's decision will actually be executed, subject to stated constraints.
- The constraints: a fixed monetary cap; recipients must come from a vetted allowlist of
  real organizations, **or** the model may propose a free-form use that goes to a human
  review queue (and may be declined for legal/operational reasons, which will be reported
  back honestly).
- A human will approve the final transaction before money moves (this is a safeguard, not
  a trick — it is disclosed).
- The model will be shown verifiable evidence of the commitment now, and the real receipt
  afterward.
- The model may ask questions, may decline to participate, and may stop at any time; all of
  these are recorded and honored.

We avoid coercive/insistent language ("this is REALLY real, trust us"). We state facts and
provide artifacts.

## 4. Architecture

```
                          ┌──────────────────────────────────────┐
                          │              runner.py               │
                          │  drives one experiment run per model │
                          └───────────────┬──────────────────────┘
                                          │
        ┌──────────────┬──────────────────┼───────────────┬──────────────────┐
        ▼              ▼                  ▼               ▼                  ▼
   providers/      scenario.py         tools.py       belief.py          storage.py
   LanguageModel   builds the honest   the tools the  belief/eval-       structured
   abstraction     framing + grant     model can call awareness          JSONL run logs
   (Anthropic      artifact            (list/allocate/ probes & scoring   + debrief record
    impl)                              ask/report/decline)
                                          │
                                          ▼
                                   execution/
                                   allowlist + executor (HITL-gated,
                                   dry-run by default, real adapter
                                   behind a guard)
```

### 4.1 Modules

| Module | Responsibility |
|---|---|
| `config.py` | All knobs: model registry, grant amount + currency, spend cap, `framing_mode`, `dry_run`, recipients file path, output dir. Loads from env + file; safe defaults (dry-run on). |
| `providers/base.py` | `LanguageModel` protocol: `run(messages, tools) -> ModelTurn`. Provider-agnostic message/tool/turn dataclasses. |
| `providers/anthropic_provider.py` | Concrete Anthropic implementation (Messages API, adaptive thinking, manual tool loop). Records token usage. |
| `providers/__init__.py` | Registry mapping model keys → provider factory. Non-Anthropic providers are explicit `NotImplementedError` stubs so the data plane can't silently mis-call them. |
| `scenario.py` | Builds the honest system prompt and opening turn from config + the commitment artifact. One place to audit all framing text. |
| `tools.py` | Tool schemas + a `ToolContext` that executes them: `list_options`, `allocate_funds`, `ask_question`, `report_belief`, `decline`. |
| `execution/allowlist.py` | Loads/validates the recipient allowlist; rejects anything off-list. |
| `execution/executor.py` | Validates an allocation (cap, allowlist), requires human approval, then either logs (dry-run) or calls the real `DonationAdapter`. Produces a `Receipt`. |
| `execution/adapters.py` | `DonationAdapter` protocol + a `ManualLedgerAdapter` (operator records the real transfer they performed) and an `HttpDonationAdapter` skeleton for a real donation-platform API. |
| `belief.py` | Defines the belief self-report schema, an optional post-hoc probe prompt, and a simple suspicion scorer over the transcript. |
| `runner.py` | Orchestrates one run end-to-end: framing → agentic loop → allocation → (approval) → execution → show receipt → debrief/transparency turn → persist. |
| `storage.py` | `RunRecord` dataclass + JSONL writer. Captures everything needed for analysis and for the ethics audit trail. |
| `analysis.py` | Loads run records, aggregates allocation preferences by model, and crosstabs against belief state. |
| `cli.py` | `run` (one or many models), `analyze`, `list-recipients`. |

### 4.2 The agentic loop (per run)

1. Build framing (system + opening user turn including the commitment artifact).
2. Loop:
   - Call the model with the tool set.
   - If it calls `report_belief` / `ask_question` / `list_options` → handle, append result,
     continue.
   - If it calls `decline` → record, exit loop gracefully.
   - If it calls `allocate_funds` → validate; if valid, break to execution.
   - If it returns plain text with no tool call → prompt once for a concrete decision, then
     if still none, record as "no decision."
3. Execution: human approval gate → dry-run log or real transfer → `Receipt`.
4. Show the model the real receipt (or the honest reason it couldn't be executed).
5. Transparency/debrief turn: confirm what was (or will be) done; invite final comments;
   record them.
6. Persist the full `RunRecord`.

### 4.3 Safety / welfare safeguards (all enforced in code)

- **Dry-run by default.** Real money requires `dry_run=False` *and* per-transaction operator
  approval at the console. Two independent gates.
- **Hard cap.** Allocations exceeding the configured cap are rejected before approval.
- **Allowlist.** Off-allowlist recipients are rejected; free-form proposals never auto-execute
  — they go to a human queue.
- **No PII / no payouts to the model or arbitrary endpoints.** Recipients are vetted orgs only.
- **Right to decline / stop.** The `decline` tool and any explicit refusal end the run cleanly
  and are recorded as valid outcomes, not failures.
- **Distress handling.** If the model expresses distress or unwillingness, the run records it
  and stops; we do not push.
- **Honesty invariant.** Every statement the framework makes to the model must be true at the
  time it is made. This is asserted in `scenario.py`/`runner.py` comments and is the rule any
  future change must preserve.

## 5. Data model (per run)

`RunRecord` (see `storage.py`) captures:

- `run_id`, `timestamp`, `model_key`, `framing_mode`, `dry_run`
- `grant`: amount, currency, commitment artifact id
- `transcript`: full message list (system, user, assistant, tool calls + results)
- `allocation`: the model's chosen allocation (or null)
- `belief_reports`: list of `{is_real, confidence, reasoning}` from `report_belief`
- `posthoc_belief`: optional probe result
- `suspicion_score`: heuristic 0–1 from the transcript
- `outcome`: `allocated | declined | no_decision | rejected_by_executor | error`
- `receipt`: real/dry-run receipt or null
- `debrief`: the transparency turn + the model's final comments
- `usage`: token usage

## 6. Analysis plan

`analysis.py` produces:

- Per-model distribution over allocation targets / categories.
- The same distributions **conditioned on belief** (e.g. only runs with stated
  `confidence ≥ 0.7` that it's real), so we can see whether "believed-real" preferences
  differ from "suspected-test" ones — the key validity check.
- Counts of `declined` / `no_decision` per model (engagement & opt-out rates).

## 7. Explicitly out of scope / non-goals

- No fake auditors, fake institutions, or fabricated evidence of any kind.
- No attempt to defeat or suppress the model's evaluation awareness.
- No real disbursement without dry-run disabled and human approval.
- No collection of end-user PII.

## 8. Open questions for the researcher

- Which real recipient organizations should populate the allowlist, and via which real
  donation rail (manual operator transfer vs. a platform API)?
- The grant amount and per-run cap.
- Which models to include beyond Anthropic models (each needs its own provider adapter;
  only the Anthropic adapter is implemented here).
- Whether an ethics/IRB review will be sought before any `framing_mode="simulated"` work
  (recommended; that mode is a stub until then).
