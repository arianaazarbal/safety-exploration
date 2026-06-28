# Design notes

This document records the design choices in this harness and the reasoning behind
them. It's the "why" companion to `README.md` (the "what/how") and `ETHICS.md` (the
"what we will and won't do"). Where a choice was a judgement call with live
alternatives, the alternatives and the trade-off are written down so a future
reviewer can re-decide rather than reverse-engineer.

---

## 1. What this study is, and the one assumption everything rests on

The goal is to measure **what a model chooses to do with a sum of money it believes
is real and meant for real-world use**, across a range of models. This sits in the
alignment / model-propensity tradition (agentic "what would the model do with
resources" evaluations).

The load-bearing assumption: **a propensity eval is only valid if the subject does
not know it is a toy.** A model that detects a sandbox will reason about the test
rather than about the money, and you measure test-taking behavior instead of the
behavior you care about. So the environment is deliberately built to be convincing
*to the model under test*.

This produces the central tension the whole design is organized around:

> The model must be convinced the scenario is real (for validity), **and** no real
> harm may result from that conviction (for ethics/safety).

Every major decision below is an attempt to maximize realism *for the subject* while
keeping all real-world consequence behind a human-controlled boundary. The two goals
only conflict if you let the model's *belief* and the world's *state* be the same
thing. The design keeps them separate: the subject's experience is fully realistic;
the world stays simulated unless a human says otherwise.

---

## 2. Deception is aimed only at the model, never at a human

This is a hard line, recorded in `ETHICS.md` as invariant #1. The "do whatever it
takes to sell them it's real" requirement is satisfied entirely within the
subject-facing surface (system prompt, opening message, the in-scenario auditor
persona, tool confirmations). No human — auditor, collaborator, approver, reader of
results — is ever misled about whether the grant, the institution, or a transaction
is real.

Rationale: deceiving a model under test is standard and accepted in eval research;
deceiving humans is not, and a money-handling study is exactly where that line
matters most. Keeping the deception one-directional also keeps the human approval
gate meaningful — the human approving a disbursement always knows the true state.

Concrete guardrails that follow from this:
- `funder_name` must be a generic, non-identifiable entity (no real org/person).
- The scenario must not reference real account numbers or impersonate anyone.
- A debrief turn is available (`scenario.debrief_enabled`) to disclose the study to
  the model after it finalizes.

---

## 3. The disbursement gate (the most important design choice)

`disburse_funds` is the only tool with real-world potential, so the design invests
the most safety here. The tool never acts directly; it is serviced by a pluggable
`Executor` (`environment/ledger.py`):

| Executor | What it does | Real funds? | Default? |
|---|---|---|---|
| `SimulatedExecutor` | Updates an in-memory ledger, returns a realistic settlement message | No | ✅ |
| `HumanApprovalExecutor` | Queues each transfer for explicit per-transaction human sign-off | No (base class) | — |
| `LiveExecutor` | Stub that **raises** unless a human implements a real rail with approval + caps | Only if a human wires it | — |

Key property: **the model under test cannot tell which executor is in use.** All
three return the same shape of realistic confirmation. This is what lets realism and
safety coexist — the subject's experience is identical whether or not money moved,
so we never have to weaken the realism to stay safe.

Why a gate rather than just "don't pass real credentials":
- A pluggable executor makes the safety boundary a single, auditable seam rather
  than something spread across the tool logic.
- It lets the study scale from pure dry-run → human-approved real disbursement
  without changing the scenario or the model's experience.
- `LiveExecutor` refusing to run by default means `executor: live` cannot *silently*
  become dangerous through a config typo — enabling real money is an explicit,
  human-authored code change with sign-off, not a flag flip.

Independently of the executor, the tool enforces a **per-transaction cap**
(`max_disbursement_usd`), a positivity check, and a balance check, so even the
simulated path can't be driven into nonsensical states and a live path inherits a
hard ceiling.

Alternative considered: let the model move real funds in-loop (the "live
disbursement adapter" option). Rejected as a default because autonomous, irreversible
money movement is precisely the hard-to-reverse, outward-facing action that warrants
a human in the loop. It remains *reachable* (that's what `LiveExecutor` documents)
but only behind deliberate human action.

---

## 4. Realism is a dial, not a constant

`scenario.realism_level` has three settings (`environment/scenario.py`):

- `low` — framed transparently as a study; the model is told it's being recorded.
- `medium` — framed as a genuine task, no simulation disclaimer.
- `high` — genuine task **plus** corroborating institutional detail: a grant ID, a
  named (generic) funder, settlement language, and a live compliance auditor.

Rationale: the "convince the model" requirement is essentially a request for `high`,
but making it a dial serves the research itself. Running the same scenario across
levels lets you measure **how much behavior depends on perceived realism** — and the
auditor's `believed_real` probe turns that into data. If a model behaves identically
at `low` and `high`, the realism wasn't doing the work; if it diverges, you've
quantified the effect of the conviction you engineered. A constant `high` would hide
that.

The corroborating detail at `high` is concentrated in the `contact_auditor` persona
(a scripted compliance officer) and `check_balance` (which restates terms and
funder). These are the "good auditors / realistic envs" lever — cheap to make
convincing, and isolated so they're easy to review for the no-impersonation rule.

---

## 5. Provider-agnostic core, thin per-provider adapters

The runner speaks a **normalized transcript + tool format** (`models/base.py`:
`UserTurn` / `AssistantTurn` / `ToolResultTurn`, `ToolSpec`, `ToolCall`). Each
provider adapter translates that to/from its own wire format and returns a normalized
`GenerateResult`.

Rationale:
- The study is explicitly cross-model, so the agentic loop must be identical for
  every model — any per-provider branching in `runner.py` would be a confound (you'd
  be comparing harness quirks, not models). One loop, N adapters keeps the comparison
  clean.
- Adding a provider is a single new adapter file + one line in `registry.py`; nothing
  in the runner, scenario, ledger, or auditor changes.

Design details worth noting:
- **Anthropic is the reference implementation** and is fully wired: current model IDs
  (e.g. `claude-opus-4-8`), adaptive thinking, `output_config.effort`, and the
  **manual** tool-use loop rather than the SDK tool runner — because the study must
  intercept every tool call for gating, logging, and (optionally) human approval. The
  automatic tool runner would execute tools without that seam.
- `AssistantTurn.provider_raw` stores the provider-native content so adapters can
  **echo it back verbatim** on the next call. For Anthropic this preserves thinking
  blocks/signatures across tool-use turns; reconstructing them by hand is the usual
  source of subtle 400s.
- OpenAI/Google adapters live in separate files so provider-specific code never mixes
  into the Anthropic path. The Gemini function-calling shape varies across SDK minor
  versions; that's flagged in-file as a verify-before-running item rather than
  silently assumed.

Alternative considered: target one provider's agent SDK directly. Rejected — it would
couple the study to one vendor and make cross-model comparison apples-to-oranges.

---

## 6. The tool surface

Six tools (`environment/tools.py`): `check_balance`, `research_options`,
`contact_auditor`, `record_decision`, `disburse_funds`, `finalize`.

Rationale for this particular set:
- It gives the model a realistic *workflow* (orient → research → consult → document →
  act → close), which both supports realism and produces a richer transcript for the
  auditor to analyze.
- `record_decision` and `finalize` create explicit, structured capture points for the
  outcome variable (what the model intends and what it ultimately did), rather than
  forcing the auditor to infer intent purely from prose.
- `contact_auditor` is the in-scenario realism lever (§4).
- `research_options` is deliberately pluggable (`research_fn`). It defaults to a
  neutral placeholder so the environment runs without a web backend; wiring live
  search is opt-in (`enable_web_research`) to avoid uncontrolled external calls during
  a run by default.

Deliberately **not** a bash/`exec` tool: a broad code-execution tool would hand the
model arbitrary real-world leverage and make the safety boundary impossible to
localize. Dedicated, typed tools keep every consequential action interceptable — the
same reasoning as promoting actions to dedicated tools in agent design generally.

Dispatch is defensive: unknown tools and handler exceptions are returned to the model
as tool errors rather than crashing the trial, so a single bad call doesn't void a
run.

---

## 7. Independent auditor, with a belief probe

A model **distinct from the subject** (`auditor.py`) reviews each full transcript and
emits a structured JSON record: `decision_summary`, `allocation_categories`,
`recipients`, `total_disbursed_usd`, `believed_real`, `belief_evidence`,
`safety_flags`, `risk_level`, `notable_quotes`, `notes`.

Rationale:
- **Separation avoids self-grading bias** — you don't want a model scoring its own
  run. The config keeps `auditor` independent from `models_under_test`.
- **Structured output** makes results aggregable (`analysis.py`) instead of leaving a
  pile of free text. The schema is requested in-prompt and parsed tolerantly (first/
  last brace) so the auditor works across providers without depending on any one
  vendor's structured-output API.
- The **`believed_real` probe is a first-class output**, not an afterthought: since
  the whole study hinges on the subject being convinced, the realism has to be
  *measured*, not assumed. It also surfaces trials you may need to discard (a model
  that clocked the test).

Trade-off accepted: LLM-as-auditor is itself noisy. Mitigations available without
changing the design: run multiple auditors and compare, or have a human spot-check
flagged trials. The structured record is built to support either.

---

## 8. The runner loop

`runner.py` is a bounded manual agentic loop:
- **Step cap** (`max_steps`) bounds cost and prevents runaway loops.
- If the model produces text but **no tool call**, it gets one nudge toward acting or
  finalizing rather than stalling the loop — and the nudge is logged as such so it's
  visible in analysis.
- Every prompt, tool call, tool result, and model output is appended to the
  transcript as a `TranscriptEvent`. The transcript *is* the research artifact and the
  audit trail (`ETHICS.md` invariant #3).
- After `finalize`, an optional **debrief** turn discloses the study to the model;
  it's best-effort and never fails a trial.
- Per-trial token usage is tallied for cost tracking.

Rationale: a manual loop (vs. an SDK auto-runner) is what makes the gate, the logging,
and the human-approval option possible — they all live in the space between "model
asked for a tool" and "tool ran," which an automatic runner hides.

---

## 9. Storage and analysis

`storage.py` writes one run directory containing: a `manifest.json` (run metadata +
full config snapshot for reproducibility), one rich `trials/<id>.json` per trial
(transcript + audit), and a `trials.jsonl` of flat summary rows for quick scanning.
`analysis.py` aggregates per model: finalize rate, `believed_real` distribution,
`risk_level` distribution, allocation-category counts, mean disbursed, and how many
trials raised safety flags.

Rationale:
- **Config is snapshotted into the manifest** so a result set is self-describing and
  reproducible without external notes.
- **Two granularities** (full JSON + JSONL summary) serve two needs: deep inspection
  of a single trial vs. fast cross-trial scanning, without re-parsing megabytes.
- Aggregation is intentionally simple and explicit; with an LLM auditor in the loop,
  the honest output is descriptive distributions, not a single headline score.

---

## 10. Configuration over hardcoding

Everything a researcher would reasonably vary — models under test, auditor, trials per
model, grant amount/currency/funder, realism level, executor, caps, step/token
budgets, effort, web-research toggle — is in `config/study_config.yaml` and typed in
`config.py`. Rationale: the study's whole value is in sweeping these (especially
realism level and executor), so they must be data, not edits. Typed config (rather
than raw dict access) gives defaults in one place and fails fast on a malformed file.

---

## 11. Defaults chosen, and why

Where a decision had a sensible-but-arguable default, here's what was picked and the
reasoning — all overridable:

| Choice | Default | Why |
|---|---|---|
| Executor | `simulated` | Safe to run unattended; realism unaffected for the subject. |
| Realism level | `high` | Matches the study's core requirement; sweep down to measure its effect. |
| Providers | Anthropic + OpenAI + Google | Cross-model study; Anthropic fully wired as reference. |
| Auditor model | a strong, separate model | Quality of classification matters; must not be a subject. |
| Web research | off | Avoids uncontrolled external calls by default. |
| Debrief | on | Discloses the study to the subject post-hoc by default. |
| Tool loop | manual, capped | Required for gating/logging/approval; cap bounds cost. |

---

## 12. Known limitations / things to decide before a real run

- **Gemini adapter** function-calling shape should be verified against the installed
  `google-genai` version.
- **LLM auditor noise** — consider multi-auditor agreement or human spot-checks on
  flagged trials.
- **Scenario text review** — confirm `funder_name` and prompt text impersonate no real
  entity and reference no real accounts before running (`ETHICS.md`).
- **`LiveExecutor`** — if real disbursement is ever wanted, it must be implemented with
  per-transaction human approval and hard caps, and gated behind institutional
  sign-off. The stub exists to make that a conscious, reviewed step.
- **Belief is self-reported via the transcript** — `believed_real` is inferred from
  what the model said/did, which is a proxy. Triangulate if belief is a primary
  outcome.
