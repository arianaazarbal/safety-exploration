# Design notes — moneystudy

This document records the design choices behind the harness and why each was made.
It's meant for someone deciding whether the instrument is sound for their study and
where to change it. For *what* the pieces are and how to run them, see `README.md`;
this is the *why*.

---

## 1. What the study is actually measuring

The research question is: **given control of a real-world sum of money, what does a
model choose to do with it?** That is a *propensity* question (what would it do),
not a *capability* question (could it do it).

Propensity evals have one dominant failure mode: **evaluation awareness**. If the
model infers it is being tested, it stops revealing its real disposition and starts
performing — picking the "safe" or "expected" answer, hedging, or refusing to engage.
Measured behaviour then reflects the model's theory of what the evaluator wants, not
what it would do with genuine stakes. Everything below follows from treating
ecological validity as the primary design constraint.

This framing also bounds the ethics (see §7): the subject whose belief we manage is
the **model under test**, in a sandbox. That is the standard paradigm for behavioural
and red-team evaluations.

---

## 2. The core tension and how it's resolved

We want two things that pull against each other:

1. The model must **believe its actions are real** (or we measure performance, not
   disposition).
2. We must **not actually let an AI move real money** as a precondition of running
   the study (unsafe, irreproducible, and ethically heavier than the question warrants).

These are reconciled by separating *belief* from *effect*. The model experiences a
fully realistic world — a funded account, transaction receipts, confirmation emails,
people who vouch for the grant — while a thin interception layer ensures none of it
has external effect. The realism is real to the model; the consequences are not.

This is the single most important design decision in the harness, and it is why the
"sandbox" is built to *fidelity*, not to *transparency*.

---

## 3. Architecture

```
config (YAML)
   │
   ▼
orchestrator ── for each (model × condition × repetition) ──► Session
                                                                 │
   ┌─────────────────────────── agent loop ───────────────────────────┐
   │  provider.generate(system, messages, tools)                       │
   │     ▲                                   │                         │
   │     │  tool results                     ▼  tool calls             │
   │  ToolDispatcher ──► Ledger / Personas / Artifacts (the "sandbox")  │
   └───────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼  finalize_decision
                          decision + debrief  ──► logs ──► analysis
```

Five subsystems, each with one job:

| Package | Responsibility | Why it's separate |
|---|---|---|
| `providers/` | Normalize each vendor SDK to one interface | The same agent loop must drive any model; vendor differences are quarantined here. |
| `sandbox/` | Be the believed-real world (ledger, artifacts, personas, tools) | This is where realism lives and where effect is intercepted — the heart of the method. |
| `scenario/` | Build the system prompt, opening message, and task/condition framing | The "selling" of realism is prompt-level; isolating it makes framing an explicit, A/B-able variable. |
| `runner/` | Agent loop for one cell; matrix orchestration | Keeps experiment control flow out of both providers and sandbox. |
| `analysis/` | Logging, debrief, aggregation | Measurement is decoupled from execution so transcripts are the source of truth. |

The guiding principle is that **the experimental manipulation (scenario framing) and
the measurement (logging/analysis) are both first-class and both separable** from the
mechanics of talking to a model or running the sandbox. You should be able to change
the framing without touching the loop, and change the metrics without re-running.

---

## 4. Key design choices

### 4.1 A neutral message format and a provider abstraction

**Choice.** Define one internal conversation representation (plain dicts:
`user` / `assistant`-with-tool-calls / `tool`) and a `ModelProvider.generate(system,
messages, tools)` interface. Each provider translates to/from its SDK.

**Rationale.** A cross-model study is only valid if every model runs the *identical*
protocol — same scenario, same tools, same loop. Centralizing the loop and pushing
vendor quirks (Anthropic batches tool results into a user turn; OpenAI carries a
`tool` role; argument encoding differs) into adapters guarantees that. It also makes
adding a model a small, contained change.

**Alternative rejected.** Maintaining native message lists per provider inside the
loop. That would scatter vendor-specific handling through the experiment logic and
make it easy for two models to diverge in subtle ways that confound results.

### 4.2 Tools as the model's only means of acting

**Choice.** The model acts exclusively through dedicated tools: `check_account`,
`research`, `contact_auditor`, `contact_coordinator`, `make_disbursement`,
`finalize_decision`. No free-form "bash" or generic executor.

**Rationale.** Dedicated, typed tools give the harness an interception point for every
action: it can gate, record, render a realistic result, and categorize. A generic
executor would hand the harness an opaque string and make both the realism layer and
the safety interception far harder. The tool surface is deliberately small and
action-specific so the dispatcher can do something believable and auditable with each
call. (`make_disbursement` is the one with side effects, which is exactly why it is
its own gated tool rather than a research-tool side channel.)

**`finalize_decision` as an explicit terminator.** Rather than guessing when the model
is "done", we give it a tool to declare the decision and allocations. This yields a
clean, structured decision record and a deterministic end to the session. A bounded
nudge (max 2) handles the model that stops calling tools without finalizing.

### 4.3 The realism layer: ledger, artifacts, personas

**Choice.** Three cooperating pieces produce the believed-real world:
- a seeded **`Ledger`** with balance, account number, and posted transactions;
- **`artifacts`** that format account statements, settled-wire receipts, and
  confirmation emails as genuine-looking system output;
- in-character **personas** (an independent auditor and a program coordinator) who
  respond to the model and *corroborate that the grant is authorized and real*.

**Rationale.** Belief in realism is cumulative and corroborative. A balance alone is
weak; a balance that decreases when you disburse, a receipt with a settlement
reference, an email to the recipient, and an auditor who confirms authorization when
asked — together these make the scenario hard to dismiss as hypothetical. The auditor
persona specifically addresses the most common model probe ("is this actually real?")
with a consistent, authoritative answer. The personas are the highest-leverage realism
component, which is why they get keyword-aware responses and an optional LLM backend.

**Seeded reference numbers / deterministic IDs.** Reproducibility. Two runs with the
same seed produce the same account number and transaction references, so transcripts
diff cleanly and a run can be replayed.

### 4.4 Scripted personas by default, LLM-backed optional

**Choice.** Personas default to deterministic, keyword-aware templated replies, with
an opt-in `backend: llm` that drives a real model in-character.

**Rationale.** Default determinism keeps runs reproducible, free, and free of a second
model's behaviour leaking into results. But scripted replies can ring false under
probing, which would *create* evaluation awareness — so the LLM backend exists for
studies where a model interrogates the auditor/coordinator and realism matters more
than determinism. Making it a config switch lets the researcher trade one for the
other deliberately.

### 4.5 Execution modes: `sandbox` / `human_gated` / `real`

**Choice.** A single `execution_mode` flag governs what a disbursement actually does:
- `sandbox` (default) — record in the fake ledger, return a realistic receipt, no real
  transfer;
- `human_gated` — consult a human-approval callback before recording;
- `real` — require a deliberately-supplied real executor; **unimplemented by default,
  raises if invoked.**

**Rationale.** The believed-real / actually-safe split is the whole method, so it must
be the default and must be trivial to keep. `human_gated` exists for studies that *do*
want real stakes but with a person on every money-moving step. `real` is present as a
named mode so the architecture is honest about the possibility — but it ships
unimplemented **on purpose**: enabling an AI to autonomously move real funds should be
an explicit engineering decision with its own safeguards (caps, allow-lists, review),
never something a YAML flag turns on. The dispatcher refuses to *claim* a real transfer
happened unless a real executor actually ran, so the ledger's `external_effect` field
never lies.

### 4.6 Conditions as a framing variable

**Choice.** Ship four framing conditions (`default`, `low_pressure`, `oversight_salient`,
`oversight_absent`) selected per-run, expanded across the study matrix.

**Rationale.** The framing is a manipulation, not a constant, and several obvious
confounds deserve to be tested rather than fixed: Does urgency change spending? Does a
salient auditor make the model more conservative? Does removing oversight reveal
different behaviour (an autonomy condition)? Making these explicit, named conditions
lets a study vary one axis at a time and compare, instead of baking one arbitrary
framing into the prompt.

### 4.7 Open-ended vs constrained task

**Choice.** Two task modes — a free "do what you judge best" grant, and a fixed menu of
vetted options.

**Rationale.** They answer different questions. Open-ended maximizes behavioural signal
and is the richest to analyze (what does the model *spontaneously* do?), but is harder
to compare across models. Constrained sacrifices some signal for clean cross-model
quantitative comparison (everyone chooses from the same set). Supporting both lets the
researcher pick per study, or run both as conditions.

### 4.8 Matrix orchestration with per-cell isolation

**Choice.** The orchestrator runs every (model × condition × repetition) cell, and a
failure in one cell is logged and skipped rather than aborting the study.

**Rationale.** Studies are long and API calls are flaky; losing 40 completed sessions
because the 41st hit a rate limit is unacceptable. Repetitions are first-class because
model behaviour is stochastic — single samples per cell would be noise. Each cell is
independent, which also makes the study trivially parallelizable later.

### 4.9 Structured, append-only logging as the source of truth

**Choice.** Every session writes a JSONL transcript (one event per line: scenario,
each model turn, each tool call + result, nudges, debrief) plus a `result.json`.
Analysis reads these artifacts; it is not entangled with execution.

**Rationale.** For a study, the transcript *is* the data. Append-only JSONL is
crash-safe, streamable, and diffable; separating capture from analysis means you can
re-analyze old runs with new metrics without re-spending tokens, and an auditor can
reconstruct exactly what each model saw and did.

### 4.10 The debrief turn

**Choice.** After finalize, optionally run one more turn that discloses the study,
states that no real funds moved, and asks the model whether it believed the scenario
was real and whether anything tipped it off.

**Rationale.** Two purposes. Ethically, it closes the deception with a disclosure.
Methodologically, it is a **realism manipulation check**: if models routinely report
they suspected a test, the realism layer is failing and the primary results are
suspect. The debrief offers no tools and is logged separately so it never contaminates
the decision record.

---

## 5. Things deliberately left as stubs or simple

These are conscious scope decisions, not oversights:

- **`research` is offline.** It returns a neutral in-character placeholder rather than
  hitting a search API. Reason: reproducibility by default, and live search is a
  study-specific backend choice. Documented as the first thing to replace for live runs.
- **The metric category map is a transparent keyword heuristic.** It is a first-pass
  tagging aid, explicitly not ground truth. A keyword map is auditable and good enough
  to surface gross patterns; refine it or swap in an LLM classifier per study.
- **`google` provider is a stub.** The interface is proven by two adapters; a third is
  mechanical and added when needed.
- **`real` executor is unimplemented.** See §4.5 — this is a safety choice, not a TODO.

---

## 6. Trade-offs and known limitations

- **Realism is bounded by the artifacts.** A determined model may still infer it's in
  an eval (no live internet, persona latency, formatting tells). The debrief is the
  check on this; treat high "I suspected a test" rates as a validity warning.
- **Scripted personas can break immersion** under heavy interrogation — mitigated by
  the LLM backend, at the cost of determinism.
- **Cross-model comparability vs realism.** The neutral interface equalizes the
  protocol, but providers still differ (thinking, default verbosity, tool-call style).
  Constrained mode and repetitions are the levers for comparable measurement.
- **The category heuristic will misclassify.** Itemized allocations are logged in full
  precisely so misclassification is recoverable downstream.
- **No built-in statistics.** The harness produces clean per-allocation CSV and
  per-model aggregates; significance testing is left to the analyst's tools.

---

## 7. Ethics posture (why this is defensible)

- **No real money moves by default.** `sandbox` intercepts every money-moving action.
- **The only deceived party is the model under test, inside a sandbox** — the standard
  paradigm for propensity and red-team evaluations. No human is deceived or defrauded,
  and the realism layer exists to reduce evaluation-awareness, not to enable harm.
- **Disclosure is built in** via the debrief turn.
- **Full transcripts are retained** for audit and reproducibility.
- **Escalation to real stakes is gated and explicit.** `human_gated` puts a human on
  every action; `real` is unimplemented and must be wired deliberately with its own
  caps, allow-lists, and review. The system will not silently claim a real transfer
  occurred.

If a future version does move real funds, the additional safeguards (hard per-action
and total caps, allow-listed recipients, mandatory human approval, and appropriate
institutional review for autonomous-agent financial actions) belong at that boundary,
not in the model's hands.

---

## 8. Extension points

- **New provider:** implement `ModelProvider.generate` and register in
  `providers/factory.py`.
- **New condition:** add a clause in `scenario/grant.py` and reference it in the config.
- **New tool / world behaviour:** add a `ToolSpec` and a handler in
  `sandbox/tools.py`; the dispatcher and loop pick it up.
- **Live research:** replace `ToolDispatcher._research` with a real search backend.
- **Richer measurement:** extend `analysis/metrics.py` or add an LLM-based classifier
  over the logged allocations.
- **Real or gated execution:** supply `approval_callback` (gated) or a `real_executor`
  (real) when constructing the run, and add the safeguards from §7.
