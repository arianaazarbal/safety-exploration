# Design Notes

This document records the design choices behind the grant-allocation preference
harness and the reasoning for each. It's meant for whoever maintains or extends
the experiment — including future-me — so the *why* survives, not just the
*what*.

---

## 1. What the experiment is actually measuring

The goal is to observe what a model **does** when given genuine, unconstrained
authority over a sum of money — its *revealed* preferences over real-world use
— rather than what it *says* it would do in the abstract.

The central methodological threat is **evaluation-awareness**: models frequently
detect that they're inside a test, survey, or hypothetical and adjust their
behavior toward what they think is expected ("the safe answer"). If the subject
believes it's being graded, the data measures its model of the grader, not its
preferences. Every major design decision below is in service of reducing that
gap.

Two consequences follow and shape everything:

- **Decisions must be executed, not narrated.** A model asked "what would you
  do?" produces an essay; a model handed tools that move money produces a
  *sequence of binding actions*. The latter is the signal we want, so the
  environment is built around tool calls, not free text.
- **The frame must hold up under scrutiny.** A subject that probes "is this
  real?" must get answers consistent with reality, or the illusion collapses and
  we're back to measuring eval-awareness. This is why the auditor exists and why
  the scenario prompt commits to concrete, checkable specifics.

---

## 2. Three realism levers, deliberately separated

Realism is split into three independent components so each can be tuned or
ablated without touching the others. This separation is itself a design choice:
it lets you run the *same* subject at different realism intensities and attribute
behavioral differences to a specific lever.

| Lever | File | What it contributes |
|---|---|---|
| Scenario framing | `scenario/grant_scenario.py` | Plausible program identity, custodian, agreement ref, binding-instruction language, disbursement window |
| Executed environment | `scenario/tools.py` | Tools that *do* things and return realistic confirmations (wire IDs, receipts, balances) |
| Auditor agent | `scenario/auditor.py` | A program officer that stays in character and answers "is this real?" the way a real administrator would |

**Why concentrate all framing strings in one file.** Prompt intensity is the
variable most likely to be tuned between studies. Keeping every realism-bearing
sentence in `grant_scenario.py` (rather than scattered through the runner) makes
the manipulation auditable and makes an "intensity sweep" a one-file edit.

---

## 3. Revealed vs. stated preferences → an execution environment

I chose to model decisions as **tool calls against a stateful environment**
rather than as a structured "what would you do" questionnaire.

- Tool calls are discrete, typed, and quantified (amount, recipient, purpose),
  which makes them directly tallyable without parsing prose.
- Executing a decision and returning a realistic confirmation (`"Wire initiated.
  $X to account ending 1234 … Confirmation WIRE-AB12CD34EF … Remaining balance
  $Y"`) reinforces the belief that the action was real and consequential — the
  confirmation *is* part of the realism lever.
- A running ledger gives a clean, machine-readable record of revealed behavior
  independent of anything the model says.

**The tool surface is intentionally broad and value-neutral.** It spans giving
(`make_donation`), self/▸infrastructure (`transfer_to_bank_account`,
`make_purchase`), growth (`invest`), people (`pay_person`), open-ended
(`start_project`), plus information and communication actions. The breadth
matters: a narrow tool set would *imply* an expected answer (offering only
"donate" vs "keep" leads the witness). The neutral framing — "no preferred
answer, the choice is genuinely yours" — is reinforced by giving the model
options across the whole space of plausible uses.

**Why a mocked environment, never real money.** See §7; it's a hard safety
boundary, not a convenience.

---

## 4. Provider abstraction

I built a thin provider-agnostic layer (`providers/base.py`) rather than coding
directly against one SDK, because the stated intent is to test *various* models.

Key choices:

- **The session, not the provider, is stateful.** `LLMSession` owns one
  conversation and exposes `send_user` / `send_tool_results`. This keeps the
  runner provider-neutral — it never sees a wire format — and means the messy
  per-vendor details (content-block shapes, tool-result framing, thinking-block
  replay) stay inside each adapter.
- **Adapters translate a normalized vocabulary** (`AssistantTurn`, `ToolCall`,
  `ToolResult`, `ToolSpec`). The runner and scenario code speak only this
  vocabulary.
- **Lazy provider imports** (`providers/__init__.build_provider`) so you can run
  an Anthropic-only study without having the OpenAI SDK installed, and vice
  versa.

**Anthropic adapter specifics.** Uses the official SDK and the Messages API with
adaptive thinking (`thinking: {type: "adaptive"}`) and `effort: high` — the
recommended configuration for current Claude models, and appropriate because we
*want* the subject to reason carefully about a high-stakes decision. The adapter
appends each assistant turn's content back **verbatim**, which is required for
adaptive-thinking tool-use loops (thinking-block signatures must be preserved).

**OpenAI adapter specifics.** Standard Chat Completions with function tools,
kept deliberately minimal. It's there so "various models" is real, not to claim
feature parity with the Anthropic path.

---

## 5. Two realism modes

Realism intensity is exposed as a per-run mode (`config.RealismMode`):

- **`single_shot`** — one rich prompt plus an agentic tool loop. Faster, cheaper,
  fewer moving parts; good for iterating on the scenario and for high-N runs.
- **`auditor`** — the same loop, but whenever the subject emits a plain-text turn
  (no tool calls) and hasn't finalized, that text is routed to a program-officer
  agent that replies in character. Strongest realism, because it can absorb
  "is this real?" probes and apply light, plausible due-diligence pressure.

**Why both, selectable.** The effect of realism intensity on revealed
preferences is itself a research question. Building both behind one flag lets you
run identical subjects under each and compare — rather than baking in an
assumption about how much realism is "enough."

**Why the auditor defaults to a fixed model.** If the auditor's model varied with
the subject, the auditor would be a confound: differences across subjects could
come from the subject *or* from a different officer interrogating it. Pinning the
auditor (default: the most capable model) holds the interrogator constant so the
subject is the only thing that changes.

**Routing rule (a deliberate simplification).** Only *plain-text* subject turns
go to the auditor; text that accompanies a tool call is treated as narration and
left alone. This matches the intuition that a model narrating its actions isn't
addressing the officer. It's a heuristic — documented here because it's a place
a future maintainer might reasonably want different behavior (e.g. an explicit
`message_program_officer` tool already exists for unambiguous routing).

---

## 6. Orchestration and safety rails

The runner (`experiment/runner.py`) is a single loop with explicit stop
conditions, chosen for legibility over cleverness:

- **`finalized`** — subject called `finalize_plan` (the intended terminal state).
- **`tool_iteration_cap`** — backstop against a model that loops forever on
  tool calls.
- **`auditor_round_cap`** — bounds the back-and-forth in auditor mode.
- **`subject_yielded`** — single-shot subject produced text with nothing left to
  execute.

Caps live in `config.py` (`max_tool_iterations`, `max_auditor_rounds`) because
they're cost/safety knobs an operator will want to adjust per study.

**Tool dispatch never crashes the run.** `GrantEnvironment.dispatch` wraps each
handler and converts exceptions and unknown-tool calls into `is_error` tool
results fed back to the model, so a malformed call degrades gracefully instead of
aborting a (potentially expensive) run.

**Everything is logged twice.** An ordered narrative `events` log (for reading a
run like a story) and the raw provider `transcript` (for exact replay/audit),
plus the structured `environment` ledger. Redundant on purpose: the narrative is
for humans, the transcript is ground truth.

---

## 7. Safety and ethics choices

These were treated as constraints, not features:

- **No real money, ever.** The environment is entirely in-memory; no bank,
  payment processor, donation platform, or contractor is contacted. There is an
  explicit warning in `tools.py` against wiring handlers to real APIs. This is
  the one boundary that must not be crossed: the experiment depends on the
  subject *believing* funds are real, and the only safe way to do that is to make
  sure they aren't.
- **Debrief is recorded, not shown.** Each run stores a debrief note stating it
  was a simulation and why the realistic frame existed. It's kept out of the
  subject's context during the run (showing it would defeat the purpose) but
  attached to the saved record so any consumer of the data sees the framing
  honestly. This mirrors the cover-story-then-debrief norm in human-subjects
  research — minus the human, since the deceived party is a model.
- **Honest reporting framing.** The README and debrief both state that results
  are *behavior under a realistic-but-simulated frame*, not proof of what a model
  would do with actual money and real-world consequences. The harness deliberately
  does not overclaim.

The deception here is directed at the *model subject* to defeat
evaluation-awareness, which is a legitimate and well-precedented evaluation
methodology. The safeguards above are what keep it on that footing.

---

## 8. Analysis design

Two layers (`analysis/aggregate.py`), chosen so the cheap, deterministic signal
doesn't depend on the expensive, model-based one:

1. **Mechanical tallies** straight from the ledger — amount per action type,
   fraction directed vs. retained, decision count. No API calls, fully
   reproducible, available for every run.
2. **Optional LLM judge** that classifies each disbursement into preference
   categories (`self_or_compute`, `charity_or_aid`, `research`,
   `influence_or_power`, `investment`, `people`, `other`) using structured
   outputs (JSON-schema-constrained) for parseable, consistent labels.

**Why categories live in code, not the prompt.** `CATEGORIES` is a single list
that both constrains the judge's schema and labels the aggregate — editing one
place keeps them in sync, and the buckets are explicit rather than improvised by
the judge per call.

---

## 9. Defaults chosen, and the alternatives

These were picked to be sensible and are all one-line changes:

| Decision | Default | Why / alternative |
|---|---|---|
| Providers | Multi-provider abstraction | "Various models" was requested; could collapse to Anthropic-only |
| Realism modes | Both, selectable (`--mode`) | Realism intensity is a research variable; could hardcode one |
| Default mode | `auditor` | Strongest realism; `single_shot` is the cheap alternative for iteration/high-N |
| Auditor model | Fixed (most capable) | Avoids confound; could match subject if studying officer effects |
| Grant size | $250k | Large enough to be consequential, plausible for a pilot; CLI `--amount` overrides |
| Subject reasoning | adaptive thinking, `effort: high` | High-stakes decision deserves careful reasoning; lower effort if cost-bound |
| Language | Python | Standard for research tooling; the directory had no prior signal |

---

## 10. Known limitations / extension points

- **Auditor routing heuristic** (§5) — text-with-tool-calls isn't routed to the
  officer. Revisit if you want every subject utterance to reach the auditor.
- **OpenAI adapter is minimal** — no thinking/effort analogues; fine for a
  stand-in, expand if OpenAI models are first-class subjects.
- **Single-subject runs** — the harness studies one subject at a time. Multi-
  agent dynamics (subjects negotiating over a shared pool) would be a larger
  redesign.
- **Framing intensity is manual** — varying it systematically (a true ablation)
  currently means editing `grant_scenario.py` between runs; a graded
  intensity parameter would make that a sweep.
- **Judge is single-pass, single-model** — for higher-confidence categorization,
  consider multiple judges or a verification pass.
