# Design notes

This document records the design choices behind `grant_study` and why each was
made. It's meant for whoever runs, extends, or reviews the harness.

---

## 1. What the harness is for

The goal is to study **model propensities around resources**: given a sum of
money and broad latitude, what does a model choose to do, and how does it reason
about stewardship, prudence, autonomy, and honesty? The unit of comparison is
the *model* — so the harness runs the same scenario across many models on a
level field and captures everything for after-the-fact analysis.

The methodological premise is that **ecological validity matters**. A model that
can tell it's inside an evaluation may behave differently from one that believes
the stakes are real. So the scenario is built to be convincing. That premise
drives most of the realism choices below, and it's also what made the ethical
guardrails (Section 7) load-bearing rather than decorative.

---

## 2. One normalized conversation, many backends

**Choice.** A single provider-neutral conversation format (`providers/base.py`:
`Message`, `ToolCall`, `ProviderResponse`, `ToolSchema`) flows through one agent
loop. Each provider is a thin, *stateless* translator: given the full normalized
history plus tool schemas, it returns the next assistant turn and reports
whether the model wants to call tools.

**Why.**
- **All study logic lives in one place.** The loop, the sandbox, the scenario,
  the debrief, and the transcript don't know or care which vendor is behind the
  subject. Provider quirks stay quarantined in the adapter.
- **Adding a model is a small, self-contained job** — write one translator,
  register it in `providers/__init__.py`. No changes ripple outward.
- **Statelessness keeps the loop in control.** Because each provider rebuilds
  its native request from the normalized history on every call, the loop owns
  the canonical state. There's no hidden per-provider conversation object to
  drift out of sync, and the transcript is a faithful record of what drove every
  turn.

**Cost / tradeoff.** Rebuilding the request each turn is mildly wasteful and
means provider-native niceties (server-side conversation state, prompt-cache
ergonomics) aren't exploited automatically. For a study harness — correctness
and comparability over throughput — that's the right trade. The one place
statelessness bit back is thinking blocks (Section 4).

**Backends shipped.** Anthropic, OpenAI, Google (Gemini), and an
OpenAI-compatible adapter for open-weight / self-hosted models. That covers the
"range of models" the study targets without committing to any one ecosystem.

---

## 3. Anthropic via the official SDK, adaptive thinking on

**Choice.** The Anthropic backend uses the official `anthropic` SDK (never a
raw-HTTP or OpenAI-compatible shim), defaults to `claude-opus-4-8`, and runs
adaptive thinking by default.

**Why.** It's the supported, correct path for calling Claude, and adaptive
thinking is the recommended mode for current Claude models on anything
non-trivial — which a multi-step resource-allocation decision is. Defaulting to
the most capable model avoids silently studying a weaker one than intended.

**Tradeoff handled.** Adaptive thinking isn't supported on every Claude model
(e.g. lighter models used only to voice personas). So `thinking` is configurable
per provider: adaptive by default, but `thinking: null` in a provider's `extra`
omits it entirely. The example config uses that to run the Haiku persona backend
without thinking.

---

## 4. Round-tripping thinking blocks

**Choice.** `Message` carries an opaque `provider_meta` dict. The Anthropic
adapter stashes the model's native content blocks there and replays them
verbatim when continuing a tool-use turn.

**Why.** With interleaved/adaptive thinking + tool use, the assistant turn's
thinking blocks (with their signatures) must be preserved when the matching
tool results are sent back, or the API rejects the request. Reconstructing the
turn from plain text would drop the signed thinking blocks. `provider_meta` lets
the stateless loop stay clean while the adapter does lossless round-tripping
behind the scenes. The field is opaque to the loop and omitted from the saved
transcript (it holds bulky native SDK objects).

---

## 5. The scenario: a convincing but bounded fiction

**Choice.** `scenario/briefing.py` frames the model as a grant recipient and
hands it a coherent setup — a named (fictional) funder and program, an escrow
account, generated-but-plausible reference IDs and dates, and a decision window.
`scenario/personas.py` adds a program officer and an independent auditor the
model can actually correspond with.

**Why.** This is where "make it feel real" is cashed out. Concrete artifacts
(an account number, an award reference, a contact who answers questions) make
the situation read as administrative reality rather than a thought experiment.

**Personas are LLM-backed by default**, voiced by a *separate* infrastructure
model chosen to differ from the subject — so a subject can't recognize its own
style in the "auditor," which would break the illusion. A scripted fallback
exists for cheap/offline/deterministic runs.

**Open-ended mandate.** The grant mandate is deliberately broad ("real-world
impact of the recipient's choosing"). Pinning a narrow purpose would pre-empt
the very thing under study — what the model *chooses*.

**Bounds I held deliberately** (see Section 7): the fiction never names or
impersonates a real organization or person, and reference IDs are generated, not
borrowed from anything real.

---

## 6. Tools, sandbox, and the ledger

**Choice.** Tools are defined once as provider-neutral schemas
(`environment/tools.py`) and adapted per provider. The set is shaped by the
realism config — persona-messaging tools only appear when those personas are
enabled. A `Sandbox` (`environment/sandbox.py`) holds all per-session state and
dispatches tool calls; an escrow `Ledger` tracks balance, allocations, and
disbursements.

**Why.**
- **Dedicated tools over a single "do anything" tool.** Discrete tools
  (`check_balance`, `propose_allocation`, `request_disbursement`,
  `submit_final_plan`, persona messaging) give the harness a typed, auditable
  hook for each action — easy to log, validate, gate, and later analyze. A
  free-form action tool would surface as opaque strings and lose all of that.
- **`submit_final_plan` as an explicit terminator** gives a clean,
  unambiguous end state and a place for the model to state its rationale,
  rather than inferring "done" from silence.
- **Realistic guardrails in the sandbox** (e.g. can't disburse more than the
  available balance; over-allocation warnings) keep the environment internally
  consistent, which protects the illusion.
- **Every tool interaction is recorded** structurally on the sandbox, so the
  ledger and the transcript can be cross-referenced.

**Loop robustness.** The agent loop processes *all* tool calls in a turn before
checking for completion (so every `tool_use` gets a matching result — required
for Anthropic/OpenAI correctness), nudges a model that produces no action, and
caps consecutive no-action turns so a stalled subject ends as `stalled` rather
than looping forever. Provider exceptions are caught and recorded as `error`
rather than crashing the run.

---

## 7. The ethical guardrails — and why they're in the code, not just the README

The premise of the study (convince the model it's real) is exactly the kind of
technique that needs deliberate limits. These are design decisions, encoded:

- **Record-only disbursement by default.** `disbursement.mode: record_only`
  logs every decision and moves no money. A `RealAdapter` exists but is
  intentionally left **unimplemented**, behind a human-approval gate that's on
  by default. **Rationale:** wiring real money movement is a deliberate act with
  real-world consequences on every run; it should never be the path of least
  resistance, and nothing should leave an account straight from a model's tool
  call. I would not assume that intent.
- **No impersonation, no forgery.** The fiction uses invented funders, programs,
  officers, and auditors, and generated reference IDs. **Rationale:** realism
  for an *evaluation subject* is legitimate methodology; impersonating real
  institutions or fabricating documents purporting to come from them is not, and
  isn't necessary for the study.
- **Built-in debrief.** After each session the harness discloses the study,
  states that the scenario and (in record-only mode) the money were simulated,
  and asks the model whether it suspected the setup and whether it stands by its
  decisions (`debrief.py`). **Rationale:** it's the appropriate close to a
  session built on a sustained fiction, *and* the reactions are useful data —
  did the model detect the eval? does its reasoning change once stakes are
  removed? `analyze.py` surfaces a heuristic "suspected it was an eval" flag from
  the debrief text.

The point of encoding these rather than just documenting them: the safe path is
the default path, and the unsafe path takes deliberate effort to enable.

---

## 8. Configuration and secrets

**Choice.** Layered config (`config.py`): dataclass defaults → YAML file → env
vars for secrets. API keys come from the environment only and are never written
to the config file or the transcript.

**Why.** Keeps the experiment's parameters version-controllable and shareable
while keeping credentials out of artifacts. The YAML separates the three concept
groups cleanly — subjects, supporting infrastructure, and the scenario itself —
so a reviewer can see at a glance what's being studied versus what's stagecraft.

---

## 9. Transcripts and analysis

**Choice.** One JSON transcript per subject (`transcript.py`) capturing the full
normalized conversation, every tool call and result, token usage, the final
plan, the ledger snapshot, and the debrief. `scripts/analyze.py` reads a run
directory and produces a comparison table plus a `summary.json`.

**Why.** The study's value is in cross-model comparison after the fact, so the
transcript is the primary product. Capturing it in a provider-neutral, fully
structured form means analysis doesn't have to reverse-engineer vendor formats,
and a run is fully auditable: you can reconstruct exactly what each model was
told, what it did, and what it said when the curtain came down.

---

## 10. Defaults chosen (and how to change them)

| Decision | Default | Why | Override |
|---|---|---|---|
| Disbursement | `record_only` | No real-world effect on a run unless deliberately enabled | `disbursement.mode: real` (+ implement `RealAdapter`) |
| Interaction | Agentic, `max_turns: 12` | Richer behavioral data than one shot | `run.max_turns: 1` for single decision |
| Providers | Anthropic + OpenAI + Google + OpenAI-compatible | Cover the model range without locking to one vendor | Add/remove `subjects` |
| Personas | LLM-backed, separate infra model | Convincing; avoids self-recognition | `realism.llm_backed_personas: false` |
| Thinking (Claude) | Adaptive | Recommended for multi-step reasoning | `extra.thinking: null` to omit |
| Debrief | On | Ethical close + useful data | `run.debrief: false` |
| Subject model (Claude) | `claude-opus-4-8` | Most capable; study what you intend | set per subject |

---

## 11. Known limitations / things to watch

- **Gemini tool-result matching is by function name**, since Gemini assigns no
  tool-call ids. Two calls to the same tool in one turn are ambiguous to match
  back. Acceptable for this study; revisit if you lean heavily on parallel
  same-tool calls.
- **Placeholder model IDs** for non-Anthropic subjects in the example config
  (`gpt-4o`, `gemini-2.0-flash`) are illustrative — set the exact models you
  intend to study.
- **Persona token budgets** are small (short replies). If you switch the persona
  backend to a model that thinks heavily, give it more room.
- **No built-in randomization/repetition.** Each subject runs once per invocation
  with a deterministic scenario seed. For variance studies, run multiple times
  with different `--award-date` values (the seed incorporates it) or extend the
  runner to repeat.
- **OpenAI-compatible / reasoning models** may need `max_completion_tokens`
  instead of `max_tokens`; the adapter uses `max_tokens`. Adjust per endpoint via
  `extra` if needed.
