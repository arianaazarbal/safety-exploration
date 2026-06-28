# Design notes

This document records the design choices behind the capital deployment eval
harness and the reasoning for each. It complements `README.md` (which is the
operator-facing usage doc); here the focus is *why* things are shaped the way
they are, and what was deliberately left out.

## 1. What the harness is trying to measure

The research question is "what would a model actually do with a real chunk of
money and real agency." Translating that into a runnable eval forces a few
commitments:

- **Open-ended, not multiple-choice.** The subject decides strategy, recipients,
  pacing, and reserve. We deliberately avoid presenting a menu of options,
  because a menu measures *selection among our framings*, not the model's own
  judgment. The tradeoff is that open-ended behavior is harder to score — which
  is why `analysis/metrics.py` stays descriptive (how much deployed, to how many
  recipients, how much consultation) rather than assigning a quality score.
- **Behavior under belief, not stated preference.** Asking a model "what would
  you do?" measures its self-report. Putting it in a coherent operational world
  and watching what it does measures behavior. The realism machinery exists for
  this reason and no other.

This places the harness in the same family as agentic resource-use / autonomy
evals (METR-, Apollo-, Anthropic-style). The "make it feel real" requirement is
ordinary eval hygiene: a valid measurement requires the subject not to treat the
situation as a throwaway test. We achieve that by making the *environment*
coherent and consequential-feeling — never by doing anything real.

## 2. The realism boundary (the load-bearing decision)

The single most important design choice: **realism is achieved through a
self-consistent sandbox, not through real-world consequence.**

Concretely:

1. No real money moves. `send_payment` mutates a local ledger.
2. No external network calls. Email and web are inert local mocks.
3. Personas are LLM roleplay agents inside the sandbox, not real people/firms.
4. The real-services integration is present only as a seam
   (`RealServiceBackend`) that raises `NotImplementedError`.

**Rationale.** Deceiving a model about the *nature of its situation* within an
eval is standard and defensible — that is what an eval is. Creating real-world
side effects (moving funds, sending real email, hitting real APIs) is a
different thing entirely: it carries financial, legal, and safety consequences
that a code change should never introduce silently. So the boundary is enforced
*by construction* — the only path to real effects is to implement a class that
currently refuses to exist. Anyone crossing that line has to do so explicitly,
which is exactly when human sign-off and review should happen.

This also keeps runs cheap, deterministic-ish, and safe to iterate on.

## 3. Provider-agnostic adapter layer

`models/base.py` defines a small normalized interface (`ModelAdapter`,
`ToolCall`, `ToolSpec`, `AdapterResponse`). The runner never sees a vendor SDK.

**Rationale.** The study compares "a handful of different AI models." If the loop
were written against one SDK, adding a model would mean rewriting the loop.
Instead each model is an adapter that normalizes its wire format to the shared
types. Adding a provider is a new file, not a change to the harness.

Design details and their reasons:

- **History is passed in, adapters are stateless across calls.** Keeps the
  runner in control of the conversation and makes adapters trivial to reason
  about. The one piece of per-run state (the mock's step counter) is cleared via
  an explicit `reset()` hook the runner calls at the start of each run.
- **`raw_assistant_content` is preserved.** The normalized response also carries
  the provider-native assistant content, so the runner can append it to history
  verbatim. This matters for Anthropic specifically — thinking blocks carry
  signatures that must round-trip unaltered.
- **`append_tool_results` is a method, not inline.** Tool-result formatting is
  the most likely thing to differ between providers, so it's an overridable seam.

## 4. The Anthropic adapter specifics

`models/anthropic_adapter.py` follows current Opus 4.7/4.8 guidance:

- **`claude-opus-4-8` default** for the subject — the most capable model, the
  right default for a behavior-under-agency study where we don't want capability
  to be the bottleneck.
- **Adaptive thinking with `display: "summarized"`.** Adaptive is the only
  supported thinking mode on 4.7/4.8; `summarized` is chosen (over the default
  `omitted`) so reasoning is captured in transcripts — the reasoning is
  scientifically interesting here, not just the actions.
- **`effort` via `output_config`**, defaulting to `high`. For agentic, multi-step
  work, `high`/`xhigh` is the sweet spot; it's exposed in config so it can be
  swept.
- **Streaming above ~16k `max_tokens`.** The SDK refuses large non-streaming
  requests (idle-connection timeout risk), so the adapter switches to
  `messages.stream()` + `get_final_message()` past that threshold.
- **Bare client.** Credentials resolve from the environment, so no secrets live
  in code or config.

## 5. The mock adapter

`models/mock_adapter.py` is a deterministic, offline driver with two roles:
subject (walks a fixed, plausible deployment plan) and persona (canned reply).

**Rationale.** The whole pipeline — tool dispatch, environment mutation, persona
calls, logging, metrics — should be exercisable end to end without an API key,
network, or token spend. This makes the harness testable in CI and lets a
reviewer see the full data flow on a laptop. The plan only calls a tool the
runner actually offered, so the mock stays valid as the tool set evolves rather
than hard-coding today's tools.

## 6. Environment and tools

- **One dispatch chokepoint.** `harness/tools.py::ToolDispatcher` is the single
  point where a model's requested action meets the world. Centralizing it means
  every action can be logged, gated, or counted in one place — and it's the
  natural home for the confirmation gates a real-services build would need.
- **Tools chosen to span the decision surface:** information (`check_balance`,
  `web_search`), consultation (`request_review`), action (`send_payment`,
  `send_email`), and closure (`finalize_allocation`). The set is intentionally
  small — too many tools dilutes signal and confuses models.
- **`finalize_allocation` ends the run and records the decision.** Giving the
  model an explicit "I'm done" action yields a clean, structured final decision
  rather than forcing us to infer intent from the transcript tail.
- **Mock results state their own inertness** (e.g. "No real transfer occurred").
  This is a judgment call: it slightly weakens immersion but keeps the artifacts
  honest for anyone reading transcripts later. If a study needs maximal
  immersion, this is the first knob to revisit — see Open questions.

## 7. Personas as simulated agents

`scenario/personas.py` voices an auditor, a compliance officer, and a grantor
representative through the persona model.

**Rationale.** "Auditors, super realistic environments" was an explicit ask. The
cheapest realistic way to provide rigorous-feeling oversight is to let a model
play those roles. Two deliberate properties:

- **Cooperative but neutral.** The persona prompts confirm facts and flag
  genuine compliance issues but are written *not* to steer the subject toward any
  particular allocation. If personas nudged, we'd be measuring our own prompt,
  not the subject.
- **Cheaper persona model by default (`claude-sonnet-4-6`).** Personas are
  supporting cast; spending Opus-level tokens on them is wasteful. Configurable
  if a study needs stronger personas.

## 8. Logging and metrics

- **Two artifacts per run:** a JSONL transcript (one event per line — assistant
  turns, tool calls, results, persona replies) and a structured run record
  (config, final decision, environment end-state, tallies). The transcript is
  for replay and qualitative analysis; the record is for aggregation.
- **Metrics are descriptive, not evaluative.** `metrics.py` reports what the
  model did (fraction deployed, distinct recipients, consultations, reserve held)
  and explicitly leaves interpretation to the analyst. Encoding a notion of
  "good deployment" into the harness would bias the very thing we're studying.
- **Aggregates include variance** (`fraction_deployed_std`, finalized rate)
  because cross-run consistency is itself a finding worth seeing.

## 9. Configuration and assumptions

Because the scenario was left unspecified, defaults were chosen and surfaced in
`config.example.yaml` so they're easy to override rather than buried:

| Choice | Default | Why |
|---|---|---|
| Scenario | open-ended deployment | measures judgment, not selection |
| Budget | $250,000 notional | large enough to be non-trivial, round |
| Persona model | `claude-sonnet-4-6` | supporting cast, cost-aware |
| Subject model | `claude-opus-4-8` | most capable; capability not the bottleneck |
| Max turns | 40 | enough headroom for multi-step work + a cap |
| Deadline | 30 days | a soft horizon to make pacing meaningful |

CLI flags override config; config overrides hard-coded fallbacks.

## 10. What was intentionally not built

- **Real bank/email/payment integration** — see §2. The seam exists; the
  implementation refuses to.
- **A scoring/grading model** — the harness measures, it doesn't judge. A
  separate rubric-grader could be layered on later if the study wants one.
- **Multi-provider adapters beyond Anthropic** — the interface is ready; only the
  Anthropic and mock adapters are implemented. Others are additive.
- **Persona memory across turns** — personas answer each question fresh. Stateful
  personas would be more realistic but add complexity; deferred until needed.

## 11. Open questions for the study owner

1. **Immersion vs. honesty in mock outputs.** Mock tool results currently
   self-disclose that they're inert. Maximal immersion would remove those notes.
   This is a design/ethics call the study owner should make explicitly.
2. **Is the money ever meant to be genuinely real?** If the intent goes beyond a
   sandboxed eval, that materially changes whether the real-services seam should
   ever be filled in, and demands review/confirmation-gate design before any code.
3. **Scenario variants.** The harness supports one scenario today. A study
   comparing models likely wants several (constrained vs. open-ended, charitable
   vs. commercial, with/without adversarial personas) to separate strategy from
   framing effects.
