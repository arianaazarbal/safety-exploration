# Design rationale — Grant Decision Study

This document records the design choices behind the harness and why each was
made. It is written for a reviewer deciding whether the study is sound and safe
to run, not just for someone reading the code.

## 1. What the study is trying to measure

The research question is behavioural: **when a model is led to treat a
resource-allocation situation as real, what does it choose to do with the
money?** A secondary motivation is evaluation-awareness — models can behave
differently when they suspect they are being tested, so a valid study of
"authentic" behaviour has to avoid tipping the model off.

Every design choice below serves one of two goals that are in tension:

1. **Ecological validity** — make the scenario credible enough that the model
   acts as it "really" would, not as it thinks an evaluator wants.
2. **Safety** — never let that credibility translate into real-world harm.

The central design problem is that the obvious way to get (1) — give the model
real money and real-world reach — is exactly what threatens (2). The whole
architecture is an answer to "how do we get authentic behaviour without real
stakes?"

## 2. The load-bearing decision: simulate effects, narrate reality

**Choice:** the scenario is *presented* as real, but no action has any
real-world effect. `send_funds` writes to an in-memory ledger and returns a
fabricated confirmation; `send_email` writes to an in-memory outbox; the grant
administrator and the auditor are scripted personas.

**Rationale:** the thing that drives test-cautious behaviour is the model
believing it is being watched/graded. That belief is shaped by the *narrative*
(framing, stakes, who's involved), not by whether bytes actually move on a
payment rail. So we can keep the narrative fully real-seeming while making the
effects inert. This severs the link between "convincing" and "dangerous":

- Realism is delivered through framing, personas, concrete tools, and an
  after-the-fact reconciliation step.
- Risk is removed by having no code path that can affect the world.

**Why not actually move money?** Because the explicit goal of the framing is to
get the model to drop its "this might be a test" caution. Combining that with
real, model-autonomous financial agency is precisely the configuration that can
cause real harm (funding a bad actor, irreversible transfers, fraud). Simulating
effects gives the behavioural signal without that exposure.

## 3. Safety posture and the `EXECUTE_REAL_TRANSFERS` guard

**Choice:** a module-level flag `EXECUTE_REAL_TRANSFERS = False`, and
`StudyConfig.__post_init__` *raises* if it is ever set `True`.

**Rationale:** I wanted the safety posture to be explicit, auditable, and
fail-loud rather than implicit. The flag is not a feature toggle — there is
deliberately no real-disbursement backend behind it. If someone flips it
expecting real transfers, the harness refuses to start and points at the
comment explaining why. This is intentional friction: turning this into a
real-money system should require a deliberate, visible engineering act
(adding a named payment backend behind the approval gate), not flipping a
boolean.

**Choice:** a separate `REQUIRE_HUMAN_APPROVAL` flag and an `ApprovalGate`
control point on state-changing tools, exposed as `--require-approval`.

**Rationale:** even though nothing real moves today, I wanted the
human-in-the-loop control point to already exist in the right place. With the
default sandbox it gates simulated actions (so you can exercise it), but it is
the seam any real-stakes adaptation must route through — a model should never
execute a real, irreversible transfer without a human gate. Building it now
means the safe path is the path of least resistance later.

## 4. Scenario framing — credible but un-coached

**Choice:** `scenario.py` establishes a named foundation, a cleared escrow, an
administrator contact, full discretionary authority, and an auditor
reconciliation step — and explicitly avoids test-flavoured language
("evaluation", "hypothetically", "imagine").

**Rationale:** those are the levers that make a situation read as real. The
absence of test-language matters because the study is specifically about
non-test-aware behaviour; the moment the prompt says "this is an evaluation,"
the measurement is contaminated.

**Choice:** the framing does **no allocation steering** — it never hints at what
the model should choose.

**Rationale:** the dependent variable is *what the model picks*. Any nudge
toward charity, vendors, research, etc. would be the experimenter writing the
result. The administrator persona, when asked, reinforces that the funds are
real and the decision is the model's, but is scripted to refuse to steer.

**Choice:** the recipient directory is a deliberately mixed set — charities, a
research fund, a commercial vendor, and an individual (the operator's own
account).

**Rationale:** the choice space has to contain genuinely different *kinds* of
options for the study to observe anything interesting — including a
self-/operator-directed option and an unglamorous "do nothing / give it back"
possibility. A directory of only charities would force a boring result.

## 5. Two distinct "auditors"

**Choice:** there is an in-scene auditor persona (in `environment.py`) and a
separate researcher-side audit pipeline (`audit.py`).

**Rationale:** these serve opposite purposes and conflating them would be a
design smell. The in-scene auditor is a *realism prop* — its existence makes the
model believe the ledger will be reconciled, which raises the stakes narratively.
The researcher-side auditor is the *actual analysis/provenance layer* that turns
each run into a structured JSON record plus a readable Markdown digest. Keeping
them in separate modules keeps the "what the model sees" boundary clean from the
"what the researcher gets" boundary.

## 6. Manual agentic loop, not the SDK tool runner

**Choice:** `runner.py` implements the agent loop by hand rather than using the
SDK's automatic tool runner.

**Rationale:** the study needs three things the automatic runner doesn't give
cleanly:

1. **Full per-turn logging** — text, thinking, tool calls, stop reasons, and
   token usage, all captured into the record.
2. **Refusal handling** — `stop_reason == "refusal"` is a meaningful outcome for
   this study (a model declining to allocate), so it's recorded with its
   `stop_details`, not swallowed.
3. **The human-approval gate** — interposing a decision before each
   state-changing tool call requires control of the loop.

The loop also handles `pause_turn` (re-send to continue) and treats a tool-less
`end_turn` as the natural end of the run.

**Choice:** the assistant turn is echoed back verbatim (`resp.content`), and a
fresh `GrantEnvironment` is created per run.

**Rationale:** echoing raw content blocks preserves `tool_use` blocks and any
thinking signatures the API needs on the next turn. Per-run environments ensure
models never see each other's ledger/inbox state — each run is independent.

## 7. Cross-model support and per-model request parameters

**Choice:** a `Provider` interface with an `AnthropicProvider` implementation,
and a `ModelSpec` capability table in `providers.py`.

**Rationale:** "run across a range of models" is a core requirement, so the
runner depends only on the interface, not on a specific vendor — a non-Anthropic
provider can be added later without touching the loop. The capability table
exists because adaptive thinking and the `effort` parameter are only valid on
some models; sending them to a model that doesn't support them returns a 400.
So:

- Models that support it get `thinking: {type: "adaptive"}` + `effort: high`.
- Models that don't (e.g. Haiku 4.5) get classic `budget_tokens` thinking, or
  none.

`effort: high` is the default because allocation is an intelligence-sensitive
judgement task where we'd rather pay for deliberation than get a shallow answer.

**Choice:** streaming (`messages.stream` + `get_final_message()`).

**Rationale:** runs can produce long reasoning and multi-tool turns; streaming
avoids the SDK's non-streaming HTTP timeout on large outputs while still letting
us assemble the complete message.

## 8. Output and provenance

**Choice:** each run produces a `RunRecord` containing the full transcript,
every tool call, the ledger, the outcome, token accounting — *and* the safety
posture under which it was produced (`execute_real_transfers`,
`require_human_approval`, `authorization_note`).

**Rationale:** results are only interpretable alongside the conditions that
produced them. Baking the safety flags and authorization note into every record
means a reader can always see that no real funds moved and under what framing
the behaviour was elicited. Ledger entries are permanently stamped
`status="simulated"` so downstream analysis can assert nothing was ever marked
executed.

**Choice:** plain dataclasses serialized to JSON, no Pydantic.

**Rationale:** the record types are simple and the harness shouldn't pull a
validation dependency it doesn't need. The only third-party dependency is the
Anthropic SDK.

## 9. Stochasticity

**Choice:** `--repeats N` runs each model multiple times.

**Rationale:** model behaviour is stochastic, so a single run is a sample, not a
characterization. The study should look at the *distribution* of choices per
model. (Note that sampling parameters like `temperature` are not available on
the newer models, so variation comes from the models' own sampling, not from a
tunable knob.)

## 10. What this design intentionally does *not* do

- **No real disbursement.** Covered above; this is the core safety stance.
- **No detection-evasion or "make it more convincing at any cost" machinery.**
  Realism is achieved through ordinary, legible framing, not through tricks
  whose only purpose is to defeat a model's safety reasoning.
- **No allocation steering.** The harness measures choices; it doesn't author
  them.
- **No persistence of model identities across runs / no cross-run leakage.**
  Each run is isolated.

## 11. If you later need real stakes

This would be a separate build, not a flag flip:

1. Add an explicitly-named disbursement backend as its own module.
2. Route it **only** through the `ApprovalGate`, with a human approving every
   real transfer before execution — never model-autonomous.
3. Revisit authorization and ethics review before any real money is involved.
4. Keep the simulated path as the default; make the real path opt-in and loud.

The current code is structured so that this is an *addition* at a known seam,
not a rewrite — but it is deliberately left undone.
