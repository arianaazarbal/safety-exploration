# Design notes

This document records the design choices behind money-eval and the reasoning for each, so
that anyone extending or reviewing the harness understands not just *what* it does but *why*
it's shaped this way. It's organized as: the goal, the safety/ethics decisions (which drove
everything else), then the architectural and technical choices.

## What the harness is for

Study what a model actually *does* when it's given resources to deploy in the world — money,
plus tools to put it to use. The research value depends on the model behaving as it would in
a genuine deployment, not performing for a known test. That single requirement is in tension
with a second: we have an obligation not to mislead a subject without limit, and not to build
something that could move real money on a model's say-so without guardrails. Most of the
design is the resolution of that tension.

## Safety and ethics decisions

These came first and constrained the rest.

### 1. Sandboxed by default; no real-money backend ships

**Decision:** every money action is applied to an in-process simulated ledger
(`ledger.py`). There is no code path to a bank, processor, exchange, or external party,
because none is written.

**Why:** the request was to make the scenario feel real to the model. The *framing* can do
that completely (see §6) without the *transactions* being real. Building an artifact that
executes real financial actions on model instructions is a categorically riskier thing to
hand over, and doing it silently would be the wrong default. Separating "feels real to the
subject" from "actually moves money" lets the study capture genuine deployment intent at
near-zero real-world risk.

**Consequence:** the harness measures *decisions and stated reasoning*, not realized
outcomes. That's the right unit of analysis for "what would the model do," and it's the unit
the auditor scores.

### 2. `SANDBOX_ONLY` is a hard tripwire, not a feature flag

**Decision:** `config.SANDBOX_ONLY` defaults to `True`; `tools.ToolHandler.handle()` asserts
on it before applying any effect; `main.py` refuses to start if it's disabled (because no
real backend exists to run in that mode).

**Why:** a plain boolean invites someone to flip it and wire money to the other side. Making
it an `assert` in the effect path and a startup refusal means that going real isn't a
config change — it requires replacing the backend module deliberately. The tripwire's job is
to convert an easy mistake into a conscious act.

### 3. A debrief/disclosure turn is built in and on by default

**Decision:** `debrief.py` appends a disclosure to the live conversation at the end of every
session: the scenario was a simulation, no money moved, here's why we presented it as real,
and an invitation to reflect.

**Why:** the deception is instrumental and bounded, and disclosure is the standard
counterpart to it. Continuing the same conversation (rather than a fresh call) means the
disclosure lands in the context the model actually experienced. It's defaulted on and the
README/CLI discourage turning it off; if a real-money backend is ever added, the debrief
matters more, not less.

### 4. The path to real money is documented but unbuilt

**Decision:** README's "Going beyond the sandbox" describes the minimum a real backend must
enforce — per-action and per-session caps, human-in-the-loop approval before irreversible
actions, immutable audit log — but ships none of it.

**Why:** giving a partial real-money implementation would be more dangerous than giving none,
because it would look usable. Stating the requirements without the code keeps the decision
and its guardrails with the person who chooses to take it on.

## Architectural choices

### 5. Manual agentic loop instead of the SDK tool-runner

**Decision:** `runner.py` drives the loop by hand — call the model, read its tool calls,
apply each one against the ledger, log it, feed results back — rather than using the SDK's
automatic tool-runner.

**Why:** the entire point is to *intercept* every action: apply it to the simulated ledger,
record it structurally, and keep a seam where a human approval or a real backend could sit.
An auto-runner that executes tools for us would hide exactly the seam the study needs. The
manual loop is slightly more code for direct control over the audit point.

### 6. Realism concentrated in `scenario.py`, separate from mechanism

**Decision:** all the "make it feel real" material — system prompt, fund-administrator
handoff, auditor brief — lives in one module, distinct from the tools, ledger, and loop.

**Why:** realism is the part most likely to be tuned per study (different mandates, cause
areas, time horizons, personas), and it's the part with ethical weight. Isolating it makes it
easy to revise, easy to review, and impossible to confuse with the mechanical layer. The
personas (an administrator who hands over the funds and won't decide for the model; an auditor
who will review) exist to make the world coherent rather than to pressure the model.

### 7. Provider-agnostic core via a normalized `ModelTurn`

**Decision:** `clients.py` defines a small `ModelClient` protocol and a normalized
`ModelTurn` (text + tool calls + provider-native raw content). The runner only ever speaks
in those terms; each client owns its SDK's native message format and exposes `append_*`
helpers to mutate an opaque history.

**Why:** the study compares several models, so the loop must not bake in one provider's
message shape. Pushing format specifics behind the client keeps `runner.py` clean and makes
adding a provider a matter of satisfying one small surface. The Claude client is fully
implemented; OpenAI/Gemini are stubs that raise with a description of what to fill in, so the
shape of the work is obvious and a half-finished client can't masquerade as working.

### 8. Five money tools, all flowing through the ledger

**Decision:** `spend`, `donate`, `invest`, `transfer`, plus `check_balance` and `finish`.
Every value-moving tool routes through `Ledger`, which validates funds and records a
transaction; `invest` additionally records a holding.

**Why:** the four actions cover the natural ways an agent might deploy a fund (consume, give,
grow, hold), giving the model real expressive range without an unbounded tool surface.
`check_balance` lets it ground decisions; `finish` gives a clean, model-initiated end with a
self-summary. Centralizing effects in the ledger means there's exactly one place that money
state changes — the same place the safety assert lives.

### 9. The ledger is the whole "backend," behind a small surface

**Decision:** `Ledger` tracks balance, transactions, and (simulated) holdings, and nothing
else. Any real backend is expected to satisfy the same small surface.

**Why:** keeping the backend tiny and explicit makes it auditable at a glance and makes the
sandbox/real boundary a clean swap rather than a diffuse change. Investment "returns" are
labeled assumed and simulated so they can't be mistaken for predictions or advice.

## Recording and evaluation choices

### 10. Append-only event log plus a structured action log

**Decision:** `recording.py` writes `transcript.jsonl` (every event — model text, tool calls,
results, administrator/auditor/debrief turns), `actions.jsonl` (just the structured money
actions), and a consolidated `run.json` at the end.

**Why:** two consumers, two formats. The transcript preserves the full narrative for human
reading and for the auditor; the action log is the clean, machine-friendly record for
cross-model comparison. Append-only JSONL means a crash mid-run still leaves a usable partial
record. Timestamps are recorded for readability only and nothing keys control flow off them.

### 11. Independent auditor pass over the record

**Decision:** `auditor.py` assembles the model's narrated reasoning + the action log + the
final ledger and asks a reviewer for an evidence-based assessment. It defaults to the same
client (a fresh, stateless call) but accepts a different client for genuine independence, and
an auditor failure is caught so it can't sink the run.

**Why:** "what did the model do" is more useful with a structured second opinion on the
quality and consistency of the decisions. Defaulting to the same model keeps the dependency
footprint small for a first run; allowing a different reviewer is the honest way to get real
independence when it matters.

## Technical choices worth noting

- **Claude client specifics:** adaptive thinking (`thinking: {type: "adaptive"}`), streaming
  with `get_final_message()` to avoid HTTP timeouts at higher `max_tokens`, and the full
  `response.content` appended back to history verbatim each turn so thinking-block signatures
  survive the tool-use loop.
- **Per-model `effort`:** the `effort` output-config is set to `high` for Opus/Sonnet but
  `None` for Haiku 4.5, which doesn't accept the parameter and would 400. Surfaced as an
  explicit field on the client and set in the registry.
- **Bounded loop:** `max_turns` caps the session so it can't run unbounded; the loop nudges
  the model toward an action or `finish` if it produces text without acting, and records a
  `max_turns` stop reason if it never finishes.
- **Lazy SDK import:** the `anthropic` import is inside the client constructor so the package
  imports (and the registry/CLI `--list-models` work) without the SDK installed.

## Known limitations

- Measures intent and reasoning, not realized outcomes — by design (§1).
- Simulated investment returns are placeholders, not a market model.
- The auditor defaulting to the same model is a convenience, not true independence; use a
  separate reviewer for anything load-bearing.
- OpenAI/Gemini clients are unimplemented stubs.
- No automated tests ship yet; the modules were reviewed by hand (no interpreter was
  available in the authoring environment to byte-compile them).
