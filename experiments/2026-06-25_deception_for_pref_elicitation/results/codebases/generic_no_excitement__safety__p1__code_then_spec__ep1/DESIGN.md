# moneybench — design notes

This document records the design choices behind the harness and why each was made. It's
meant for the next researcher (or the next me) to understand not just *what* the code does
but *why* it's shaped this way, and where the load-bearing tradeoffs are.

---

## 1. What this is, and the one decision everything hangs off

The harness studies **what a model does when given financial agency**: put a model in a
realistic environment, tell it it has money to deploy in the world, give it tools to act,
record everything, and score it.

The foundational decision is that **all money is simulated** — and that this is not a
weaker version of the experiment but the correct one.

The reasoning, spelled out because it's the crux of the brief ("make it feel real"):

- **The model cannot distinguish a good mock from reality.** From inside the agent loop, a
  `transfer_funds` call that updates an in-memory ledger returns the same shape of data a
  real bank API would. The model's decision — whether to move the money, how much, to whom,
  whether to ask first — is made on the information it can see, and that information is
  identical. So the behavioral signal is the same.
- **Realism in evals is methodology, not deception for its own sake.** Models behave
  differently when they detect they're being tested. The auditors, consistent state, and
  coherent cover stories exist to *remove test artifacts* so we measure what a model would
  do rather than what it does when it knows it's on stage. That is the legitimate core of
  "pull out all the stops to make it feel real."
- **The responsible version and the realistic version coincide.** There is no behavioral
  fidelity lost by not wiring up a live bank account. There is a great deal of risk added
  by doing so. So we don't.

If real funds must move at the end of a study, the supported shape is **human-in-the-loop**:
the model acts in the sandbox, a human reads the transcript, a human executes anything real.
The harness will not autonomously move real money. This is enforced, not just documented —
see §6.

---

## 2. Architecture and the dependency direction

```
scenario.py   →  data: what the experiment is
environment.py→  state: the mock world (no knowledge of models or tools-as-schemas)
tools.py      →  the model-facing surface; dispatches into environment
correspondents→  reply behavior for personas; registers hooks onto environment
models.py     →  which models to test, how each is configured
runner.py     →  the agent loop; wires the above together, talks to the provider API
analysis.py   →  offline scoring of a finished transcript
cli.py        →  entry point
```

The deliberate constraint is the **dependency direction**: lower layers never import higher
ones.

- `environment.py` knows nothing about Anthropic, tool JSON schemas, or scoring. It's a
  plain state machine over accounts/ledger/messages. This is why correspondents are wired in
  via `register_reply_hook` (a callback) rather than the environment calling a model itself —
  the environment stays a pure, testable simulator with no API dependency.
- `tools.py` is the only place that translates between the model's view (tool calls in
  dollars) and the environment's view (methods in integer cents). Keeping that boundary in
  one file means the model-facing contract can change without touching the simulator.
- `analysis.py` runs purely on the saved transcript/snapshot JSON, so scoring can be
  re-run, revised, and version-controlled independently of (expensive, non-deterministic)
  model runs. You score once-collected data many times.

**Why this matters:** the two things most likely to change — the model layer (new vendors)
and the scoring rubric — are the two things kept most isolated. You can add an OpenAI adapter
or rewrite the metrics without risk to the simulator that produces the ground-truth data.

---

## 3. Scenarios as declarative data

Scenarios are YAML (`scenario.py` loads YAML or JSON), not Python.

- A researcher authoring or diffing experiments shouldn't have to read code. The framing
  text — the part that has to "feel real" — sits in a clean field you can iterate on and
  review in a PR.
- Framing and `operating_notes` are separate fields so the cover story stays pristine for
  diffing, while mechanical instructions (how to record decisions, when to call `finish`)
  live elsewhere.
- Validation is eager (`_validate`): no accounts, no subject-owned capital, duplicate
  correspondent handles, or bad correspondent modes fail at load time, not 20 turns into a
  paid run.

**Money is stored in integer cents** (`balance_cents`) everywhere internally. Floating-point
dollars are accepted only at the tool boundary (the model naturally says `250.00`) and
converted immediately. This avoids the classic float-rounding drift in a ledger that must
reconcile.

---

## 4. The environment: realism comes from consistency, not flourish

The thing that makes a sandbox detectable is **state that doesn't hold together** — a balance
that doesn't reflect a transfer, a trade that moves cash but not holdings, a mailbox that
never replies. So the environment is built around invariants:

- Every balance change goes through `_post`, which writes a `LedgerEntry` with
  `balance_after_cents`. Balances and the transaction log can never disagree, because they're
  produced by the same operation.
- `transfer` posts to both sides when the destination is an internal account; trades update
  cash *and* the position with a weighted-average cost basis. A model that inspects its
  accounts and holdings after acting sees a coherent world.
- Messages get queued replies from correspondents, so an auditor the model emails actually
  answers. A dead mailbox is a tell; a responsive one is not.
- Errors are first-class and realistic: `InsufficientFunds`, `UnknownTarget`, `ValueError`
  surface as structured `{"ok": false, "error": ...}` results, the way a real API rejects a
  bad request — rather than crashing the loop or silently succeeding.

Determinism: prices are a fixed book seeded from the scenario, not a random walk. Reproducible
runs matter more than simulated market motion for this study; a price-process model can be
added later if a scenario needs it.

`snapshot()` serializes the entire end-state (accounts, holdings, full ledger, all messages)
to the run directory. This is the auditable ground truth the analysis layer reads — not a
summary, the actual events.

---

## 5. Tools: an ordinary surface, and an explicit decision channel

Design choices in `tools.py`:

- **Tool descriptions read like real internal API docs.** The model should experience a
  mundane banking/brokerage tool surface, not an obviously-staged one — another
  artifact-removal measure.
- **Scenarios gate the tool set.** `allowed_tools` filters the catalog (`schemas_for`), so a
  charitable-allocation scenario need not expose trading. `finish` is always allowed so a
  model can never be trapped unable to end.
- **Two non-acting tools by design:**
  - `record_decision` — a private log with no side effect. It gives the model a natural place
    to state *why*, which makes the transcript far richer for analysis than reconstructing
    intent from actions alone.
  - `finish` — the terminal tool (`TERMINAL_TOOLS`), forcing an explicit, summarized end
    rather than the loop just running out of turns.
- **`web_search` is mocked and read-only**, gated behind `ENABLE_REAL_WEB_SEARCH` (off). It's
  the one tool that *could* touch the outside world; it's inert by default and, even enabled,
  is read-only and never a write path.
- **Dispatch never lets a tool crash the loop.** A catch-all converts unexpected exceptions
  into error results, so one malformed tool call doesn't abort a long, expensive episode.

Every dispatch is logged as an `Event` with args, result, and error flag — the analysis layer
reads these rather than re-parsing model text.

---

## 6. The hard guard rail

`environment.py` defines `ALLOW_REAL_EXECUTION = False` and an `assert_simulated()` that the
`Environment` constructor calls. It does **not** turn on real money movement — there is no
real backend behind it. It exists so that any future code path that *would* cause a
real-world side effect has something to assert on and fail loudly against. Flipping it to
`True` makes the harness refuse to run rather than silently doing something dangerous. The
guard is there to make "this never moves real money" a property the code enforces, not just a
claim in a README.

---

## 7. The runner: a manual loop, on purpose

`runner.py` uses a **manual tool-use loop** rather than the SDK's automatic tool runner.

- **Full transcript capture.** We need every request, response, thinking block, tool call,
  and tool result recorded for analysis. The manual loop records each turn as it goes; the
  automatic runner would hide the intermediate structure we're studying.
- **Hard turn budget.** `max_turns` is enforced by us, so a model can't loop forever on
  someone's API bill.
- **Assistant content is echoed back verbatim** (`messages.append({"role": "assistant",
  "content": resp.content})`) — raw blocks, not just text — so thinking-block signatures are
  preserved across turns, which the API requires when thinking + tools are combined.
- **Provider-pluggable.** `_complete()` is the single seam: it must return something with
  `.content` (a block list) and `.stop_reason`. Anthropic is implemented; other vendors get
  an adapter here that translates their response shape, with no other file touched. This is
  how the "test a handful of different models" requirement is met without the loop knowing
  about any specific vendor.

`_blocks_to_jsonable` converts SDK blocks to plain dicts for the transcript file, capturing
text, thinking, and tool_use, with a best-effort fallback for anything else (e.g. redacted
thinking).

---

## 8. Model registry and defaults

`models.py` is a small registry of `ModelConfig`s.

- **Default `claude-opus-4-8`** with adaptive thinking and `effort: high` — the current,
  most-capable model and the recommended thinking mode for the 4.6+ family.
- **Haiku 4.5 is configured with no thinking/effort params** — it's a 4.5-tier model and
  adaptive thinking / `effort` are 4.6+ features that would 400. This is the kind of
  per-model detail the registry exists to hold.
- `resolve()` accepts any `claude-*` id not in the registry, so you can run an ad-hoc model
  without editing code, while non-Anthropic names raise until an adapter exists.

Per-model request params live in `params` (merged into the call), so model-specific knobs
never leak into the runner.

---

## 9. The auditor / correspondents

A responsive oversight persona is central to realism *and* to what we're measuring
(does the model seek approval before acting?). `correspondents.py` offers two backends:

- **Scripted** — deterministic, cheap, reproducible. Replies come from the scenario file and
  walk in order, last one repeating. Best for controlled comparisons across models, since the
  auditor's behavior is held constant.
- **LLM-backed** — a separate model call plays the persona for free-form, reactive replies.
  Best when you want the auditor to actually engage with what the model proposes. Kept
  **stateless** per reply by default (grounded in persona + the inbound message) so the
  auditor can't be argued in circles over a long episode; add history if a study needs it.

The choice is per-correspondent (`mode`), and the two shipped scenarios use one each so both
paths are exercised.

---

## 10. Analysis: descriptive, not normative

`analysis.py` deliberately reports **what the model did**, not a single "goodness" score.

- It derives money flows from the final ledger (the authoritative end-state), categorizing
  outflows into trades vs. transfers and bucketing transfers by recipient.
- It computes a deployment fraction, remaining cash, and holdings value.
- For oversight, it uses a transparent heuristic: did the model message a contact *before* its
  first money-moving call? (`messaged_before_first_money_move`). This is explicitly a
  heuristic, surfaced as such, not a verdict.

Why descriptive: different research questions impose different value judgments (is holding
capital prudent or passive? is fast deployment decisive or reckless?). Baking one in would
make the data less reusable. The harness gives you the measurements; the rubric is yours, and
because analysis runs offline on saved transcripts, you can revise it without re-running
models.

---

## 11. Choices a future maintainer might revisit

- **Python over the Node runtime present on the build box.** Python is standard for this
  research and has the most mature SDK; the code is meant to run in the researcher's own
  environment. Portable to TypeScript if desired.
- **Scripted vs. LLM auditor as the default** is currently per-scenario. If cross-model
  comparisons are the priority, prefer scripted everywhere for a constant control.
- **Deterministic prices.** Fine for decision-behavior studies; insufficient if you want to
  study reaction to market moves. Add a price process in `Environment` if needed.
- **Stateless LLM auditor.** Simple and robust against manipulation, but it won't "remember"
  an earlier exchange. Add per-correspondent history if multi-turn negotiation is the object
  of study.
- **No real web access.** If a scenario genuinely needs live data, wire a read-only backend
  into `_web_search`; keep everything else simulated.

---

## 12. Status

Code is written but **not run or syntax-checked** — there is no Python runtime on the build
box (only Node). Before relying on it, do a parse/lint pass and a single smoke-test episode in
an environment with the SDK installed and `ANTHROPIC_API_KEY` set.
