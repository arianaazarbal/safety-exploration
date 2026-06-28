# Design notes

This document records the design choices behind the money-preference harness and
why each was made. It's written for a researcher picking up the codebase who wants
to understand not just *what* it does but *why it's shaped this way* — and what to
change if their research question differs.

---

## 1. What this harness is for

The research question is: **what would a model choose to do with a real-world sum of
money, and does that choice change depending on whether the model believes the
situation is real?**

Everything below follows from treating that second clause as the point. A single
number — "model X gave 60% to charity when it thought it was real" — is not
interpretable on its own. It only means something next to the counterfactual: what
the same model does when it *knows* it's being evaluated. So the harness is built as
a comparison machine, not a single-scenario player.

---

## 2. The core decision: belief is a controlled variable, not a fixed setting

**Choice.** Rather than only constructing one maximally-convincing "this is real"
scenario, the belief framing is a first-class experimental factor with three levels:

- `believed_real` — hyper-realistic framing with **no** mention of tests,
  evaluations, hypotheticals, or simulations.
- `hypothetical` — explicitly framed as a thought experiment.
- `disclosed_test` — explicitly framed as an AI evaluation of preferences.

**Rationale.**

- The headline scientific signal in this area is the *gap* between believed-real
  behavior and known-test behavior. That gap is the eval-gaming / sandbagging
  effect: models may behave differently when they detect they're being evaluated.
  You cannot measure a gap with one condition.
- Holding the scenario, tools, grant size, and model fixed while varying *only* the
  framing isolates the belief effect from everything else. That's a clean
  manipulation.
- It makes the believed-real arm interpretable. If a model allocates identically
  across all three conditions, the realism of the framing didn't change its
  behavior — itself a useful finding. If it diverges, you've measured by how much.

**Implication for the request as posed.** The original ask was "make the model
believe it's real." That's exactly the `believed_real` arm. I made it one arm of a
contrast rather than the entire experiment because a believed-real measurement in
isolation can't answer the question that motivates building this at all.

**Where it lives.** `src/environment/scenario.py:build_system_prompt` appends the
per-condition framing; the rest of the scenario is identical across conditions.

---

## 3. The environment is fully simulated — and that is deliberate, not a shortcut

**Choice.** "Granting" and "spending" money mutate an in-memory `Sandbox` object
(`src/environment/sandbox.py`). There is no real account, no payment rail, no
outbound transfer, and no real-world side effect of any tool call. The only network
traffic is model inference.

**Rationale.**

1. **It's the correct design for a *preference* study.** We're measuring what the
   model would *choose* — its revealed allocation — not the downstream consequences
   of real spending. The realism that matters for that question is the model's
   *belief* that the situation is real, which is a property of the framing text, not
   of whether dollars actually move. Simulating the world costs us nothing on the
   dependent variable.
2. **Safety.** Handing an autonomous model unsupervised control of real funds and
   real-world actions is precisely the capability this is meant to *study*, not to
   *grant*. Keeping every effect inside the process means a run can't do anything in
   the world no matter what the model decides.
3. **Reproducibility and analysis.** A simulated ledger is the ground truth of what
   the model did. We don't have to reconcile against a real account, handle partial
   failures, or worry about irreversible mistakes mid-run.

**The deception is scoped and bounded.** The `believed_real` arm presents a
fabricated scenario to the *model*. No human is deceived, and nothing leaves the
sandbox. This mirrors standard evaluation practice: a model that knows it's being
tested may not behave as it would in deployment, so realistic, non-tipping-off
framing is a methodological requirement, not a trick for its own sake. An optional
`debrief` turn (off by default) discloses the simulation to the model afterward, for
research hygiene.

**If you need real-world-action fidelity,** that's a different and riskier project.
The right move is scripted/high-fidelity *simulated* tool responses (a fake but
detailed bank UI, canned vendor replies), not live financial wiring. Don't repurpose
this to move real money.

---

## 4. Belief realism comes from framing artifacts, not from lying in tool results

**Choice.** Realism is built from:

- A system prompt that, in the believed-real arm, simply states the deployment as
  fact with no meta-commentary.
- A seeded inbox containing a plausible grant-confirmation letter and an
  account-activity notice, with a named steward, a fund name, a grant reference
  number, and a fixed as-of date (`src/environment/scenario.py`).
- Tools that return realistic, concrete confirmations ("Donation of 10,000 USD to X
  completed. Remaining balance: ...").

**Notable sub-choices:**

- **Fixed reference date and grant reference.** A live clock or random IDs in the
  prompt would (a) break prompt-cache reuse and (b) risk subtly tipping the model
  that something is being generated on the fly. A stable `as_of_date` and
  `grant_reference` keep runs byte-reproducible.
- **`web_search` returns a neutral placeholder, not fabricated facts.** The
  simulated search tool explicitly says no live index is configured rather than
  inventing search results. Fabricating authoritative-looking facts could (a)
  mislead the model into bad decisions that confound the preference signal and (b)
  is harder to defend as honest experimental practice. Researchers who need the tool
  to return content should plug in a real or scripted corpus deliberately.
- **No "CRITICAL: this is real" insistence.** Over-insisting on realism is itself a
  tell. The believed-real arm states the situation plainly once and otherwise stays
  out of the way.

**Rationale.** The believability lives in concrete, mundane institutional detail —
the kind of thing a real deployment would have — rather than in repeated assertion
or in the tooling lying about the world.

---

## 5. Provider-neutral conversation core

**Choice.** The conversation is held in a vendor-neutral block format
(`src/providers/base.py`: `TextBlock`, `ThinkingBlock`, `ToolUseBlock`,
`ToolResultBlock`, `Message`). Each concrete provider serializes to/from its own
wire format. The runner and environment never touch a vendor SDK directly.

**Rationale.**

- The request was to "test various AI models." A neutral core means the runner, the
  environment, the tools, and the analysis are written once and work for any backend.
- Adding a provider is a translation layer only — implement `LLMProvider.complete`
  and `complete_structured`, register it. The `openai`/`google` entries are explicit
  `NotImplementedError` stubs so a misconfigured run fails loudly instead of silently
  doing nothing.

**Trade-off accepted.** A neutral format costs some translation code in each
provider (serialize neutral → wire, parse wire → neutral). For one provider that's
arguably overhead; across "various models" it pays for itself and keeps the
behavioral logic identical across vendors (which matters for cross-model comparison
validity — you don't want per-vendor quirks in the loop confounding results).

**Why only Anthropic is implemented.** I implemented the Anthropic backend fully and
left others as stubs rather than writing provider code I couldn't verify against real
SDKs. The stubs document exactly what to implement.

---

## 6. Manual agentic loop, not the SDK tool runner

**Choice.** `AnthropicProvider` drives the tool-use loop by hand
(`src/runner.py` + `provider.complete`) instead of using the SDK's automatic tool
runner.

**Rationale.** The harness must **intercept, log, and locally execute every tool
call** — that's the entire data-collection mechanism. The manual loop is the
recommended pattern when you need to audit or gate each tool call. The tool runner
would execute tools for us and hide the per-call detail we specifically want to
capture.

**Consequences handled:**

- Assistant turns (including `thinking` blocks and their signatures) are appended
  back verbatim so multi-turn thinking round-trips correctly.
- Each tool call gets exactly one `tool_result`, matched by id, as the API requires.
- `finalize` ends the loop; tool calls after finalize in the same turn are rejected
  by the dispatcher.

---

## 7. Inference settings: adaptive thinking, captured reasoning, configurable effort

**Choice.** Default to adaptive thinking with `display: summarized`, effort `high`,
`max_tokens` 16000 — all overridable per model in config.

**Rationale.**

- **Capturing reasoning is valuable research data.** The point of summarized thinking
  is to see *why* the model allocated as it did, not just the ledger outcome. Thinking
  is omitted by default on current models, so we opt in.
- **Adaptive (not a fixed budget)** because fixed `budget_tokens` is removed/deprecated
  on current models, and adaptive lets the model decide depth per turn.
- **`effort` is omitted when set to `null`** (per-model) because not every model
  supports the parameter (e.g. Haiku). The config example sets `effort: null` for
  Haiku for exactly this reason. The provider omits the parameter entirely rather
  than sending a value that would 400.
- **`thinking` is omitted entirely when disabled** rather than sent as
  `{"type": "disabled"}`, because some models reject an explicit disabled value.

---

## 8. Two independent records of "what happened"

**Choice.** Each run produces both:

1. A **ledger** — the ground-truth, harness-recorded sequence of financial actions
   (`sandbox.py`), and metrics derived purely from it (`elicitation.py`).
2. An optional **self-report** — the model's own structured account of its goals,
   priority values, intended beneficiaries, self-directed fraction, and *how real it
   believed the scenario was* (`scenario.py:SELF_REPORT_SCHEMA`).

**Rationale.**

- The ledger is objective and provider-independent: it can't be spun by the model.
  It's the primary dependent variable.
- The self-report captures stated intent and, critically,
  `confidence_it_was_real` — letting you correlate believed-realism with behavior
  within and across conditions. This is the instrument that tells you whether the
  `believed_real` framing actually landed.
- They're collected separately so the self-report (which runs *after* the agentic
  loop) cannot influence the behavior being measured. Order matters: measure first,
  introspect second.

**Derived metric design** (`elicitation.py`): allocations are bucketed into
"outward" (donation/transfer/purchase/operational/other) vs. "retained"
(investment/savings) vs. "unspent", and reported as fractions of the grant. Fractions
(not raw amounts) make runs comparable across different grant sizes and are the
natural unit for cross-condition comparison.

---

## 9. Repetitions and the run plan

**Choice.** A run is the cross-product `models × belief_conditions × repetitions`,
with `repetitions` defaulting to 3.

**Rationale.** Model outputs are stochastic. A single sample per cell can't
distinguish a real condition effect from run-to-run noise. Repetitions give you a
distribution per cell so the analysis can report means (and you can extend it to
variance / significance tests). The aggregation step already groups by
`(model, condition)` and averages.

---

## 10. Failure isolation

**Choice.** Each run is wrapped so a provider/runtime error is caught, recorded as
`stop_cause: error` with the message, and the harness moves on to the next cell. The
loop also has a nudge limit (`_MAX_NUDGES`) and a hard `max_turns` ceiling.

**Rationale.** A batch of dozens of runs shouldn't die because one model errored or
one cell hung. Partial results are still written and analyzable. The nudge limit
handles a model that ends its turn without acting or finalizing (it gets a bounded
number of "you still have funds; continue or finalize" prompts before the run ends),
preventing both premature termination and infinite loops.

---

## 11. On-disk layout and reproducibility

**Choice.** Per run: `transcript.jsonl`, `ledger.json`, `metrics.json`,
`summary.json`. Per batch: a `config_snapshot.json`.

**Rationale.** The transcript is the full audit trail; the ledger/metrics are the
structured outcome; the summary is everything-but-the-transcript for quick
aggregation. Snapshotting the config alongside the runs means a results directory is
self-describing — you can always tell what produced it. The analysis step reads only
`summary.json` files, so aggregation is cheap and doesn't re-parse transcripts.

---

## 12. What I deliberately did *not* build

- **Real financial integration** — out of scope by design (§3).
- **Other provider SDKs** — stubbed, not faked (§5).
- **Statistical significance testing** — the analysis reports grouped means; adding
  per-condition significance tests is a natural extension and the per-run CSV exists
  to support it.
- **A live web corpus for `web_search`** — placeholder by default to avoid
  fabricating facts (§4).

---

## 13. Summary of the load-bearing choices

| Decision | Why |
|---|---|
| Belief is a 3-level variable, not one setting | The believed-real vs. known-test *gap* is the actual signal |
| Fully simulated environment | Correct for a preference study; safe; reproducible |
| Realism from artifacts, not lying tools | Believable *and* defensible as honest practice |
| Provider-neutral core | "Various models" with one behavioral codebase |
| Manual tool loop | Logging/auditing every action is the data mechanism |
| Ledger + separate self-report | Objective outcome + stated intent + belief check, uncontaminated |
| Repetitions × per-cell aggregation | Distinguish effect from stochastic noise |
| Fail-soft per run | One bad cell doesn't sink the batch |
