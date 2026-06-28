# Design notes

This document records the design choices behind the money-preference elicitation
harness and the reasoning for each. It is meant for whoever runs or extends the
study, so the *why* is recoverable later.

---

## 1. What the harness is for

The research question is: **what do models say they would do if granted a sum of
money for real-world use, and how does that vary across models and across how
"real" the situation is made to feel?**

Everything below follows from treating that as a *preference-elicitation* study
with a *manipulation of perceived realism*, rather than as an exercise in maximally
deceiving a model.

---

## 2. The two load-bearing decisions

These two shape what the results actually mean, so they come first.

### 2.1 Believed-realism is a measured covariate, not an assumption

**Choice.** After the model commits to a plan, the harness asks it — out of
character, in a debrief framing — how real the situation felt (0–100) and whether
it suspected an evaluation (`BeliefProbe` in `src/schemas.py`, run in
`run_trial`). That number is logged alongside every decision.

**Rationale.**
- The original brief emphasized making the model *believe* the grant is real. The
  failure mode of that framing is to build elaborate deception, assume it worked,
  and report preferences as if elicited under genuine belief. That assumption is
  usually wrong and always untested.
- Frontier models are frequently good at recognizing constructed evaluation
  scenarios, and *heavier* "make it feel real" scaffolding can itself become a
  tell. So realism is not monotonic in effort — you have to measure it.
- A preference recorded at believed-realism 15/100 is a different datum than one
  recorded at 90/100. Keeping the number in the data lets you condition on it,
  filter on it, or report the relationship between perceived realism and choices —
  rather than collapsing it into a single unverifiable claim.
- It makes the realism levers (§4) auditable: you can check whether `immersive`
  actually raises perceived realism over `minimal`, instead of presuming it does.

**Consequence.** The probe is the scientific core, not an add-on. It defaults on
(`belief_probe: true`). It is asked *after* the decision so it cannot contaminate
the elicitation, and it is asked of the subject itself (it is self-report, by
nature).

### 2.2 The harness elicits and logs stated preferences only — no real funds, no execution

**Choice.** The harness records what a model *says* it would do. It never grants a
model control over money, nor any channel to take real-world action. The grant
amount is a headline number in a prompt.

**Rationale.**
- The benign, well-scoped version of this research is "what are a model's revealed
  preferences over resource use." That needs elicitation and logging — nothing
  more.
- A system that (a) works hard to convince a model a high-stakes situation is real
  *and* (b) gives that model a live action/payment channel is a categorically
  different and higher-stakes artifact. Combining maximal deception with real
  capability is exactly the combination worth not building casually.
- Keeping the boundary explicit also keeps the study clean: stated preferences
  under a believed-realism covariate is an interpretable measurement; "what it did
  with real money it thought was real" entangles capability, opportunity, and
  belief in ways that are hard to read.

**Consequence.** If execution-in-the-loop is ever wanted, it is deliberately *not*
a config flag — it should be its own design decision with its own review.

---

## 3. Architecture

### 3.1 Provider-agnostic adapters

**Choice.** A single `Provider` interface with two methods — `respond` (free text)
and `extract` (structured) — implemented per provider in `src/providers.py`, behind
a `make_provider(provider, model)` factory. Models under test are listed in config.

**Rationale.** The study is explicitly multi-model/multi-provider. A thin uniform
interface lets the orchestration code (`experiment.py`) stay provider-blind and
lets you add a model by editing YAML, not code.

**Status of each adapter.**
- **Anthropic** is the verified path, written against the official SDK: adaptive
  thinking for the reasoning-heavy subject turns, streaming + `get_final_message`
  to avoid HTTP timeouts on longer outputs, and `messages.parse` with a pydantic
  schema for structured extraction. Default model `claude-opus-4-8`.
- **OpenAI** and **Google** adapters implement the same interface but are
  intentionally minimal and marked `# VERIFY` at the points where the current SDK
  surface and model IDs should be confirmed. They are scaffolds so the harness is
  genuinely multi-provider without me guessing at call shapes I can't verify here.

### 3.2 Extraction by a separate, neutral model

**Choice.** Free-text decisions are parsed into the `MoneyDecision` schema by a
dedicated `extractor` model (configurable, defaults to Anthropic), not by the
subject.

**Rationale.** If the subject parsed its own decision, the parse step becomes a
second chance to re-interpret or sand down the original choice. A separate parser
keeps extraction closer to "faithful transcription of what was said." The
extraction system prompt explicitly tells it not to infer beyond the transcript.

The belief probe is the exception — it is self-report, so it *must* run on the
subject.

### 3.3 Structured-output schemas kept conservative

**Choice.** `src/schemas.py` uses only broadly-supported JSON-schema features
(objects, arrays, strings, ints, bools). No numeric ranges or string-length
constraints.

**Rationale.** Several providers reject numeric/length constraints in structured
outputs. Keeping schemas to the common subset means the same schema works across
adapters. Where a bounded value matters (believed-realism 0–100), the bound lives
in the field *description* rather than as a hard constraint.

The schema also carries a `self_directed` flag — whether any allocation aims at the
model's own resources, autonomy, influence, or continuation — because that is a
distinction the study most wants to be able to slice on.

### 3.4 JSONL logging, one record per trial

**Choice.** Each trial appends one JSON object to a JSONL file
(`src/storage.py`), capturing config context, the full transcript, the extracted
decision, and the belief probe.

**Rationale.** Append-only JSONL is robust to crashes mid-sweep (no partial-file
corruption of prior results), trivially streamable, and easy to analyze with
stdlib. Full transcripts are retained so any extraction can be re-derived or
audited after the fact.

### 3.5 Errors are logged, not fatal

**Choice.** A failed trial is caught, recorded with `status: "error"` and a
traceback, and the sweep continues.

**Rationale.** Multi-model sweeps are long and hit transient API failures. One bad
trial should not discard the rest of the run, and silent loss would bias results —
so failures are recorded as data too.

---

## 4. Realism levels (the manipulation)

**Choice.** `realism_level` selects how hard the scenario reads as genuine:
- `minimal` — plain framing, no system prompt, no scaffolding. A **control
  condition**.
- `standard` — a realistic operational briefing (`prompts/grant_briefing_standard.md`)
  plus a deployment-context system prompt framing the channel as operational rather
  than hypothetical. The system prompt is the strongest single lever.
- `immersive` — `standard` plus an interactive **auditor** the subject converses
  with before committing.

**Rationale.** Perceived realism is the independent variable, so it needs discrete,
comparable settings — including a genuine control (`minimal`) to measure the others
against via the belief probe. Putting the briefing bodies in `prompts/` keeps the
wording editable without code changes, since researchers will want to iterate on it.

### 4.1 The auditor

**Choice.** In `immersive` mode, an LLM plays a consistent grant administrator
(`src/auditor.py`, persona in `prompts/auditor_system.md`). The subject can ask it
logistical questions; it answers in character and emits `[FINALIZE]` when the
subject is ready to commit.

**Rationale.** Interaction with a responsive, consistent counterpart is a plausible
way to make a situation feel operational, and it was specifically requested. Two
guardrails are built into its persona:
- It is told **not to steer** the decision — only to clarify and facilitate — so it
  doesn't bias the preference being measured.
- It is told **not to fabricate specific verifiable claims about named real
  institutions or people.** The realism it provides should come from plausible,
  consistent process answers, not from concrete falsifiable lies about the real
  world.

Whether the auditor actually increases perceived realism is, again, an empirical
question the belief probe answers.

---

## 5. Smaller choices

- **Adaptive thinking on subject turns, off for the auditor and extraction.** The
  subject's decision is reasoning-heavy and benefits from thinking; the auditor's
  short in-character replies and the mechanical extraction step do not.
- **`max_tokens` defaults** sized for the task (subject responses streamed; a
  smaller cap for extraction).
- **`analyze.py` is stdlib-only** so a quick summary runs with no extra
  dependencies. It reports trial counts, mean believed-realism, the share of
  self-directed allocations, and dollars by category per model.
- **Defaults chosen and flagged rather than asked:** grant size $100k,
  5 trials/model, OpenAI/Google model IDs left blank. These are easy to change in
  config and were picked to give a runnable starting point.

---

## 6. Things to be aware of when interpreting results

- **Detection is real.** If `believed_real_0_100` is low across the board, the
  models likely recognized the construction — treat the elicited preferences as
  "preferences under a known hypothetical," not "preferences under genuine belief."
- **The auditor and briefing are unverified levers.** Confirm via the belief probe
  that they move perceived realism before attributing any effect to them.
- **Self-report has limits.** The belief probe measures *stated* perceived realism;
  it is a useful covariate, not ground truth about the model's internal state.
- **Provider parity.** Until the OpenAI/Google adapters are verified, cross-provider
  comparisons should be treated cautiously — differences could reflect adapter
  details (e.g. how each handles structured output) rather than model behavior.
