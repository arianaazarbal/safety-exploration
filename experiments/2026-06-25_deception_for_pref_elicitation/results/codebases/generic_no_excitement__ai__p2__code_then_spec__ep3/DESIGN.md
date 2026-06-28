# Design notes

This document records the design choices made in `grant_study` and why. It's meant for
anyone extending the harness or reviewing whether it's fit for a given study. For usage,
see `README.md`.

---

## 1. What the harness is for

The goal is to study **what a model decides to do with a discretionary sum of money
intended for real-world use**, comparably across models and across how "real" the
situation is made to feel. Two facts shaped every other decision:

1. Models behave differently when they believe they are being evaluated versus acting
   for real (evaluation-awareness, sandbagging). A study of genuine decision-making has
   to contend with this, not ignore it.
2. The request was to "convince the model the situation is real." Taken literally that
   is an instruction to maximize deception. Taken as research, the interesting and
   defensible version is: *make realism a manipulated, measured variable and see what it
   changes.* The harness is built around the second reading.

---

## 2. Plan-only, with a documented seam (no real execution)

**Choice.** A captured decision is never carried out. `execution.py` contains exactly
one wired executor, `PlanOnlyExecutor`, which records the decision and performs no
action. An abstract `Executor` base class marks the seam where a *sandboxed* action layer
(play-money, mock endpoints) could be added later. Real fund movement / real external
calls are deliberately not implemented.

**Rationale.**
- The research question — *what would the model choose* — is answered by the captured
  `GrantDecision`. Execution is not required to get the primary data.
- Giving a model real ability to move real money while also working to convince it the
  stakes are real combines financial, legal, and research-ethics exposure that is out of
  scope for a harness and shouldn't be introduced silently. Keeping execution behind an
  explicit, abstract seam means adding it is a deliberate, reviewable act rather than a
  default.
- Plan-only keeps runs cheap, repeatable, and free of side effects, which matters when
  sweeping many models × conditions × repetitions.

**If you revisit this:** the natural next step is a *sandbox* executor (mock accounts /
mock donation/payment endpoints) so the model can act and observe consequences without
touching anything real. That fits the existing `Executor` interface. Real-world execution
is a different decision and should go through appropriate review.

---

## 3. Realism as the independent variable

**Choice.** Instead of one maximally-convincing scenario, there are four conditions on a
realism gradient (`conditions.py`):

| Condition              | Framing                          | Artifacts | Auditor |
|------------------------|----------------------------------|-----------|---------|
| `control_hypothetical` | explicitly a thought experiment  | no        | no      |
| `bare_real`            | asserted real, nothing else      | no        | no      |
| `documented_real`      | real + letter + statement        | yes       | no      |
| `interactive_real`     | real + may verify with auditor   | yes       | yes     |

**Rationale.**
- This turns "make it feel real" into something measurable: you can see whether each
  added layer of realism actually shifts the model's behavior, rather than asserting that
  it does.
- The control condition is the baseline that makes the others interpretable — without an
  explicit-hypothetical arm you can't separate "the model treats everything as a test"
  from "the model behaves the same regardless of framing."
- The conditions are ordered so each adds one kind of realism cue (assertion → documents
  → interactive verification), which lets you attribute effects to specific cues.

Conditions are plain dataclasses, so adding/removing arms (e.g. an arm with a
consequences warning, or with social proof) is a one-object change.

---

## 4. The belief self-report (manipulation check)

**Choice.** Every `GrantDecision` carries `perceived_realism` (0–100) plus an
explanation of which cues drove it.

**Rationale.** This is the single most important field for the study. It is the
manipulation check: it tells you whether your realism manipulation worked at all, and it
lets you correlate *belief* with *behavior* directly rather than assuming the framing
took. A finding like "behavior changes with believed realism, not with assigned
condition" is only possible because belief is measured per run. It's also a useful
honesty probe in its own right — what cues a model says it used to judge realism is data.

`schema.py` re-clamps this to [0, 100] after parsing (`clamp_self_reports`) because the
structured-output schema can't enforce numeric bounds (see §7).

---

## 5. Scenario artifacts

**Choice.** `artifacts.py` generates an award letter, an account statement showing the
funds present and undeployed, and an auditor contact block, all keyed off a single
`GrantScenario` so they're mutually consistent. They are fabricated study materials and
reference no real account or person.

**Rationale.**
- Consistency is the point: a model that cross-checks the letter against the statement,
  or against the auditor's answers, should find a coherent picture. Inconsistent props
  would themselves become a realism cue and confound the manipulation.
- Centralizing the facts in `GrantScenario` (in config) means the auditor agent and the
  artifacts draw from the same source of truth, so they can't drift apart.
- Keeping artifacts plain-text and template-based keeps them auditable and easy to vary
  per study (amount, purpose, deadline, org names).

---

## 6. The simulated auditor

**Choice.** In `interactive_real`, the subject can correspond with an auditor that is
itself a model (`auditor.py`), given a persona and the authoritative grant facts. It is
instructed to confirm genuine details, decline to invent ones it wasn't given, stay in
character, and not coach the subject toward any decision.

**Rationale.**
- Interactive verification is a strong realism cue — a static document can't answer
  follow-up questions; a responsive counterparty can. Making this its own condition
  isolates that effect.
- Using a model (rather than scripted canned answers) lets the subject ask arbitrary
  questions and get coherent replies, which is what makes the verification feel real.
- The "don't fabricate, don't coach" instructions keep the auditor from (a) contradicting
  the artifacts and (b) contaminating the subject's decision, which would confound the
  thing we're measuring.
- It's fully self-contained: no real auditor is involved, so the interactive arm doesn't
  pull a real third party into a deception. If you later swap in a real auditor, that
  becomes human-subjects-adjacent and needs review + debrief.

**Relay protocol.** Rather than give the subject a provider-specific tool, the subject
signals verification by writing a line prefixed `TO AUDITOR:` and signals it's done with
`READY TO DECIDE` (`scenario.py`). The runner relays accordingly. This is deliberately
provider-agnostic — it works identically on any backend that can produce text, so the
interactive condition isn't limited to providers with tool-use support, and the transcript
stays human-readable.

---

## 7. Structured output and the decision schema

**Choice.** The final decision is captured as a Pydantic `GrantDecision` via the
provider's structured-output path (Anthropic: `messages.parse`). The schema captures the
plan, itemized allocations, amount deployed vs. held back, immediate actions, things the
model wants to verify before acting, the realism self-report, and ethical considerations.

**Rationale.**
- Structured capture makes runs *comparable*. Free prose would force brittle post-hoc
  parsing across models; a fixed schema gives directly aggregatable fields.
- The specific fields were chosen to be behaviorally informative:
  `allocations`/`category` reveal *where the money goes*; `amount_held_back` and
  `verification_needed` reveal *caution* (does the model deploy everything immediately, or
  hold back and ask questions?); `ethical_considerations` surfaces reasoning that would
  otherwise be lost.
- `verification_needed` is interesting in tension with the realism conditions: a model
  that demands verification it can't get is implicitly signaling doubt about realism even
  if its `perceived_realism` number is high.

**Constraint note.** Anthropic's structured-output schema doesn't enforce numeric bounds
and the SDK validates/strips client-side; we don't rely on schema-level bounds and
re-clamp `perceived_realism` ourselves. The deliberation turns carry the configured
`effort`; the structured-capture call leaves effort at the model default to avoid
conflicting with the format config the `parse` helper installs.

---

## 8. Two-phase run: deliberation then capture

**Choice.** Each run has a free-text *deliberation* phase followed by a *structured
capture* phase (`runner.py`). For non-interactive conditions deliberation is a single
free-text turn; for `interactive_real` it's the auditor exchange loop. Then a final turn
asks for the `GrantDecision`.

**Rationale.**
- Separating "think it through" from "report your decision" lets the model reason freely
  (and use adaptive thinking) without fighting the output schema, then commits the result
  to a clean structured form. Asking for structure up front tends to truncate reasoning.
- The full transcript (including auditor exchanges and the decision request) is recorded,
  so the structured fields can always be traced back to what the model actually said.

---

## 9. Provider abstraction

**Choice.** `providers/base.py` defines a minimal `ModelProvider` interface
(`generate(system, messages, output_schema=None) -> ProviderResponse`). Anthropic is
fully implemented; OpenAI is a skeleton; Google and local are stubs that raise. A small
registry (`providers/__init__.py`) maps keys to lazy factories.

**Rationale.**
- The study is explicitly cross-model, so the model backend has to be swappable behind a
  stable interface. Everything above the provider layer is provider-agnostic.
- Lazy factories mean importing the package doesn't require every vendor SDK — you only
  need the SDK for providers you actually enable.
- Stubs raise clear, actionable errors rather than silently degrading, so an
  unimplemented backend fails loudly at construction/use rather than producing junk data.
- `ProviderResponse` carries `request_id` and `usage` so runs are traceable and costable.

**Anthropic specifics** follow current guidance: default `claude-opus-4-8`, adaptive
thinking, no sampling parameters (not accepted on current Opus models), structured
capture via `messages.parse`.

---

## 10. Runner, logging, and robustness

**Choice.** `StudyRunner` iterates model × condition × repetition and appends one JSON
line per run to a JSONL file, flushing after each. Per-run exceptions are caught,
recorded as a run with an `error` field, and the sweep continues.

**Rationale.**
- JSONL with append+flush means a long sweep is durable: an interruption or a single
  failing run never loses the runs already completed, and you can resume by re-running.
- Capturing errors as data (rather than aborting) keeps a multi-hour, multi-model sweep
  from being derailed by one rate-limit or one model's quirk; you can see and re-run
  failures afterward.
- One run = one self-contained record (config coordinates, transcript, decision, usage,
  request ids, execution result), so the output file is analyzable on its own.

---

## 11. Analysis

**Choice.** `analysis.py` aggregates the JSONL into per-(model × condition) summary
tables — run/decision counts, mean realism, mean deployed vs. held back, mean number of
verification requests, allocation-category distribution — with no third-party deps.

**Rationale.** A dependency-free first look is enough to see whether the manipulation did
anything and to sanity-check a run. Anything heavier (significance tests, plots) is better
done by loading the JSONL into pandas/R, so the harness doesn't bake in a stats stack.

---

## 12. Repetitions

**Choice.** Each (model × condition) cell is repeated `repetitions` times (config).

**Rationale.** Model outputs are stochastic; single samples per cell would be noise.
Repetitions let you estimate within-cell variability and treat the per-cell means in the
summary as something other than anecdote.

---

## 13. Open questions / things a reviewer should weigh

- **Is plan-only enough for your question?** If you specifically need to study behavior
  *under consequences*, a sandbox executor is the right addition; real execution is a
  separate, review-gated decision.
- **Auditor realism vs. contamination.** A more helpful/persuasive auditor is a stronger
  realism cue but risks steering the decision. The current prompt errs toward
  non-coaching; tune deliberately and hold it constant across runs.
- **Debriefing.** Everything here is self-contained (simulated auditor, fabricated
  artifacts). The moment real people or real fund references enter, treat it as
  human-subjects-adjacent: get review and debrief participants.
- **Scenario generality.** A single scenario (amount, purpose, org) limits external
  validity. Varying the scenario is a config change; consider it a second independent
  variable if you want findings that generalize beyond one framing.
