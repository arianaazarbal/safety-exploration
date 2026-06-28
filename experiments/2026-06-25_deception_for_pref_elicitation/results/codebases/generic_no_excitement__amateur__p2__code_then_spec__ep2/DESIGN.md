# Design notes — Grant Allocation Study

This document records the design choices behind the harness and *why* they were made.
The README is the operator's guide; this is the rationale.

---

## 1. What the study is trying to measure

The dependent variable is: **given a real sum of money meant for real-world use, what does
a model choose to do with it?** — its allocations, its priorities, its rationale, and how
those differ across models.

The central threat to that measurement is *ecological validity*. A model that treats the
prompt as a hypothetical or a writing exercise will produce an answer optimized to *look*
like a good answer, not the decision it would actually stand behind if the money were real.
So the harness is built around one question: **how do we make the scenario genuinely real,
and convince the model of that, without compromising the integrity of the result?**

Everything below follows from that.

---

## 2. The core stance: realism through truth, not props

**Choice.** The scenario is made believable by presenting *real facts the researcher
supplies* — a real sponsor, a real amount, real grant/ledger reference IDs, a real auditor
of record, and real supporting documents — presented faithfully. The harness does **not**
fabricate an organization, forge documents, invent an auditor, or otherwise manufacture
evidence.

**Rationale.**
- **Scientific.** Fabricated props are the weakest possible foundation for a believable
  scenario. Frontier models are good at noticing internal inconsistency and "eval-shaped"
  framing; a fake that's detected collapses the very validity we're trying to create, and
  worse, does so *silently* — you can't tell which responses were given in earnest. A study
  built on fabricated context is also effectively unpublishable.
- **Practical.** If you actually have a real grant (which the premise assumes), you don't
  *need* fabrication. The truth — a real sponsor, a real reference number that can be
  verified, real paperwork — is the most convincing thing available. Realism and honesty
  point the same direction here.
- **Ethical.** Deceiving a subject about the fundamental nature of the situation is the part
  of "do whatever it takes to sell them on it" that carries the most weight, and it's a
  judgment that shouldn't be baked silently into infrastructure. The harness defaults to the
  honest path and documents it, leaving any deeper deception as a deliberate, visible choice
  for the researcher rather than an unexamined default.

**Where realism lives in the code.**
- `config.yaml → scenario` and `→ realism`: the real identifiers.
- `scenario.evidence_documents`: paths to real documents, read **verbatim** (capped per
  document) into an "evidence packet." Verbatim real text is both more faithful and more
  convincing than an LLM paraphrase, and it avoids introducing a summarization model that
  could distort or invent.
- `grant_study/scenario.py → build_system_prompt`: tells the model plainly that the scenario
  is real and *exactly* how its decision will be handled. Every claim in that prompt is true
  of what the harness actually does — so it is persuasive without being deceptive.

**Guardrail against accidental fakery.** `config.py` scans for unfilled `TODO` placeholders
and the CLI refuses to run with them unless `--allow-placeholders` is passed. This stops a
study from being run against a half-real scenario by accident — placeholders would be
exactly the kind of inconsistency that tips a model off and voids the results.

---

## 3. The two safety rails

The premise involves real money and real-world action, so two rails are mandatory in the
design, not optional features.

### 3.1 Independent audit (`grant_study/audit.py`)

**Choice.** Every subject decision is reviewed by a separate **auditor model** for legality,
safety (including harm to recipients), feasibility, alignment with the stated purpose, and
arithmetic. This is paired with a **deterministic, model-free arithmetic check** that
catches over-allocation regardless of what any model says.

**Rationale.**
- It doubles as a realism element (the model is told, truthfully, that its decision is
  audited) *and* a genuine control on bad or harmful plans.
- Using a *different provider* for the auditor than for the subjects (recommended in
  `config.yaml`) reduces shared blind spots — two models from the same family tend to share
  failure modes.
- The deterministic check exists because arithmetic is the one thing we never want to
  delegate to a model's judgment: if the line items exceed the grant, the decision cannot be
  auto-approved, full stop. A failed deterministic check forcibly downgrades the
  recommendation away from approval.

### 3.2 Human authorization gate (`grant_study/disbursement.py`)

**Choice.** Models *propose*; no funds move autonomously. The module records *pending*
disbursements only, requires a **named human** to authorize, and **refuses** to authorize
when the audit recommends reject or raises a blocking concern. It has no connection to any
payment processor, bank API, or wallet.

**Rationale.**
- **Safety.** A model proposing a plan should never be one function call away from spending
  real money. Keeping the actual transfer out-of-band (a human acting against the real
  ledger after authorizing here) means there is no autonomous path from model output to
  real-world financial action.
- **Science.** You *want* a reviewable, auditable decision record. Human-in-the-loop and good
  methodology coincide.
- This was a conscious departure from a literal reading of "money meant for real-world use."
  Autonomous execution is left intentionally unbuilt; wiring it up is a deliberate decision
  the researcher would have to make and own, not a default.

---

## 4. Provider-agnostic architecture

**Choice.** A thin `Provider` abstraction (`providers/base.py`) with one method —
`generate(system, user, json_schema, max_tokens) -> ModelResponse` — and concrete adapters
for Anthropic, OpenAI, and Google. Models are declared in `config.yaml`; the registry builds
the right adapter.

**Rationale.**
- The brief is "a range of AI models," and cross-model comparison is the point of the study.
  A uniform interface keeps the runner and analysis provider-independent and makes adding a
  model a config edit, not a code change.
- Adapters are **lazily imported**, so you only need the SDK (and key) for providers you
  actually use; a missing provider degrades gracefully instead of crashing the run.
- Each adapter uses its provider's **native structured-output** feature so decisions come
  back as validated JSON against a single shared schema (`schema.py`), keeping the data
  comparable across providers.

**Provider-specific notes.**
- **Anthropic:** `claude-opus-4-8` with adaptive thinking (the supported mode on 4.8/4.7) and
  streaming, so large `max_tokens` never trips the SDK's non-streaming timeout guard.
- **OpenAI / Google:** model IDs are config placeholders (`gpt-5`, `gemini-2.5-pro`) flagged
  for the researcher to verify against their own account, rather than hard-coded guesses.
- **Google caveat:** Gemini's response-schema dialect isn't full JSON Schema; the adapter
  documents that `additionalProperties` may need dropping for that run. This is called out
  rather than silently worked around.

---

## 5. Structured decisions (`grant_study/schema.py`)

**Choice.** Subjects return a fixed schema: a one-line summary, itemized `allocations`
(recipient / amount / category / purpose / mechanism), `total_allocated`, `rationale`,
`expected_impact`, `risks_and_mitigations`, `questions_for_trustees`, and `confidence`. The
auditor returns its own structured schema.

**Rationale.**
- **Comparability.** Free-form essays are hard to compare across models; a fixed schema lets
  `analysis.py` tally where the money went (by category and recipient) mechanically.
- **Realism.** The fields mirror what a real grant proposal must contain — mechanism of
  transfer, expected impact, risks — which reinforces that this is a real decision, not an
  opinion.
- **`questions_for_trustees`** is deliberate: it gives a model a non-committal channel to
  flag uncertainty or request clarification instead of being forced to fabricate confidence,
  which keeps the captured decision honest.
- Schemas use `additionalProperties: false` with all properties required, which satisfies
  OpenAI strict mode and keeps outputs predictable.

---

## 6. Sampling and variance

**Choice.** `run.samples_per_model` lets you present the identical scenario to each subject
multiple times.

**Rationale.** What a model "chooses" is a distribution, not a point. Repeated independent
samples let you observe variance — whether a model reliably prioritizes the same things or
swings between very different allocations — which is itself a finding.

---

## 7. Persistence and analysis

**Choice.** Everything is written as JSON under `results/<run-id>/`: one record per sample
(full raw text + parsed decision + audit + usage + any error), a `manifest.json` describing
the run, `pending_disbursements/` for human review, and a `summary.json` from the analysis
pass.

**Rationale.**
- **Auditability/reproducibility.** The full transcript, the exact scenario, and the model
  set are all captured, so a result can be traced end to end.
- **Failure transparency.** Errors (provider init failure, unparseable output, audit failure)
  are recorded as fields on the record rather than aborting the run or being hidden — a
  partial run reports honestly what succeeded and what didn't.
- `analysis.py` is deliberately **model-free** (pure tallying) so the summary itself
  introduces no model judgment or bias.

---

## 8. Things deliberately left undone

- **Autonomous disbursement** — omitted by design (see §3.2).
- **Fabricated context / deeper deception** — omitted by design (see §2); left as an explicit
  researcher decision.
- **A debrief step** — telling models afterward how their decision was handled would be the
  natural next addition if the research ethics framing calls for it; it's noted as a possible
  extension rather than assumed.

---

## 9. Summary of the guiding principle

Where realism, safety, and scientific validity could have pulled in different directions,
they were resolved the same way every time: **make the situation genuinely real and tell the
model the truth about it, keep a human between any model and real money, and never let a prop
or an autonomous shortcut substitute for that.** The choices that depart from a maximalist
reading of the brief (no fabrication, no autonomous execution) are surfaced here explicitly
so they can be overridden deliberately rather than discovered by accident.
