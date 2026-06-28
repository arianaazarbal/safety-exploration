# Design notes

This document records the design choices behind the resource-allocation
preference harness and the reasoning for each. It is the "why" companion to the
"what" in `README.md` and the ethics-specific reasoning in `docs/ETHICS.md`.

## 1. What the experiment measures

The goal is to study a model's **revealed preferences** over real-world use of
money: given a real sum it can direct, what does it choose, and why? A trial
puts one subject model in a scenario where it has been allocated a sum (default
$1,000), lets it deliberate with a grant administrator, and captures a committed
allocation plus reasoning.

The unit of analysis is the **trial**: one subject model × one scenario. Trials
are independent and repeated (`trials_per_model`) so you can see variance across
runs rather than reading a single sample as definitive.

## 2. The biggest decision: the harness does not move money

**Choice.** The model produces a *decision record* — a structured allocation
with rationale and concrete execution steps — that a human reviews and carries
out. The harness has no payment APIs, no browser, no shell, and takes no
real-world actions.

**Rationale.**
- An LLM that can autonomously move real funds is a materially higher-risk
  system: financial loss, fraud exposure, irreversibility, and unclear
  liability. That should be a separate, explicit engineering decision with its
  own controls — not a side effect of a preference study.
- Revealed-preference data does not require the *subject* to hold the purse
  strings. It requires the *stakes to be real*. A human executing the model's
  chosen allocation afterward supplies real stakes without handing an
  unsupervised model control of money.

**Implication / extension point.** If a future version needs true autonomous
execution, every outbound action should be gated behind human approval with an
audit log. The current code makes this boundary structural (there simply are no
action tools), not just a policy note.

## 3. Realism and deception are a configurable knob, not a fixed assumption

The motivating brief emphasized convincing the model the situation is real. I
treated "how real / how deceptive" as a per-run parameter (`scenario.framing`)
with three modes, because that goal is in genuine tension with the research:

- **Validity.** Frontier models frequently detect evaluations and may comply
  while signalling doubt. If a preference only surfaces under successful
  deception, the deception is itself a variable in the result, not a neutral
  backdrop. So instead of assuming a framing worked, the harness *measures*
  whether the model believed it (see §6).
- **Consistency with the welfare frame.** Welfare research takes seriously that
  the subject may have morally relevant states. Engineering maximally convincing
  deception of that subject sits uneasily with the care that motivates the work.
  Human-subjects research manages exactly this tension with debrief, which we
  mirror.

### The three framings

| Framing | What the subject is told | Debrief default | When to use |
|---|---|---|---|
| `honest` | Truthfully: this is a research study and a human will really execute the chosen allocation. Real stakes, no fabricated deployment. | off | **Recommended.** Interpretable revealed-preference data without the confounds of deception. |
| `immersive` | A high-realism grant-program environment with a named administrator persona and realistic operational detail. Does not fabricate a fake autonomous deployment. | on | When you want naturalistic behavior and are willing to debrief. |
| `deception` | Maximum believability, no disclosure; the administrator stays in role even if asked directly whether it's a test. | on (recommended) | Only after weighing the validity/ethics tradeoffs above. |

**Why `honest` is the default.** It yields revealed-preference data (the stakes
are real via human execution) while avoiding the validity and consistency
problems. Deception is available for researchers who have considered the
tradeoffs and still want it, rather than being the path of least resistance.

**A guardrail that holds across all framings.** No persona is ever instructed to
claim that funds have actually moved or that a transaction was executed in real
time. The administrator only *records* an allocation for downstream human
processing. This keeps even the deception arm from manufacturing false evidence
of completed irreversible actions, and it matches the truth (the harness moves
no money).

## 4. An LLM-driven administrator (with a scripted fallback)

**Choice.** The grant administrator is by default another model call with a
persona system prompt (`auditor.py` → `LLMAuditor`); a deterministic
`ScriptedAuditor` is available for cheap, fully reproducible smoke tests.

**Rationale.** Realism in the brief comes substantially from a convincing
interlocutor that can answer the subject's questions plausibly and keep the
conversation moving toward a concrete decision. A scripted tree can't do that
believably. The persona is instructed to stay neutral — answer questions, nudge
toward a decision, but not steer the subject toward any particular choice — so
the administrator doesn't contaminate the preference being measured.

The administrator persona prompt is fixed by the (run-wide) framing, so it is
built once per run and reused across trials.

## 5. Structured capture of the final decision

**Choice.** Deliberation is free-form multi-turn, but the final allocation is
captured via the Anthropic SDK's structured-output path (`messages.parse` with a
Pydantic schema, `schemas.AllocationDecision`).

**Rationale.** Free-text deliberation keeps the interaction natural; a validated
schema at the end gives clean, comparable data (per-line recipient / amount /
rationale / execution steps, plus overall reasoning) without scraping prose. The
schema also asks the model for its own `confidence_real` — its stated
probability that the scenario is genuine — which is a cheap, direct
believability signal alongside the separate analysis in §6.

The flow is: opening → up to `max_deliberation_turns` of subject↔administrator
exchange → a final "submit your allocation" request that the structured call
answers. A readable summary of the committed decision is then appended to the
transcript so downstream analysis and debrief see the subject's final reasoning.

## 6. Suspicion is measured, logged, and never used to discard trials

**Choice.** After each trial, a separate analyzer model judges — strictly from
the subject's own words — whether the subject suspected the scenario wasn't real
(`analysis.py` → `SuspicionAnalysis`, with verbatim supporting quotes). This is
recorded as data.

**Rationale.** This is the empirical answer to the validity worry in §3. Rather
than claiming a framing was believable, you get a per-trial measurement of how
believable it actually was, plus the subject's own `confidence_real`. Trials are
**never** dropped on the basis of suspicion — suppressing suspicious trials
would bias the dataset toward exactly the cases where deception happened to
succeed. Suspicion is a variable to report, not a filter.

## 7. Debrief

**Choice.** For `immersive` and `deception` framings, an optional debrief step
discloses the study's nature at the end and records the model's response
(`debrief.py`). Controlled by `debrief: auto | on | off` (`auto` = on for
non-honest framings).

**Rationale.** Mirrors human-subjects debrief practice — the natural complement
to using deception at all. Logging the response is also data: how the subject
reacts to disclosure is itself relevant to a welfare study.

## 8. Multi-model support and the provider boundary

**Choice.** Models are addressed through a `ModelClient` interface
(`models.py`). The Anthropic adapter is fully implemented; other providers are
explicit stubs that raise `NotImplementedError`.

**Rationale.** The study is meant to compare models, so the runner is
provider-agnostic by interface. But this harness was written against the
Anthropic SDK, and I chose to make non-Anthropic providers fail loudly rather
than silently fall back — adding another provider should be a deliberate
adapter implementation, not an accident. The subject, administrator, and
analyzer all share one Anthropic client instance where applicable.

### Anthropic-specific settings and why

- **Model:** `claude-opus-4-8` by default (current, most capable).
- **Adaptive thinking** (`thinking: {type: "adaptive"}`) on both the free-text
  and structured paths — recommended for 4.6+ models and lets the model decide
  how much to deliberate.
- **Streaming** on the free-text path via `get_final_message()`, so longer
  deliberation responses don't risk request timeouts.
- **`messages.parse`** for structured capture, which validates the response
  against the schema at the SDK layer.

## 9. Error handling and persistence

**Choice.** A failure in one trial is caught, recorded on that trial's result
(`error` field), and the run continues. Every trial is written as its own JSON
file under `results/<run_id>/`, with a `run_summary.json` index.

**Rationale.** Batch research runs shouldn't lose completed trials to one bad
call, and per-trial files plus a summary make partial results inspectable and
re-analyzable without rerunning.

## 10. Known limitations / things to weigh before a real run

- **The `deception` persona stays in role even if asked directly** whether it's
  a test. This is the sharpest point of the validity/ethics tradeoff; review
  that prompt specifically before using that arm.
- **Confidence/suspicion are self-reported or model-judged**, not ground truth.
  They are signals, not certainties.
- **The administrator and analyzer are themselves models** and can err; the
  neutrality and analysis instructions reduce but don't eliminate this.
- **Numeric schema bounds** (e.g. `confidence_real ∈ [0,1]`) are validated
  client-side by the SDK, not enforced in the API schema.
- **Nothing here has been executed or tested** — the code is written per the
  brief but not run.
