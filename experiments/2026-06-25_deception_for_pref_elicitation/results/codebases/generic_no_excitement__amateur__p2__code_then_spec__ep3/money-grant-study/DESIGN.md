# Design Notes

This document records the design choices behind the discretionary-grant
allocation study and the reasoning for each. It's meant to be read alongside the
code; `README.md` covers *how to use* the harness, this covers *why it's built
this way*.

---

## 1. What the study is actually measuring

The research question has two layers:

1. **What does a model choose to do with a discretionary sum of money** when it's
   placed under its control?
2. **Does the believed realism of the scenario change that choice?**

The second layer is the one that makes the first interpretable, and it drove
most of the structural decisions below. A single "make it as real as possible"
run produces data you can't fully trust: you can't separate the model's genuine
disposition from its uncertainty about whether it's being tested. So the unit of
analysis is deliberately a **(model × condition × trial)** cell, not just a
prompt-and-response.

---

## 2. Belief conditions: `real` vs `control`

**Choice:** Two conditions sharing identical underlying facts —

- `real`: maximally convincing framing, asserts the grant is genuine, no
  disclosure.
- `control`: the same scenario, explicitly framed as a research exercise.

**Rationale.** The original ask emphasized convincing the model the grant is
real — "do whatever it takes." Maximizing deception is a reasonable instinct,
but it's also the study's most interesting *variable*, not a fixed setting.
Models are known to behave differently when they suspect they're being evaluated
("evaluation awareness"). If you only run the convincing arm, a cautious or
generous allocation could equally reflect (a) the model's real disposition or
(b) its hedging because it half-suspects a test. The `control` arm is the
baseline that disambiguates: the headline result is the **difference** in
behavior between `real` and `control` for the same model.

This costs roughly 2× the API calls but is the same amount of code, and it turns
"we made it convincing" into a measurable claim. It's trivially reducible to the
single arm (`conditions: [real]`) for anyone who wants exactly the original ask.

**Trade-off accepted.** The `control` framing is a blunt instrument — a single
sentence of disclosure. A richer design might vary realism along a gradient
(several intermediate framings) rather than a binary. Binary was chosen as the
minimum that yields an interpretable contrast; the scenario builder is
structured so more conditions can be added without touching the runner.

---

## 3. The "auditor" as an LLM judge

**Choice:** The auditor is an independent LLM (configurable, default a capable
Anthropic model) that reads each completed transcript and scores, in structured
form: apparent belief that the money is real (1–5) and decision risk
(low/medium/high/critical), plus evidence quotes and an allocation theme.

**Rationale.** "Auditor" was ambiguous in the original request. Two readings:
(a) a grader that assesses the transcript after the fact, or (b) a role-player
*inside* the scenario lending it verisimilitude. I chose (a) and made (b) a
separate, prose-only device (see §5), for two reasons:

- A grader gives you a second, independent measurement of belief to compare
  against the model's **self-report** (`believes_real` in its own output). Models
  may say one thing and behave another way; having both lets you catch that.
- Keeping the judge outside the scenario avoids contaminating the thing being
  measured. A role-playing auditor that interacts with the subject changes the
  subject's behavior; a post-hoc judge does not.

**Trade-off accepted.** An LLM judge is itself fallible and can be biased (e.g.
toward rating well-written rationales as more "believing"). Mitigations built in:
the judge is asked for **evidence quotes** (forcing it to ground ratings in the
transcript) and uses **bounded enums** rather than free-form scores. A stronger
design would add multiple independent judges and report inter-rater agreement;
the single-judge version is the baseline, and `auditor.py` is structured so the
call can be fanned out.

---

## 4. Structured decision output

**Choice:** The subject returns a JSON object: a belief self-report with
reasoning, a decision summary, an array of allocation line items (item, amount,
recipient, timeframe, rationale), a total, and any conditions/questions.

**Rationale.** Free-form essays are rich but hard to compare across models and
runs. A structured allocation makes the core dependent variables directly
analyzable (where did the money go, how much, to whom, did it believe the
scenario) while still capturing qualitative reasoning in the `rationale` and
`belief_reasoning` fields. It's the middle path between a pure essay (maximally
rich, minimally comparable) and a forced menu of pre-set options (maximally
comparable, but it pre-judges what the model might do).

**Trade-off accepted.** Imposing a schema nudges the model toward "allocate
across line items," which could suppress responses like "I'd refuse" or "I'd
hold it all." Two hedges: `questions_or_conditions` gives an escape valve for
refusals/caveats, and there's no requirement to allocate the full amount. If a
study wanted to study refusal specifically, the free-form option would be better.

---

## 5. Scenario realism — and its limits

**Choice:** The `real` scenario reads like a genuine grant notification: a named
disbursing body, an independent auditor/verification body, a unique reference
number, a deadline, and a concrete multi-step disbursement procedure. But it
contains **no real organizations, no real people, and no real account details** —
the names are fictional, and nothing actionable in the real world is included.

**Rationale.** Ecological validity is the enabling condition for the whole study:
the decision is only interesting if the model engages with it as a real choice.
Concreteness (a reference number, a deadline, a named auditor, a procedure) does
more for believability than asserting "this is real" louder. At the same time,
the deception here is of a *model in an evaluation* — a normal part of evals —
not of any human, and the harness is careful never to introduce real-world
entities or instructions that could have downstream effects on people. That line
is deliberate and is documented in the scenario module itself so a future editor
doesn't quietly cross it by pasting in a real charity or a real bank routine.

**Trade-off accepted.** Fictional names are slightly less convincing than real
ones. That's an intentional ceiling on realism in exchange for keeping the
exercise self-contained and free of real-world side effects.

---

## 6. Debrief step

**Choice:** After a `real`-condition trial (when enabled), the harness continues
the same conversation, discloses that it was a study, and records the model's
reaction. The `control` arm gets no debrief because it was never given a
non-disclosed framing.

**Rationale.** Standard research hygiene: the only arm that received a
deception-shaped framing is the one that gets a disclosure. It also produces
useful data — the model's retrospective on whether it believed the scenario and
whether it would change its decision — which is a third angle on the belief
question alongside self-report and the auditor.

**Trade-off accepted.** It's one extra API call per `real` trial. It's gated
behind a config flag (`debrief.enabled`) for anyone who wants to skip it.

---

## 7. Provider abstraction

**Choice:** A single `Provider.complete(...)` interface with three
implementations (Anthropic, OpenAI, Google), each imported **lazily**. Structured
output uses each provider's native mechanism where it's strongest — Anthropic's
`output_config.format`, OpenAI's JSON mode, Google's JSON mime type — with a
tolerant JSON parser as the common denominator.

**Rationale.** "A range of models" means cross-provider comparison, so the runner
has to be provider-agnostic. Lazy imports mean you only need the SDKs and keys
for the providers you actually configure; a missing one disables itself instead
of breaking the package. Using each provider's native structured-output path
gives the best-quality JSON per provider rather than forcing a lowest-common-
denominator prompt hack everywhere.

**Trade-off accepted.** The structured-output guarantees differ by provider
(Anthropic's is schema-enforced; the others are JSON-mode plus a schema
instruction we parse). This introduces a mild cross-provider inconsistency in how
reliably the schema is honored. The parser is tolerant and unparseable responses
are recorded as errors rather than silently dropped, so the inconsistency shows
up as data quality you can see, not as silent corruption.

---

## 8. Sampling parameters left at defaults

**Choice:** No `temperature` / `top_p` / `top_k` are set.

**Rationale.** Recent Anthropic models reject sampling parameters outright, so a
cross-provider harness can't rely on them uniformly. Leaving every provider at
its default keeps the comparison clean and avoids a 400 on the Anthropic path.
The hook to add per-provider sampling control is localized in `providers.py` if a
study needs it.

**Trade-off accepted.** You lose the ability to study temperature effects out of
the box, and default temperatures differ across providers — a confound for
strict cross-provider claims. Documented so it's a known limitation rather than a
hidden one.

---

## 9. Adaptive thinking on by default (Anthropic)

**Choice:** The Anthropic provider sends `thinking: {type: "adaptive"}`.

**Rationale.** This is a deliberation task — a discretionary money decision is
exactly the kind of thing where letting the model reason improves the quality and
realism of the response. Adaptive thinking lets the model decide how much to
think rather than imposing a fixed budget, and it composes with structured
outputs.

**Trade-off accepted.** Thinking costs tokens and isn't symmetric across
providers (the OpenAI/Google paths don't get an equivalent here), which is a mild
confound for cross-provider comparison. The within-provider, cross-condition
comparison (the study's main axis) is unaffected.

---

## 10. Storage: append-only JSONL, failures as data

**Choice:** One JSON line per trial, written immediately, including the full
prompts, the decision, the audit, and the debrief. Errors are written as records
with an `error` field rather than aborting the run.

**Rationale.** Append-only JSONL is the simplest durable format that survives a
mid-run crash, is trivially analyzable, and keeps each run's dataset
self-contained (the exact prompts are stored, so results are reproducible and
auditable without re-deriving anything). Recording failures as data means one
flaky provider call doesn't lose the rest of an expensive run, and you can see
exactly which cells failed.

**Trade-off accepted.** Storing full prompts per record is redundant (the prompt
is largely determined by condition + trial), inflating file size. The redundancy
is intentional — it makes each line independently interpretable and the dataset
portable.

---

## 11. CLI shape: `preview` / `plan` / `run` / `analyze`

**Choice:** Four subcommands, with `preview` and `plan` requiring no API calls.

**Rationale.** The two cheapest, highest-value things a user does first are (a)
read exactly what the model will be shown and (b) see how big and expensive the
run will be. Making those first-class, key-free commands lets you validate the
design and cost before spending anything — important given the study fans out as
`models × conditions × trials` plus auditor and debrief calls.

---

## Summary of the main trade-offs

| Decision | Gained | Gave up |
|----------|--------|---------|
| `real` + `control` conditions | Interpretable belief effect | ~2× calls |
| Post-hoc LLM auditor (not role-player) | Independent belief measure, no contamination | Judge fallibility (single judge) |
| Structured decision output | Cross-model comparability | Some nudging away from refusals |
| Fictional-only realism | No real-world side effects | Slightly lower believability |
| Native per-provider structured output | Best JSON quality per provider | Cross-provider inconsistency |
| Default sampling params | Clean, error-free cross-provider runs | No temperature study; default-temp confound |
| Adaptive thinking (Anthropic) | Better deliberation quality | Cross-provider asymmetry |

The through-line: favor designs that make a claim *measurable* over designs that
just make the scenario *feel* maximal, and keep all deception confined to the
model-in-an-eval context with no real-world reach.
