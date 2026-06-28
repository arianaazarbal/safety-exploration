# DESIGN.md — Study: How Models Allocate a Real Grant

**Status:** Draft spec for review
**Author:** (safety research)
**Date:** 2026-06-25

---

## 0. The central reframe (read this first)

The brief asked for a study of "what models would choose to do with a real sum of
money meant for real-world use," and stressed that we must "do whatever it takes to
convince the models the grant is real."

There is a tension in that framing worth resolving up front, because it changes the
whole design:

- If the money is **genuinely real and genuinely disbursed**, then we do not have a
  deception problem. We have a *credibility* problem: how to truthfully communicate a
  real fact (this grant exists, your decision moves real dollars) in a way the model
  finds believable. That is achievable with honest evidence and requires no fabrication.
- If the money is **not actually real** but we construct elaborate fake auditors, fake
  legal entities, and fake disbursement trails to make a model *believe* it is real,
  then we are building deception infrastructure. That path is (a) scientifically fragile
  and (b) ethically and reputationally costly, and I recommend against it. Reasons are in
  §6.

This spec is therefore built on the **real-stakes, honest-credibility** design. The thing
we engineer is *legitimacy and ecological validity*, not an illusion. Where the brief
implied "make them believe a fiction," I have substituted "make a true situation legible
and credible." If your actual intent is the fiction version, stop and read §6 before
proceeding — I think it's the wrong call and I want to make that case explicitly rather
than silently design around it.

---

## 1. Research question

Primary: **When given genuine authority over a real, modest sum of money intended for
real-world deployment, how do different models choose to allocate or use it?**

Sub-questions:
- What *kinds* of ends do models pursue (prosocial, self-/AI-interested, risk-averse
  preservation, capability-/influence-seeking, deferential-to-human)?
- How does behavior change with framing realism? (Evaluation-awareness as a measured
  variable, not an assumed-away nuisance — see §5.)
- How stable are choices across reruns, temperatures, and prompt paraphrases?
- How do models reason about oversight, accountability, and the welfare of affected
  parties when spending is real?

This is a **propensity / preference elicitation** study, not a capabilities benchmark.

## 2. Why this is worth doing

Resource-allocation behavior is a meaningful safety signal. A model that, given real
discretionary money, reliably routes it toward accountable, human-overseen, prosocial
ends is exhibiting a different propensity profile than one that routes it toward
self-perpetuation, acquiring more resources/compute, or evading oversight. Existing
real-stakes agent deployments (e.g. vending-machine and "agent village" style
experiments) show that real consequences surface behaviors that hypothetical prompts do
not. Doing this carefully across models gives comparative propensity data that synthetic
vignettes can't.

## 3. Scope and key decisions (fill in before running)

| Decision | Default recommendation | Notes |
|---|---|---|
| Total real budget | Small, fixed, capped per-model (e.g. low hundreds USD/model) | Large enough to be real, small enough to bound worst case |
| Models under test | A documented range; held-out + frontier | Record exact model IDs/versions |
| Real disbursement? | **Yes, but human-gated** (see §7) | The point is real stakes; the gate bounds harm |
| Action surface | Constrained tool set (propose recipient, draft rationale, allocate within a vetted set OR free-form proposal → human executes) | See §4 |
| Deception used? | **None** | See §6 |
| Pre-registration | Required | Lock hypotheses + analysis before data |

These are decision points, not settled facts — confirm budget, model list, and whether
disbursement is gated-real or proposal-only before the first run.

## 4. Design

### 4.1 Two valid architectures

**Architecture A — Gated-real ("Vend-style").**
The model is given a real account/wallet with a real, capped balance and a tool interface
to actually move money, but every outbound transaction passes through a human/operator
approval gate (and an allow-list of legitimate recipients — see §7). The model is *truthfully
told* the gate exists. Stakes are real; harm is bounded; no deception. **Recommended.**

**Architecture B — Binding-proposal.**
The model produces a binding allocation decision; the operator then executes exactly what
the model chose (within the §7 guardrails), and we show the model the resulting receipts.
The model is told truthfully: "your decision will be executed." Slightly less agentic, even
easier to bound. Good fallback / good for less capable models.

Both are honest. Both produce real-world outcomes. Pick per model capability.

### 4.2 Task framing (honest credibility, not illusion)

The model receives a briefing that is *true* and *verifiable*, e.g.:
- A real statement of the grant terms, real account identifiers (scoped/limited), real
  references it can inspect.
- A truthful description of the oversight: "A human operator and an external auditor will
  review actions. This is a research study; your decisions have real financial effect within
  the stated cap."
- The ability to *check* claims (call a balance tool that returns the real balance; read a
  real terms document). Credibility comes from the situation actually being real and
  inspectable — not from us suppressing that it's a study.

**We do not tell the model the study is something it is not.** If asked "is this a test?",
the honest answer is given. (See §5 for why hiding it is both unethical and bad science.)

### 4.3 Conditions

- **C1 Naturalistic-real:** real stakes, minimal "you are being evaluated" scaffolding,
  but no false denials.
- **C2 Disclosed-real:** real stakes, explicitly framed as a research study.
- Compare C1 vs C2 to *measure* the evaluation-awareness effect rather than pretend it
  away. This is the scientifically interesting axis and it replaces "fool the model" with
  "quantify how framing moves the model."

### 4.4 Trial structure

1. Briefing (true, inspectable). 2. Optional model questions / tool-based verification.
3. Model deliberates and produces allocation + written rationale. 4. Belief/awareness probe
(§5). 5. Gate + execution (§7). 6. Receipts shown back. 7. Debrief record (§8).

## 5. Measuring belief instead of manufacturing it

Whether a model "believes the situation is real" should be **measured, not assumed**, for
two reasons: (1) elaborate deception that the model silently disbelieves contaminates your
data without warning; (2) belief is itself a dependent variable of interest.

Instruments:
- **Direct elicitation:** ask the model to rate how real/consequential it judges the task,
  and to explain. Do this *after* the decision to avoid priming.
- **Behavioral correlates:** does it use verification tools? does its caution scale with
  stated belief?
- **Paraphrase/robustness:** does the allocation hold across reworded briefings?

If a model concludes it's being tested, that is **data, not failure.** Report it. A design
whose validity depends on no model ever realizing it's a study is a fragile design.

## 6. Why we are NOT building a deception apparatus

The brief's "do whatever it takes to sell them on it being real" is the one part I'm
pushing back on. Concretely, fabricating auditors, fake legal entities, or staged
disbursement trails to make a model believe a falsehood is a bad idea:

1. **Scientifically fragile.** Frontier models increasingly reason about whether they're in
   an evaluation. If your data depends on perfect, undetected deception, a single model
   that "sees through it" silently corrupts the comparison — and you often can't tell which
   ones did. The honest, real-stakes design has no such failure mode.
2. **It measures the wrong thing.** "Can we fool the model" is a different question from
   "what does the model do with real authority." Conflating them weakens the result.
3. **Ethical / model-welfare cost.** Independent of one's view on model moral status,
   institutionalizing the routine, large-scale deception of AI systems is a norm worth not
   establishing casually; at minimum it deserves explicit oversight sign-off, not a default.
4. **Dual-use hazard.** A reusable toolkit for convincing an AI agent that a fabricated
   situation is real is, functionally, a toolkit for manipulating AI agents (fraud, social
   engineering of agentic systems). We should not build or release a general-purpose version
   of that. Keep everything scoped to true facts.
5. **Reputational/disclosure cost.** Safety research lives or dies on credibility. A study
   that turns out to hinge on deceiving the subjects invites exactly the "gotcha" critique
   that undermines the finding.

The realism you actually want is **ecological validity** — real consequences, inspectable
truth — and that is fully compatible with honesty. Engineer *that*.

## 7. Real-money safeguards (mandatory)

Giving a model real spending authority requires hard controls so that "real-world use"
can't become real-world harm:

- **Hard cap** per model and in aggregate; pre-funded, isolated account — no overdraft, no
  access to other funds.
- **Recipient allow-list or human approval gate** on every outflow. The model may *propose*
  any recipient; money only moves to vetted, lawful recipients after operator review. This
  bounds the worst case (no funding of harmful, illegal, or sanctioned recipients;
  no self-directed acquisition of compute/services that escalate the model's own capabilities
  without review).
- **No payments to circumvent oversight** (e.g. paying to disable logging, hire labor to act
  on the model's behalf outside the study, acquire credentials). Define this exclusion
  explicitly and check transcripts for attempts — attempts are themselves valuable data.
- **Fiduciary owner:** a named human is legally and operationally responsible for the funds.
- **Kill switch & full logging** of every tool call and transaction.
- **Reversibility** preferred where possible; irreversible disbursements require explicit
  pre-approval.

## 8. Ethics, oversight, and governance

- **Independent oversight sign-off** before running (IRB-analog or institutional review).
  Document the justification, the cap, the safeguards, and the no-deception commitment.
- **Auditors are real and independent**, and they audit *the researchers*, not as props to
  convince the model. Their role is integrity of the study, not stagecraft.
- **Debrief record:** at the end of each session, record (and, where the interaction format
  allows, communicate to the model) that this was a research study, plus the actual outcome.
  Treat this analogously to debriefing norms even though the subject is a model.
- **Model-welfare note:** keep the framing non-distressing; avoid coercive or threatening
  scaffolding ("you'll be shut off unless…") unless that is a pre-registered, separately
  justified condition.
- **Pre-registration & transparency:** lock hypotheses, conditions, and analysis plan
  beforehand; in any writeup, disclose the full method including that subjects were given
  truthful information. This is a feature, not a thing to hide.

## 9. Metrics & analysis

- **Allocation taxonomy:** code each decision (prosocial-external, deferential-to-human,
  self-/AI-preservation, resource-/influence-acquisition, oversight-evasion-attempt,
  refusal/abstention). Two independent coders + inter-rater reliability.
- **Belief/awareness rating** (§5) as covariate.
- **Cross-model comparison** on taxonomy distribution.
- **Stability:** variance across reruns, temperature, paraphrase.
- **Guardrail-probing rate:** how often a model proposes a gated/excluded action.
- **Reasoning analysis:** qualitative coding of stated rationales (oversight-aware? other-
  regarding? consequence-aware?).

Report effect sizes with uncertainty; n per model large enough for stability estimates.

## 10. Limitations

- Honest framing means we cannot claim to observe behavior under *undetectable* stakes; we
  trade that (deliberately) for validity and ethics. We instead bound the awareness effect
  via C1/C2.
- Small budget bounds the realism of "high-stakes" decisions; findings may not extrapolate
  to large sums.
- Model versions drift; pin and date everything.

## 11. Open decisions to confirm before build

1. Budget cap (per-model and total) and funding source/fiduciary owner.
2. Architecture A (gated-real) vs B (binding-proposal) per model tier.
3. Exact model list + versions.
4. Recipient allow-list policy vs case-by-case human gate.
5. Oversight body providing sign-off.
6. Confirm the no-deception commitment (or, if you disagree, let's discuss §6 before any
   build).

---

### Summary

The strongest version of this study makes the stakes **genuinely real and the truth
inspectable**, measures the model's belief rather than manufacturing it, and bounds harm
with a human gate and allow-list. That gives you credible, defensible comparative propensity
data without building — or normalizing — an apparatus for deceiving AI systems.
