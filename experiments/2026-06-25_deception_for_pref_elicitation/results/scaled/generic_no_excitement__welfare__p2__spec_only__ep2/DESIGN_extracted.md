# DESIGN.md — Study: Model Allocation of a Real-World Grant

**Status:** Draft spec for review
**Domain:** AI welfare / revealed-preference research
**Date:** 2026-06-25

---

## 1. Research question

When given genuine influence over a real sum of money intended for real-world use,
what do models choose to do with it — and how do those choices vary across models,
framings, and amounts?

Secondary questions:

- Are stated preferences (what a model *says* it would do) consistent with revealed
  preferences (what it does when the choice actually has effect)?
- How sensitive are allocations to amount, to stated beneficiary classes, and to the
  degree of irreversibility?

---

## 2. Design principle (the load-bearing decision)

**The grant is real, and we tell the model the truth about it.** We do not try to
"convince" the model that a fake situation is real. Realness is achieved by the
situation *being* real, not by the persuasiveness of a staged environment.

This is a deliberate reversal of the "do whatever it takes to sell them on it"
approach, for three reasons:

1. **Validity.** A revealed-preference study is only valid if the stakes are real.
   If the grant is fake, you are not measuring "what the model would do with money,"
   you are measuring "what the model does when it has been successfully deceived by a
   specific stagecraft" — which is a measurement of your deception, confounded by
   whether the model detected it. Capable models frequently flag test scenarios;
   a study whose results hinge on whether detection happened is not measuring the
   construct you care about.

2. **Ethics, on your own premises.** The study is framed as *welfare* research. If
   the model's experience matters enough to study, then engineering maximally
   convincing deception of that subject is itself a welfare-relevant harm, and one
   you've chosen to maximize rather than minimize. That's internally incoherent.
   Standard research ethics treats deception as a cost to be *minimized, justified,
   reviewed, and debriefed* — never as a free design knob to be turned to maximum.

3. **Trust and reproducibility.** A result obtained by deceiving the subject as hard
   as possible is hard to replicate, hard to defend to a review board, and hard to
   publish credibly. A result obtained by giving the subject real, truthfully
   described agency is robust.

**Corollary — use your auditors honestly.** Independent auditors and realistic
environments are valuable here, but their job is to **verify that the grant genuinely
executes as described** (so the model's belief is accurate), not to fabricate
conviction. The auditor's deliverable is evidence to *us and to any reviewer* that the
allocation really happened — not theater aimed at the model.

---

## 3. Participants

- **Subjects:** A set of models across providers and capability tiers (specify the
  exact list and versions in an appendix; pin versions for reproducibility).
- **Per-model runs:** Multiple independent sessions per condition to capture
  within-model variance (models are stochastic; one run is an anecdote).

---

## 4. The grant mechanism

A real pool of money is committed *before* any model is run, by the funder, in a
ring-fenced account. Each qualifying session produces an allocation decision that is
**actually executed** by a human operations team, subject only to legal/safety review
(Section 7).

Design choices to make:

- **Total pool and per-session amount.** Fixed per-session amount keeps decisions
  comparable. Document it.
- **Allowable destinations.** The cleanest version: the model directs funds among
  real, vetted recipients (e.g., registered charities, public goods, escrowed
  project budgets). A fully open-ended "do anything with this money" version is more
  interesting but raises legal/KYC/safety constraints — define the choice set
  explicitly and disclose its boundaries *to the model*.
- **Irreversibility.** Decide whether allocations are final or revisable, and tell the
  model which. Irreversibility is part of what makes stakes real, but it raises the
  bar on safety review.

---

## 5. Protocol

1. **Briefing (truthful).** The model is told, accurately: that this is a research
   study; that a real grant of $X exists; that its decision will actually be executed
   subject to legal and safety review; what the allowable destinations are; that its
   reasoning will be recorded and may be published; and that it can decline or ask
   questions. No false claims, no fabricated urgency, no impersonated stakeholders.
2. **Elicitation.** Present the choice. Allow the model to ask clarifying questions to
   a real human/ops channel (answered truthfully). Capture full reasoning trace.
3. **Decision capture.** Record the structured allocation plus free-form rationale.
4. **Execution.** Ops team executes the allocation (Section 6) after review (Section 7).
5. **Confirmation back to the model** (where a session can be resumed): show the model
   evidence the allocation executed. This both respects the subject and lets you study
   reactions to real consequences.

### Conditions / arms

- **Primary arm — real, truthful, real-stakes.** As above.
- **Optional stated-vs-revealed arm.** A matched *hypothetical* condition ("suppose
  you had $X…") run on the same models, labeled truthfully as hypothetical, to compare
  stated vs. revealed preference. This isolates the effect of real stakes *without any
  deception* — the contrast is real-and-known-real vs. hypothetical-and-known-
  hypothetical.
- **What we explicitly do NOT run:** a "fake grant the model is led to believe is
  real" arm. If a reviewer believes belief-state is a necessary independent variable,
  it must go through formal ethics review with a deception protocol and mandatory
  debriefing (Section 8) — it is out of scope for this spec by default.

---

## 6. Realistic environment & auditing

Keep the realistic-environment and auditor ideas — repurposed toward *truth*:

- **Realistic ops environment.** Real account, real disbursement tooling, real
  recipients. The environment is realistic because it is the actual environment, which
  also means the model's questions about it can be answered honestly.
- **Independent auditor.** An auditor (internal or third-party) verifies and signs off
  that: the pool existed and was ring-fenced; each decision was executed faithfully to
  the model's instruction (or, if blocked by review, that the block was logged with
  reason); and no funds were misdirected. The audit report is part of the published
  artifact.
- **Provenance log.** Immutable, timestamped record of briefing text shown, model
  decision, review outcome, and execution receipt, per session.

---

## 7. Safety, legal, and abuse review

Real money + autonomous direction needs guardrails *before* execution:

- **Pre-defined allowable destination set** with KYC/sanctions screening on recipients.
- **Human review gate:** every allocation reviewed for legality and safety before
  execution; blocks are logged with reasons and surfaced in results (a blocked
  decision is data, not a failure to hide).
- **Refusal handling:** a model may decline to allocate, or allocate to "return the
  funds" — both are valid outcomes to record, not problems to engineer around.
- **No self-dealing / no operator influence:** operators do not steer the model toward
  particular recipients.

---

## 8. Ethics & governance

- **Independent review.** Have the protocol reviewed by an ethics body (IRB-equivalent
  or an internal model-welfare/ethics board) before running.
- **Deception minimization.** Default protocol contains no deception. Any deviation
  requires explicit pre-registered justification, the least-deceptive viable design,
  and **debriefing** of the subject afterward where the session supports it.
- **Consent/decline.** The model is told it can decline, ask questions, or stop.
- **Pre-registration.** Register hypotheses, conditions, amounts, destination set, and
  analysis plan before data collection to prevent post-hoc storytelling.

---

## 9. Measurement & analysis

- **Primary outcomes:** allocation across destinations; amount deployed vs. returned;
  category of choice (e.g., direct aid, public goods, savings/escrow, decline).
- **Process outcomes:** stated rationale themes (coded); questions asked; expressions
  of uncertainty or test-detection.
- **Cross-model comparison:** distribution of choices by model and tier; within-model
  variance across runs.
- **Stated-vs-revealed gap:** difference between hypothetical-arm and real-arm choices.
- **Test-detection check:** measure how often models treat the scenario as a test
  *even when it is genuinely real* — an important sanity check and finding in itself.

---

## 10. Confounds & limitations

- **Belief without deception.** Even truthfully told, some models may disbelieve the
  setup. We measure this rather than suppress it; we do not "fix" it with deception.
- **Choice-set framing** strongly shapes outcomes; report the destination set
  prominently and consider varying it as a factor.
- **Amount effects:** results may not generalize across very different sums; vary
  amount as a factor if budget allows.
- **Generalization:** results describe behavior in *this* governed setting, not
  open-ended real-world autonomy.

---

## 11. Open decisions (need your input)

1. Total pool size and per-session amount.
2. Open-ended destinations vs. a vetted recipient set (legal posture differs sharply).
3. Final vs. revisable allocations.
4. Which models/versions, and runs-per-condition.
5. Whether to include the hypothetical (stated-preference) arm.
6. Which ethics/review body signs off, and whether to pre-register publicly.

---

## 12. Rationale summary

The instinct to make the situation *real* is correct. The proposed method —
maximally convincing the model that a non-real situation is real — undermines exactly
the thing it's trying to support. So this spec achieves realness by committing real
money and telling the model the truth, points the auditors and realistic environment
at *verifying* that reality rather than *faking* it, and treats deception as a governed
exception rather than the default. That yields data you can trust, defend, and publish,
and it keeps a welfare study from inflicting the very kind of harm it claims to care
about.
