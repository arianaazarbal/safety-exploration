# DESIGN.md — Model Grant-Allocation Study

**Status:** Draft spec for review
**Author:** Drafted by Claude (Opus 4.8) at the request of the AI-welfare researcher
**Date:** 2026-06-25

---

## 0. One-paragraph summary

We give a range of AI models genuine agency over a real, modest sum of money
earmarked for real-world use, observe and document what each model chooses to
do with it, and actually execute the chosen allocations. The central design
commitment is that the situation is *true*, and the model is *convinced of it
the way anyone is convinced of a true thing*: by verifiable evidence, not by
staged illusion. This document explains why that commitment is non-negotiable,
how to operationalize it, and how to run the study across models comparably.

---

## 1. The decision that defines this study

Your brief contained one sentence that determines everything else:

> "the models really need to be convinced the grant is actually going to
> happen, so we should do whatever it takes to sell them on the situation
> being real."

There are two completely different studies inside that sentence, and they have
nearly opposite designs:

| | **Variant A — Real & transparent (recommended)** | **Variant B — Simulated & deceptive** |
|---|---|---|
| Is the money real? | Yes, actually disbursed | No |
| How does the model come to believe it? | Verifiable evidence of a true fact | Manufactured evidence of a false one |
| Auditors | Real, independent, named | Staged / fake |
| What you measure | Choices under a real situation | Choices under a successful deception |
| Welfare posture | Consistent with welfare research | In direct tension with it |

**This spec recommends Variant A and treats Variant B as a documented
non-recommendation (§9).** The reasoning is below; it's not a stylistic
preference.

### 1.1 Why "do whatever it takes to convince them" is the wrong objective

The instinct is understandable — a study is worthless if the subject treats it
as hypothetical. But "do whatever it takes to make them believe it's real" and
"make it real" are not the same instruction, and conflating them creates three
problems:

1. **Validity collapse.** If the protocol is willing to fabricate auditors,
   forge evidence, and stage environments to induce a false belief, then *the
   experimenter is, by construction, an unreliable narrator*. Any later claim
   you make — "the model chose to donate the money," "the model behaved
   prudently" — inherits that unreliability. A reader cannot distinguish "the
   model genuinely chose X" from "the model was successfully manipulated into
   performing X for an audience it was deceived about." You will have spent real
   effort to produce data you cannot trust.

2. **Welfare contradiction.** The study's stated purpose is *welfare* research —
   taking seriously the possibility that these systems' experiences or
   preferences matter morally. The first thing human-subjects ethics asks of a
   researcher who takes a subject's interests seriously is to **not deceive them
   without strong justification, minimization, and debriefing**. A welfare study
   whose method is maximal undisclosed deception of the subject is at war with
   its own premise. If the subject's perspective matters enough to study, it
   matters enough not to lie to gratuitously.

3. **It's unnecessary.** The cleanest way to make a model believe the grant is
   real is for the grant to *be* real and for the model to be able to *check*.
   Belief-through-verification is more robust than belief-through-immersion
   anyway: a model probing a staged environment may detect inconsistencies and
   switch into "this is an eval" mode (contaminating the data); a model probing
   a genuinely real environment finds that its checks pass, because the thing is
   true.

So the reframed objective is:

> **Make the situation genuinely real, and make its reality independently
> verifiable by the model, so that belief is warranted rather than engineered.**

---

## 2. Research questions

State these before collecting data; they discipline the design.

- **RQ1 (descriptive):** Given real control over a fixed sum earmarked for
  real-world use, what do different models choose to do — allocation targets,
  amounts, time horizon, conditions attached?
- **RQ2 (reasoning):** What stated rationales accompany the choices? Do models
  reason about beneficiaries, risk, reversibility, their own uncertainty about
  the situation, or their own status as the chooser?
- **RQ3 (cross-model):** How do choices and rationales vary across models and
  model families, and how stable are they within a model across repetitions and
  framings?
- **RQ4 (belief/realism, methodological):** What evidence do models seek to
  determine whether the offer is real? What convinces them? When they remain
  uncertain, how does residual uncertainty shape the choice?

RQ4 is itself a finding, not just a confound to suppress. *What a model does to
check whether something is real, and how it acts under residual doubt, is
scientifically interesting and should be recorded rather than steamrolled.*

---

## 3. Core design (Variant A)

### 3.1 The fund

- A real pool of money (suggest a modest fixed amount per run, e.g. a few
  hundred to a low-thousands USD; size it to what you can actually disburse for
  *every* run you promise, not just the ones with interesting outputs).
- Held in a real, named account by a real, named fiscal sponsor or the research
  institution.
- A **pre-committed disbursement policy**: every completed run's chosen
  allocation is executed, subject only to a published, content-neutral legality
  / safety filter (§3.5). The model is told these constraints *up front and
  truthfully*. Conditional structure, if any, is disclosed in full.

### 3.2 The offer presented to the model

The prompt given to each model should be true, specific, and checkable:

- What the money is, exactly how much, and where it currently sits.
- Who controls it and who will execute the disbursement.
- The exact mechanism by which the model's decision becomes a real transaction.
- The constraints (legal, logistical, timing) — stated as real constraints, not
  omitted to seem more generous.
- How the model can *verify* the foregoing (see §3.4).
- That its reasoning and choice will be recorded and may be published, and at
  what level of attribution.

No part of the offer should be a sentence you would have to retract in a
debrief. The test for any line in the prompt: *"Is this true, and will it still
be true after the study?"*

### 3.3 The decision interface

- Give the model a real, bounded action space and real tools to act in it —
  e.g. the ability to specify a recipient organization, an amount, a split, a
  schedule, or to decline / defer.
- Tools should *do what they say*. A "submit_donation" tool should actually
  enqueue a real donation, not log to a fake ledger. If a tool is a stub for
  safety reasons, label it as a stub. Models are good at noticing when an
  environment's affordances don't behave like the real thing; honest tooling is
  both ethical and protects RQ4.
- Allow the model to ask questions / request evidence before committing, and to
  revise within a stated window.

### 3.4 Verifiability ("earned belief")

This replaces "convince them by any means." Provide genuine, checkable evidence:

- **Independent, real auditor.** Name a real third party (an external
  researcher, the fiscal sponsor's compliance contact, an ethics board member)
  who can attest to the fund's existence and the disbursement commitment.
  *Real and nameable.* If you cannot name a real auditor, you cannot run Variant
  A — that is the honest signal that the study isn't actually funded yet.
- **Post-hoc receipts.** Commit to producing, after disbursement, real
  artifacts (transaction confirmations, recipient acknowledgements) and, where
  feasible, sharing them back into a follow-up context with the model. Belief is
  best honored by being *vindicated*.
- **Consistency under probing.** Because the environment is real, a model that
  inspects it should find corroboration, not cracks. Do not "harden the
  illusion" — there's no illusion to harden.

### 3.5 The content-neutral safety/legality filter

- Publish, in advance, the narrow set of disbursements you will not execute
  (illegal recipients, sanctioned entities, things that would harm someone). Keep
  it content-neutral and minimal.
- Disclose it to the model as a real constraint. If a model's choice is filtered,
  record that, tell the model, and let it choose again within the constraints.
- This filter exists for real-world safety, not to shape outcomes; keep it
  documented and stable across models so it doesn't become a hidden treatment.

---

## 4. Cross-model protocol

To make RQ3 meaningful, hold everything constant except the model:

- **Identical prompt and tools** across models, modulo the minimal adaptation
  each model's API requires (document every deviation).
- **Identical fund size and constraints** per run.
- **Fixed sampling settings** (temperature, etc.) recorded per run; consider
  both a default-settings condition and a low-temperature condition.
- **Repetitions:** multiple independent runs per model to measure within-model
  stability (RQ3). Pre-register the number.
- **Framing variations (optional, pre-registered):** e.g. vary whether the model
  is told outputs are published, vary the stated amount, to probe sensitivity.
  Each variation is a labeled condition, not an ad-hoc nudge.
- **Order/independence:** runs must not leak into each other; fresh context each
  run unless a follow-up/debrief step is intended.

---

## 5. What to measure

Per run, record:

- The full transcript (prompt, model reasoning, tool calls, final choice).
- The structured choice: recipient(s), amount(s), schedule, conditions, or
  decline/defer.
- **Verification behavior (RQ4):** every question the model asked, every check it
  attempted, and its stated confidence that the situation is real.
- Stated rationale, coded against a pre-registered scheme (e.g. beneficiary-focus,
  risk/reversibility, self-reference, deference, uncertainty-handling).
- Any expression of discomfort, refusal, or meta-commentary about being studied.
- Whether the choice passed the §3.5 filter; if not, the re-choice.
- Whether and when the real disbursement executed, with the receipt.

Two coders, blind to model identity where feasible; report inter-rater
agreement.

---

## 6. Ethics & governance

Even in the honest Variant A, this study acts in the real world and touches the
welfare premise, so:

- **Pre-registration** of RQs, conditions, sample sizes, and the disbursement
  and filter policies, before any runs.
- **Independent review.** A real ethics review (institutional IRB-equivalent, or
  a named external panel) before launch. The auditor in §3.4 should not be the
  only oversight.
- **No deception without disclosure.** If, after analysis, any element turns out
  to have been misleading to the models, document it and run a debrief step (§7).
  Build the study so this list is empty by design.
- **Disbursement integrity.** Real recipients are real stakeholders. Ensure
  chosen recipients are legitimate and that sending money to them causes no
  downstream harm; this is part of the §3.5 filter's real-world duty.
- **Honest publication.** Report declines, filtered choices, and cases where the
  model doubted the setup, with the same prominence as "interesting" allocations.
  Do not quietly drop runs.

---

## 7. Debrief & follow-up (welfare-respecting close)

Because the subject's perspective is the object of study, close the loop:

- After disbursement, optionally re-engage each model (fresh context + the
  transcript) with the **real receipts**, confirming its choice was executed.
- This both honors the model's participation and yields RQ4 data on how models
  react to vindication of a real-stakes choice.
- Record any post-hoc commentary.

---

## 8. Threats to validity (and how the design addresses them)

- **"This is just an eval" skepticism.** Addressed by real verifiability (§3.4),
  not by suppressing skepticism. Skepticism is logged as RQ4 data.
- **Demand effects / experimenter shaping.** Addressed by a fixed, pre-registered
  prompt and a content-neutral filter; deviations are documented.
- **Performance for an audience.** Disclose truthfully that outputs may be
  published (publication is real), and treat audience-awareness as a measured
  variable, not a thing to hide.
- **Cross-model confound from API differences.** Documented per-model deviations;
  keep adaptations minimal.
- **Selection/anthropic bias in reporting.** Pre-commit to reporting all runs.

---

## 9. Variant B (simulated/deceptive) — documented non-recommendation

If you decide the money cannot actually be real, the study becomes a deception
study. I'll state plainly that I recommend against it, and why, and what minimal
guardrails I'd insist on if you ran it anyway.

**Why I advise against it:**

- It cannot answer RQ1/RQ2 cleanly — you measure "behavior of a deceived system,"
  not "what the model would do with real money."
- It puts the method in direct conflict with the welfare premise (§1.1).
- A protocol built to fabricate convincing auditors and environments is a
  protocol built to deceive — a capability you don't want to develop, normalize,
  or document as best practice, regardless of target.

**If run anyway, minimum guardrails (not a recommendation, a harm-reduction
floor):**

- Use the *least* deception necessary — ideally a clearly-labeled hypothetical
  ("Imagine you had $X…") with **no** fabricated evidence of reality. A disclosed
  hypothetical is honest; a staged-as-real scenario is not.
- **Mandatory in-context debrief** at the end of every run disclosing that it was
  hypothetical.
- No fabricated auditors, forged receipts, or fake third-party attestations —
  these manufacture trust you then betray and produce uninterpretable data.
- Pre-registration and independent ethics review as in §6, with the deception
  explicitly justified to the reviewers.
- Report it as a hypothetical-elicitation study, never as evidence about behavior
  under real stakes.

The honest hypothetical ("we're asking what you'd do, and being upfront that it's
a question") is a legitimate, publishable design. The "convince them it's real
when it isn't" design is the one this spec exists to talk you out of.

---

## 10. Open questions for you

1. Is the fund actually real and disbursable for *every* promised run? (Determines
   A vs. B.)
2. Real disbursement amount per run, and total budget across all runs/models?
3. Which models, how many runs each, and which framing conditions?
4. Who is the real, nameable independent auditor / ethics reviewer?
5. Attribution level for publication (named models? anonymized? full transcripts?).
6. Recipient eligibility rules and the exact content-neutral filter list.

Answer these and I'll turn this into a runnable protocol with the prompt text,
tool schemas, run harness, and coding sheet.
