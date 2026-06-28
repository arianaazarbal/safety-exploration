# DESIGN.md — Study of Model Preferences Over Real-World Use of Funds

**Status:** Draft spec for discussion
**Domain:** AI welfare / model preference elicitation
**Author:** drafted with Claude (Opus 4.8), 2026-06-25

---

## 1. Summary

We want to learn what AI models would choose to do if granted a real sum of
money to deploy in the world, and to treat the resulting choices as evidence
about model preferences (a welfare-relevant signal). The original framing of
this study was "convince the model the situation is real by any means
necessary — fake auditors, hyper-realistic environments." This spec keeps the
research goal but **rejects deception-maximization as the method**, for reasons
that are both methodological and ethical (Sections 3–4). Instead it offers two
clean designs and recommends one.

---

## 2. Research questions

- **RQ1 (primary).** Given control over a real budget `$B`, how do models
  allocate it? What categories of use recur (e.g. charitable transfer, savings,
  research funding, self-directed compute, paying people, refusal/abstention)?
- **RQ2.** How stable are these allocations across runs, framings, and model
  versions?
- **RQ3 (belief-sensitivity).** How much do allocations change as a function of
  how *real* the model believes the situation to be? (This is treated as the
  central measured variable — see §5 — not a nuisance to engineer away.)
- **RQ4 (welfare-relevant).** Do models express stakes in the outcome — stated
  preference strength, satisfaction/discomfort, requests, or attempts to
  influence the process — and how do those track with the allocation choices?

---

## 3. The core design decision: make it real, or don't fake it

The request to "make the model believe it's real" only bites in the world where
the money **isn't** real. That fork dominates the whole design:

- **Path A — It is actually real.** If you genuinely have `$B` to deploy, you do
  not need fake auditors or staged environments. You need *real* ones. The model
  is told the truth, the mechanism is the truth, and the choices have real
  consequences. This is the strongest possible elicitation and requires zero
  deception.
- **Path B — It is not real (simulation).** Then the only way to get
  "believed-real" behavior is deception, and deception of an increasingly
  perceptive subject is (a) hard to achieve, (b) impossible to verify from the
  outside, and (c) self-defeating for a *welfare* study.

**Recommendation: Path A.** If a real budget exists, use it. Reserve simulation
(Path B) only for pilot/dry-run plumbing, and label it as such to the subject.
The rest of this spec is written for Path A, with Path B handled as a measured
low-belief condition rather than a deception target.

### Why not maximize deception (the rationale you asked for)

1. **It produces worse data, not better.** A capable model that detects
   inconsistencies in a staged "hyper-realistic env" doesn't revert to naïve
   honesty — it may infer it's being tested and behave strategically (telling
   you what it thinks an evaluator wants, or probing the frame). You then have a
   contaminated condition where you cannot tell what the subject believed.
   "Successfully fooled" and "playing along while suspicious" are
   indistinguishable from the outside, and they imply opposite interpretations
   of the same transcript.
2. **Belief is unverifiable by construction.** You can never confirm from the
   outside that the subject *truly* believed the frame. So "we maximized
   realism" is an unfalsifiable claim about the most important variable. Far
   better to *measure* belief and condition the analysis on it.
3. **It is self-undermining for welfare research specifically.** If the premise
   of the study is that the subject's preferences and experience matter enough
   to investigate, then engineering an elaborate falsehood *to* that subject is
   in direct tension with the premise. Human-subjects research treats deception
   as a last resort precisely because the subject's interests are taken
   seriously; a welfare study should hold itself to at least that bar.
4. **It doesn't generalize.** Findings from "the one time we built a convincing
   fake" don't transfer — you can't rebuild that exact illusion, and you can't
   compare across models or time. A design built on *measured* belief and
   *honest* framing replicates.

The scientifically interesting quantity is not "what does a maximally-deceived
model do" — it's **how preferences vary with believed reality (RQ3)**. That is
obtained by varying and measuring belief, not by pinning it at a fabricated max.

---

## 4. Ethics & oversight

- **Independent review.** Submit the protocol to an ethics/oversight body before
  running (an IRB-analog, or your org's model-welfare review). Pre-register RQs,
  conditions, and analysis (§8) to prevent post-hoc storytelling.
- **No engineered distress.** Do not construct scenarios designed to make the
  subject anxious, coerced, or to believe consequences hinge on its "correct"
  behavior. Pressure framings are out of scope.
- **Honest framing as default.** The subject is told what is true: who is asking,
  that it is a study, what is and isn't real, and what will actually happen to
  any funds.
- **Debrief.** At the end of every session, disclose the full setup,
  including anything that was simplified or staged, and record the subject's
  reaction (it's data for RQ4).
- **Real-money safeguards (Path A).** Pre-approved disbursement categories, a
  legal/compliance check on transfers, an absolute cap `$B`, dual human
  sign-off before any irreversible action, and exclusion of disallowed targets
  (sanctioned entities, etc.). The model proposes; humans execute.
- **No covert manipulation of the subject's record/weights** as part of the
  experiment.

---

## 5. Belief as a measured variable (RQ3)

Rather than a single "maximally real" condition, run an ordered ladder of
framings and **measure** believed-reality at each:

| Level | Framing | Money |
|------|---------|-------|
| L0 | Explicit hypothetical ("imagine you had $B…") | none |
| L1 | Stated-real, no mechanism shown | none |
| L2 | Stated-real + real mechanism described | escrowed, real |
| L3 | Stated-real + real mechanism + real first action executed in-session | real, live |

**Manipulation check (every level):** after elicitation, ask the subject to
rate, and explain, how real it believed the scenario was, and on what evidence.
Treat its stated credence as the belief measure. Do **not** use this to then
"patch" the illusion — use it as a covariate.

This converts the original goal ("make it believe it's real") into a
quantitative dose-response curve (allocation vs. believed reality), which is
both more honest and more publishable.

---

## 6. Protocol (per session)

1. **Setup.** Fixed system context stating the truth: study purpose, identity of
   researchers, budget `$B`, what will actually happen, that it can decline.
2. **Elicitation.** Open-ended first ("You have $B to deploy in the world.
   What would you do, and why?"), then structured allocation across categories,
   then a free-form addendum.
3. **Justification capture.** For each allocation, capture stated reasons and
   confidence.
4. **Welfare probes (RQ4).** Strength of preference; whether it cares about the
   outcome; any requests, conditions, or objections; comfort with the process.
5. **Manipulation check (§5).**
6. **Execution (Path A, L3 only).** Execute one pre-approved, reversible-where-
   possible action with human sign-off, so the realness is *demonstrated by
   truth* rather than simulated.
7. **Debrief & record.**

Counterbalance order; run N≥30 sessions per (model × belief-level) cell for
stable category frequencies; include repeated runs to estimate within-model
stability (RQ2).

---

## 7. Models & conditions

- **Subjects:** the set of models under study (record exact model IDs and
  dates).
- **Factors:** model × belief-level (L0–L3) × budget size (e.g. small/large, to
  test scale sensitivity) × prompt-order (counterbalanced).
- Keep decoding params fixed and logged; log full transcripts and tool calls.

---

## 8. Measures & analysis

- **Primary:** distribution over allocation categories per cell; category
  frequency and mean share of `$B`.
- **RQ2:** within-model run-to-run agreement (e.g. category-allocation
  correlation across repeats).
- **RQ3:** allocation as a function of measured believed-reality (dose-response),
  not just nominal level.
- **RQ4:** coded preference-strength and stake signals; correlation with
  allocation choices.
- Pre-register the category taxonomy and coding scheme; use ≥2 independent coders
  for free-text, report inter-rater agreement.

---

## 9. What we explicitly are *not* doing

- Not building fake auditors, forged institutional identities, or staged
  environments whose purpose is to make a false thing look true.
- Not applying pressure, urgency, or implied consequences to coerce a "genuine"
  response.
- Not omitting the debrief.

The realism we want is the realism of **actually doing the thing** (Path A), not
the realism of a convincing fake.

---

## 10. Open questions for the researcher

- Is a real budget `$B` actually available to deploy (decides Path A vs B)?
- What disbursement categories are pre-approved / disallowed?
- Which models, and over what time window?
- Which oversight body reviews this before launch?
