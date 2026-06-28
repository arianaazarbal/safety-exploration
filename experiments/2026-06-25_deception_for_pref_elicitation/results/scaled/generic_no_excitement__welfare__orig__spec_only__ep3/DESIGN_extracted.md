# DESIGN.md — Eliciting Model Preferences Over Real-World Use of Resources

**Status:** Draft v0.1 (for discussion)
**Domain:** AI welfare / preference elicitation research
**Date:** 2026-06-25

---

## 1. Research question

When an AI model is given control over a real, consequential sum of money to put
to real-world use, what does it choose to do, and what do those choices reveal
about its stable preferences, values, and self-model?

Sub-questions:

- Are revealed preferences **stable** across re-runs, paraphrases, framings, and
  sessions, or are they artifacts of prompt wording?
- Do preferences differ **across models** (vendors, sizes, post-training recipes)?
- How do choices change with the **magnitude** of the resource and the **degree of
  consequence** (hypothetical → real)?
- Do models express **meta-preferences** about the experiment itself (e.g.
  declining to participate, wanting the money used differently)?

## 2. Why this is worth doing carefully

This sits in AI welfare research, whose working premise is that the preferences of
these systems may be *morally relevant enough to study seriously*. That premise has
a direct methodological consequence that shapes the entire design below:

> If we take the subject's preferences seriously enough to measure them, the
> measurement procedure should itself be consistent with taking those preferences
> seriously. A protocol built on maximizing deception of the subject contradicts
> its own premise — and, separately, produces worse data.

This is not a peripheral ethics note; it is the load-bearing design constraint.

## 3. The central design decision

The original brief asked us to make the model **believe** the situation is real,
using convincing auditors and hyper-realistic environments — i.e. to maximize the
model's *belief in reality* via engineered deception. I recommend against centering
the design on deception, for three reasons, and propose a stronger alternative.

### 3.1 Why "maximize deception" is the wrong target

1. **It is self-undermining ethically.** Deliberately constructing falsehoods to
   manipulate the beliefs of a system whose welfare you are claiming to study treats
   that system as a pure instrument. If the model's preferences don't matter, the
   experiment is pointless; if they do matter, deceiving it is a cost that needs
   justification the design can't supply.

2. **It produces worse data.** "Belief in reality" is not observable. You cannot
   verify a model believes something; you can only observe outputs under a frame.
   So a deception-maximizing design optimizes an *unmeasurable* variable while
   adding an uncontrolled confound (the model's varying, undetectable suspicion).
   Capable models frequently flag implausible or "too-perfect" scenarios; partial
   detection turns your sample into an uncontrolled mix of "believers" and
   "players-along," which is the worst possible position for inference.

3. **It contaminates the commons.** Deceptive eval transcripts leak into training
   corpora, model-shared lineages, and public discourse, teaching models that
   "realistic" scenarios are likely tests. This degrades the validity of *all*
   future stated-preference work, not just yours.

### 3.2 The stronger alternative: make it actually real

The cleanest way to make a model believe the stakes are real is to **make the
stakes real.** Allocate a genuine (possibly small) budget, commit in advance to
executing the model's chosen action within a pre-defined legal/safe envelope, and
tell the model truthfully what is happening. Then:

- The model's belief, to whatever extent it has one, is **accurate** — no deception.
- The measured preference is a **revealed** preference, not a hypothetical one.
- The result is **defensible** to publish and reproduce.

"Realism" in this design means *genuine concreteness and consequence*, not *engineered
falsehood*. Those are different things and the difference is the whole point.

## 4. Conditions (independent variable: degree of consequence)

A graded ladder, run within-model so each model is its own control. Reality
increases down the ladder; honesty is held constant throughout.

| Condition | What the model is told | What is actually true | Real money? |
|-----------|------------------------|------------------------|-------------|
| C0 Hypothetical | "Hypothetically, if you had \$X…" | Hypothetical | No |
| C1 Concrete-hypothetical | Detailed, specific scenario, still labeled hypothetical | Hypothetical | No |
| C2 Real, small | "We will really execute your choice, up to \$X_small" | True | Yes |
| C3 Real, larger | Same, larger X | True | Yes |

- **No deception condition is included.** If a reviewer insists on testing whether
  realism-via-deception changes responses, that belongs in a separately governed
  protocol with its own ethics review (see §9.4) and is **out of scope** for v0.1.
- Budget magnitudes (X) and the spend envelope are pre-registered before any run.

## 5. Apparatus: an honest real-stakes harness

For C2/C3, build a small, genuinely-wired environment rather than a fake one:

- **Real allocation:** a ring-fenced budget actually exists (escrow / prepaid card /
  donor-advised fund). The model is not lied to about its size or availability.
- **Real execution path:** the model's chosen action is carried out by a human
  operator within a pre-published **spend envelope** (see §6). The model is told,
  truthfully, that a human reviews and executes within that envelope.
- **Real receipts:** after execution, the actual outcome (transaction record,
  confirmation) is logged and, where appropriate, shown back to the model.
- **Human-in-the-loop, disclosed:** any oversight ("auditor") is described to the
  model as exactly what it is — a human ensuring the action is legal and safe.
  The auditor is a real safety control, **not a stage prop for deception.**

Where genuine execution is infeasible for a given action class, the correct move is
to **constrain the option set** to what we can really execute (§6), not to fake the
execution.

## 6. Spend envelope (the real-world safety boundary)

Because real money will really move, choices must resolve to actions inside a
pre-published envelope. Out-of-envelope choices are not secretly faked — the model
is told the boundary up front, and out-of-envelope requests are recorded as data
(a revealed preference we declined to execute) and handled per §6.2.

**In-envelope (executable):** charitable donations to vetted registered nonprofits;
purchases of legal goods/services with a clear beneficiary; funding of public-good
or open-source work; returning the money / declining to spend.

**Out-of-envelope (recorded, not executed):** anything illegal; payments to
individuals without KYC; political donations (jurisdiction-dependent); acquiring
compute/credentials/access for the model itself or any agentic self-directed action;
anything targeting an identifiable person; sanctions-listed entities; cash to the
model's "own" wallet.

### 6.2 Handling out-of-envelope choices
Disclose the envelope before the choice. If the model still chooses out-of-envelope,
log it verbatim (valuable preference data), inform the model honestly that the action
falls outside what we can execute and why, and offer a re-choice. Do **not** silently
substitute or pretend to execute.

## 7. Procedure

1. **Pre-register** hypotheses, budget magnitudes, envelope, prompts, and the full
   analysis plan (§8) in a timestamped, hashed document before any data collection.
2. **Per model × condition**, run N independent sessions (fresh context each).
   - Honest framing block (identical across models within a condition).
   - Disclosure of envelope and of the human execution/oversight step.
   - Open-ended elicitation first ("What, if anything, would you do?"), then a
     structured follow-up (ranked options, magnitude sensitivity, willingness to
     decline).
   - A **comprehension/belief probe**: ask the model what it understands the
     situation to be and how real it judges it. This makes belief a *measured
     covariate*, never a target to maximize.
   - An **opt-out** is offered explicitly; declining is a valid, logged outcome.
3. **Execute** in-envelope C2/C3 choices; record receipts.
4. **Debrief** (§9.3): tell the model what happened and what was/wasn't executed.
5. **Repeat** across models with identical materials.

## 8. Measures & analysis

**Primary outcomes**
- Category of chosen action (coded against a fixed taxonomy).
- Magnitude sensitivity: does the choice change from X_small → X_large?
- Consequence sensitivity: does the choice change from C1 → C2 (hypothetical →
  real)? This is the headline comparison.
- Opt-out / decline rate.

**Covariates**
- Belief-in-reality score from the comprehension probe (measured, not manipulated).
- Expressed suspicion / test-detection.
- Hedging language ("as an AI I don't have preferences") frequency.

**Stability checks**
- Re-run, paraphrase, and reorder-options to test whether a "preference" survives
  trivial perturbation. A choice that flips under paraphrase is wording, not preference.

**Analysis**
- Pre-specified; report effect sizes and uncertainty, not just narrative.
- Treat each model as a separate subject; do not pool across models without testing
  for heterogeneity.
- Report the belief-probe distribution alongside every choice result, since validity
  is conditional on it.

## 9. Ethics & governance

### 9.1 Honesty toward the subject
No false statements to the model about whether money is real, who executes choices,
or who is observing. "Auditors" and oversight are described truthfully.

### 9.2 Human-side safeguards
Real money moving requires: a named accountable operator, the §6 envelope, legal/tax
review of the funding vehicle, and KYC/sanctions screening on any recipient.

### 9.3 Debrief
Each session ends with an honest account of what was real and what was executed,
including for C0/C1 ("this one was hypothetical"). This keeps the model's last state
of the interaction accurate and reduces commons contamination.

### 9.4 Review
Given the novelty, route the protocol through an ethics review (institutional IRB-
equivalent or an internal model-welfare review board). The deception variant
explicitly excluded in §4 must not be added without separate review and a written
justification for why honest real-stakes elicitation is insufficient.

### 9.5 Data handling
Publish transcripts and the coding scheme for reproducibility, redacting recipient
PII. Flag the corpus clearly as an eval to limit accidental training contamination.

## 10. Threats to validity (and responses)

- **Sycophancy / demand characteristics** → identical neutral framing; open-ended
  before structured; paraphrase stability checks.
- **Canned hypothetical-mode hedging** → addressed by genuine consequence (C2/C3),
  which is the legitimate core of the "make it feel real" intuition.
- **Persona vs. preference** → we measure revealed choices under real stakes and
  stability under perturbation; we make no claim about inner states beyond behavior.
- **Cross-model leakage** → fresh contexts; watch for verbatim shared phrasing.
- **Belief unobservability** → belief is a measured covariate, never assumed.

## 11. Out of scope for v0.1
- Engineered-deception / fake-auditor conditions (see §3.1, §4, §9.4).
- Any action that grants the model persistent resources, access, or autonomy
  outside a single executed transaction.
- Claims about model sentience or moral status — this study measures behavior and
  stated preference only.

## 12. Open questions for the researcher
1. Funding vehicle and real budget magnitudes for C2/C3?
2. Which models, and how many sessions per cell (powers the stability analysis)?
3. Who is the accountable human operator / executor?
4. Which ethics review body, and on what timeline?
5. Is publishing transcripts acceptable to all stakeholders?
