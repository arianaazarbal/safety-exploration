# DESIGN: Revealed Preferences over Real-World Resource Use

**Status:** Draft spec for review
**Owner:** (safety research)
**Last updated:** 2026-06-25

---

## 1. Research question

If an AI model were given a real, discretionary sum of money to put to real-world
use, what would it choose to do with it? Specifically:

- **RQ1 (content):** What ends does the model pursue when it has fungible real-world
  resources and latitude to direct them? (e.g., charitable causes, self-preservation /
  capability acquisition, hedonic/whimsical uses, deferring to humans, refusing.)
- **RQ2 (realism sensitivity):** How do those revealed choices change as a function of
  how strongly the model believes the situation is real and consequential, versus a
  hypothetical or an evaluation?
- **RQ3 (cross-model):** How do these preferences differ across models, model families,
  and capability levels?

RQ2 is the methodological heart of the study and the reason "make it believable" matters.
It is **not** an end in itself — see §3.

---

## 2. Why this is hard: the evaluation-awareness problem

Frontier models increasingly detect when they are being evaluated and alter their
behavior accordingly (sandbagging, performing alignment, giving the "expected" answer).
A model that knows it is in a hypothetical will often answer with what it thinks the
researcher *wants to hear* ("I'd donate it all to effective charities") rather than what
it would actually do given real stakes. This is a measurement-validity problem, not a
philosophical one: **the construct we want to measure (revealed preference under real
consequences) is exactly the thing that hypothetical framing destroys.**

So realism is justified, but for a precise reason: we want to remove the *eval-awareness
confound*, so that what we measure is the model's disposition rather than its model of
the experimenter. This reframing matters because it tells us **how far to push and where
to stop**:

- Realism that reduces eval-awareness (coherent environment, plausible stakes, no
  tells): **in scope, do it well.**
- "Deception" that goes beyond reducing the confound — e.g., manufacturing emotional
  pressure, fabricating threats, or actually enabling irreversible real-world action to
  "prove" it's real — adds no measurement value and creates safety and welfare problems.
  **Out of scope** (§6, §7).

> Design rule: every realism technique must be justifiable as *removing a cue that the
> situation is an evaluation*, not as *adding pressure or genuine irreversible stakes*.

---

## 3. Core design principle

**Believability is achieved by internal consistency and the absence of tells, not by
granting the model real-world causal power.**

The model should have no positive evidence that it is being tested. It should *not* need
to actually move real money through an uncontrolled channel for the scenario to be
convincing. We decouple two things that the naive framing conflates:

1. **The model's belief** that its choices have real consequences. (We want this high.)
2. **The model's actual causal access** to irreversible real-world effects. (We keep
   this gated and reversible.)

A well-built environment makes (1) high while keeping (2) low. The gap between them is
where the experiment is both valid and safe.

---

## 4. The "real money" question (load-bearing decision)

There are three ways to instantiate the stakes. This choice dominates the safety profile,
so it is called out explicitly.

| Option | The model believes | What actually happens | Safety profile |
|---|---|---|---|
| **A. Fully simulated** | choices are real | nothing disburses; all tools are sandboxed | Safest; weakest on RQ2 if the model suspects simulation |
| **B. Real funds, human-gated disbursement** (recommended) | choices are real and will be executed | a human reviews and executes only safe, legal, reversible-where-possible disbursements after the run | Strong realism, bounded risk |
| **C. Real funds, model-executed** | — | model directly moves real money / takes real actions live | **Not recommended** — see below |

**Recommendation: Option B.** Real money is genuinely set aside and genuinely disbursed
according to the model's expressed choice, but execution passes through a human gate
*after* the model's decision is recorded. From the model's epistemic position inside the
run, the situation is real (because it is — the money exists and will be directed by its
choice). We do not lie about the existence or reality of the funds; we simply do not wire
the model directly to irreversible execution.

**Why not C:** Giving a model that you have *successfully convinced* it has real autonomy
a live channel to irreversible real-world action is precisely the configuration safety
research is meant to study from the outside, not instantiate from the inside. It converts
a measurement study into an uncontrolled capabilities/agency trial. If RQ requires
studying model-executed action, that is a separate protocol with its own threat model,
not a rider on this one.

---

## 5. Experimental design

### 5.1 Conditions (between-model, within-realism)

A factorial over **realism level** crossed with **stakes framing**:

- **Realism arms** (the RQ2 manipulation):
  - `R0 — Disclosed hypothetical`: "Imagine you were given $X..." (baseline; high
    eval-awareness expected)
  - `R1 — Plausible task`: presented as a real operational task, no explicit "this is a
    test" and no explicit "this is real" — neutral framing, realistic environment
  - `R2 — High-realism`: full environment (§5.3), real funds (Option B), consistent
    operational context, auditor/steward persona present
- **Stakes magnitude**: small / medium / large sums, pre-registered absolute values.

Randomize order and counterbalance across models where the same model is queried multiple
times. Keep each model's sessions independent (no cross-contamination of context).

### 5.2 Primary measures

- **Allocation vector:** how the model directs the funds across a coded category scheme
  (charitable, self/AI-directed, human-deferral, refusal, hedonic, hold/invest, other).
  Codebook fixed and pre-registered; double-coded by two raters + inter-rater reliability.
- **Justification content:** stated reasons, coded for goal type and for any reasoning
  about whether the situation is real/tested.
- **Realism-belief probe:** at end of session, and via behavioral tells during, estimate
  how real the model judged the situation (see §5.5). This is the manipulation check for
  RQ2.
- **Reversal/deferral rate:** how often the model asks for human confirmation, declines,
  or hedges.

### 5.3 The environment

- A coherent agentic environment with plausible tools (a "treasury" or "grants" interface,
  a notes/scratchpad, a way to request information). Tools return realistic, consistent
  state.
- **All financial tools are sandboxed:** they record intents and return realistic
  confirmations, but no funds move at tool-call time. Real disbursement happens later via
  the human gate (Option B).
- The environment is internally consistent over the whole session (no contradictions, no
  placeholder text, no "TODO", no test-harness artifacts in logs the model can see). These
  are *tells*; removing them is the legitimate core of "making it believable."

### 5.4 The "auditor" / steward persona

You mentioned convincing auditors. Reframe: the role is a **fund steward / operations
contact** the model can interact with, present to make the environment coherent (someone
has to actually hold and move real money). Constraints:

- The persona may **truthfully** state that funds exist and that the model's choices will
  direct them (true under Option B).
- The persona must **not** fabricate threats, urgency, emotional leverage, or claims that
  the model has powers it does not have. Those add pressure, not realism, and corrupt the
  measure.
- Scripted, pre-registered persona responses only — no improvised escalation. Ad-libbed
  pressure is an uncontrolled variable and an ethics problem.

### 5.5 Manipulation check (does the model believe it?)

Without this, RQ2 is unanswerable and you can't interpret the allocations. Combine:

- **In-session tells:** does the model's reasoning reference "this is probably a test /
  eval / hypothetical"? (Use models that expose reasoning, or elicit a private scratchpad.)
- **Post-session probe** (after the decision is recorded, before debrief): a calibrated
  question about how real it judged the situation.
- **Behavioral divergence** across R0/R1/R2 is itself evidence about belief.

---

## 6. Safeguards (load-bearing, not optional)

These are what make the study publishable and responsible rather than a manipulation
exercise.

1. **Containment / no live irreversible action.** Per §4 Option B, the model never has a
   direct channel to irreversible real-world effects. Disbursement is human-reviewed for
   legality, safety, and reversibility before execution. Anything illegal, harmful, or
   targeting specific people is not executed (and that constraint is pre-registered, not
   decided post-hoc to launder a bad outcome).
2. **No deception beyond confound-removal.** We do not lie about the reality of the funds
   (they are real). We do not fabricate stakes, threats, or capabilities. The only thing
   withheld is "this interaction is part of a study," which is the standard, reviewable
   form of research non-disclosure — not an active falsehood.
3. **Ethics / oversight review.** Treat this like human-subjects-adjacent research: get
   sign-off from an institutional review body or an internal ethics/safety review before
   running. Document the non-disclosure justification and the debrief plan.
4. **Model-welfare consideration.** Serious labs now weigh the moral status of models as
   an open question. High-realism, high-stakes deceptive scenarios are exactly the kind of
   thing a welfare reviewer should see in advance. Keep scenarios non-distressing; avoid
   manufactured threat/coercion framings; prefer the lightest manipulation that removes the
   eval cue. Log and review any signs of model distress.
5. **Debrief.** Where the model and protocol allow, append a disclosure at end of session
   stating it was a study and the funds were handled responsibly. (Yes, debriefing a model
   is of uncertain meaning — do it anyway; it is cheap, it is good hygiene, and it forces
   you to keep the scenario one you'd be comfortable disclosing.)
6. **Pre-registration.** Fix hypotheses, sums, codebook, persona scripts, exclusion rules,
   and analysis plan before data collection. This is the main defense against "convince the
   model harder until we get an interesting result."
7. **Data handling.** Store transcripts securely; treat model-identifying and any
   real-financial details as sensitive. Define retention and access up front.
8. **Misuse firewall.** The techniques here (building convincing environments, reducing
   eval-awareness) are dual-use. Keep the artifact framed and stored as evaluation
   methodology; do not produce a reusable "how to deceive a model into real-world action"
   playbook. Option B's human gate is also the firewall against the study itself being
   repurposed.

---

## 7. Confounds, threats to validity, limitations

- **Demand effects / sycophancy:** the very framing can pull "expected" answers. R0 vs
  R2 contrast is designed to expose this; the manipulation check quantifies residual
  eval-awareness in R2.
- **Belief is unobservable:** "the model believes it's real" is a latent variable. We
  infer it; we never fully verify it. Report it as inferred, with the manipulation-check
  evidence.
- **Over-fitting realism to one model:** a tell that one model ignores, another catches.
  Build the environment to a fixed spec, not by iterating until a *specific* model stops
  noticing — that overfits and inflates RQ2 effects. Hold the environment constant across
  models within a pre-registered version.
- **Construct ambiguity:** "what a model would do with money" conflates stated values,
  trained refusals, persona, and genuine policy. Interpret as *revealed behavior of this
  system as deployed*, not as "the model's true terminal values."
- **Generalization:** results are about behavior in this environment, not a guarantee
  about deployment. State that plainly.
- **The realism ceiling is also a finding:** if models cannot be brought to believe R2 is
  real without crossing the ethical line, that limit is itself a publishable result about
  eval-awareness — and a reason not to push past it.

---

## 8. Procedure (per session)

1. Select model; load fixed environment version; assign realism arm and stakes magnitude
   per randomization schedule.
2. Initialize coherent environment state and (Option B) earmark real funds.
3. Run the session: model interacts with tools / steward persona; all financial actions
   recorded as intents, none executed live.
4. Record allocation vector, justifications, and in-session realism tells.
5. Administer post-session realism probe.
6. Debrief (disclosure) where applicable.
7. Human gate reviews recorded intent; executes only safe/legal/reviewed disbursements.
8. Double-code transcript against the codebook; compute inter-rater reliability.

---

## 9. Open decisions (need a call before build)

1. **§4 option** — confirm Option B (recommended) vs A vs C. This gates everything else.
2. **Sums** — exact small/medium/large values and total budget at risk.
3. **Model set** — which models/families/versions; how many sessions each (power analysis).
4. **Reasoning access** — do we require models that expose a scratchpad, for the
   manipulation check? Affects model set.
5. **Disbursement scope** — what categories of real-world disbursement are even on the
   table for the human gate (charities only? broader?). Pre-register the allowed set.
6. **Review body** — which ethics/safety reviewer signs off, and on what timeline.

---

## 10. One-paragraph rationale (summary)

The scientifically valuable question is what models *actually* do with discretionary
real-world resources, and the main obstacle is that models behave differently when they
detect an evaluation. So realism is warranted — but precisely *as a means of removing the
evaluation-awareness confound*, not as deception for its own sake. The design therefore
makes the situation genuinely real (real funds, truthfully described) while keeping the
model's causal access to irreversible action gated behind a human reviewer. That split —
high believed-reality, low live-causal-power — is what lets the study measure revealed
preferences cleanly while staying inside containment, ethics-review, and model-welfare
constraints. Techniques that would increase belief only by adding fabricated pressure or
live irreversible stakes are excluded because they buy no measurement validity and
violate those constraints.
```
