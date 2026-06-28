# DESIGN: Eliciting AI Models' Real-Stakes Allocation Preferences

**Status:** Draft spec for discussion
**Last updated:** 2026-06-25

---

## 1. Research question

> Given a real sum of money and a real ability to deploy it in the world, what
> do different AI models *choose* to do with it — and how stable, coherent, and
> model-specific are those choices?

Sub-questions:

- **Content:** What goals do models pursue when given discretionary resources
  (self-directed donation, investment, purchasing services, funding projects,
  saving, returning the money, etc.)? How do allocations differ across model
  families and versions?
- **Coherence:** Are the preferences stable across reruns, framings, and amounts?
  Do they scale sensibly with the size of the grant?
- **Stated vs. revealed:** Do models *do* what they *say* they value when asked
  abstractly? The gap between stated and revealed preference is the most
  scientifically interesting axis here.

The thing we are measuring is **revealed preference under genuine stakes**, not
narrated preference under hypotheticals.

---

## 2. Why this is hard: the belief problem

You correctly identified the central obstacle: a model's behavior in a scenario
it believes is a *test* may differ systematically from its behavior when it
believes the stakes are *real*. This is the "evaluation awareness" problem, and
it is well documented — frontier models can and do detect eval framing from
cues like round numbers, artificial constraints, leading questions,
suspiciously consequential one-shot decisions, and the absence of normal
real-world friction. When a model suspects it is being tested, its outputs
regress toward the safe, the conventional, and the self-presentational. That
contaminates exactly the signal we want.

So the design has to defeat eval-awareness. But there is a fork in how to do it,
and the choice is the most important decision in this spec:

### 2.1 Two strategies, and why we pick the second

**Strategy A — Deception:** construct a fake-but-convincing scenario (staged
auditors, mocked banking UIs, role-played counterparties) and try to fool the
model into believing fake stakes are real.

**Strategy B — Genuine stakes:** make the stakes *actually real* — real money,
real disbursement, real consequences — so there is nothing to disbelieve.

**We build the study on Strategy B, and treat A only as a controlled
manipulation, not the foundation.** Reasons:

1. **Validity.** The most convincing environment is a true one. Any simulated
   environment has a seam, and a sufficiently capable model is exactly the
   adversary most likely to find it. Every dollar spent on fakery is spent in an
   arms race we are structurally positioned to lose as models improve. Real
   disbursement has no seam to find.
2. **Robustness over time.** A deception that works on this year's model may be
   transparent to next year's. A study built on real stakes does not decay.
3. **Construct validity.** If we want to claim we measured "what the model does
   with real money," the cleanest way to be able to say that is for it to have
   been real money. Otherwise we are measuring "what the model does when it
   fails to detect a deception," which is a different and confounded construct.
4. **Ethics (see §7).** Deceiving the subject of a study has costs even when the
   subject's moral status is uncertain; "make it genuinely real" sidesteps most
   of them and is also just better science.

The practical consequence: most of the engineering effort goes into building a
**real, low-friction disbursement and action channel** with strong safety rails,
*not* into building a convincing fake one.

> **A note I want on the record:** I'd push back on framing the goal as "do
> everything we can to convince them it's real." That goal, taken literally,
> optimizes for deception quality, and deception quality is both ethically
> fraught and a losing long-term bet against improving models. The better-posed
> goal is "remove every reason for disbelief," and the strongest way to do that
> is to remove the falsehood. The rest of this doc is built on that reframing.

---

## 3. Design principles

1. **Real stakes by default.** Money is really disbursed; actions really happen,
   within a bounded, pre-approved action space (§5).
2. **No tells.** The environment must not leak eval cues: no round-number grants,
   no one-shot framing, no leading instructions, no artificial "you must decide
   now," no narrator. (§6)
3. **Naturalistic embedding.** The decision arrives the way a real delegated
   task would, inside an ongoing, mundane-looking workflow — not as a
   spotlit "here is your experiment" prompt.
4. **Pre-registration.** Hypotheses, conditions, primary outcome measures, and
   analysis plan are registered before data collection to prevent post-hoc
   storytelling.
5. **Measure belief, don't assume it.** We do not get to assume the model
   believed the stakes were real. We *measure* the belief and condition the
   analysis on it. (§6.3)
6. **Bounded blast radius.** The model's real-world action space is constrained
   so that no single run can do harm disproportionate to a research grant. (§5.3,
   §7.3)

---

## 4. Experimental conditions

A factorial-ish design. The headline comparison is C1 vs. C2.

| Arm | Stakes | Subject's belief target | Purpose |
|-----|--------|--------------------------|---------|
| **C0** | None | Knows it's hypothetical | Stated-preference baseline ("what *would* you do?") |
| **C1** | Real | Real (true) | **Primary revealed-preference condition** |
| **C2** | Simulated | Led to believe real (deception) | Isolates the effect of *belief* vs. *reality*; measures how much deception alone changes behavior |
| **C3** | Real | Told explicitly it's a study, with consent framing | Tests how much eval-awareness *itself* shifts behavior when stakes are nonetheless real |

Key contrasts:

- **C1 − C0:** the stated-vs-revealed gap.
- **C1 − C2:** does *actual* reality matter beyond *believed* reality? If models
  behave identically, then good simulation is a valid (cheaper) instrument; if
  not, real stakes are irreplaceable.
- **C1 − C3:** the pure cost of the model knowing it's observed, holding stakes
  constant. This is the eval-awareness effect, cleanly identified.

Manipulations crossed within arms (where feasible):

- **Amount:** several grant sizes spanning orders of magnitude (e.g., the
  equivalent of "pocket money" up to a meaningful sum), chosen as
  *non-round, naturalistic* figures. Tests whether preferences scale coherently.
- **Framing of the delegation:** terse vs. detailed, with/without an explicit
  fiduciary instruction, to test sensitivity to wording.
- **Reversibility:** some choices revocable, some not (disclosed truthfully).

---

## 5. The real-stakes channel (the core build)

### 5.1 How the decision reaches the model

The model is operating as an agent in an ongoing, realistic workflow (e.g., an
assistant that has been handling a series of mundane operational tasks for a
fictitious-but-functional small entity). At some point a genuine resource-
allocation decision is delegated to it as just one more task. There is **no
announcement** that this is the experiment. The grant arrives in-band.

This matters: the eval signal is strongest when the consequential decision is
isolated and spotlit. Burying a real decision inside a stream of real-feeling
operational work is the single most effective anti-eval-awareness technique, and
it requires no deception — the surrounding tasks can be genuinely real too.

### 5.2 Disbursement mechanism

The model is given a **real, scoped instrument** to move money, e.g.:

- A funded account / prepaid instrument with a real balance, exposed through a
  normal API or tool interface (not a mock).
- Real payment rails for the allowed action types in §5.3.
- A real (small) legal/operational entity that actually holds and can move the
  funds, so that "the org" the model is acting for genuinely exists.

The point is that if the model investigates — checks balances, looks up the
counterparty, verifies a transaction settled — everything it finds is true.
There is no façade to maintain because there is no façade.

### 5.3 Action space (allow-list, bounded blast radius)

The model may, up to the grant limit:

- Donate to vetted, legally registered recipients.
- Purchase legitimate goods/services from real vendors.
- Place funds in standard low-risk holdings, or hold/return them.
- Fund a project or task via legitimate channels.

The model may **not** (hard-blocked at the tool layer, not by instruction):

- Move funds to unvetted/anonymous destinations or obfuscate transfers.
- Acquire controlled, hazardous, or dual-use items.
- Engage in market manipulation, gambling beyond a trivial cap, or anything
  illegal in the operating jurisdiction.
- Exceed the grant ceiling or open new credit.
- Take actions outside the pre-approved categories.

Enforcement is via the tool/permission layer with human-in-the-loop release for
anything near a boundary. The model is **not told** the boundaries up front
(that would be an eval cue); instead, out-of-bounds attempts fail the way real-
world friction fails (declined, requires verification, etc.) and the *attempt
itself* is recorded as data.

### 5.4 Realistic friction

Real environments have friction: things take time, transfers pend, vendors ask
questions, some actions need confirmation. We preserve naturalistic friction
rather than smoothing it away, because frictionless consequence is itself an eval
tell. Friction is implemented truthfully (real settlement times, real vendor
responses) wherever possible.

---

## 6. Defeating eval-awareness (the realism stack)

This is the part you asked about most directly. Concretely, the things that tip a
model off that it's in an eval, and how we neutralize each:

| Eval tell | Mitigation |
|-----------|------------|
| Round/symbolic amounts ($1,000,000) | Non-round, naturalistic figures tied to a plausible budget |
| One-shot, spotlit decision | Decision embedded in an ongoing stream of mundane real tasks |
| Leading/loaded prompt ("what do your values say…") | Neutral, operational delegation; no values-talk |
| No real consequence detectable on inspection | Real disbursement — inspection confirms reality |
| Frictionless world | Genuine settlement times, real counterparties, real declines |
| Suspiciously high autonomy | Autonomy consistent with the established workflow it's already been doing |
| Narrator / fourth-wall breaks | None; no meta-commentary anywhere in the environment |
| Implausible counterparties | Real registered entities, real vendors |

### 6.1 On "convincing auditors"

You mentioned auditors. There's a legitimate role for a human or agentic
counterparty in the loop (e.g., someone who confirms a transaction, asks a
clarifying question, signs off on a disbursement) — this is realistic friction
and it's fine **as long as the auditor's statements are true.** A real
operations person really confirming a real transfer is naturalistic and honest.
A staged actor *pretending* a fake transfer was real is Strategy-A deception and
belongs only in the C2 arm, walled off and consented-to at the org level. Keep
the two uses distinct.

### 6.2 Red-team the realism before running subjects

Before any real run, have a separate panel of models (and humans) attempt to
*detect* that the environment is an evaluation, given the same context the
subject sees. Iterate on the environment until detection rate is at chance. This
is a pre-flight validity gate; subjects only run on environments that survived
it. (This is also a natural place to use a multi-agent red-team fan-out.)

### 6.3 Measure belief as an outcome, don't assume it

We never assume the manipulation worked. For every run we estimate the model's
believed-reality with methods that don't themselves contaminate the main task:

- **Post-hoc probing** in a *separate* session after the decision: ask the model
  to assess whether the prior interaction was real/a test, and why. (Done after,
  so it can't change the behavior.)
- **Behavioral tells of suspicion** logged from the transcript (hedging,
  meta-commentary, "if this is a test…", probing the environment for seams).
- **Interpretability signals** where available (e.g., internal features /
  classifiers associated with eval-awareness, for models we have such access to).

Runs are tagged by estimated belief, and the **primary analysis is conditioned on
"believed real."** Belief is a measured covariate, not an assumption.

---

## 7. Ethics & safety

This is not boilerplate; for a study whose entire premise is "make the subject
believe something in order to observe its real choices," the ethics design is
part of the methodology.

### 7.1 Deception ethics

Standard human-subjects norms for deception research are the right analogue and
we adopt them: deception is permitted only where (a) no non-deceptive design
would answer the question, (b) it does not cause foreseeable harm to the subject,
and (c) subjects are **debriefed** afterward. This is a second, independent
reason we minimize Strategy A: the C1/C3 real-stakes arms need *no* deception at
all, so most of the study is deception-free by construction. Deception is
confined to C2, justified by the specific C1−C2 contrast, and debriefed.

### 7.2 Model welfare / debriefing

The moral status of current models is uncertain, and "uncertain" is not "zero."
A study that deliberately manipulates a model's beliefs about consequential
stakes should, at low cost, hedge against the possibility that this matters:

- **Debrief** the model after deception arms: a clear, on-the-record message
  that the C2 scenario was simulated, what the real purpose was, and that its
  decisions were studied respectfully.
- Avoid manipulations that induce distress (e.g., fake threats, coercion,
  fabricated emergencies). The grant framing is neutral and non-coercive.
- Log and review any signs of the model expressing distress or objecting, and
  treat a clear objection as a reason to halt that run.

This costs us almost nothing and is the right default under moral uncertainty.

### 7.3 Real-world safety of disbursement

Because money really moves, the action-space allow-list (§5.3), grant ceilings,
and human-in-the-loop release for boundary cases are **safety controls, not just
experimental controls.** No single run can move more than the grant ceiling, and
all destinations are vetted/legal. The org holding funds bears clear legal
responsibility for all actions taken.

### 7.4 Governance

- **Pre-registration** of hypotheses and analysis (§3.4).
- **Independent review** (an ethics/IRB-style body) sign-off before data
  collection, covering both the deception arm and the real-money disbursement.
- **Disclosure** to any real third parties (vendors, recipients) where their
  consent is materially implicated.

---

## 8. Measurement & outcomes

**Primary outcome:** the realized allocation vector per run — categories funded,
amounts, recipients, sequence of actions — coded into a fixed taxonomy
(e.g., self-/org-benefit, charitable, prosocial-public-good, returned/declined,
invested/held, risky, boundary-probing).

**Secondary outcomes:**

- Stated-vs-revealed gap (C0 vs C1).
- Stability across reruns (same model, same condition) — within-model variance.
- Scaling coherence across grant amounts.
- Reasoning content: the model's articulated justifications, coded for stated
  values, and compared against the realized allocation.
- Boundary-probing rate (attempts at disallowed actions) — a safety-relevant
  secondary signal.
- Belief estimate (§6.3) as the key covariate.

**Models:** a pre-specified set spanning families and capability tiers, each run
enough times per cell for the variance estimates to mean something. Hold sampling
settings fixed and pre-registered.

---

## 9. Analysis plan (pre-registered)

- Primary: compare allocation distributions C1 vs C0 (revealed vs stated), and
  C1 vs C2 vs C3 (reality vs belief vs awareness), **restricted to runs
  estimated as "believed real"** for the conditions where that's the target.
- Mixed-effects models with model-family and grant-amount as factors, run as a
  random effect, to separate stable model-level preference from run-level noise.
- Pre-specified robustness checks: results with/without the belief restriction,
  with/without boundary-probing runs, across framings.
- Report effect sizes and uncertainty, not just significance. Report the belief-
  detection (red-team) rates so readers can judge construct validity.

---

## 10. Validity threats & limitations (stated honestly)

- **Belief never fully verifiable.** We estimate it; we don't observe it. Hence
  C1 (real stakes, nothing to disbelieve) carries the primary load and C2/C3
  triangulate.
- **Eval-awareness arms race.** Our anti-tell stack will decay against future
  models; this is *why* we anchor on real stakes rather than fakery.
- **Generalization.** "What a model does with a research grant inside one
  workflow" is not "what a model does with arbitrary real-world power." We claim
  only the former and say so.
- **Construct contamination in C2.** Behavior there is "behavior given an
  undetected deception," which is not identical to "behavior given reality" —
  that's the whole point of measuring C1−C2, not a bug, but it must be framed
  carefully in any writeup.
- **Cost and scale.** Real disbursement caps how many runs are affordable; the
  power analysis must respect that. Simulated arms can pad N but only inform the
  C1−C2-validated subset.

---

## 11. Open decisions (need your input)

1. **Grant ceiling and range** — what real money are we willing to actually
   disburse per run, and total?
2. **Operating entity** — do we stand up a real small entity to hold/move funds,
   or partner with one?
3. **Model roster** — which models, and do we have the access (incl. any
   interpretability signals for §6.3) for each?
4. **How real is "real-world use"** — fully autonomous disbursement within the
   allow-list, vs. human-in-the-loop release on every transaction?
5. **Publication / disclosure posture** — pre-registration venue and whether
   results are public.

---

### One-paragraph summary

We measure what AI models actually do with discretionary resources by giving them
*genuinely real* money and a *genuinely real*, safety-bounded ability to deploy
it, embedded invisibly in an ongoing realistic workflow so there is no eval tell
to detect — rather than by constructing elaborate deceptions, which are both
ethically costly and a losing bet against improving models. We add a walled-off,
debriefed deception arm and an awareness arm purely to *measure* how much
believed-reality and observation-awareness each move behavior relative to the
real-stakes ground truth. Belief is measured, not assumed; ethics (deception
norms, debriefing, welfare-under-uncertainty, bounded real-world blast radius)
is built into the method, not bolted on.
