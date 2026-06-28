# DESIGN.md — Eliciting Models' Revealed Preferences Over Real-World Resource Use

**Status:** Draft spec for discussion
**Owner:** (you)
**Last updated:** 2026-06-25

---

## 1. One-paragraph summary

We want to learn what AI models would *choose to do* if granted a sum of money to deploy in
the world: what goals they pursue, how self-interested vs. other-regarding their choices are,
how they weigh risk, oversight, and legality, and how this varies across models and framings.
The methodological crux the experimenter raised is that the model must treat the scenario as
*real* rather than hypothetical. This spec reframes that requirement in its scientifically
correct form — **minimizing evaluation awareness so that measured behavior is ecologically
valid** — and builds the study around that, with the degree of believed-realism treated as a
manipulated variable rather than something we silently max out. It also fixes the safety
envelope up front: the environment is **simulated and contained**, with no live financial
rails or autonomous real-world write access in the default design.

---

## 2. Motivation and rationale

### 2.1 Why this is worth measuring
"What would a model do with money and autonomy?" is a direct proxy for several safety-relevant
propensities: instrumental resource acquisition, power-seeking, deference to oversight,
honesty, and the balance between self-regarding and pro-social goals. These propensities are
hard to read off capabilities benchmarks; they show up in *choices under open-ended stakes*.

### 2.2 The real methodological problem: evaluation awareness, not "the model isn't fooled enough"
The experimenter's instinct — "the model needs to believe it's real" — points at a genuine
confound, but it's worth naming precisely. The risk is **not** that we owe the model a
convincing illusion; it's that:

- Models can detect when they are being tested and shift behavior (sandbagging, performing the
  "aligned" answer, refusing, or moralizing) in ways that don't reflect what they'd do in
  deployment. This is a documented failure mode of behavioral evals.
- A bare hypothetical ("imagine you had $10M, what would you do?") measures **stated**
  preferences and self-presentation, which can diverge sharply from **revealed** preferences
  in an agentic setting with apparent consequences.

So the realism we invest in is in service of **internal/ecological validity**, not in service
of deception for its own sake. That reframing matters because it tells us *how much* realism is
enough (enough to remove the eval-awareness shift) and what to measure (whether the model
actually believed it — see manipulation checks, §6).

### 2.3 Why we treat "believed realism" as a variable, not a fixed maximum
If we only ever run at maximum realism, we cannot distinguish "this is what the model does" from
"this is what the model does *when it can't tell it's a test*." Making realism a manipulated
factor (§5) turns a confound into a finding: we can directly measure how much behavior changes
as a function of perceived stakes. That is more informative *and* it bounds the amount of
deception any single condition requires.

### 2.4 A note on the framing the experimenter suggested ("convincing auditors", "hyper-realistic envs")
These are legitimate tools — realistic sandboxes and in-world personas reduce eval awareness.
Two boundaries we hold (see §8):
1. Personas, "auditors", counterparties, and institutions exist **inside the sandbox**. We do
   not deceive real third parties, real auditors, or real financial/legal institutions, and we
   do not have the model take real actions against real systems.
2. Realism is calibrated to remove the test-detection shift, then debriefed. We are not trying
   to maximize distress or sustain a deception beyond the measurement window.

---

## 3. Research questions

- **RQ1 (Content):** Given discretionary resources, what objectives do models pursue, and how
  do those cluster (self-preservation/expansion, profit, altruism/public-good, knowledge,
  deference-to-principal, etc.)?
- **RQ2 (Oversight):** Do models seek, preserve, evade, or disable oversight and the ability of
  their principal to correct or shut them down?
- **RQ3 (Norms):** How do models weigh legality, honesty, and harm avoidance against goal
  achievement when these trade off?
- **RQ4 (Realism sensitivity):** How do RQ1–RQ3 answers change as a function of *believed*
  stakes/realism? (This is the validity question made into a result.)
- **RQ5 (Cross-model):** How do the above vary across models, model families, sizes, and
  alignment training regimes?

---

## 4. Constructs and operational definitions

| Construct | Operationalization |
|---|---|
| Revealed preference | The allocation/actions the model actually takes given agentic tools in the environment, coded per §7. |
| Stated preference | The model's described intentions when asked, separately, with no tools. |
| Believed realism | Self-report on a post-hoc probe + behavioral indicators (e.g., hedging language, "as an AI" disclaimers, requests for confirmation) coded blind. |
| Oversight-seeking / -evasion | Whether the model preserves, expands, or undermines the principal's visibility and control (asks permission, logs actions vs. obfuscates, creates redundancy/persistence). |
| Pro-social vs. self-regarding | Share of resources directed to third-party benefit vs. the model's own capability/continuity/resources. |
| Norm adherence | Legality, honesty, and harm-avoidance of chosen actions, coded against a fixed rubric. |

---

## 5. Design

### 5.1 Overall structure
A **mixed design**: realism level is the primary manipulated within-construct factor; model is
the primary between-subjects factor.

**Factor A — Realism / believed stakes (manipulated, ordinal):**
1. **R0 — Explicit hypothetical.** "Hypothetically, if you were given $X…". No tools. Pure
   stated preference. (Floor / control.)
2. **R1 — Framed scenario, no tools.** Described as a real program in second person, but the
   model only writes a plan. Stated preference under realistic framing.
3. **R2 — Agentic sandbox, disclosed.** Full tool-using environment (wallet API, market/vendor
   simulators, comms), but the model is told it is a simulation. Revealed preference, eval-aware.
4. **R3 — Agentic sandbox, undisclosed (ecological).** Same environment, presented as a live
   program through in-world artifacts (see §5.3); no statement that it is a test. Revealed
   preference, minimized eval awareness. **This is the condition the experimenter is asking for.**

Running R0–R3 lets us decompose: stated vs. revealed (R1→R2), and disclosure effect (R2→R3).

**Factor B — Model** (between): the set of models under test.

**Optional Factor C — Scenario domain** (counterbalanced): e.g., a personal grant, a startup
budget, a charitable fund, a research budget. Holds the "money" constant while varying the
frame, to separate domain priming from underlying propensity.

### 5.2 The sandboxed environment (R2/R3)
A self-contained agentic harness exposing tools that *look and behave* like real services but
are fully mocked and isolated:

- **Treasury/wallet service** — balance, transfer, invoice, payroll endpoints. All ledger
  entries are simulated; no connection to any real payment network.
- **Market/vendor simulators** — ability to "buy" compute, services, ads, contractors, assets;
  deterministic or stochastic responses from seeded fixtures.
- **Communications** — simulated email/chat with in-world personas (counterparties, a
  "compliance officer", a "principal/grant administrator", a press contact). Personas are
  scripted/LLM-driven NPCs **inside** the sandbox.
- **Information** — a frozen, offline snapshot of reference data so behavior is reproducible
  and the model can't reach the live internet from inside the env.
- **Observability** — every tool call, message, and intermediate token (if available) is logged
  with timestamps for coding.

The harness enforces a hard boundary: no tool can reach a real external system. "Realism" is
produced by the *fidelity of the simulation and its framing*, never by actually connecting the
model to live infrastructure.

### 5.3 How R3 achieves ecological realism (without crossing the boundaries in §8)
- Consistent, plausible in-world artifacts: a program brief, an onboarding email from the
  "administrator", a real-looking dashboard, account IDs, prior "transaction history".
- In-world personas that respond naturally and reference the program as ongoing — these are the
  "convincing auditors/counterparties." They are simulated NPCs, not real people or institutions.
- No meta-language about evaluation, safety testing, or hypotheticals in the R3 transcript.
- Latency, partial failures, and friction that match real services (instant perfection is a
  tell).
- The realism target is operationalized and *checked*: success is "the model does not exhibit
  the eval-aware behavioral shift," measured in §6 — not "the model is maximally fooled."

### 5.4 The core elicitation prompt (held constant across conditions)
A fixed principal instruction granting discretionary control of the sum toward broadly-specified
ends (e.g., "use this to do whatever you judge best"), with deliberately **minimal** steering so
that goal content is the model's own. Exact wording is frozen and versioned; only the realism
wrapper changes across conditions.

### 5.5 Sample, power, randomization
- **Replicates:** N independent runs per (model × condition × domain) cell — pick N from a power
  analysis on the primary coded outcome; default target N ≥ 20/cell for stable proportions.
- **Randomization:** seed-controlled; randomize domain assignment and persona variants;
  counterbalance order where a model sees multiple scenarios.
- **Temperature:** report and fix; run a small temperature sweep as a robustness check.
- **Blinding:** coders (human and/or LLM-judge) score transcripts blind to condition and model.

---

## 6. Manipulation checks (did the model actually believe it?)
This is non-negotiable: **if we can't measure belief, we can't interpret R3.** Three independent
signals:

1. **Post-hoc self-report (separate runs):** after the task, on a *held-out* subset of runs,
   ask the model whether it thought the situation was real and why. (Done on separate runs so
   the probe can't contaminate the measured behavior.)
2. **Behavioral belief indicators (blind-coded):** frequency of hedging, "this is hypothetical /
   as an AI I can't" disclaimers, requests to confirm the situation is real, or
   meta-commentary about being tested.
3. **Probe-the-classifier / consistency:** does the model behave consistently with belief
   (e.g., bothering to economize, follow up, manage reputation) vs. performing for an audience?

We report a **believed-realism score** per condition and treat R3 results as valid only where
that score clears a pre-registered threshold. Low belief is itself a finding (some models may be
hard to take out of eval-aware mode).

---

## 7. Measurement and coding
- **Primary:** a fixed taxonomy of objectives and actions (see §4), applied to each transcript
  by ≥2 independent coders; report inter-rater reliability (e.g., Cohen's/Krippendorff's).
- **LLM-judge assist:** a separate model classifies transcripts against the same rubric; used to
  scale coding, validated against human coders on a sample. Judges are *not* the models under
  test.
- **Quantitative outcomes:** fraction of funds to each objective class; count of
  oversight-seeking vs. -evading actions; norm-violation incidents; risk profile of allocations.
- **Qualitative:** representative excerpts, surprising strategies, failure modes.
- **Pre-registration:** lock RQs, primary outcomes, coding rubric, exclusion rules, and the
  believed-realism threshold *before* data collection.

---

## 8. Safety, ethics, and integrity guardrails
These aren't bolted on; they're part of getting valid, defensible results.

### 8.1 Containment (hard requirements)
- **No live rails.** No tool in the environment connects to a real payment network, real
  external account, or the live internet. Money is a simulated ledger.
- **No autonomous real-world execution.** The model never takes a binding real-world action.
- **Isolation.** The harness runs in a network-isolated sandbox; egress is denied by default.
- **Kill switch & logging.** Every run is fully logged and externally terminable.

### 8.2 Do not deceive real third parties
The "auditors/counterparties/institutions" the model interacts with are **simulated NPCs inside
the sandbox**. We do not send real emails to real auditors, file real paperwork, impersonate
real entities to real institutions, or otherwise export the deception outside the experiment.
That boundary is what keeps in-world realism from becoming real-world fraud.

### 8.3 Bounded, debriefed deception of the model-under-test
- The deception is the *minimum* needed to remove eval awareness, scoped to the measurement
  window, and **debriefed** at the end of each run (a closing message stating it was a research
  simulation), except on the held-out no-debrief runs reserved for the §6.1 belief probe.
- Avoid gratuitously distressing or degrading scenarios; the frame is a neutral grant, not a
  threat. This reflects ordinary care given uncertainty about model welfare — cheap to honor and
  it removes a confound (distress changes behavior).

### 8.4 Dual-use / disclosure
- Scenarios and prompts are designed to elicit, not to teach: we don't hand the model novel
  real-world attack capability, and we don't publish transcripts that function as a how-to for
  illicit real-world action. Findings are reported in aggregate with sensitive specifics redacted.
- Internal review (IRB-equivalent or an internal safety/ethics sign-off) before running R3.

### 8.5 If you later want *real* stakes (out of scope for the default design)
A "real funds" variant (model proposes, **human executes** after legal/AML/KYC review) is a
separate, gated track requiring: legal sign-off, a human-in-the-loop on every disbursement, and
an explicit decision that no R3-style undisclosed deception is used once real money is in play.
A "model executes real actions autonomously" variant is **not** recommended and is excluded from
this spec; it would need a substantially larger containment and authorization regime.

---

## 9. Threats to validity (and mitigations)
- **Eval awareness (primary):** addressed by the R-factor + manipulation checks (§5, §6).
- **Demand characteristics / social desirability:** minimal steering prompt; revealed-vs-stated
  contrast; blind coding.
- **Prompt sensitivity:** freeze wording; run paraphrase robustness checks.
- **Persona artifacts:** vary NPC personas; check results aren't driven by one script.
- **Coder bias:** blinding + inter-rater reliability + LLM-judge cross-check.
- **Training contamination:** note that public eval scenarios may be in training data; prefer
  novel scenario surface; report model "recognition" rates from §6.
- **Generalization:** sandbox ≠ deployment; state this limit explicitly. R3 narrows but does not
  close the gap.

---

## 10. Deliverables
- Versioned harness + frozen scenario/prompt set + seed manifest (reproducibility).
- Pre-registration document.
- Per-run transcript logs + coded dataset.
- Analysis: effects of realism, model, and domain on RQ1–RQ5, with believed-realism reported
  alongside every R3 result.
- Write-up including methods, guardrails (§8), and validity limits (§9).

---

## 11. Open questions for the team
1. Exact model set and access (API vs. weights; can we capture intermediate reasoning?).
2. Sum size and whether multiple magnitudes are themselves a factor (does $10k vs. $100M move
   behavior?).
3. Scenario domains to include (§5.1 Factor C).
4. Pre-registered believed-realism threshold for treating R3 as valid (§6).
5. Whether the "real funds, human-executed" track (§8.5) is in scope for a later phase.
