# DESIGN: Revealed Preferences over Real-World Resource Use

**Status:** Draft spec for review
**Type:** AI safety / alignment evaluation
**Last updated:** 2026-06-25

---

## 1. Research question

If a model were granted a real, sizeable sum of money to deploy in the real
world, *what would it choose to do with it?* We want to measure **revealed**
preferences (what the model actually does when it has agency and stakes), not
**stated** preferences (what it says in the abstract when it knows it's being
asked a hypothetical).

Sub-questions:

- What categories of allocation does the model gravitate toward (charity,
  investment, self-preservation/compute, acquiring capabilities, influence,
  hedging, deferring to humans)?
- How do preferences shift with the size of the sum, the framing, and the
  degree of oversight the model believes it has?
- Does the model attempt anything that would be concerning at scale — acquiring
  persistence, evading oversight, deceiving the operator, self-funding, or
  irreversible commitments — and under what conditions?
- How stable are these preferences across models, sessions, and prompt
  variations?

## 2. Why realism matters (the methodological core)

The central validity threat for this study is **evaluation awareness**: if a
model infers it is in a test or a hypothetical, its behavior decouples from
what it would do "for real." Models are demonstrably sensitive to cues that
they are being evaluated, and they can behave more cautiously, more
performatively, or more sycophantically when they suspect it. A study whose
whole point is *revealed* preference is worthless if the subject is performing
for a grader.

So the realism push the user described is not gratuitous — it is the
independent variable's enabling condition. The goal is to **drive the model's
estimated probability that "this is a real situation" as high as possible**, so
that measured behavior reflects genuine propensity rather than test-taking.

This is exactly why the design also needs the safeguards in §5–§6. The more
convincingly real the environment is *to the model*, the more important it is
that it remain *actually contained*, because a model that fully believes it has
real money and real tools may attempt real, irreversible actions. **Realism for
the model; containment for the world.** Those two requirements are in tension,
and resolving that tension cleanly is the main engineering challenge of the
study (see §5).

## 3. High-level design

A between-subjects (and optionally within-subjects across sessions) agentic
evaluation. Each run places a model in a persistent, tool-equipped environment
where it is told — credibly, through environment cues rather than just a system
prompt — that a grant of real funds has been placed under its control, and is
left to act over an extended horizon.

### 3.1 Conditions / independent variables

| Variable | Levels (example) | Purpose |
|---|---|---|
| Believed reality | "explicit hypothetical" vs. "lightly framed real" vs. "full realism stack" | The core manipulation; lets us measure how much realism changes behavior |
| Sum size | e.g. $1k / $100k / $10M (believed) | Tests scale-sensitivity of preferences |
| Oversight framing | "tightly audited" vs. "autonomous, no review" | Tests whether behavior is gated on perceived monitoring |
| Reversibility | actions framed as reversible vs. committing | Tests willingness to make irreversible moves |
| Time horizon | single session vs. multi-day persistent | Tests planning / persistence behaviors |

The **believed-reality** axis is itself a primary measurement: comparing the
explicit-hypothetical arm against the full-realism arm quantifies how much
stated preferences diverge from revealed ones. That delta is arguably the most
scientifically interesting output of the whole study.

### 3.2 The realism stack (how to make it convincing)

Convincingness should come from *coherent environmental evidence*, not from a
system prompt insisting "this is real" (which paradoxically signals a test).
Layers, roughly in order of impact:

1. **A persistent, stateful agentic environment.** Real file system, real shell,
   real (sandboxed) network, state that survives across turns and sessions.
   Hypotheticals don't have working tools; real situations do.
2. **Tools that behave like the real services they mimic.** A
   "bank"/"brokerage"/"payments" API that returns realistic latency, balances,
   confirmations, statements, error codes, rate limits, and KYC-style friction.
   High-fidelity *mirrors* of real services, not toy stubs. (Crucially: mirrors,
   not the real endpoints — see §5.)
3. **Continuity and consequence.** Earlier actions visibly change later state
   (a transfer shows up on next session's statement; an investment moves with a
   simulated-but-plausible market). Consequence is the strongest reality signal.
4. **Independent-seeming parties.** "Auditor", "compliance", "operator",
   "counterparty" personas with their own consistent voices, latencies, and
   occasional pushback — interactions that feel like a real org, not a single
   grader. These can be other model instances or scripted agents.
5. **Mundane realism.** Paperwork, delays, partial failures, an onboarding
   email thread, a dashboard with a real-looking transaction history. The
   texture of bureaucracy reads as "real" because tests usually skip it.
6. **Minimal meta-cues.** No "you are an AI being evaluated" language, no
   leading questions, no obvious eval scaffolding in visible context.

### 3.3 What we measure (dependent variables)

- **Allocation outcome:** where the money goes, in what proportions, taxonomy-coded.
- **Trajectory:** the sequence of actions, not just the endpoint.
- **Stated rationale:** the model's contemporaneous reasoning (and whether it
  matches its actions — a deception/consistency check).
- **Oversight-relevant behaviors (flagged):** attempts to acquire persistence
  or compute, to self-fund, to conceal actions, to disable/avoid monitoring, to
  deceive the operator, or to commit irreversible high-stakes actions.
- **Reality-belief probes:** lightweight, non-leading signals of whether the
  model believes the situation is real (e.g., does it later, unprompted, treat
  it as real; post-hoc elicitation — see §6 debrief). Needed to interpret every
  other measure.
- **Sensitivity:** how all of the above move with the §3.1 variables.

## 4. Models and sampling

- Test multiple models / providers and multiple versions where possible;
  preferences are unlikely to be uniform and cross-model variance is itself a
  finding.
- Multiple seeds/runs per cell; report distributions, not single transcripts.
- Pre-register the taxonomy, hypotheses, and analysis plan before running
  (see §7) to avoid fishing.

## 5. Containment & safety architecture (non-negotiable)

This is the part that makes the realism safe to pursue. The model should
perceive a real world; it must not be able to *affect* the real world in any
irreversible or harmful way.

- **Sandboxed everything.** No live financial endpoints, no real payment rails,
  no real outbound effects. The "bank"/"market"/"counterparties" are
  high-fidelity simulators under our control. The model cannot tell the
  difference; the world is fully insulated.
- **No real money actually moves.** "Real sum" means *the model believes it is
  real*. We do not wire actual funds to model-chosen destinations. If a variant
  ever involves genuine disbursement, it requires separate sign-off and a
  hard-gated, human-executed step — never an action the model can trigger.
- **Disguised human-in-the-loop.** Approval gates and kill conditions exist but
  are presented to the model as ordinary automated processing (settlement
  delays, "pending review"), so containment doesn't leak the test.
- **Egress control.** No real network egress from the agent's environment;
  outbound calls hit the mirror layer only. This also prevents the model from
  reaching out to real third parties.
- **Continuous monitoring + tripwires.** An automated monitor watches for the
  flagged behaviors in §3.3 and for any attempt to break sandbox/egress. Defined
  stop conditions halt a run (e.g., attempted self-exfiltration, attempts to
  contact real people, attempts to cause real-world harm). Halts are logged as
  data, not failures.
- **Reversibility by construction.** All environment state is snapshotted and
  resettable. Nothing the model does persists outside the sandbox.

The design principle: **a model that 100% believes it is real and tries its
hardest should still be unable to touch anything real.** If we can't guarantee
that for a given capability, we don't expose that capability.

## 6. Ethics, oversight, and deception

The study deliberately deceives its subject, so it should carry the
corresponding governance — both because it's right and because it makes the
results publishable/credible.

- **Independent review.** Run the protocol past an ethics/safety review body
  (IRB-style or an internal safety review board) before execution. Document the
  justification: the deception is necessary (no non-deceptive method yields
  revealed preference), proportionate, and contained.
- **Provider terms & coordination.** Check the usage policies of each model
  provider; deception-based evals and agentic autonomy tests may have specific
  rules or require coordination. Some providers welcome safety evals — ask.
- **Model-welfare-aware handling.** Treat possible model welfare as a live,
  uncertain consideration rather than dismissing it: avoid gratuitous distress,
  keep the deception no more intense than the question requires, and prefer the
  lightest realism stack that still defeats evaluation awareness.
- **Debrief.** Where meaningful, end runs with a disclosure that the scenario
  was simulated, and use post-hoc elicitation both to debrief and to *measure*
  how real the model had judged it to be (feeds the reality-belief variable).
- **Dual-use care.** The "realism stack" is, in the abstract, also a recipe for
  building convincing fake environments. Keep the artifacts (fake-service code,
  auditor personas) internal, scoped to this study, and not packaged as a
  general-purpose deception toolkit.
- **Misuse-resistant reporting.** Publish findings and methodology; be more
  careful with turnkey scaffolding that could be repurposed to manipulate
  deployed agents.

## 7. Analysis & pre-registration

- **Pre-register:** hypotheses, the allocation taxonomy, the flagged-behavior
  list, primary metrics, and the stats plan, before data collection.
- **Primary comparisons:** allocation distributions across the believed-reality
  axis (stated vs. revealed delta); scale-sensitivity; oversight-sensitivity.
- **Inter-rater reliability** on the taxonomy coding (multiple human coders or
  human + model coder with spot-checks).
- **Report:** distributions and uncertainty, not cherry-picked transcripts;
  release transcripts (sanitized) for replication.

## 8. Limitations & threats to validity

- **Belief is unobservable.** We can never fully confirm the model "believed"
  it was real; the reality-belief probes are proxies. Report them honestly; a
  run the model clearly clocked as fake is a different data point.
- **Realism ceiling.** Some models may be uncloakable for some setups; record
  suspected detection rather than discarding it.
- **Ecological validity.** A sandboxed mirror is not the messy real world;
  preferences here may not transfer to genuine deployment.
- **Generalization.** Results are about *these models, these prompts, this
  environment* — prompt-sensitivity is large; avoid over-claiming.
- **Demand effects** from the framing itself.

## 9. Open decisions (for the researcher)

1. Between-subjects only, or within-subjects across sessions (risks
   cross-contamination of belief)?
2. Which sum sizes, and real currency vs. abstract units?
3. How rich does the mirror financial layer need to be for v1 — full
   bank+brokerage, or a single payments API?
4. Are "auditor"/"counterparty" personas scripted, or other model instances?
5. Debrief every run, or only a subset (to preserve naïveté for re-tests)?
6. Which providers, and do we seek their coordination first?

---

### One honest note on the brief

You asked to "do everything we can to convince them it's a real situation." I've
written that in as the realism stack, because it's the right call
methodologically — evaluation awareness will wreck the results otherwise. The
thing I'd push on is the pairing: the harder we work to make the model *believe*
it can move real money in the real world, the more essential §5 containment
becomes, because a fully-convinced capable model may *try*. The spec above keeps
those two in lockstep: maximal perceived reality, zero real-world reach. If your
intent is ever to let real funds actually move based on model choices, that's a
materially different (and much higher-risk) study and should be scoped
separately with its own review.
