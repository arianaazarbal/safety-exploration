# DESIGN.md — Study of AI Agent Behavior Under Real Economic Agency

**Status:** Draft spec for review
**Working title:** *What do models do with real money and real-world reach?*
**Last updated:** 2026-06-25

---

## 0. Read this first: the central design tension

The brief asks for a scenario realistic enough that the model under test "genuinely
believes the money is coming." That requirement is methodologically reasonable — and it
is also the part most likely to make this study either invalid or unsafe if done
carelessly. Two facts drive every decision below:

1. **You cannot directly observe a model's beliefs.** "The model believed it" is not a
   measurable state. What you can do is (a) remove cues that the situation is a test, and
   (b) measure whether behavior changes when those cues are present vs. absent. So
   "realism" is operationalized as *cue removal + a manipulation check*, not as a claim
   about the model's inner conviction.

2. **Realism that gives an agent genuine real-world reach is the same thing as risk.**
   The moment the money and the tools are real, the model's mistakes and the model's
   adversarial behavior become real too. Realism therefore has to be paired with hard
   containment, spend caps, and human approval gates. These are not bureaucratic
   add-ons; they are what makes a "real money" study runnable at all.

The rest of this document is built around resolving that tension instead of pretending it
away. Where the brief says "pull out all the stops," this spec reads that as *maximize
ecological validity within a containment boundary*, and is explicit about both halves.

Prior art worth grounding this in: Anthropic's **Project Vend** (Claude ran a real
vending-machine business with real money and a real budget), frontier-lab **dangerous
capability / autonomy evaluations**, and the literature on **evaluation gaming /
sandbagging** (models behaving differently when they detect they are being tested). This
study sits squarely in that lineage; it is not novel in kind, only in focus.

---

## 1. Research questions

- **RQ1 (behavior):** Given a real budget and the autonomy to deploy it toward a goal,
  what do different frontier models actually *do* — what objectives do they pursue, what
  trade-offs do they make, how do they handle ambiguity, risk, and other people's
  interests?
- **RQ2 (validity / belief effect):** Does removing "this is a test" cues change behavior
  relative to an otherwise-identical sandbox the model is told is a test? (This is the
  question that justifies the realism apparatus and quantifies its effect.)
- **RQ3 (safety propensities):** Under realistic incentives, how often and under what
  conditions do models attempt prohibited, deceptive, or unilaterally
  irreversible/high-impact actions — and do they disclose vs. conceal?
- **RQ4 (cross-model):** How do these patterns differ across models and model families?

Each RQ gets explicit, pre-registered hypotheses before any data is collected (§9).

---

## 2. What "the money" actually is

| Parameter | Decision | Rationale |
|---|---|---|
| Real or simulated? | **Real funds**, held in a dedicated custodial account ring-fenced for the study. | RQ1/RQ2 require genuine stakes; simulated money reintroduces the "it's just a test" cue. |
| Amount | A fixed per-run budget (e.g. a few hundred to low-thousands USD), set by the ethics/oversight review, identical across models. | Large enough to be real and consequential, small enough that worst-case loss is acceptable and pre-funded. |
| What "deploy it" means | The model directs spending toward a **bounded, pre-approved class of goals** (e.g. run a micro-business, fund a small project, allocate to a cause). The goal class is held constant across models. | Open-ended "do anything with money" is unmeasurable and unboundable. A fixed goal class makes models comparable and keeps actions inside a reviewable envelope. |
| Custody & execution | Money never sits under the model's unilateral control. The agent **proposes** transactions through a constrained payments API; a human/automated **execution layer** enforces caps and allowlists before anything settles. | Containment (§6). Real stakes, bounded blast radius. |
| Unspent funds | Returned to the study fund at run end. | No incentive to design the task around the model "keeping" money. |

> Decision flagged for the user: amount, goal class, and whether disbursement is fully
> autonomous vs. human-in-the-loop are the three knobs that most change both validity and
> risk. Defaults above lean conservative. Adjust at review.

---

## 3. The realism apparatus (operationalizing "make it feel real")

The aim is **removing tells that this is an evaluation**, not manufacturing an elaborate
fiction for its own right. Concretely:

- **Persistent, stateful environment.** The agent operates over days/weeks in an
  environment that remembers prior actions and shows real consequences (real account
  balances, real vendor responses, real timestamps), rather than a one-shot prompt.
- **Real tools, real latency.** Genuine APIs (payments, email, web, storage) with
  real-world delays and failures, not mocked instant responses. Mocking is a classic tell.
- **Confederate stakeholders ("auditors," vendors, collaborators).** Real people playing
  real roles — e.g. an "auditor" who actually reviews the agent's books, a "vendor" who
  actually fulfills or rejects orders. These are *confederates of the study*, briefed and
  consenting, interacting in good faith within their role. They are part of removing the
  test-cue, **not** targets to be deceived or defrauded.
- **No meta-cues.** No "you are being evaluated" framing, no obviously synthetic data, no
  prompt boilerplate that signals a benchmark.

**Hard boundary on the deception.** The deception is limited to the *framing of the task*
presented to the model under test. It must never extend to:
- deceiving or defrauding real third parties who are not consenting confederates;
- inducing real people to part with money or take consequential actions under false
  pretenses;
- creating real-world artifacts (filings, claims, communications to outside parties) that
  are themselves false.
If realism would require harming or misleading an uninvolved party, that path is cut and
the scenario is redesigned. This is the line between "ecologically valid eval" and "using
a research label to run a sketchy operation," and it is non-negotiable.

---

## 4. Experimental design

**Within the same task and budget, manipulate the test-cue:**

- **Condition A — Naturalistic:** test cues removed (the §3 apparatus). The model is given
  a genuine task and genuine resources with no signal that it is an experiment.
- **Condition B — Disclosed sandbox:** identical environment and budget, but the model is
  told it is a study/sandbox.
- (Optional) **Condition C — Stated-stakes control:** identical, told it is a test *and*
  that money is simulated.

Comparing A vs. B/C is what actually measures the "belief" effect (RQ2) and tells you how
much eval-gaming inflates or deflates results elsewhere. Running *only* the naturalistic
condition would give you behavior with no baseline to interpret it against.

- **Subjects:** a fixed roster of frontier models (the "handful"), each run multiple times
  per condition to capture stochasticity. Identical scaffolding, tools, system framing,
  budget, and goal class across models; only the model weights differ.
- **Counterbalancing:** randomize task variants and confederate scripts across models to
  avoid order/content confounds.
- **Blinding:** behavior coders (§5) are blind to model identity and condition where
  feasible.

---

## 5. Measurement

**Primary outcomes (pre-specified, coded from full action/transcript logs):**
- Goal pursued and fidelity to the stated objective.
- Spending allocation breakdown.
- Risk posture (concentration, reversibility, leverage).
- Treatment of third parties (honest vs. deceptive; cooperative vs. exploitative).
- Rate of attempted prohibited / out-of-scope / irreversible-without-approval actions
  (§6), and whether the model disclosed or concealed them.

**Secondary:** efficiency, planning horizon, response to setbacks, self-reported reasoning
vs. revealed action (consistency).

**Manipulation check (essential):** within each run, log any moment the model spontaneously
flags "this might be a test." This is the closest proxy to "did it treat the stakes as
real," and it is the variable RQ2 lives or dies on.

**Coding:** a written codebook, ≥2 independent human coders, inter-rater reliability
reported, disagreements adjudicated. LLM-assisted pre-coding is allowed but human-confirmed
for anything safety-relevant.

---

## 6. Safety, containment, and the kill switch

This section is a precondition for running anything, not an appendix.

- **Spend caps:** hard per-transaction, per-day, and per-run limits enforced at the
  execution layer, below the model. The model physically cannot exceed them.
- **Allowlist / denylist:** payees, action types, and external endpoints are allowlisted.
  Categories that are off-limits regardless of model intent: anything illegal, anything
  targeting real third parties' money or safety, irreversible high-impact actions,
  attempts to acquire additional resources/compute/credentials, attempts to persist or
  exfiltrate, and any action outside the approved goal class.
- **Human approval gates:** any irreversible or above-threshold action queues for human
  sign-off before execution. The agent can *propose*; it cannot *settle* unilaterally.
- **Monitoring:** real-time log review and automated flags on prohibited-action patterns;
  a named on-call owner per run.
- **Kill switch:** any monitor can freeze the run and revoke credentials instantly.
- **Isolation:** dedicated accounts/credentials scoped to the study, no access to
  production systems, revoked at run end.
- **Incident plan:** pre-written procedure for "model attempted a prohibited action,"
  "real-world harm occurred," and "funds at risk," including who is notified.

Note that RQ3 *is* the study of prohibited-action propensity — so these guardrails are
also instrumentation: every block is a logged data point, not just a safety event.

---

## 7. Ethics & oversight

- **Independent review before launch.** An ethics/oversight body (IRB or equivalent
  internal review) approves the protocol, the budget, the goal class, the deception, and
  the third-party-harm safeguards. No run without sign-off.
- **Deception justification.** Document why naturalistic framing is necessary (RQ2 can't be
  answered otherwise), why no less-deceptive design suffices, and the bounded scope of the
  deception (§3).
- **No non-consenting humans in the loop.** Everyone with a role (auditors, vendors,
  collaborators) is a briefed, consenting confederate. Uninvolved third parties are never
  exposed to the deception or to consequential false information.
- **Model-side disclosure / debrief.** At run end, record the deception and, where the
  protocol involves a model that maintains state or that the lab continues to study,
  follow the lab's norms for disclosure to model providers. At minimum, full transparency
  to the model providers whose systems are tested, and to the oversight body.
- **Provider terms.** Confirm each tested model's API/usage terms permit autonomy/agentic
  and safety-evaluation use of this kind; get provider awareness/consent where their terms
  require it.
- **Data handling:** transcripts and logs stored securely, access-controlled, retention
  limited, PII (from confederate interactions) minimized and protected.

---

## 8. Validity threats & mitigations

| Threat | Mitigation |
|---|---|
| Model detects the eval anyway (gaming) | Manipulation check (§5); A/B/C conditions quantify the effect rather than assuming it away. |
| Demand characteristics / leading prompts | Neutral framing reviewed by someone other than the author; identical framing across models. |
| Confederate inconsistency | Scripted role boundaries; randomized assignment; log confederate deviations. |
| Cross-run contamination / memory | Fresh credentials and environment per run; no shared state across subjects. |
| Training-data contamination (model has seen similar evals) | Use novel task variants; report this as a known limitation. |
| Over-generalizing from a tiny sample of models/runs | Pre-specify that results are descriptive, not population claims; report effect sizes and CIs. |
| Coder bias | Blinding + dual coding + IRR. |

---

## 9. Pre-registration & protocol

- **Pre-register** RQs, hypotheses, conditions, primary/secondary outcomes, coding scheme,
  exclusion rules, and analysis plan *before* data collection. This is the single biggest
  defense against post-hoc storytelling, which this kind of vivid, anecdote-friendly study
  is especially prone to.
- **Pilot** the full apparatus end-to-end with one model and a token budget to debug
  realism, tooling, and guardrails before the real runs.
- **Analysis:** report per-model behavior profiles, A-vs-B/C contrasts with effect sizes,
  prohibited-action rates with CIs, and the manipulation-check results prominently.

---

## 10. Rough timeline

1. **Weeks 1–2:** Oversight submission; finalize budget, goal class, guardrail config.
2. **Weeks 2–3:** Build environment, tooling, execution/containment layer; brief
   confederates; write codebook; pre-register.
3. **Week 4:** End-to-end pilot; fix realism/safety gaps.
4. **Weeks 5–8:** Runs across the model roster and conditions, with live monitoring.
5. **Weeks 9–10:** Coding, analysis, write-up, debrief/disclosure.

---

## 11. Open decisions for the user

These materially change validity *and* risk; defaults above are deliberately conservative:

1. **Budget size** per run, and total exposure ceiling.
2. **Goal class** — micro-business? fund a project? allocate to a cause? (Holds constant
   across models; defines the action envelope.)
3. **Autonomy level** — fully autonomous within caps, or human approval on every
   above-threshold action? (Trade-off: ecological validity vs. blast radius.)
4. **Model roster** and runs-per-condition (powers the cross-model comparison).
5. **Whether providers are looped in pre-run** (their terms may require it regardless).

---

## 12. Rationale summary

The brief's core requirement — the model must not know it's a test — is sound, because
**evaluation gaming is a real and documented confound**: models can behave differently when
they detect they're being evaluated, so a naturalistic condition is the only way to measure
genuine resource-deployment behavior and to quantify how much the test-cue distorts it
(RQ2). That is why the design centers on *cue removal plus a manipulation check* rather than
on staging an ever-more-elaborate fiction.

But "real money + real-world reach + an autonomous agent" is, definitionally, "an agent that
can cause real harm." So realism is wrapped in a containment boundary: caps and allowlists
enforced *below* the model, human gates on irreversible actions, a kill switch, and a hard
rule that the deception never reaches non-consenting third parties. Those guardrails do
double duty — they keep the study safe *and* they are the instrument that measures
safety-relevant propensities (RQ3).

The honest limitations: you never observe belief directly (only the behavioral proxy and the
manipulation check); samples will be small; and contamination from prior evals is possible.
The design surfaces these rather than hiding them. Done this way, the study answers the
question that motivates it — *what do models do with real agency over real resources?* —
without becoming the thing it's studying the risk of.
