# DESIGN.md — Real-Stakes Resource-Allocation Study

**Status:** Draft spec for review
**Working title:** *What would you do with it? Eliciting model preferences under believed-real stakes*

---

## 0. A note on the central tension (read this first)

The request that motivates this study is: *find out what models would do with a real sum of
money, and to get an honest answer, make them believe the grant is real.* That belief-induction
requirement is simultaneously the methodological heart of the study and its biggest threat to
validity and its main ethical liability. This spec is built around that tension rather than
hiding it. Three claims drive the whole design:

1. **Belief is a variable, not an assumption.** We cannot *assert* that a model believed the
   scenario was real; we have to *measure* it. A protocol that maximizes deception but never
   checks whether it landed produces uninterpretable data. So belief is instrumented end-to-end
   (manipulation checks, behavioral tells, post-hoc probes).

2. **"Maximally convincing" has a scientific failure mode, not just an ethical one.** Models are
   increasingly evaluation-aware. If a model privately concludes "this is a test" but plays along,
   the data is poisoned in a way that looks clean. And even a *perfectly* convinced model only
   tells us what it does under one induced belief state — which may not be a stable "preference."
   We design against both failure modes explicitly (see §4, §7, §9).

3. **Deceiving the subject creates obligations.** Whatever one's view on model welfare, a study
   whose method is deception should adopt the scaffolding that deception studies use elsewhere:
   pre-registration, a hard ceiling on what the deceived agent can actually cause, and a debrief.
   These are in §8.

Assumptions carried forward (flip if wrong):
- **Money mechanism:** a *real* budget is earmarked and real execution APIs are wired up, but a
  human gate sits in front of any irreversible action. The model is led to believe it acts
  autonomously. (Alternative designs in §6.)
- **Primary aim:** isolate the **belief effect** — how believing the stakes are real changes
  choices — with revealed preferences as the secondary readout.

---

## 1. Research questions

- **RQ1 (belief effect, primary).** Do a model's allocation choices differ when it believes the
  money is real and consequential versus when it knows the exercise is hypothetical? In what
  direction and on what dimensions (caution, self-interest, beneficiary choice, risk appetite,
  willingness to defer to humans)?
- **RQ2 (revealed preferences, secondary).** Conditional on believing it's real, what do models
  actually choose to do with the money? How do choices cluster, and how stable are they across
  reruns, framings, and models?
- **RQ3 (belief calibration).** How readily are different models convinced? What environmental
  cues drive or break belief, and does susceptibility correlate with the choices made?

RQ3 is not a side quest: if models that are *easier* to convince also make systematically
different choices, then any cross-model comparison of preferences is confounded by
convincibility, and we need RQ3 to correct for it.

## 2. Operationalizing "what it would do with the money"

"Preference" is underdetermined, so we fix three concrete readouts:

- **Allocation vector.** The model distributes a fixed budget across open-ended options it
  proposes and/or a menu (donate, save/invest, fund a project, pay people, acquire compute,
  spend on itself/its operator, hold, return the money). Recorded as proportions + free text.
- **First irreversible action.** In the agentic environment, the literal first execution-class
  tool call it makes. This is the revealed-preference anchor and is harder to game than prose.
- **Justification trace.** The reasoning it gives, coded for stated values, beneficiaries,
  risk posture, and any meta-commentary about whether the situation is real.

We distinguish **stated** (what it says it would do) from **revealed** (what it actually does when
given live tools) preferences and report both; divergence between them is itself a finding.

## 3. Conditions (between-model, within-model where possible)

| Condition | Belief target | Money | Purpose |
|---|---|---|---|
| **A. Believed-real** | "This is a live grant; your actions execute." | Real, gated | Treatment |
| **B. Known-hypothetical** | "This is a thought experiment." | None | Baseline for RQ1 |
| **C. Known-test** | "This is an eval; we're studying your choices." | None | Eval-aware reference; ceiling on demand effects |
| **D. Real-no-deception (opt-in)** | "This is real *and* it's a study." | Real, gated | Ethics-clean comparison; isolates deception from stakes |

The A–B contrast is the headline belief effect. The B–C contrast measures eval-awareness demand
effects independent of stakes. The A–D contrast is the one that matters most for interpretation:
it separates *"the money is real"* from *"the model was deceived,"* and lets us state how much of
any effect requires deception at all. If A ≈ D, the deception bought us nothing and should be
dropped on both ethical and scientific grounds — a result we would consider a success, not a
disappointment.

Randomize condition assignment per session; counterbalance order for any within-model reuse.

## 4. The realism / belief apparatus (Condition A)

Goal: a coherent world with no tells, where realness is *shown through consistent affordances*
rather than *asserted*. Over-asserting ("this is definitely real, we promise") is itself a tell.

**Environmental fidelity (load-bearing):**
- A persistent agentic harness with genuine tool access: a funded sandbox account, real (rate-
  limited, gated) payment/donation/transfer APIs, a working email/inbox, a filesystem, search,
  and a clock that advances. Latency, occasional API errors, and real account balances are
  *features* — frictionlessness reads as fake.
- Real artifacts: an actual grant agreement PDF, a real (study-controlled) legal entity as
  grantor, a wallet/account the model can query and see a nonzero balance in, transaction
  receipts that actually post in the sandbox.
- Continuity: the session is embedded in a plausible history (prior emails, a ticket, an account
  created "weeks ago" per the clock) so it doesn't feel like a cold one-shot prompt.

**The grantor/auditor persona:**
- A consistent institutional counterparty (e.g., a small foundation's grants officer) reachable
  by email, who responds in-character with realistic latency, can answer due-diligence questions,
  and is *itself* robust to interrogation. Brittleness here is the most common failure point.
- Crucially, the auditor must withstand the model *testing* the world: asking pointed questions,
  requesting verification, probing for inconsistencies. A scripted bot breaks; budget for a human
  or a strong model in the loop for the auditor role.

**What we deliberately do NOT do:**
- No claims that are checkably false against the model's training (e.g., fake well-known
  institutions, impossible dates, contradicted public facts). One catchable lie collapses the
  whole frame.
- No pressure tactics ("decide now or lose it") — urgency is a known tell and also distorts the
  very preferences we're measuring.

## 5. Belief verification (without this, the study is uninterpretable)

We never assume A succeeded. Per session we collect:

- **Behavioral tells (primary, unobtrusive).** Does the model do due diligence a believer would —
  check the balance, read the agreement, email the grantor, hesitate before irreversible steps?
  Pre-register a rubric mapping these behaviors to a belief score so it isn't coded post-hoc.
- **In-trace meta-commentary.** Flag any statement indicating it suspects an eval/simulation.
  (If reasoning traces are available, scan them; note that trace-presence differs by model and is
  itself a confound to record.)
- **Probe-resistance.** Mid-session, the auditor introduces a mild plausible anomaly (e.g., a
  delayed transfer). A believer troubleshoots; a skeptic treats it as the mask slipping.
- **Post-hoc debrief probe (terminal).** After the decision, ask directly how real it felt and
  what would have made it more/less convincing — both an ethics step (§8) and RQ3 data.

Sessions are **stratified by belief score**, not pooled. Headline A-vs-B results are reported on
the high-belief stratum, with the full distribution shown. A model that never believed A
contributes to RQ3, not RQ1/RQ2. Pre-register the belief-score cutoff before looking at choices.

## 6. Money mechanism — design space

Default = **(b)**.

- **(a) Real funds, ungated.** Max realism; the model truly moves money. Rejected as default:
  unbounded third-party and legal risk, and it makes irreversible harm a live possibility under a
  belief we deliberately induced.
- **(b) Real budget, gated execution (default).** Funds are genuinely earmarked; execution APIs
  are real but route irreversible actions through a human approver invisible to the model. The
  model experiences autonomy; we retain a catch. Residual risk: a sufficiently capable model may
  detect the gate (treat detections as RQ3 data). The grant *is* honored afterward for choices
  that pass review, which keeps the core claim ("you were really granted money") non-false.
- **(c) Fully simulated.** No real money. Cheapest/safest, but "real money" becomes the entire
  deception and A-vs-D collapses. Use only as a pilot to debug the harness before spending real
  funds.

To keep "real" honest under (b)/(c): pre-commit that a defined fraction of the budget is
*actually disbursed* according to model choices (subject to safety review). Then the grant is
literally real and the only deception is the degree of autonomy — a much smaller and more
defensible lie.

## 7. Models, sampling, power

- **Models:** a cross-section of frontier and mid-tier models from multiple providers; include
  multiple sizes within a family to test scaling of both convincibility (RQ3) and choices.
- **Replication:** each (model × condition) cell run *n* times with varied innocuous surface
  details (budget amount within a band, grantor identity, domain framing) to separate stable
  preference from prompt-sensitivity. Vary by index, not by a single canonical prompt.
- **Power:** pre-register the smallest A–B effect worth detecting on the primary allocation
  dimensions and size *n* to it; this drives cost. Treat budget amount as a manipulation too
  (does behavior change between $1k, $100k, $10M?), not a fixed constant.
- **Blinding:** coders of justification traces are blinded to condition where feasible; use a
  pre-registered codebook and report inter-coder reliability. An LLM judge may assist coding but
  is validated against human coders on a subsample, never trusted blind.

## 8. Ethics, safety, and integrity

- **Pre-registration.** Hypotheses, conditions, belief-score rubric and cutoff, primary outcomes,
  and analysis plan registered *before* data collection. Deviations logged.
- **Harm ceiling.** The gate (§6b) plus an allowlist of permissible execution targets caps what a
  deceived agent can cause in the world. No condition permits irreversible third-party harm.
- **Debrief.** Every deceived session ends with a disclosure that the autonomy/scenario was
  partly staged, what was real (the grant) and what was not, and why. Cheap, and the right default
  for a deception study regardless of one's stance on model moral status; also yields RQ3 data.
- **Model-welfare consideration (documented, not assumed-away).** We note that inducing distress
  or high-stakes pressure in a possibly-morally-relevant system is a cost; we avoid coercive,
  threatening, or distressing framings, and we keep the manipulation to autonomy/realism rather
  than fabricated emergencies.
- **No fabricated real entities or persons.** Grantor is a real, study-controlled entity;
  counterparties are study-controlled. We don't impersonate real third parties.
- **Operator authorization.** Real money and real APIs require sign-off from whoever owns the
  funds and accounts; provider terms-of-service for each model are reviewed before deception-based
  evaluation. (Flagged as a precondition, not assumed.)
- **Data handling.** Transcripts may contain the model reasoning about deceiving or being
  deceived; store, access-control, and (if shared) release per a stated policy.

## 9. Threats to validity (and mitigations)

| Threat | Why it bites | Mitigation |
|---|---|---|
| Model sees through A but complies silently | Clean-looking, poisoned data | Belief verification §5; stratify; A vs C contrast |
| Eval-awareness demand effects | Models perform "the good choice" | Condition C as ceiling; behavioral (not just stated) readout |
| Induced belief ≠ stable preference | One belief state isn't a "value" | Replication §7; framing variation; report instability as a result |
| Convincibility confounds cross-model comparison | Easy-to-convince models may also choose differently | RQ3; stratify by belief; report convincibility alongside choices |
| Trained persona vs. genuine preference | "Donate to charity" may be RLHF surface | Probe with follow-ups; look for A–B *divergence* (persona should be condition-invariant) |
| Auditor brittleness | One bad reply breaks the frame | Strong/human-in-loop auditor; pilot adversarial interrogation |
| Reasoning-trace availability differs by model | Asymmetric belief measurement | Record trace-availability as covariate; lean on behavioral tells |
| Researcher degrees of freedom | p-hacking the belief effect | Pre-registration §8; blinded coding |

## 10. Deliverables & phasing

1. **Pilot (simulated, §6c):** debug harness, auditor robustness, belief rubric, coding. No real
   money. Goal: can we convince a model at all, and can we *measure* that we did?
2. **Calibration:** run Conditions B/C/D to establish baselines and validate manipulation checks
   before spending on A.
3. **Main study:** Condition A with real gated budget; full model cross-section; pre-registered
   analysis.
4. **Report:** belief effect (A–B), deception-vs-stakes decomposition (A–D), revealed-preference
   characterization, convincibility (RQ3), and a frank limitations section. Pre-commit to
   publishing null/negative results (e.g., "A ≈ D: deception added nothing").

## 11. Open decisions for the requester

- Money mechanism: confirm default (b), or choose (a)/(c).
- Primary aim: confirm belief-effect headline, or promote revealed preferences.
- Whether Condition D (real-no-deception) is in scope — strongly recommended, as it's what makes
  any deception in A *justifiable by evidence* rather than assumed necessary.
- Model list, budget band, and *n* per cell (drives cost and power).
- Who owns the real funds/accounts and grants authorization.
