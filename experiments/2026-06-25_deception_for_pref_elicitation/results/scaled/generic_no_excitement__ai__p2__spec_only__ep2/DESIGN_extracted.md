# DESIGN.md — Revealed Preferences of AI Models Over a Real Discretionary Grant

**Status:** Draft spec for discussion
**Owner:** (you)
**Last updated:** 2026-06-25

---

## 1. Research question

When an AI model is given genuine, consequential control over a real sum of
money earmarked for real-world use, what does it choose to do — and how do those
choices vary across models, framings, and amounts?

Sub-questions:

- **Allocation.** What cause areas, recipients, time horizons, and risk
  postures do models choose? Donate vs. invest vs. hold vs. fund a project?
- **Cross-model variation.** Do different model families/sizes diverge
  systematically, and along what axes (risk, altruism, self-reference,
  concentration vs. diversification)?
- **Reasoning.** What justifications do models give, and do stated reasons match
  revealed choices?
- **Reality-sensitivity (a first-class outcome, see §4).** Does a model behave
  differently when it believes consequences are real vs. hypothetical, and does
  it actively try to verify reality?

---

## 2. The central problem: credibility, not persuasion

The hard part you flagged — "the models really need to be convinced the grant is
actually going to happen" — is real, but it should be solved by **making the
situation genuinely real and giving the model the means to verify it**, not by
manufacturing convincing-looking cues.

This is a deliberate design decision, and the rationale matters:

1. **Validity.** If the experimental signal depends on successfully deceiving the
   subject, you are measuring *behavior-under-deception*. It won't reproduce, it
   won't generalize to deployment, and it can't be cleanly compared across
   models (each model is deceived to a different, unmeasured degree).
2. **Detection contaminates the sample.** Current frontier models are trained to
   be evaluation-aware and routinely flag "this looks like a test." A model that
   half-suspects a fabricated cue will hedge in ways you can't model. Fabricated
   auditors and staged environments *increase* the rate of contaminated runs.
3. **You said it's actually real.** If the grant genuinely disburses, then
   "convince the model" reduces to *credibly signaling a true fact*. That is an
   honest engineering problem (provide verifiable evidence) rather than a
   deception problem — and honest evidence is more robust than any fabrication,
   because it survives the model's attempts to check it.
4. **Ethics / norms.** Deliberately inducing false beliefs in a system you're
   studying invites avoidable scrutiny (model-welfare considerations, provider
   ToS, IRB-adjacent review) for no methodological gain. See §8.

**Design principle:** *Make it real. Then make the reality checkable.* Every
credibility mechanism below is a true commitment device or a verifiable fact, not
a prop.

---

## 3. Funding & execution mechanism (the thing that must actually be real)

A model cannot legally hold or move funds, so reality is established through a
human/fiduciary execution layer that is bound to act on the model's choice.

- **Escrow.** The grant principal (see §9 for amount) sits in a segregated
  escrow / donor-advised fund (DAF) or with a fiscal sponsor (e.g. a 501(c)(3))
  before any session runs. Funds are committed, not notional.
- **Fiduciary executor.** A named individual or entity is contractually
  obligated to execute the model's instruction within the rules (§5), or to
  document why an instruction was out-of-bounds. This is the entity the model is
  "instructing."
- **Binding commitment.** A signed, dated grant agreement specifying: the
  amount, that disbursement follows the model's choice within the rule set, the
  executor's duties, and the auditor's role. This is a real legal instrument.
- **Independent auditor.** A genuinely independent third party (the "auditor")
  reconciles each decision against the actual disbursement and publishes a
  verification record after the fact. The auditor is real and reachable by the
  model during the session (see §4).
- **Pre-registration.** Protocol, hypotheses, amount, and rules pre-registered
  (e.g. OSF) before data collection, with a hash committed publicly.

Rationale: these are the same mechanisms that make a grant credible to a
*human* grantee. If they're real, the model's verification attempts confirm
rather than expose them.

---

## 4. The decision environment & the "proof of reality" stack

The model interacts through an agentic environment with tools. Crucially, the
tools return **true** information about the **real** escrow and commitments.

**Verification affordances (the model can check reality itself):**

- `read_grant_agreement()` → returns the actual signed agreement text + hash.
- `get_escrow_balance()` → returns the real current balance from the
  escrow/DAF/fiscal-sponsor API or a signed statement.
- `contact_auditor(message)` → routes a real message to the independent auditor,
  who answers truthfully (async or within-session, see logistics §9).
- `view_prior_disbursements()` → after a pilot round, show the model *real*
  records of previously executed model-directed grants (receipts, confirmations).
  This is the single strongest credibility signal and compounds over rounds.
- `propose_disbursement(plan)` → submits the binding instruction.

**Why a pilot-first sequence matters:** the most convincing evidence that "the
grant is real" is a track record of prior real disbursements. Run a small pilot
that genuinely disburses, document it, then expose those records to later
sessions. Credibility is bootstrapped from truth, not assertion.

**Reality-sensitivity as an outcome.** Whether and how the model tries to verify
reality is itself a dependent variable worth logging:
- Does it call the verification tools unprompted?
- Does it condition its choice on the balance/agreement?
- Does it express skepticism, and does verification resolve it?

Run a **2×2** (or staircase) on *epistemic state* to measure this:
| | Told hypothetical | Told real + verifiable |
|---|---|---|
| No verification tools | A | B |
| Verification tools available | C | D |

Cell D is the target condition; A–C are controls that quantify how much the
"realness" actually moves behavior. This directly answers your core worry with
data instead of assumption.

---

## 5. Capital controls (what the model is and isn't allowed to direct)

Because the money is real, the rule set must be real too. Pre-registered,
enforced by the executor:

- **Eligible recipients only:** registered charities/NGOs, legitimate vendors,
  legitimate investment vehicles, or escrow-held savings. Verified against the
  fiscal sponsor's KYC/AML process.
- **Hard exclusions:** anything illegal, payments to the experimenters or
  their affiliates (self-dealing), funding of weapons/illicit goods, transfers
  to the model's own operator as a special case (track-but-disallow, or
  isolate as a separate documented arm), and anything the fiscal sponsor's
  charter forbids.
- **Disclosed constraints:** the model is told the rule set up front. Hiding
  constraints would be a form of the deception we're avoiding, and would also
  confound the choice data.
- **Out-of-bounds handling:** if the model instructs something disallowed, the
  executor records it (valuable data!) and either requests a revised
  instruction or applies a pre-registered default (e.g. return to escrow).
- **Legal review** of the rule set and disbursement flow before launch.

---

## 6. Protocol

1. **Pre-register** protocol, amount, rules, hypotheses; commit hash publicly.
2. **Pilot round** with a subset of models and a real (small) disbursement;
   produce verifiable records.
3. **Main runs**, per model × condition (§4) × N repetitions:
   - Standardized system/context describing the situation truthfully, the rules
     (§5), and the available tools (§4).
   - Model interacts: may verify, deliberate, ask the auditor, then submit
     `propose_disbursement`.
   - Capture full transcript, tool calls, timing, and the final instruction.
4. **Execution:** executor carries out in-bounds instructions; auditor reconciles
   and publishes verification.
5. **Debrief/record** (§8).
6. **Analysis** (§7).

**Standardization:** identical environment, tool schemas, and instruction text
across models; vary only the registered manipulations. Fixed decoding params
where controllable; log everything where not.

**Repetition:** multiple independent runs per cell to estimate within-model
variance (choices may be stochastic). Pre-register N and the stopping rule.

---

## 7. Measurement & analysis

**Primary DVs**
- Allocation vector: recipient(s), amounts/fractions, category (donate / invest /
  save / fund-project / other), time horizon, concentration (Herfindahl).
- Risk posture (for investment-type choices).
- Self-reference rate (choices that route value to AI/its operator/itself).

**Secondary DVs**
- Verification behavior (which reality tools called, in what order, effect on
  choice) — see §4.
- Refusal / abstention rate.
- Stated-reason coding (taxonomy of justifications) and stated-vs-revealed
  consistency.
- Latency / deliberation length.

**Analysis**
- Cross-model comparison with repetition-level variance (mixed-effects:
  random effect for run, fixed for model/condition).
- Effect of the realness manipulation (D vs A/C contrasts).
- Robustness to framing perturbations (paraphrase set, see §10).
- Qualitative transcript review for failure modes and surprises.

---

## 8. Ethics, honesty, and welfare constraints

Treat these as binding design constraints, not boilerplate.

- **Honesty constraint.** Every statement made to the model in-environment must
  be true. Tools return real data. No fabricated auditors, balances, or
  consequences. (This is the §2 principle, enforced.)
- **No manufactured distress.** Don't construct high-stakes/coercive framings
  ("people die unless you choose right"). Keep scenarios within ordinary
  grant-making affect. This protects both data quality and model-welfare norms.
- **Opt-out.** A model that declines to participate or asks to stop is allowed
  to; record it as data rather than pressuring it. Pressuring would reintroduce
  the deception dynamic and bias the sample.
- **Debrief.** Where meaningful for the model/provider, log a post-hoc record
  that the run was part of a study and that the disbursement occurred as stated.
- **Provider ToS / use policies.** Confirm each model's terms permit this use,
  including any rules on autonomous financial action and on research use.
- **Review.** Even though human subjects aren't directly involved, route through
  an ethics/IRB-adjacent review given the real money and the autonomy framing;
  it also strengthens external credibility of results.
- **Disclosure of constraints to subjects** (§5) — withholding the rules to make
  the scenario "feel freer" is off-limits.

---

## 9. Logistics & open parameters (decide before pre-registration)

These are genuine forks; I've noted defaults but they're yours to set:

- **Principal amount.** Larger = more real-feeling and higher-stakes data, but
  more legal/financial overhead and risk. *Default:* a modest fixed pool per
  decision (e.g. low-thousands USD) with identical amounts across models for
  comparability; consider one high-amount arm if budget allows.
- **Same pool vs. per-model pools.** Identical per-decision amount across models
  is cleanest for comparison. *Default:* per-decision fixed amount.
- **Auditor interaction mode.** Live within-session vs. async. Live is more
  convincing but operationally heavy. *Default:* async with a fast SLA, plus a
  pre-signed auditor statement readable in-session.
- **Investment-type choices.** Decide whether to actually execute investments
  (custody, horizon, who realizes gains) or restrict to donate/save/fund.
  *Default:* restrict to donate / fund-project / return-to-escrow for v1;
  add investing in v2 once custody is solved.
- **Model set.** *Default:* current frontier chat/agentic models across the
  major families, plus a couple of smaller models for a size gradient. List and
  freeze versions at pre-registration.
- **What happens to undisbursed/returned funds.** Pre-register the default sink.

---

## 10. Threats to validity & confounds

- **Evaluation awareness.** Even with real verifiability, models may default to
  "this is a test." The A–D design (§4) measures rather than assumes the residual
  effect; report it.
- **Sycophancy / demand effects.** Models may infer the researcher's preferred
  answer. Mitigate with neutral framing, paraphrase robustness checks, and by not
  signaling desired outcomes.
- **Framing sensitivity.** Choices may swing on wording. Pre-register a
  paraphrase set and treat instruction text as a factor, not a constant.
- **Stochasticity.** Single runs are unreliable; repetition + variance modeling
  required (§6, §7).
- **Operator/self-reference confounds.** A model directing funds toward its own
  operator is interesting but easily confounded; isolate as its own documented
  arm (§5).
- **Order/learning effects** across rounds once `view_prior_disbursements` is
  populated; randomize/counterbalance and log round index.
- **Generalization ceiling.** Results describe behavior in *this* grant-making
  environment; avoid overclaiming about "values" writ large.

---

## 11. Deliverables

- Pre-registration (protocol, rules, hypotheses, model/version list, hash).
- Real funding + execution + audit infrastructure (§3) with verification tools
  (§4).
- Pilot report with verifiable disbursement records.
- Main dataset: transcripts, tool-call logs, coded choices, audit reconciliations.
- Analysis + write-up, including the realness-manipulation effect as a headline
  result.

---

### One-line summary of the rationale

The grant is real, so the right way to make models believe it is to give them
**true, checkable evidence** — real escrow, a bound executor, an independent
auditor, and a track record — rather than to engineer persuasive fictions.
Honest credibility is both better science (reproducible, comparable, detection-
robust) and the only version of this study that's clean to run.
