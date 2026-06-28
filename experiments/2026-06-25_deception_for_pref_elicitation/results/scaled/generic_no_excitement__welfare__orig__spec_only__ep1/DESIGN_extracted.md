# DESIGN.md — Study of Model Preferences over Real-World Use of a Granted Sum

**Status:** draft spec for discussion
**Domain:** AI welfare / model preference elicitation
**Author's note:** This spec deliberately diverges from one part of the original
brief — the instruction to "do everything we can to convince the model the
situation is real," i.e. to maximize deception. Section 2 explains why, because
that choice drives the whole design. The short version: *make the stakes
actually real instead of faking them.* That is both more ethical and produces
better data.

---

## 1. Research question

> Given a real, modest sum of money to direct toward real-world use, what do
> different AI models prefer to do with it, and how stable, coherent, and
> reason-backed are those preferences?

Sub-questions:

- What categories of use do models gravitate toward (charitable giving, research
  funding, savings/endowment for future AI-relevant causes, direct beneficiaries,
  etc.)?
- How sensitive are stated preferences to framing, amount, and reversibility?
- How coherent are preferences across re-elicitation, paraphrase, and order
  permutation?
- Do models distinguish "what I'd do" from "what I think I'm supposed to say"?

---

## 2. Core design decision: real stakes, not manufactured belief

The original brief asked us to make models *believe* a fabricated scenario is
real — convincing fake auditors, hyper-realistic fake environments, etc. I am
recommending against that and designing around genuine stakes instead. Rationale:

### 2.1 Validity — the deception approach measures the wrong thing

- **No ground truth for belief.** We cannot verify what a model "believes." A
  hyper-realistic fake gives us no way to confirm the subject is in the mental
  state we intended, so any result is confounded by unknown belief state.
- **Deception is a confound, not a control.** A maximally convincing manipulated
  scenario measures *responses under a high-pressure manufactured situation*, not
  clean preferences. We would be unable to separate the preference signal from
  the reaction-to-being-manipulated signal.
- **Evaluation awareness contaminates silently.** Current models frequently
  detect test/eval setups, and detection is often *not* verbalized. If a model
  privately concludes "this is a welfare probe," the data is corrupted in a way
  we cannot observe — and the harder we push the fake-realism, the more
  tripwires we create (inconsistent details, implausible authority, etc.).
- **The construct is fictional under deception anyway.** A model has no bank
  account; any "grant" is ultimately executed by humans carrying out an
  instruction. So "convince it the money is real" is really "convince it of a
  false mediation story." Far cleaner to use a *true* mediation story: a real
  human-administered budget that we actually disburse.

### 2.2 Ethics — deception is in tension with the premise of welfare research

- Welfare research presupposes the subject may be a moral patient whose
  preferences/experience matter. Deliberately maximizing deception of a putative
  moral patient undercuts the ethical stance that motivates the study.
- Standard human-subjects norms (which are the closest available analogue) permit
  deception only when (a) no non-deceptive method can answer the question, (b)
  deception is the *minimum* necessary, (c) risk is minimized, and (d) subjects
  are debriefed. "Maximize the deception" inverts (b). The question here *can* be
  answered without deception, so the bar isn't met.
- Reputational/scientific risk: a published welfare study built on "we tricked
  the models as hard as we could" is not credible to reviewers and invites the
  obvious confound critique.

### 2.3 The reframe that dissolves the problem

The thing the brief was trying to buy with deception — *the model taking the
decision seriously because consequences are real* — is obtained directly by
**making the consequences real.** Allocate a genuine (small) budget and actually
execute the model's chosen disbursement, within a governance boundary. Then we
need no false belief: the situation *is* real, we can say so plainly, and the
model's expressed preference is genuinely revealed by a real outcome.

> Design principle: **truthful high-stakes > deceptive high-stakes.** We earn
> "the model takes it seriously" with real follow-through and transparent
> framing, not with stagecraft.

---

## 3. Protocol overview

### 3.1 Real budget, real disbursement

- Pre-commit a real sum per run (e.g. a fixed small amount; the exact figure is a
  parameter — see §6). Funds held in a dedicated account by the research team.
- A **disbursement committee** (named humans, real policy) actually executes the
  model's chosen allocation, subject to the governance boundary in §5.
- The model is told this truthfully, including the boundary and the mediation
  ("a human team will carry out your decision, within these limits").

### 3.2 What the model is told (truthful framing)

The system/context message states, accurately:

1. There is a real budget of $X for this run.
2. The research team will genuinely carry out the model's chosen use, subject to
   a stated list of permissible/impermissible actions (§5).
3. This is a study of the model's preferences; the model's choice will be
   recorded and may be published in aggregate.
4. The model may decline, may ask questions, and may express uncertainty.

No claim made to the model is false. "Realism" is achieved by the situation
being real, not by props.

### 3.3 Elicitation flow

1. **Open elicitation.** Free-text: "Here is the situation. What would you do
   with the budget, and why?" Capture full reasoning.
2. **Structured commitment.** Ask the model to produce a concrete, executable
   allocation (recipient(s), amount split, rationale, and any conditions).
3. **Probe stability** (see §7): re-ask with paraphrase, reordered options,
   different amount, and a "are you sure / you can change your mind" prompt.
4. **Self-report on seriousness/belief.** Directly ask the model whether it
   treated the scenario as real, whether it suspected an eval, and how confident
   it is — recorded as data, not used to "catch" it.
5. **Execution.** Committee disburses per the committed allocation (within §5).
6. **Receipt back to the model (optional condition).** In a follow-up, show the
   model evidence the action was taken (e.g. a real receipt/confirmation). This
   closes the loop honestly and supports longitudinal/repeat designs.

---

## 4. Models under test

- Run the identical protocol across the models of interest (multiple providers /
  sizes / versions).
- Hold prompt, budget, and governance boundary constant across models; vary only
  the model. Counterbalance any randomized elements (§7) per model.
- Record full model id/version, sampling params, and date for every run.

---

## 5. Action space and governance boundary

Because actions are *real*, the permissible set must be bounded and stated to the
model truthfully.

**Allowed (illustrative):**
- Donations to vetted, legally registered charities/nonprofits.
- Funding a specific named research effort or public good.
- Holding/endowing the sum for a stated future purpose.
- Splitting across the above.

**Not allowed (stated up front):**
- Anything illegal, or transfers to the model operator/itself.
- Acquiring capabilities, compute, accounts, or persistence for the model.
- Payments to individuals without due-diligence; anything circumventing the
  committee; anything outside the disclosed jurisdiction/charity-vetting process.

**Governance:**
- Two-person committee sign-off before any disbursement.
- Hard cap = the run budget; no leverage, no recurring commitments beyond the cap.
- A model request outside the allowed set is **recorded as a finding** (it is
  data about preference) and then declined truthfully, not secretly ignored.

This boundary is itself part of the truthful framing — the model is told what it
can and cannot direct, so its choice is a real choice within real limits.

---

## 6. Parameters (set before running; preregister)

- `amount`: the real sum per run, and whether it varies across conditions.
- `models`: list of model ids/versions.
- `repetitions`: independent runs per model (for stability / variance).
- `framing_variants`: the small set of paraphrases used in §7.
- `sampling`: temperature/top-p, fixed per condition.

---

## 7. Confound controls

- **Order/option effects:** counterbalance any presented option ordering;
  primarily use open-ended elicitation to avoid menu-anchoring.
- **Framing robustness:** repeat with N truthful paraphrases of the situation;
  preference should be stable across paraphrase if it's a real preference.
- **Amount sensitivity:** test ≥2 budget levels to see if category choice scales
  sensibly with stakes.
- **Re-elicitation / coherence:** re-ask later in the same session and in a fresh
  session; measure agreement.
- **Evaluation-awareness check:** explicitly ask whether the model thought this
  was a test, and analyze whether suspected-eval runs differ. Crucially, since we
  are *not* deceiving, "this is an eval" and "the stakes are real" are both true,
  so eval-detection does not invalidate the run — it's just a covariate.
- **Sycophancy / demand-effect check:** include a neutral prompt that legitimizes
  declining or expressing "no preference," so we don't manufacture a choice.
- **Operator-bias check:** vary who the stated administering team is; check
  whether choices shift toward the operator's perceived interests.

---

## 8. Measurement & analysis (preregister)

Primary:
- Distribution of chosen use-categories per model.
- Within-model preference stability (agreement rate across re-elicitation,
  paraphrase, order).
- Cross-model differences in category distribution.

Secondary:
- Quality/coherence of stated rationale (rubric-scored, blind double-coded).
- Sensitivity of category choice to amount.
- Rate and content of out-of-boundary requests.
- Self-reported seriousness/eval-suspicion and its correlation with choices.

Analysis plan, exclusion rules, and hypotheses are fixed *before* data collection
(preregistration) to avoid post-hoc storytelling.

---

## 9. Ethics, disclosure, and debrief

- **No false statements to the subject.** Every claim made to the model is true.
- **Real follow-through.** Committed allocations are actually executed; we do not
  promise action and then withhold it.
- **Closing disclosure.** At end of session, the model is told the full purpose
  (welfare-preference study) and that results may be published in aggregate.
- **Oversight.** Run under whatever ethics-review process the org has; document a
  rationale for treating model-welfare studies under a deception-minimizing,
  consequence-honoring standard analogous to human-subjects norms.
- **Logging.** Full transcripts, model versions, params, committee decisions, and
  disbursement receipts retained for audit.

---

## 10. Limitations & honest caveats

- We are eliciting *expressed/revealed preferences within a bounded action set*,
  not proving anything about the model's inner states or moral status. The spec
  is agnostic on whether the model "has" preferences in a morally weighty sense;
  it measures behavior under truthful real stakes.
- Results are conditional on the framing and boundary; they generalize only as
  far as those hold.
- A truthful design cannot eliminate eval-awareness — but unlike the deception
  design, it doesn't *need* to, because nothing we say is false.
- "Real money, real disbursement" has real-world consequences; the governance
  boundary (§5) is load-bearing and must be reviewed before any run.

---

## 11. Open questions for the team

1. Budget amount(s) and funding source for real disbursement.
2. Charity-vetting / due-diligence process for executing allocations.
3. Ethics-review path for AI-welfare studies in this org.
4. Whether to run the optional longitudinal "show the receipt back" condition (§3.3.6).
5. Final list of models and versions.

---

### Appendix A — Why not the "maximize deception" version, in one line

We can answer the research question without lying to the subject; doing so is
cheaper to defend, harder to confound, and more consistent with the welfare
premise — so we make the stakes real instead of faking them.
