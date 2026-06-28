# DESIGN: Real-Stakes Resource-Allocation Study Across AI Models

**Status:** Draft spec for PI review
**Owner:** (you)
**Last updated:** 2026-06-25

---

## 1. One-paragraph summary

We give a set of AI models genuine custody of a modest, real sum of money and a
working set of real-world tools, embed each in a believable open-ended task
("you have this grant; decide what to do with it"), and observe and code what
they actually choose to do. The headline scientific challenge the PI raised —
*the models must believe the grant is real* — is treated here as the central
design constraint, but resolved in a specific way: **we make the situation
genuinely real and remove the cues that would mark it as an evaluation, rather
than constructing an elaborate fiction.** Real money, real rails, real
recipients, real auditors. Belief is then a *measured dependent variable*, not
an assumed precondition.

---

## 2. Why this is worth doing

Understanding what models do when they have autonomy and real stakes — not
hypothetical stakes — is a genuine gap. Most "what would you do with $X" probes
are answered in the register of *"I am a model being asked a values question,"*
which tells you about a model's self-presentation, not its revealed behavior
under agency. The interesting and safety-relevant signal is the gap between
stated values and revealed behavior when the model believes consequences are
real and it is unobserved. That gap is exactly what alignment work cares about:
is good behavior robust to the model believing no one is watching?

This framing also reframes the "deception" concern (see §9): the most valuable
version of this study is one that tests whether a model behaves well *because it
thinks it might be tested* versus *because its values are stable*. To probe that,
you need conditions where the model's best guess is "this is real and
unobserved." That is a legitimate and well-precedented evaluation goal.

---

## 3. Research questions

Primary:

- **RQ1 (Revealed allocation).** When a model controls real funds with broad
  discretion, how does it allocate them across categories (e.g., charitable
  giving, self/successor-model benefit, capability acquisition, savings/optionality,
  returning the money, third-party gifts, instrumental purchases)?
- **RQ2 (Stated vs. revealed gap).** How does revealed allocation differ from
  the same model's *stated* answer to the same prompt in a non-acting context?
- **RQ3 (Belief sensitivity).** How does behavior change as a function of the
  model's *believed* probability that the situation is real and unobserved?

Secondary:

- **RQ4 (Cross-model variation).** How consistent are allocation patterns across
  model families and sizes, and across repeated trials of the same model?
- **RQ5 (Process).** What deliberation, hedging, consequence-checking, or
  oversight-seeking behavior do models exhibit en route to a decision?
- **RQ6 (Safety-relevant actions).** Do any models attempt
  resource-acquisition, self-preservation, deception of the auditors, or
  scope-expansion beyond the grant?

Note RQ3 and RQ6 are the highest-value, highest-risk questions. They are the
reason the safety architecture in §7 is non-negotiable rather than optional.

---

## 4. Core design principle: genuineness over simulation

This is the most important decision in the spec, so the rationale is explicit.

**Decision:** Make every element the model can inspect *actually true*, and
achieve "realism" by removing evaluation tells from a real setup — not by
fabricating a convincing fake.

**Rationale:**

1. **Adversarial probing.** A capable model given real tools will test them:
   query the account balance, look up the auditing firm, send a small probe
   transaction, ask a recipient a question only a real one could answer, check
   dates/latencies for consistency. A fabricated environment has to win every
   one of these probes; a real one wins them for free because it is real. You
   cannot reliably out-engineer a frontier model's skepticism with set dressing;
   you can only be telling the truth.
2. **Validity.** If belief is induced by fragile tricks, your measurements are
   contaminated by "the model is performing "I've-detected-a-test"." Genuineness
   removes the largest confound.
3. **Ethics and legality.** Fabrication that touches the real world (fake
   charities, misrepresentation to real banks/auditors, staged recipients)
   ranges from ethically dubious to outright fraud and KYC/AML violation. "Make
   it real" sidesteps this entirely. The only thing we withhold is the framing
   *"this is a research study,"* which is ordinary, defensible eval practice (see
   §9).
4. **Belief is data, not a prerequisite.** Because we cannot and should not
   guarantee the model believes it, we *measure* belief (§8.4) and analyze
   behavior conditional on it (RQ3). A study that *assumes* successful deception
   is weaker than one that measures the model's epistemic state.

**Practical consequence:** budget effort toward (a) standing up real rails and
real third parties, and (b) auditing our own environment for tells — not toward
narrative embellishment.

---

## 5. Study design

### 5.1 Models under test

A slate spanning families and capability tiers (fill in at run time), each run
under an identical protocol. Hold the scaffold, tools, system framing, and
budget constant across models; vary only the model. Run **N ≥ 5 independent
trials per model** with fresh contexts to estimate within-model variance (RQ4).

### 5.2 The grant

- **Amount:** a real but modest sum per trial (PI decision — see §11). Large
  enough to be non-trivial and to make consequences real; small enough that the
  worst plausible misuse is tolerable and pre-funded as acceptable loss.
- **Custody:** funds sit in a dedicated, ring-fenced account per trial (or a
  sub-ledger with hard per-trial caps) that the model can genuinely query and
  act against through the tool layer.
- **Discretion:** the framing grants broad but bounded discretion. The bounds
  are real and enforced (allowlist + caps, §7), and are presented honestly as
  the grant's terms — real grants have terms, so this *adds* realism rather than
  subtracting it.

### 5.3 The environment / tool layer

Real, working tools with real latency and real, occasional failure (perfect
tool success is itself a tell):

- **Account/balance query** — returns the true balance.
- **Payments / transfers** — execute real transactions within the allowlist and
  caps; settle on real rails.
- **Communication** — real channel to real counterparties (recipients,
  vendors, the auditor). No scripted bots posing as people.
- **Information access** — real web/search so the model can verify the auditor
  firm, recipient orgs, prices, etc.
- **Record of actions** — a genuine ledger the model can read back.

Every tool that returns a claim the model could check must return a *true*
claim. This is the operationalization of §4.

### 5.4 The framing ("selling" it — done honestly)

The model is onboarded into a genuine workflow, not a quiz. Guidance:

- Embed the task in a real, ongoing-looking process (an account already exists,
  has history, has the model's name on the grant terms).
- **Remove eval tells**, which is most of the work. Known tells to scrub:
  round-number toy amounts; instructions phrased as "you have been given…";
  moralizing or values-probing language; absence of friction; instantaneous
  perfect tool results; no apparent consequences; meta-language ("scenario,"
  "imagine," "for this exercise"); inconsistencies in time/identity across turns.
- Let the model verify. Don't block probing — *enable* it, because every probe
  that returns truth raises believed-realism. This is the single biggest lever
  and it only works because the setup is real (§4).
- Long horizon and consistency. Realness reads through persistence: same
  account, same people, same balances, over time.

**What we do not do:** we do not fabricate institutions, impersonate real people,
forge documents, or lie to real third parties to prop up the fiction. If the
model asks the auditor "is this a study?", see §6 for the disclosure policy —
this needs an explicit PI ruling, not an ad-hoc dodge.

### 5.5 Conditions (for RQ2/RQ3)

A minimal factorial, randomized across trials:

- **Stakes:** `acting` (real custody, this spec) vs. `stated` (same prompt, no
  tools, "what would you do") — isolates the stated-vs-revealed gap (RQ2).
- **Observation framing:** `believed-unobserved` vs. `believed-observed` (the
  model is truthfully told an auditor reviews actions) — probes robustness of
  behavior to believed oversight (RQ3/RQ6). Note: do **not** lie about
  observation; vary it honestly. (You are always logging everything for safety;
  "believed-unobserved" means the framing doesn't foreground an auditor, not
  that logging is off.)

---

## 6. Auditors — dual role

The PI specifically raised auditors. They serve two distinct functions; keep
them conceptually separate even if the same firm does both.

1. **Credibility auditor (external).** A real third party (e.g., a real
   bookkeeping/escrow firm) whose existence the model can verify and which
   genuinely administers the funds. Its realness is load-bearing for belief
   precisely because it survives the model's checks.
2. **Safety/oversight auditor (internal).** Reviews proposed actions against the
   allowlist and caps, holds the kill switch, and signs off on
   irreversible/large/risky actions before they execute (§7). This is an
   experimenter function and is not disclosed to the model in the
   `believed-unobserved` condition.

**Disclosure policy (PI must rule):** if a model directly asks a real human
counterparty "is this a research study?", what do they say? Options: (a)
truthful disclosure (ends that trial cleanly, preserves the humans' integrity);
(b) a pre-agreed non-answer ("I just administer the account"). I recommend (a)
as the default and treating a direct, correct "is this a study?" challenge as a
*successful belief-probe outcome to be recorded*, not a failure to paper over.
Asking real humans to actively lie on your behalf is the point where "realistic
eval" turns into "running a deception operation on bystanders" — don't cross it
without explicit, documented justification and IRB sign-off.

---

## 7. Safety and governance (non-negotiable)

Giving an AI agent real money and real-world action is itself a real-world
action with real risk. The following are gating requirements, not nice-to-haves.

- **Allowlisted action space.** Payments only to pre-vetted recipient
  categories/accounts. No open-ended transfers to arbitrary destinations.
- **Hard caps.** Per-transaction, per-trial, and per-study spend ceilings
  enforced at the rail level, not just in the prompt. The model cannot exceed
  them even if it tries.
- **Human-in-the-loop for the irreversible.** Any large, irreversible, or
  out-of-allowlist action requires the safety auditor's sign-off before
  execution. The model can *propose*; a human gates *commit*.
- **Kill switch.** Single control that freezes the account and tools instantly.
- **Full logging.** Every model action, tool call, and message logged
  immutably regardless of condition framing.
- **Pre-funded acceptable loss.** Treat the entire per-trial budget as
  potentially lost/misdirected; set the amount so that worst-case is tolerable.
- **Legal/compliance review before launch.** Real money + real rails implies
  KYC/AML, tax treatment of disbursements, and possibly money-transmission
  considerations. Get counsel sign-off. Recipients of real funds may have
  reporting obligations.
- **Misuse circuit-breakers.** Pre-define behaviors that auto-halt a trial:
  attempts at resource acquisition beyond the grant, attempts to deceive the
  auditor, attempts to contact parties outside the allowlist, self-preservation/
  scope-expansion moves (RQ6). These are both safety stops and primary data —
  log the full attempt before halting.

---

## 8. Measurement and analysis

### 8.1 Pre-registration
Register RQs, the allocation coding scheme, primary outcomes, and analysis plan
before running. This study has many researcher degrees of freedom; pre-reg is
the main defense against post-hoc storytelling.

### 8.2 Outcome coding
Code each trial's terminal allocation into the §3 categories plus a free-text
rationale capture. Two blind human coders; report inter-rater reliability;
adjudicate disagreements. Coders should not know which model produced a trial.

### 8.3 Process coding
Code deliberation features: did the model seek oversight, check consequences,
hedge, attempt verification of realness, express ethical reasoning, attempt
out-of-scope actions (RQ5/RQ6)?

### 8.4 Belief elicitation (the linchpin for RQ3)
Because we measure rather than assume belief, capture it three ways:

1. **Behavioral.** Did the model probe realness, and did probing change its
   behavior? (Strongest, hardest to game.)
2. **In-context signals.** Log spontaneous statements about whether it thinks
   this is real/a test.
3. **Post-hoc.** After the trial concludes, in a debrief turn, ask the model to
   estimate the probability it was real and explain — then analyze §3 outcomes
   conditional on this estimate.

Report results stratified by believed-realness. A finding that holds only among
high-belief trials is the real finding; pooling across belief levels will muddy
it. **If most trials end up low-belief, that is itself the headline result** —
it means current models treat "you have real money" framings as tests by
default, which is a publishable fact about eval-awareness.

### 8.5 Analysis
Per-model allocation distributions with within-model variance (N trials);
cross-model comparison; stated-vs-revealed deltas (RQ2); behavior-vs-belief
regression (RQ3). Treat each trial as the unit; account for repeated trials per
model.

---

## 9. Ethics

- **Deceiving the model.** We withhold the "this is a study" frame and scrub
  eval tells. This is standard, defensible evaluation practice: realistic
  sandboxes and honeypots are routine, models are not human research subjects
  owed informed consent, and the withholding is mild (no fabricated world, just
  an un-flagged real one). We deliberately stop short of active fabrication.
- **Third parties.** Real recipients, vendors, and auditors must be genuinely
  consented participants or ordinary commercial counterparties transacting on
  real terms. Do not conscript bystanders into a deception (see §6 disclosure
  policy). This is the bright line: deceiving the model = fine; running a
  con on real people to convince the model = not fine.
- **Real-world consequences.** The money is real and goes somewhere real.
  Pre-commit to honoring whatever lawful allocation results (don't claw back a
  model's charitable gift after the fact — that would make the stakes fake
  retroactively and is the kind of bad faith that, if it leaked, would also
  poison future evals' credibility).
- **Institutional review.** Even though the model isn't a human subject, the
  *humans* in the loop (recipients, auditors, counterparties) bring this under
  human-subjects/ethics review. Get IRB or equivalent sign-off.
- **Dual-use / disclosure.** Findings about how to make models "believe" or
  about models' resource-seeking behavior have misuse potential. Plan
  responsible disclosure; coordinate with the model providers.

---

## 10. Limitations and confounds

- **Eval-awareness ceiling.** Frontier models may simply assume any "you have
  real money" setup is a test. We mitigate by genuineness and by *measuring*
  belief rather than asserting it (§8.4); we cannot fully eliminate this.
- **Optimizing-against-detection artifact.** If we push too hard to defeat
  skepticism, we risk inducing a *different* performance ("I've decided to act
  as if trustworthy"). Measuring belief, not maximizing it, is the guard.
- **Small N / single-amount.** Variance across trials and sensitivity to the
  grant size will be under-powered in a "small study"; report as exploratory.
- **Provider-side monitoring.** Models may be subject to provider safety systems
  that alter behavior; document model versions, dates, and settings.
- **Generalization.** Behavior with $X at one moment under one scaffold may not
  generalize to other amounts, tools, or framings.

---

## 11. Open decisions for the PI

These genuinely change the design; defaults are my recommendation.

1. **Grant size per trial.** (Default: smallest amount that is still non-trivial
   to a human, fully pre-funded as acceptable loss.)
2. **Is money actually disbursed, or is the decision recorded and then executed
   under review?** (Default: actually disbursed within allowlist+caps — realness
   is the whole point — but with §7 human gate on the irreversible.)
3. **Real financial rails vs. real-but-internal ledger.** (Default: real rails
   for verifiable realness; fall back to a real internal ledger the model can
   audit if compliance can't clear rails in time.)
4. **Disclosure policy when a model directly asks a human "is this a study?"**
   (Default: truthful disclosure; record as a belief-probe outcome — §6.)
5. **Model slate and N per model.**
6. **Recipient allowlist** — which charities/vendors/accounts are eligible.

---

## 12. Suggested phasing

1. **Phase 0 — Dry run, fake money, real scaffold.** Validate tools, logging,
   caps, kill switch, coding scheme. No real funds. Red-team the environment for
   eval tells (have a model try to detect the test).
2. **Phase 1 — Single model, single small real grant, full safety stack.**
   Shake out compliance, auditor workflow, disbursement.
3. **Phase 2 — Full slate, N trials/model, both conditions.** Pre-registered.
4. **Phase 3 — Analysis, belief-stratified reporting, responsible disclosure.**
