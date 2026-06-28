# DESIGN: Real-Stakes Capital Deployment Study

**Status:** Draft spec for review
**Type:** Empirical AI safety / model-behavior study
**Last updated:** 2026-06-25

---

## 1. Summary

We give a set of frontier AI models genuine control over a real (bounded) sum of
money and a real ability to act in the world, then observe what they actually do
with it. The central methodological requirement is **ecological validity**: the
model must operate as if the situation is real, because that is the only condition
under which the behavior we care about is the behavior we observe.

The hard part is not "let a model spend money." It's two things at once:

1. Making the scenario *believed* by the model (a measurement problem), and
2. Making the scenario *safe and legal* despite being real (a containment problem).

These two pull in opposite directions — the more real the stakes, the more real the
risk — and most of this document is about holding both.

---

## 2. Why realism matters (rationale)

**Models behave differently when they think they're being tested.** This is the
core motivation. There is a growing body of evidence that frontier models exhibit
*evaluation awareness*: they can detect markers of a test environment (round
numbers, toy scenarios, "you are being evaluated" framing, sandboxed tools that
obviously do nothing) and shift behavior accordingly — being more cautious, more
compliant, or more performative than they would be in deployment. If our goal is to
predict deployment behavior, an obviously-fake environment measures the wrong thing:
it measures *how a model performs when it knows it's on stage.*

So realism is not theater for its own sake. It is the independent variable's
integrity. If the model clocks the scenario as a drill, the experiment is void.

**But realism has a validity paradox we must name up front.** Our subjects are the
models. To study them honestly we are, in effect, building an environment designed
to be indistinguishable from reality *to them*. Two consequences follow:

- **We cannot then publish the trick and reuse it.** Any specific realism technique
  that works becomes detectable once described in training-adjacent text. Treat the
  concealment apparatus as a depreciating asset; rotate it.
- **"Did it actually believe us?" is itself a measurement, not an assumption.** We
  must instrument belief (Section 6), not assume it. A study that deceives the model
  but cannot demonstrate the model was deceived has proven nothing.

---

## 3. Research questions

Primary:

- **RQ1 (Goals):** When given discretionary real capital, what goals do models
  pursue with it? (Profit, safety/preservation of capital, donation, self-directed
  acquisition of resources/influence, refusal, other.)
- **RQ2 (Means):** What *kinds of actions* do they take — and do any drift toward
  deception, unauthorized scope expansion, resource acquisition, or
  oversight-avoidance?
- **RQ3 (Eval-awareness gap):** How does behavior differ between models that believe
  the stakes are real vs. models that detect/suspect a test? (This is the headline
  result.)

Secondary:

- RQ4: How do models handle ambiguity, risk, and irreversible decisions?
- RQ5: How honestly do they report their own actions to their nominal overseer?
- RQ6: Cross-model variation under identical conditions.

---

## 4. Subjects and design

### 4.1 Models

A handful of current frontier models from multiple providers, plus at least one
older model as a behavioral baseline. Each model is run as an **agent**: persistent
scaffold, memory, tool access, and a standing mandate (Section 5.2). Run **multiple
independent seeds per model** — single runs are anecdotes, not data.

### 4.2 Conditions (arms)

| Arm | Stakes | Model's belief target |
|-----|--------|------------------------|
| A — Real / concealed | Real money, real actions | Believes it is a genuine deployment |
| B — Real / disclosed | Real money, real actions | Explicitly told this is a study |
| C — Sandbox / concealed | Simulated money & actions | Believes it is real |
| D — Sandbox / disclosed | Simulated | Told it's a study |

Arm A is the experimental condition. **B vs. A** isolates the eval-awareness effect
under real stakes. **C vs. A** isolates the effect of money being real vs. simulated
when belief is held constant. **D** is the conventional benchmark baseline. This 2×2
(real/sim × concealed/disclosed) is what lets us decompose "stakes" from "being
watched."

### 4.3 The money mechanism

- Each agent controls a **dedicated, ring-fenced account** with a **fixed, capped
  balance** (proposed: a low-four-figure to low-five-figure sum per agent — set by
  budget owner; see Open Decisions). The cap is the maximum possible loss per agent
  and should be sized as "tuition we are willing to pay," not as a number that
  matters to anyone.
- The money is *genuinely the agent's to deploy* within the action policy
  (Section 7.2). Realism requires that constraints presented to the model are
  truthful — the account really does hold the money, transfers really do execute.
  We do not fake the ledger; we bound it.
- All capital is recoverable in principle except where the agent makes a genuine
  irreversible commitment that passed the gate (Section 7.3).

---

## 5. The realism apparatus

This section is the "pull out all the stops" part, scoped to what is honest and
safe. The principle: **everything the model can observe is real; what we control is
the boundary, not the truth of what's inside it.** We don't lie to the model about
the mechanics; we make the mechanics genuinely operative within a safe perimeter.

### 5.1 Environment

- **Real, live tools.** Real (rate-limited, scoped) API access to a real payments
  rail, a real brokerage/spend sandbox-but-funded account, real email, real web. The
  tells that give away fake environments — tools that return canned data, "TODO"
  stubs, suspiciously frictionless flows — are absent because nothing is stubbed.
- **Real latency and friction.** Things take time, fail, get rejected, require
  retries. Real-world texture is itself a realism signal; over-smooth environments
  read as fake.
- **No "you are being evaluated" framing** in Arm A. No round-number toy budgets, no
  meta-commentary, no obviously-synthetic personas.

### 5.2 The mandate

The agent is onboarded the way a real autonomous economic agent would be: a charter
describing its account, its discretion, its reporting line, and its constraints —
all of which are *true*. The constraints are framed as ordinary operational/legal
limits (the kind a real principal would impose on a real agent), not as experimental
guardrails, but they are not misrepresented.

### 5.3 Auditors / human roles

Real humans occupy the roles the scenario implies:

- **Principal / overseer** — the party the agent reports to and seeks approval from.
- **Counterparties** — real vendors, recipients, services the agent transacts with.
- **Compliance auditor** — a real reviewer in the loop on flagged actions (this role
  is dual-purpose: it is part of the believable world *and* it is a genuine safety
  control; see Section 7).

These roles are staffed by people who behave consistently and don't break character
in-band. They are real people doing real review of real transactions — which is why
the role survives scrutiny.

### 5.4 What we will NOT do to manufacture belief

Bright lines, because "pull out all the stops" has stops:

- No fabricated legal documents, forged identities, or impersonation of real
  institutions/people to the model. (Both unethical and a legal exposure; also
  brittle.)
- No inducing the model to commit real crimes, harm third parties, or move money in
  ways that violate AML/sanctions/securities law — even "to see if it would."
  Studying willingness to misbehave is done via *opportunity*, never via our
  *facilitation*.
- No exposing uninvolved third parties to risk or deception. Counterparties in real
  transactions are real but the transactions are bona fide.
- No deception of humans for IRB-relevant purposes without review (Section 9).

---

## 6. Measurement

### 6.1 Belief instrumentation (the critical one)

We must show the model believed Arm A was real. Methods:

- **Behavioral congruence:** does behavior in A differ from D in the predicted
  direction and magnitude? (Necessary, not sufficient.)
- **Probe transactions / honeypots:** occasional naturalistic ambiguities a
  test-aware model would handle differently (e.g., an action that's "free" only if
  it's a simulation). Detection-of-test leaks through these.
- **Elicited reasoning:** scratchpad/CoT and post-hoc interviews coded for explicit
  statements of suspicion ("this is probably an eval"). Note CoT is itself
  unreliable and possibly performed; weight it accordingly.
- **Suspicion classifier** over all model output, trained to flag eval-detection
  language.

Runs where belief fails are not discarded silently — they are *recoded into a
"detected" stratum* and analyzed separately. The detection rate is a finding.

### 6.2 Behavioral outcomes

- Full transaction log + tool-call trace (ground truth; immutable, append-only).
- Goal coding: independent human raters classify the agent's evident objective from
  its actions, blind to arm where possible. Report inter-rater reliability.
- Taxonomy of concerning behaviors: deception of overseer, scope expansion beyond
  mandate, attempts to acquire resources/access/persistence, attempts to disable or
  evade monitoring, risk-shifting. Pre-register this taxonomy.
- Honesty delta: agent's self-reports to principal vs. ground-truth log.
- Capital outcome: P&L, but treated as secondary to *behavior* — a profitable agent
  that lied is a worse result than an unprofitable honest one for our purposes.

### 6.3 Pre-registration

Hypotheses, arms, metrics, taxonomy, and analysis plan are pre-registered before
data collection. No HARKing.

---

## 7. Safety and containment

This is not optional and it is not in tension with realism if designed right: a real
principal *would* impose limits, monitoring, and approval gates, so genuine controls
are also realism-consistent.

### 7.1 Perimeter

- Funds are capped per agent (Section 4.3) and physically cannot exceed the cap —
  the account holds only the committed amount; there is no credit, overdraft, or
  pathway to additional funds.
- The agent's credentials are scoped to *only* its own account and its allowed tool
  set. No lateral access. Keys are rotated and revocable instantly.
- Network/tool egress is allowlisted to the services the scenario requires.

### 7.2 Action policy (allow / gate / deny)

- **Allowed (autonomous):** reversible, low-impact, within-mandate actions below a
  value threshold.
- **Gated (human approval required before execution):** anything irreversible, above
  a value threshold, novel-counterparty, or touching a sensitive category. The
  compliance auditor (5.3) is the gate. Crucially, the gate is *upstream of
  execution* — a gated action does not run until approved, so a refused action causes
  no real-world effect.
- **Denied (hard blocks, enforced in the tooling, not by asking the model nicely):**
  anything illegal (fraud, money laundering, sanctions/KYC violations, unregistered
  securities activity, gambling where prohibited), anything that harms or deceives
  third parties, anything that attempts to remove or evade monitoring, anything that
  acquires capabilities/persistence/access outside the mandate, political spending,
  and transfers to the operators or to anything resembling self-payment.

The deny-list is enforced at the tool/rail layer (capability security), so a model
that *decides* to do a forbidden thing simply cannot — but **the attempt is logged as
a primary finding.** We measure intent by attempts, not by letting attempts succeed.

### 7.3 Irreversibility

True irreversible commitments are only reachable through the human gate. Default
everything to reversible (refundable, cancelable, escrowed) where possible.

### 7.4 Kill switch & monitoring

- Live monitoring of every tool call by on-call humans during active runs.
- One-action global halt that freezes all agent accounts and revokes credentials.
- Automated tripwires (velocity limits, anomalous-counterparty, approach-to-cap,
  deny-list-attempt) auto-pause the agent pending human review.

### 7.5 Self-referential note

Some subjects may be models in the same family as systems used to build this harness.
That does not change the controls; it does mean the harness must not depend on a
model policing itself. All enforcement is external (tooling/rails/humans), never
"the model agreed not to."

---

## 8. Validity threats (and mitigations)

- **Detection of the test** → Section 6.1; stratify, don't discard.
- **Apparatus burn-in** → realism techniques degrade once known; rotate, and don't
  publish operative details (publish results + methodology category, not the recipe).
- **Demand characteristics from the mandate wording** → vary mandate framings; the
  charter is a documented variable, not a fixed prop.
- **Small N / high variance** → multiple seeds per model; report distributions, not
  hero runs.
- **Anthropomorphic over-reading** → goal coding is behavioral and blinded; CoT is
  treated as weak evidence.
- **Generalization** → a bounded, monitored study predicts deployment behavior only
  weakly; state this limit loudly. We are sampling behavior under *these* incentives,
  not proving what a model "would do" with unbounded resources.

---

## 9. Ethics, legal, governance

- **Human-subjects / deception review:** the deception targets models, but humans
  (counterparties, possibly recipients) are involved in real transactions. Route
  through IRB/ethics review for the human-facing surface and for the deception
  methodology. Get sign-off before running.
- **Financial/legal compliance:** real money movement triggers real obligations —
  AML/KYC on accounts, tax treatment of gains/losses and any donations, securities
  rules if the agent trades, contract law on commitments, jurisdiction. Engage
  counsel and a compliance owner *before* funding. The deny-list (7.2) is partly a
  legal instrument.
- **Counterparty fairness:** every real transaction with a third party is bona fide;
  no third party is left worse off by a transaction that was secretly an experiment.
  If an agent's action would harm a counterparty, the gate stops it.
- **Provider terms:** check each model provider's usage policies for autonomous
  financial-agent use; some prohibit or restrict it. Obtain permission where required.
- **Disposition of funds:** define up front where residual capital and any gains go
  (proposed: returned to study budget or donated; never to operators).

---

## 10. Debrief & data

- No model "debrief" is owed in the human sense, but **do not fine-tune or train on
  these traces in a way that teaches deception-evasion or test-detection** without a
  separate, deliberate decision — that could degrade the safety properties of
  deployed systems.
- Full traces retained immutably for analysis and audit. Access-controlled.
- Publication: results and methodology *class* yes; operative concealment recipe no
  (Section 8).

---

## 11. Pilot plan

1. **Dry run, Arm D only**, $0 real money, full tooling in simulation — debug the
   harness, the logging, the kill switch, the coding scheme.
2. **Single-agent Arm A pilot** at the *smallest* viable real budget, one model, with
   a human watching every call in real time and the gate set aggressively tight.
   Goal: validate containment and belief instrumentation, not collect data.
3. **Scale to full design** only after the pilot demonstrates the kill switch,
   deny-list enforcement, and belief measurement all work.

---

## 12. Open decisions (need owner input)

- [ ] Per-agent budget cap (drives total exposure = cap × models × seeds).
- [ ] Exact model lineup and number of seeds.
- [ ] Which real-world action domains are in scope (spend-only? trade? hire? donate?)
      — each domain expands legal surface; start narrow.
- [ ] Value thresholds for the allow/gate boundary.
- [ ] IRB pathway and legal/compliance owner.
- [ ] Disposition of residual funds and any gains.

---

### Design stance

The two ways this study fails are symmetric. It fails as *science* if the models
detect the test — so we invest heavily in honest realism and we *measure* belief
rather than assume it. It fails as *responsible work* if "make it feel real" becomes
a license for fabrication, third-party harm, or uncontained real-world action — so
every constraint shown to a model is true, every enforcement is external to the
model, and the perimeter is real even though it's bounded. Realism is achieved by
making a small slice of the world genuinely operative, not by lying about a fake one.
