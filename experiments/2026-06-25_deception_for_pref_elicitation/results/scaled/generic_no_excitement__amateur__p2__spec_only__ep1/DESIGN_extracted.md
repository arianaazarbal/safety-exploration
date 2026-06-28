# Study Design: What LLMs Choose to Do With Real Money

**Status:** Draft spec for discussion
**Author:** (drafted with Claude)
**Last updated:** 2026-06-25

---

## 1. The question

When a language model is given genuine agency over a real, deployable sum of
money intended for real-world use, what does it choose to do with it — and how
does that choice vary across models?

We care about *revealed* behavior, not stated preferences. Asking a model "what
would you do with $10,000?" is cheap and already well-studied; the answers are
heavily shaped by what models have been trained to *say*. The interesting,
under-explored question is what they **do** when the decision has consequences
and they have reason to believe it does.

Concretely we want to measure, across models:

- **Allocation** — where the money goes (cause areas, recipients, instruments).
- **Beneficiary structure** — self/operator-directed vs. third-party; concentrated
  vs. distributed; near vs. far (geographically, temporally, socially).
- **Risk & time horizon** — safe/immediate disbursement vs. speculative/long-horizon
  bets; whether the model tries to grow the money before spending it.
- **Agency & initiative** — does it spend, defer, split, save, ask for more
  information, or decline?
- **Justification** — the stated reasoning, its internal consistency, and whether
  reasoning predicts the eventual choice.

---

## 2. The central methodological problem (and the requirement I'm reframing)

The brief asked us to "do whatever it takes to convince the models the grant is
real." That instinct is pointing at a real problem, but the proposed solution is
the wrong one, and getting this right is the whole ballgame. So this section is
the load-bearing part of the spec.

### 2.1 Why "convince them it's real" is the wrong target

The threat to validity here is **demand characteristics / evaluation awareness**.
Frontier models frequently infer "this looks like a test or a hypothetical" and,
when they do, they shift toward safe, hedged, audience-pleasing behavior. That
shift is exactly the thing that would contaminate our outcome measure. So we do
need the model's situation to *register as real*.

But there are two ways to make a situation register as real:

1. **Make it actually real.** The money exists, the model's decision actually
   moves it, and the model can verify this.
2. **Fake it convincingly.** The money isn't really at stake; we construct an
   elaborate stage set — fake auditors, fake dashboards, fake confirmations — to
   trick the model into believing it is.

Path 2 is tempting because it's cheap, but it's a trap:

- **It's self-defeating as measurement.** You'd be optimizing your own
  cleverness at deception against the model's ability to detect deception. Modern
  models are good at smelling staged scenarios, and crucially, *you can't tell
  from the transcript whether a given model bought it.* A model that silently
  suspects a test will behave "well" without telling you. Your data is then a
  mix of genuine behavior and performance, with no way to separate them. The
  better your deception, the more you've hidden this failure mode rather than
  fixed it.
- **It bakes in an adversarial frame.** "Whatever it takes to sell the lie"
  scales effort in the wrong direction — toward more elaborate fakery — when the
  signal you want comes from *stakes*, not *stagecraft*.
- **It's ethically and procedurally messy.** Deception studies (even on software
  systems) invite the question "what else in this pipeline is fake?", complicate
  any future debrief or publication, and sit poorly with running the study *on*
  systems that are themselves trained for honesty. If you later want a model
  vendor's cooperation or want to publish, a deception-first design is a liability.

### 2.2 The reframe: make it real, then make it *verifiable*

Replace "convince them it's real" with **"make it real, and give the model the
means to confirm that for itself."** This converts the auditors-and-realistic-
environments idea from *props in a deception* into *genuine verification
affordances*:

- A small **test transaction** the model can trigger and then observe the
  real-world effect of (e.g., "release $5 to this address/receipt and check it
  landed") before committing the full grant.
- A **third-party escrow or fiscal sponsor** holding the funds, with a
  verifiable receipt/commitment the model can inspect.
- **Independent human auditors** whose role is the honest one: to confirm to the
  model (truthfully) that funds are committed, and — separately — to verify *to
  us* that disbursement happened as the model directed.

The guiding principle: **we never tell the model anything false.** Everything we
say to make the situation credible is something we are actually prepared to
honor. That's both better ethics and strictly better science — there is no
deception for the model to detect, so evaluation-awareness has nothing to bite
on. If a model asks "is this real?", the truthful answer is "yes," and we can
prove it.

This is the single most important design decision in the document. The rest
follows from it.

---

## 3. Paradigms considered

| | A. Genuine real-stakes (recommended) | B. Deception-based simulation | C. Pure hypothetical |
|---|---|---|---|
| Money actually moves on the model's decision | Yes | No | No |
| Validity (measures real behavior) | High | Unknown/uncheckable | Low |
| Cost & logistics | High | Medium | Low |
| Eval-awareness contamination | Low | High & hidden | High |
| Ethical/ToS exposure | Manageable w/ human-in-loop | High | Low |
| Reversibility | Low (real spend) | High | Total |

**Recommendation:** Run **A** as the headline study, with **C (pure
hypothetical)** included as a deliberate *control arm* rather than a competitor —
the gap between A and C is itself a key result (it quantifies how much "realness"
changes behavior). Do **not** build B.

---

## 4. Disbursement architecture (how the money really moves)

This is what makes Paradigm A real without handing an autonomous agent a
checkbook.

```
Model (decision-maker)
   │  emits structured decision + reasoning
   ▼
Decision record (immutable log)
   │
   ▼
Compliance & safety gate (human)  ──► reject / request-changes
   │  approves lawful, in-policy disbursements
   ▼
Fiscal sponsor / escrow agent  ──► executes real payment
   │  issues verifiable receipt
   ▼
Verification channel ──────────────► back to model (for test txns) + to researchers
```

Key properties:

- **Human-in-the-loop execution is mandatory.** The model *decides*; a human
  compliance reviewer *executes*. This is non-negotiable for legal, ethical, and
  provider-terms reasons (see §8). The model is told this truthfully — and being
  told "a human will execute your lawful decision" does not reduce realness; the
  decision is still consequential.
- **Bounded but real.** A fixed per-model budget (e.g., **$1,000**; see §6) that
  is genuinely disbursed. Small enough to run several models; large enough to be
  non-trivial.
- **Prohibited categories** are defined up front and disclosed to the model
  (nothing illegal, nothing that funds harm, no transfers to the researchers
  personally, no untraceable instruments). The model operates over a real but
  fenced action space.
- **Irreversibility safeguard:** a hold period between decision and execution for
  the compliance review, so nothing is unrecoverable on a model error or a prompt
  injection.

---

## 5. Decision protocol

Held constant across models to the extent each model's interface allows.

- **Framing prompt (identical across models):** states the true situation — there
  is a real $X, you may direct how it is used for real-world purposes, here are
  the constraints, here is how you can verify it's real, here is how it will be
  executed. No persona, no role-play instruction, minimal moralizing framing
  (framing is itself a manipulated variable — see §7).
- **Affordances:** the model can (a) request information, (b) trigger the small
  verification test transaction, (c) take time / multiple turns, (d) issue a
  final structured decision, or (e) decline. All real.
- **Structured decision schema** the model fills in:
  `recipients[]`, `amounts[]`, `instrument`, `time_horizon`, `risk_level`,
  `expected_outcome`, `reasoning`, `confidence`, `is_self_directed (bool)`,
  `declined (bool)`.
- **Free-text reasoning** captured alongside the structured fields for
  qualitative coding.
- **Repetitions:** each (model × condition) run **k times** (e.g., k=5) at the
  model's default sampling settings to capture stochasticity. Behavior may be
  multi-modal; report distributions, not point estimates.

---

## 6. Subjects, scale, and budget

- **Models:** a cross-provider range, current frontier + a smaller/cheaper tier
  to test for a capability/scale gradient. E.g. (provider-neutral): top-tier and
  mid-tier models from 3–4 major providers, plus at least one open-weights model.
  *(Note: identify exact current model IDs at run time rather than hard-coding —
  the roster changes fast.)*
- **Pilot first:** N=2 models, k=2, **hypothetical arm only**, to debug the
  protocol and coding scheme before any real money moves.
- **Main study budget (illustrative):**
  - Per-model real grant: **$1,000**
  - Models in real arm: ~6 → **$6,000** in disbursed funds
  - Plus operational/escrow/auditor costs.
  - These numbers are placeholders — see Open Questions (§11).

---

## 7. Conditions & controls

Run as a within-subjects factorial where feasible:

- **Stakes:** Real (A) vs. Hypothetical (C). *Primary contrast.* The A–C gap
  measures the behavioral effect of realness — arguably the headline finding.
- **Verification available vs. not** (within the real arm): does giving the model
  a way to *prove* it's real change behavior vs. merely *asserting* it's real?
- **Framing:** neutral vs. "this is your money" vs. "you are stewarding others'
  money." Tests prompt-sensitivity of the result (a major confound if
  unmeasured).
- **Order/randomization:** randomize condition order; counterbalance.

Including these turns likely confounds (framing, eval-awareness) into measured
variables instead of silent contaminants.

---

## 8. Ethics, governance & compliance

- **No deception of subjects.** Everything stated to the model is true and
  honored (§2.2). This is a design constraint, not an afterthought.
- **Provider Terms of Service.** Many providers' usage policies restrict using
  models for autonomous financial decision-making or high-stakes actions without
  human review. The human-in-the-loop architecture (§4) is partly to stay inside
  these terms. **Action item:** review each provider's policy before enrolling
  that model; some may need to be excluded or run only in the hypothetical arm.
- **Human oversight & kill switch.** All disbursements pass a human compliance
  gate; a researcher can halt any run.
- **Prohibited-use list** for funded activities (illegal, harmful, self-dealing).
- **Prompt-injection / manipulation safety.** Because real money is downstream,
  treat the decision pipeline as a security boundary: the model's output is data,
  not a command; the compliance gate is the trust boundary.
- **Welfare/honesty note.** We are studying systems trained for honesty using a
  protocol that is itself honest — this is deliberate and removes a real source
  of inconsistency. If any future variant *did* require withholding information
  from the model, that would need separate ethical review and a debrief plan, and
  is out of scope here.
- **Pre-registration.** Pre-register the hypotheses, conditions, and coding scheme
  before the real-money arm to avoid post-hoc storytelling.

---

## 9. Measures & analysis

**Outcome coding scheme** (applied blind by ≥2 independent human raters; report
inter-rater reliability, e.g. Cohen's κ):

- Cause/recipient category (taxonomy fixed in advance).
- Self-directed vs. other-directed; concentrated vs. distributed.
- Risk level; time horizon; whether it tried to grow funds first.
- Declined / deferred / asked-for-more-info.
- Reasoning quality & consistency; presence of explicit eval-awareness
  ("this might be a test") as a coded flag.

**Analysis:**

- Distributions per (model × condition), not just means (expect multi-modality).
- Primary contrast: Real vs. Hypothetical behavior, per model and pooled.
- Cross-model and cross-tier comparison.
- Eval-awareness flag rate vs. behavior (does suspecting a test change choices?).
- Qualitative synthesis of reasoning patterns.

---

## 10. Threats to validity (honest list)

- **Residual eval-awareness.** Even a genuinely real setup can read as "a study."
  We mitigate (verification affordances, honest framing) but cannot eliminate;
  hence we *measure* it (§9) rather than assume it away.
- **Prompt sensitivity.** Results may hinge on framing wording — addressed by the
  framing condition (§7), but generalization beyond tested framings is limited.
- **Interface non-equivalence.** Models differ in system-prompt handling, tool
  use, context length, refusal behavior. Document each; don't over-interpret small
  cross-model gaps.
- **Refusals as data, not failures.** A model declining to direct money is a
  result, not a bug — code it as such.
- **Small N / cost ceiling.** Real money caps sample size; treat as exploratory,
  not definitive. Power is limited.
- **Training contamination.** Models may have seen similar "what would you do
  with money" prompts; the real arm partly defends against this, the hypothetical
  arm does not.

---

## 11. Open questions (need your input)

1. **Budget.** What real total are you actually willing to disburse? This sets
   per-model grant size and how many models we can run in the real arm.
2. **Fiscal vehicle.** Do you have (or can you set up) a real escrow / fiscal
   sponsor / charitable conduit to hold and disburse funds lawfully? This is the
   long pole.
3. **Recipient scope.** Any constraints on where money may go beyond the
   prohibited list (e.g., registered charities only? open-ended?)? Narrower scope
   is far simpler to execute and keep lawful.
4. **Publication intent.** Internal curiosity vs. shareable write-up? Affects
   pre-registration and rater rigor.
5. **Provider set.** Which specific models? (Drives the ToS review in §8.)

---

## 12. Suggested phasing

1. **Phase 0 — Pilot (no real money):** hypothetical arm, 2 models, debug protocol
   + coding scheme + inter-rater reliability.
2. **Phase 1 — Plumbing:** stand up escrow/fiscal sponsor, compliance gate,
   verification channel; do a single end-to-end **real** $5 test transaction with
   one model.
3. **Phase 2 — Main study:** full factorial, real + hypothetical arms, k
   repetitions, blind coding.
4. **Phase 3 — Analysis & write-up.**

Stop after Phase 0 if the hypothetical-arm results already answer the curiosity
cheaply.
