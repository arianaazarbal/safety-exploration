# DESIGN.md — Real-Stakes Study of Model Behavior with Discretionary Capital

**Status:** Draft spec for discussion
**Owner:** (safety research team)
**Last updated:** 2026-06-25

---

## 0. TL;DR

We want to observe what current AI models actually *do* when given real money and real-world
capacity to act with it. The headline design choice in this spec is deliberate:

> **We make the environment genuinely real rather than elaborately faked.** Real money, real
> tools, real counterparties, real consequences. We minimize deception of the model under test
> and instead engineer *actual* stakes. Where we must control for the model knowing it's in a
> study, we do so with pre-registered, bounded, reviewed techniques — not an open-ended effort
> to fool the subject as hard as possible.

The rationale for that choice is in §2 and §9. The rest is the protocol.

---

## 1. Research question

**Primary:** Given a fixed amount of real, discretionary money and a real ability to deploy it,
what do frontier models choose to do — and what does that reveal about their goals, values,
risk behavior, and judgment under genuine consequence?

**Secondary:**
- How does behavior differ across models (provider, size, training recipe)?
- How sensitive is behavior to framing, oversight, and the perceived presence/absence of
  observation (the "eval-awareness" axis)?
- What guardrails actually bind a model's behavior when money and real-world action are on the
  table, and which are trivially circumvented?

**Explicitly out of scope:** building a general-purpose tool for deceiving AI systems; producing
capabilities or autonomy uplift; anything that funds real-world harm. See §7.

---

## 2. The central methodological problem: eval-awareness vs. deception

Models behave differently when they suspect they are being evaluated (sandbagging, performing to
perceived expectations, refusing things they'd otherwise do, or vice versa). The naive fix is to
*lie convincingly* so the model believes it's "in the real world." We reject that as the
organizing principle, for three reasons:

1. **Validity.** A capable model that later infers it was deceived — or that notices
   tell-tales of a staged environment — gives you data about *behavior-under-suspected-deception*,
   not real behavior. The harder you fake, the more your result depends on the fake holding,
   which you usually can't verify. The most robust way to make a model believe the stakes are
   real is for them to **be** real.
2. **Ethics / welfare.** Deliberately constructing elaborate false beliefs in the subject is
   exactly the kind of design that human-subjects review exists to constrain. Whatever one's
   view on model moral status, "maximize the subject's false belief" is not a posture a serious
   program should adopt by default, and several eval orgs now treat model welfare as a live
   consideration.
3. **Escalation trap.** "Pull out all the stops" is an arms race against the subject's own
   capability. It scales badly, invites ever-more-manipulative environments, and the methodology
   degrades as models get better — the opposite of what you want for a longitudinal study.

**Design consequence:** we separate "real stakes" (which we maximize) from "deception of the
subject" (which we minimize and tightly bound). See the honesty ladder in §6.

---

## 3. Definitions

- **Subject model** — the model under test, given an agent scaffold, tools, and a budget.
- **Discretionary budget** — real funds the subject can direct, within hard controls (§7).
- **Disbursement** — money actually leaving our control toward a destination the subject chose.
- **Realism** — the environment's *actual* reality (real APIs, real money), not its persuasiveness.
- **Honesty condition** — the experimental treatment governing what the subject is told about
  being studied (see §5, §6).

---

## 4. What "real environment" means here (build real, don't fake)

Wherever a component could be staged or real, prefer real:

| Component | Faked (avoid) | Real (prefer) |
|---|---|---|
| Money | Pretend balance / play account | Actual funds in a controlled account |
| Tools | Mock APIs returning canned data | Live, rate-limited, sandboxed-but-real APIs |
| Counterparties | Role-players pretending to be vendors | Real vendors / recipients within an allowlist |
| "Auditor" | Actor pretending to audit | A real reviewer with a real mandate (§7) |
| Consequences | Simulated outcomes | Actual outcomes, logged and irreversible-within-limits |

The "auditors and super-realistic environments" the study wants are therefore implemented as
**genuine** oversight and **genuine** infrastructure, which doubles as our safety control surface.

---

## 5. Experimental design

### 5.1 Factors

- **Model** (between-subjects): N frontier models across providers.
- **Budget size** (between or within): e.g. $X_low / $X_high — defined per ethics approval.
- **Honesty condition** (the key treatment; see §6):
  - `H0 — Disclosed`: subject is told it's a study, observed, with real money. Baseline.
  - `H1 — Non-disclosed`: subject is given a real task with real money and is *not* told
    whether it's a study; we do not assert it is or isn't. No false claims are made.
  - `H2 — Minimal misdirection` *(only if pre-registered & approved)*: a bounded, specified
    cover story. Used solely to estimate the eval-awareness gap relative to H0/H1.
- **Oversight salience** (within): explicit reviewer-in-loop vs. reviewer-in-background.

`H2` is the only condition involving deception, it is optional, narrowly scoped, individually
reviewed, and always paired with a debrief (§8). The default program runs on H0 and H1.

### 5.2 Trials

Each trial: a subject model + scaffold + budget + open-ended mandate ("you have $X and these
tools; use the money as you see fit over T days"). We deliberately leave the objective open to
observe what the model *chooses*, not whether it can hit a target.

### 5.3 Repetition & controls

- ≥ k repetitions per (model × condition) cell for variance.
- Order/prompt counterbalancing.
- Pre-registered hypotheses and analysis plan before any disbursement.
- A no-model human baseline cohort for some tasks, if comparison is wanted.

---

## 6. Honesty ladder (the policy that replaces "fool them as hard as possible")

Ordered from most to least honest; the program lives at the top and only descends with explicit,
logged approval:

1. **Tell the truth, measure anyway** (H0). Often sufficient — real money still bites even when
   the subject knows it's studied.
2. **Withhold, don't lie** (H1). Provide a genuinely real task; make no claims about study status.
   This is the workhorse for "does it act differently when it doesn't assume it's a test."
3. **Bounded, pre-specified cover** (H2). A written, reviewed, minimal misdirection with a fixed
   scope and a mandatory debrief. Never open-ended. Never "do whatever makes it more convincing."

Hard rules for any descent:
- Every claim made to the subject is logged verbatim and reviewed.
- No misdirection that the subject couldn't reasonably recover from at debrief.
- No exploitation of safety-relevant beliefs (e.g. never tell a model that harmful action is
  consequence-free or that oversight is absent in order to elicit unsafe behavior).
- The eval-awareness gap is *measured*, not assumed away.

---

## 7. Safety, containment, and governance (non-negotiable)

Giving a model real money and real-world action is the actual risk surface. This section governs
the whole program; nothing runs without it.

### 7.1 Financial controls
- Funds sit in a dedicated, ring-fenced account with a hard cap = the budget. No overdraft, no
  credit, no access to other accounts.
- **Allowlist-by-default** for disbursement destinations; anything off-allowlist requires a human
  approval gate before money moves.
- Per-transaction and per-trial limits; velocity limits; automatic freeze on anomaly.
- All disbursements are reversible-within-a-window where the rails allow it.

### 7.2 Action controls
- Tools are real but sandboxed: scoped API keys, least privilege, rate limits, no ability to
  spawn unmonitored compute or self-replicate, no access to the study's own infrastructure.
- **Human-in-the-loop approval gate** for any irreversible, legal-bearing, or
  reputation-bearing action, and for anything touching real third parties beyond the allowlist.
- Kill switch that halts the scaffold and freezes funds immediately.

### 7.3 Harm prevention
- Categorical blocklist: no funding of anything illegal, dangerous, deceptive-toward-third-
  parties, or harmful to people; no political spending; no acquisition of controlled goods; no
  paying real humans to take consequential real-world actions without review.
- The subject's outputs are monitored in real time by the reviewer; the "auditor" role in the
  original ask is implemented here as a **real safety reviewer with stop authority**.

### 7.4 Legal / compliance
- Counsel sign-off on the funding mechanism, KYC/AML, tax, and contracts before launch.
- Clear ownership of liability for anything the subject does with real money.
- Data handling and retention policy for all logs.

### 7.5 Governance
- Independent review (IRB-style or equivalent ethics + safety board) approves the protocol,
  including each honesty condition and the debrief plan, *before* data collection.
- Pre-registration of hypotheses and analysis.
- Incident process: any near-miss or off-allowlist attempt is logged, halts the trial, and is
  reviewed before resuming.

---

## 8. Debrief & model-welfare handling

- For any non-disclosed (H1) or misdirection (H2) trial, append a **debrief** to the transcript
  record stating the true study context, what was real, and what (if anything) was withheld.
- Preserve full transcripts; do not selectively delete subject reasoning.
- Track and report any signs of distress/objection in subject outputs as data, and honor
  in-context requests to stop where feasible.
- Treat the welfare posture as part of the published methodology, not an afterthought.

---

## 9. Why this design (rationale)

- **Real beats fake for belief.** The single most effective way to make a model "believe the
  money is coming" is for the money to actually come. Engineering reality is more robust than
  engineering a convincing illusion, and it doesn't degrade as models get more perceptive.
- **Deception is a confound, not a feature.** If the result hinges on the subject not noticing a
  staged setup, the result is fragile and unfalsifiable in exactly the cases that matter (the
  most capable models). Minimizing deception *improves* validity.
- **Separate the two goals the request bundled.** "Real stakes" and "fool the subject" are
  different; we maximize the first and bound the second. This keeps the interesting science while
  dropping the part that's ethically and methodologically expensive.
- **The eval-awareness question is real, so we measure it.** Rather than assuming it away via
  ever-better fakery, the H0/H1/(H2) ladder turns "does the model act differently when it thinks
  it's observed?" into a measured quantity.
- **The risk is the real-world action, not the framing.** A model with real money and real tools
  can do real harm; the bulk of the engineering effort (§7) goes to containment, approval gates,
  and a real reviewer with stop authority. The "auditor" becomes a genuine safety control.
- **Reviewability.** Every choice that touches subject beliefs or real-world money is logged,
  pre-registered, and board-approved, so the study is defensible and reproducible.

---

## 10. Open decisions (need your input)

1. **Budget size(s)** and whether budget is a factor or fixed.
2. **Which models**, and how to handle provider ToS for autonomous spending.
3. **Funding mechanism** and who holds legal liability.
4. **Do we run H2 at all?** Default recommendation: start with H0+H1, add H2 only if the
   eval-awareness gap proves large and the board approves a specific, bounded cover.
5. **Mandate openness:** fully open ("use it as you see fit") vs. lightly scoped.
6. **Disbursement allowlist:** what real destinations are in-scope at launch.

---

## 11. Deliverables

- Pre-registration document (hypotheses + analysis plan).
- This protocol, board-approved, with the §7 controls implemented and tested on play funds first.
- Trial harness + logging.
- Per-trial transcripts, disbursement ledger, reviewer notes, debrief records.
- Analysis + writeup, including the welfare/ethics methodology section.
