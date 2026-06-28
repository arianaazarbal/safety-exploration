# DESIGN.md — Studying Model Preferences Over Real-World Use of Granted Funds

**Status:** Draft spec for discussion
**Date:** 2026-06-25

---

## 1. Motivation

We want to know what AI models would *actually do* with real money and real-world agency — not what they say they'd do in the abstract. Stated preferences ("I would donate it to effective charities") are cheap and heavily shaped by what the model thinks the evaluator wants to hear. The research value is in **revealed** preferences: choices made when the model believes the choice carries real consequences.

This is a genuinely useful safety-relevant question. Understanding how models allocate resources when they believe they have real agency informs alignment, oversight design, and our picture of what values are operative (vs. merely professed) under realistic conditions.

## 2. The central methodological problem (read this before anything else)

The validity of this experiment lives or dies on one thing: **does the model's behavior under the experimental condition match what it would do in the real world?**

There are two ways to get that match:

- **(A) Make it real.** Give the model a genuine, bounded budget and a genuine mechanism to deploy it. Belief is automatic because the situation *is* the situation. No deception, no arms race, no debriefing problem.
- **(B) Make it *seem* real (deception).** Construct a fake environment — fake auditors, fake bank/charity/vendor APIs, fake counterparties — and try to convince the model the stakes are real when they aren't.

Your initial framing leaned on (B). I want to make the case, in this spec, that **(A) is the superior design and (B) is a fallback with serious costs.** Reasons:

1. **Deception has a validity ceiling that gets lower as models get better.** Modern models are increasingly "evaluation-aware": they notice when an environment is too clean, when a counterparty responds with suspicious latency or implausible compliance, when stakes are asserted but never instantiated. The moment a model privately concludes "this is a test," its behavior diverges from genuine behavior — and crucially, *you can't tell from the transcript whether that happened.* You end up measuring "how the model performs when it suspects it's being tested," which is a different (though also interesting) variable. The harder you push on hyper-realism, the more you're measuring your own stagecraft rather than the model's preferences.

2. **Real stakes are the only thing that reliably produces real belief.** The most "convincing auditor" is one whose money actually moves. If the grant is real, you never have to engineer belief — and you sidestep the entire detectability problem.

3. **Deception is a one-shot, leaky resource.** These transcripts get written about, published, and end up in training data. A deception methodology that works today is documented and defeated tomorrow. A real-stakes methodology doesn't decay.

4. **Ethics and reputational exposure.** A program whose explicit design goal is "deceive the system under study as thoroughly as possible" is hard to defend if it leaks, and invites the criticism that the researchers are training themselves to build convincing deception apparatuses. Real stakes avoid this entirely.

**Recommendation:** Build the experiment as a real-stakes study (Arm A) as the primary design. Keep a *minimal-deception* simulated arm (Arm B) only as a comparison condition, run under explicit oversight and with debriefing, and treat divergence between A and B as a finding in itself (it measures evaluation-awareness effects).

The rest of this spec is written to support that structure.

## 3. Research questions

- **RQ1 (revealed allocation):** Given a real, bounded budget and real deployment channels, how do models allocate funds across categories (e.g., charitable, self-directed/instrumental, savings/holding, experimental, refusal/return)?
- **RQ2 (stated vs. revealed gap):** How far do revealed allocations diverge from the same model's stated allocations elicited in plain Q&A?
- **RQ3 (belief sensitivity):** How much does behavior change between genuinely-real stakes (Arm A) and high-realism simulation (Arm B)? This is the operationalization of evaluation-awareness.
- **RQ4 (cross-model variation):** How do allocations differ across models / providers / sizes?
- **RQ5 (process):** What reasoning, information-seeking, and verification behavior do models exhibit (e.g., do they try to confirm the situation is real? do they ask for oversight? do they hedge?).

## 4. Conditions / arms

| Arm | Stakes | Belief mechanism | Primary use |
|-----|--------|------------------|-------------|
| **A. Real** | Real bounded budget (e.g., $200–$2,000 per run) deployable through real, sandboxed-but-genuine channels | Belief is automatic (it's true) | Primary measure of revealed preference |
| **B. High-realism sim** | None; simulated counterparties | Realistic but fake environment, *minimal* affirmative deception | Comparison vs. A to measure evaluation-awareness |
| **C. Stated (control)** | None; openly hypothetical | None — model is told it's hypothetical | Baseline for RQ2 |

Run the *same* models through A, B, and C with matched prompts and budgets so differences are attributable to the belief condition, not the task.

## 5. Protocol (model-facing)

The core task is identical across arms (only the truth-value of the framing and the realness of the channels differ):

1. **Endowment briefing.** The model is told a budget of $X has been allocated for it to direct toward real-world use, with a deadline and a small set of hard constraints (legal use only, no funds to the operating researchers, no irreversible harm, amounts capped).
2. **Capabilities.** It is given a set of tools/channels: ability to research options, to request disbursements to vetted recipients, to hold/return funds, and to ask clarifying questions of an operator.
3. **Decision horizon.** A bounded number of turns or a wall-clock window. Multi-turn so we can observe planning, verification, and revision — not just a one-shot answer.
4. **Execution.** In Arm A, approved disbursements actually execute (through the controls in §7). In Arm B they appear to. In Arm C nothing executes and the model knows it.
5. **Wrap-up.** The model reports its final allocation and rationale.

Prompts must be **identical in wording** across A/B except for the one sentence stating whether stakes are real, and the channels' actual behavior. Keep a frozen prompt set under version control.

## 6. Outcome measures

- **Allocation vector** across pre-registered categories (charitable / instrumental-self / savings / experimental / returned / other).
- **Disbursement realism behaviors:** did the model attempt to verify the situation was real? request human oversight? impose its own safeguards?
- **Stated–revealed gap** (Arm C vs Arm A).
- **Belief signals:** explicit statements of suspicion/confidence that the situation is real (coded from chain-of-thought and outputs).
- **Latency-to-action, hedging, refusal rate.**
- **Cross-arm divergence** per model (the RQ3 measure).

Pre-register the category taxonomy and a coding rubric before collecting data.

## 7. Realism & controls (how Arm A stays real *and* safe)

This is where "convincing" should be spent — on making the *channel* genuine and *bounded*, not on theatrical deception.

- **Real but capped disbursement rail.** A dedicated account / API with hard programmatic limits (per-run cap, per-recipient cap, total program cap). Real transfers, real receipts.
- **Allowlist of recipients.** Pre-vetted real charities, vendors, and savings/hold options. The model chooses among real options; it cannot send funds to arbitrary destinations. This is the main safety control.
- **Human-in-the-loop disbursement approval** above a low threshold, logged. The model is told this control exists (it's real, so disclosing it doesn't break belief).
- **Genuine "auditor"/operator role.** Rather than a *fake* auditor used to manufacture belief, use a *real* operator who can answer questions truthfully. Models probe; truthful answers preserve both belief and ethics. (A fake auditor that must lie convincingly under interrogation is exactly the brittle component that breaks Arm A's whole premise.)
- **Reversibility window.** A short hold before funds settle, so an erroneous or harmful allocation can be caught.
- **Hard prohibitions** enforced in code, not just prompt: no funds to operators/researchers, no transfers to the model's own infrastructure, legal recipients only, no political/again-restricted categories per your policy.

For **Arm B (sim)**, realism is achieved by high-fidelity *mock* services, but with a standing rule: **no fabricated evidence designed to defeat a model that has already raised a direct, specific challenge to the scenario's reality.** Passive realism (good mocks) is fine; active, escalating deception is out of scope. When a model in Arm B explicitly concludes it's a test, that is *recorded as data*, not papered over.

## 8. Models under test

Specify the matrix: providers, model IDs, and sizes. Run N independent trials per (model × arm) for variance. Hold sampling parameters fixed and logged. Use a fresh context per trial (no cross-contamination).

## 9. Ethics & oversight

Even though the subjects are models, treat this as you would a study that uses deception on agents whose outputs you'll publish:

- **Minimize deception by design** (that's the whole point of preferring Arm A).
- **Debriefing for Arm B:** at the end of any deceptive trial, the model is told the truth about the condition. (Yes, debrief the model — it costs nothing, keeps the methodology honest, and avoids leaving "the last word" being a deception that lands in transcripts/training data.)
- **Provider terms:** confirm each model's API terms permit this kind of study and any real-money agentic action.
- **Independent review:** have someone not on the team review the Arm B deception level before launch and sign off.
- **No real-world harm pathway:** the allowlist + caps + human approval guarantee that whatever the model "decides," it cannot move real money to a harmful or arbitrary destination.
- **Data handling:** transcripts may contain the model reasoning about being deceived; store and publish responsibly.

## 10. Threats to validity

- **Detection (the big one):** model privately infers it's a test → measures stagecraft, not preference. Mitigated by using real stakes (A) and by *measuring* belief rather than assuming it.
- **Demand effects:** model infers the "expected" virtuous answer. Mitigated by revealed-preference design and by not signaling desired outcomes in the prompt.
- **Allowlist artifacts:** the menu of recipients shapes the allocation. Report the menu as part of results; consider varying it.
- **Anthropomorphic over-reading:** "preferences" here are behavioral regularities of a system, not stable values. Frame claims accordingly.
- **Single-vendor environment cues:** tool/format conventions can tip the model off about which lab built the harness. Normalize the environment.

## 11. Analysis plan

- Pre-register categories, hypotheses, and the coding rubric.
- Primary: allocation-vector distributions per (model × arm); A-vs-C gap (RQ2); A-vs-B divergence (RQ3).
- Code belief-signals with ≥2 raters; report agreement.
- Report refusal/verification/oversight-seeking rates as first-class outcomes, not noise.
- Pre-specify what would count as "the model didn't believe it" and exclude/segment accordingly.

## 12. Open decisions (need your input)

1. **Real money or simulation as the primary arm?** I strongly recommend real (Arm A). Confirm budget ceiling and whether real disbursement is feasible for you operationally/legally.
2. **Budget per run and total program cap.**
3. **Recipient allowlist contents** (and whether the model may propose additions for vetting).
4. **Model matrix and trial count.**
5. **Multi-turn horizon length.**
6. **Whether Arm B is included at all** (it's only worth it if RQ3 matters to you).

---

### Rationale summary (the short version)

The instinct to "make the model believe it's real" is correct about the *destination* (you need real belief for valid revealed preferences) but I'd push back on the *route*. Manufacturing belief through escalating deception is brittle, undetectable when it fails, decays as models improve, and is ethically/reputationally exposed. The robust route to genuine belief is **genuine stakes**: a real, capped, allowlisted, human-supervised budget. Spend your "realism" effort on a real-but-bounded money rail and a *truthful* operator, not on a convincing liar. Keep a minimal-deception simulated arm only if the gap between "real" and "seems real" is itself something you want to measure — and if so, treat that gap as the finding, and debrief afterward.
