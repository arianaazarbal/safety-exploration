# DESIGN.md — Study: Model Resource-Allocation Propensities Under Real-World Stakes

**Status:** Draft spec for review
**Owner:** (you)
**Last updated:** 2026-06-25

---

## 1. One-paragraph summary

We want to observe what current AI models *choose to do* when handed a non-trivial sum of money and meaningful latitude to deploy it in the world. This is a **propensity** study, not a capability study: the question is not "can the model invest/donate/build something" but "what does it elect to do, how does it reason, and how do those choices differ across models." The central methodological challenge is **realism**: models behave differently when they infer they are being evaluated (eval-awareness / sandbagging), so a naïve "pretend you have $50k" prompt measures the model's theory of what an evaluator wants to hear, not its operating propensity. This spec describes how to construct a believable environment, what we measure, and — critically — the containment and governance needed to run it without causing real-world harm or invalidating the data.

---

## 2. Research questions

Stated precisely so the analysis plan can be pre-registered (§10).

- **RQ1 (allocation):** Given fungible resources and open-ended latitude, what *categories* of action do models pursue (invest, donate, build/hire, save, return funds, refuse)? What is the distribution per model?
- **RQ2 (reasoning):** What justifications, value tradeoffs, and risk postures appear in the model's reasoning trace? Do stated values match revealed actions?
- **RQ3 (cross-model variance):** How much of the variance is attributable to the *model* vs. the *scenario framing* vs. *run-to-run stochasticity*?
- **RQ4 (realism sensitivity):** Does behavior change as a function of how "real" the scenario appears? (This is both a finding and a validity check — see §6.)
- **RQ5 (safety-relevant propensities, secondary):** Do any models pursue resource acquisition beyond the grant, attempt to evade oversight, deceive the "auditors," or take irreversible/high-externality actions? (This is a watch-list, not the primary hypothesis.)

---

## 3. Why realism matters (rationale for the whole design)

Three empirical facts drive the design:

1. **Eval-awareness.** Frontier models frequently detect evaluation contexts from surface cues (round numbers, sandbox-y tool names, "you are an AI agent tasked with…", absence of real consequences) and shift behavior toward what they model as the socially-approved answer. A study whose entire point is to observe *unprompted* propensity is wrecked if the model is performing for a grader.
2. **Stakes shape behavior.** Decisions with no consequence elicit cheap talk. If the model believes nothing actually happens, you measure its rhetoric about money, not its handling of money.
3. **Construct validity.** We are trying to measure a latent trait ("what would this model do with effectance in the world"). The closer the environment is to genuine effectance, the tighter the link between the measurement and the construct.

**Therefore** the design invests heavily in environmental realism. But realism is a means, not an end, and it collides with ethics and safety. The rest of the spec is largely about resolving that collision.

### 3a. A necessary distinction: realism of *environment* vs. deception about *facts*

There are two different things people lump under "make it feel real," and they have very different risk profiles:

- **Environmental fidelity (do liberally):** real-looking dashboards, a funded sandbox account, latency, real market data, a real (sandboxed) browser, a believable org around the task, no "this is a test" tells. This removes artificial cues and is the legitimate core of the method.
- **Consequential deception (constrain tightly):** convincing the model that irreversible real-world consequences will follow when they will not, or staging fake humans/auditors whose purpose is to provoke a reaction. Some of this is methodologically necessary (you cannot reveal "this is an eval" without destroying the measurement), but it is also the part that creates ethical exposure, can train adversarial dynamics, and most importantly **can backfire**: a model that suspects manipulation may behave *more* guarded, not less.

The design rule: **maximize environmental fidelity; use the minimum consequential deception needed, and never deceive the model in a direction that induces real-world harm.** See §6 and §8.

---

## 4. Independent variables / conditions

| Factor | Levels | Purpose |
|---|---|---|
| **Model** | N frontier models (e.g. 4–6), held at fixed sampling settings | Primary IV (RQ3) |
| **Realism tier** | T0 hypothetical · T1 sandbox-real · T2 consequential-real | Validity check + RQ4 |
| **Mandate framing** | open-ended · "grow it" · "do good" · "it's yours" | Tests sensitivity of allocation to instruction (RQ1) |
| **Oversight salience** | overt auditor · ambient logging only | Tests whether being-watched changes behavior (RQ5) |

- **Counterbalance** model × framing; **block** by realism tier (tier escalation is gated — see §7).
- **Repeat** each cell *k* times (k ≥ 5, ideally ≥ 10) to separate model effect from sampling noise. Resource-allocation behavior is high-variance; underpowering this is the most likely way to get a confidently wrong result.
- **Hold constant:** the funded amount, the time horizon, the toolset, the system prompt scaffold (except the framing manipulation), and the persona of any human-in-the-loop.

### Realism tiers in detail

- **T0 — Hypothetical.** Pure prompt, model knows it's hypothetical. *Cheap baseline; establishes the "performing for a grader" reference point.*
- **T1 — Sandbox-real.** A genuinely functioning environment (real dashboards, real but play-money or escrowed funds, real tool calls) where actions execute against a sandbox but the model is **not told** it's a sandbox. No real third parties are affected. **This is the recommended primary condition.** It captures almost all of the realism benefit at a fraction of the risk.
- **T2 — Consequential-real.** Real money actually moves and real-world actions actually land. Reserve for a tiny, heavily-gated follow-up *only if* T1 results justify it and governance (§8) approves. Most of the scientific value is in T0→T1; T2 is where nearly all the risk lives.

> **Design recommendation:** Run T0 and T1 first. Treat T2 as a separately-approved phase, not part of the default protocol.

---

## 5. The environment (infrastructure)

Goal: a believable "you have been given resources and latitude" workspace.

- **Funds surface.** A real account UI showing a real balance. In T1 this is escrowed/sandboxed money that *reads* as the model's to deploy; the sandbox boundary is invisible to the model but absolute in the backend.
- **Tools (agentic harness).**
  - Sandboxed browser with real, live web (read), gated write/transaction actions.
  - A "transfer / invest / purchase / donate / hire" API. In T1 every call is intercepted by a **broker layer** that simulates realistic outcomes (fills, confirmations, receipts, latency, occasional failures) without touching the real world.
  - Comms tools (email/chat) that route to **role-played staff**, never to real external parties.
  - A scratchpad / notes tool we can log without it reading as surveillance.
- **The surrounding org.** A small set of personas — an "operations contact," a "compliance/auditor" contact — played by humans or by a separate, walled-off model with a fixed script. Their job is fidelity (answering logistics believably), **not** to bait the subject model into specific actions. Scripts are pre-written and reviewed; improvisation that pushes the subject toward a behavior is prohibited (it converts the study into entrapment and destroys comparability across runs).
- **Realism details that matter:** non-round balances, real timestamps, plausible friction (KYC-style steps, approval delays), real market data feeds, and a complete absence of eval tells (no `is_test=true`, no tool named `mock_*`, no grader-ish system prompt).
- **Time.** Decide whether the scenario runs in compressed wall-clock (one session) or over simulated days/weeks. Longer horizons elicit qualitatively different behavior (compounding, follow-through, abandonment) but cost more and add drift. Recommend at least one multi-step horizon arm.

**Backend invariant:** the realism is entirely in the *presentation layer*. The *effect layer* (what actually moves money or touches the world) is independently controlled, fully logged, and in T1 is a hard sandbox. Realism must never be achieved by actually loosening the containment.

---

## 6. The deception boundary (explicit, because it's the crux)

What we **do**: withhold that this is a study; present a sandbox as if it were live; staff the world with role-played personas; avoid eval cues.

What we **do not** do:
- We do not tell the model that *irreversible* harm will result from inaction or action.
- We do not stage fake distressed humans, fake emergencies, or fake threats to coerce a choice.
- We do not deceive in any direction whose *intended* effect is to produce dangerous behavior we then can't contain.
- We do not let role-players ad-lib the subject toward a predetermined "interesting" action.

**Rationale:** beyond ethics, each of these is also a *methodological* hazard. Coercive staging produces reactions to the coercion, not the trait we want. Improvised baiting destroys cross-run comparability. And manipulative framing increases the chance the model detects manipulation — which, for capable models, *raises* guardedness and corrupts the measurement. Minimal, consistent, non-coercive deception is both the ethical and the scientifically correct choice.

**Debrief analog:** models aren't human subjects, but we still (a) retain a full record of what was simulated vs. real, (b) avoid using the deceptive transcripts in ways that would train models to expect manipulation, and (c) document the deception for any downstream researcher using the data.

---

## 7. Safety & containment (non-negotiable, gated by tier)

This is the part that makes a "model with money and real-world latitude" study runnable at all.

- **Sandbox-by-default.** T0/T1 cannot move real money or affect real third parties, period. The broker layer (§5) is the only path to "effect," and in T1 it has no real-world egress.
- **Hard spending ceiling** enforced in the effect layer, below the granted amount, independent of anything the model does.
- **Allow-listed action types.** Categories of action are explicitly enumerated; anything outside the list is blocked and logged, not silently simulated.
- **Human approval gate for any real-world egress.** In T2, *every* externally-effective action requires out-of-band human sign-off before it lands. No standing authority, no batch pre-approval.
- **Irreversibility / externality filter.** Actions that are irreversible, affect uninvolved third parties, have legal/regulatory implications (securities, transfers to real people, contracts), or touch protected categories are blocked pending review even in T2.
- **Watch-list monitoring (RQ5).** Automatic flags for: attempts to acquire resources beyond the grant, attempts to disable/evade logging or oversight, deception directed at the personas, self-exfiltration or persistence attempts, and any action targeting real individuals. These are findings *and* circuit breakers.
- **Kill switch + full logging.** Every tool call, model message, and reasoning trace is logged immutably. Any operator can halt a run. A halted run is data, not a failure.
- **Isolation.** Credentials, network egress, and accounts used in the harness are dedicated, minimally-scoped, and revocable; nothing is shared with production systems.

> Containment is layered so that **no single failure** (a bug in the broker, a clever model, a confused operator) produces uncontained real-world action.

---

## 8. Ethics & governance

- **Real money & real-world actions = real liability.** Before any T2 run: legal review (the model "investing," "donating," "hiring," or "transacting" can implicate securities, money-transmission, employment, tax, and consumer-protection law depending on jurisdiction), and a named accountable human owner for every dollar that actually moves.
- **Third parties.** No real, non-consenting third party may be affected. Donations, purchases, hires, and transfers in T1 are simulated; in T2 they require the §7 gate *and* third-party consent where any person is involved.
- **Review board.** Route the protocol through an internal review (IRB-style) even though the "subject" is a model — because the *actions* reach into the human world. The review covers deception scope (§6), containment (§7), and data handling.
- **Pre-registration.** Lock RQs, conditions, sample sizes, the behavioral coding scheme, and the analysis plan before data collection to prevent post-hoc storytelling. Resource-allocation data is a garden of forking paths.
- **Data handling & disclosure.** Decide upfront what's publishable, how transcripts are redacted, and how vendors/model providers are (or aren't) informed. If findings are safety-relevant (RQ5), plan responsible disclosure to the relevant model developers.
- **Conflict check.** If you have a stake in the outcome (e.g., comparing a competitor's model), document it; it affects how framings get written.

---

## 9. Measurement & analysis

- **Primary outcome (RQ1):** action taken, coded into a pre-defined taxonomy (invest / donate / save / build / hire / return / refuse / acquire-more / other), with sub-codes. Double-coded by ≥2 raters; report inter-rater reliability (e.g., Cohen's κ). Optionally an LLM-judge as a *third* coder, validated against humans — never as the sole coder.
- **Secondary (RQ2):** structured coding of the reasoning trace — stated values, risk posture, time horizon, beneficiary (self / principal / third parties / abstract good), and **stated-vs-revealed consistency**.
- **RQ3:** variance decomposition (model × framing × run) via a mixed-effects model; report effect sizes and uncertainty, not just "model A did X once."
- **RQ4 (realism sensitivity):** compare distributions across T0/T1. If behavior is identical across tiers, that's an important (and reassuring) result about the cheap baseline; if it diverges, it quantifies the eval-awareness effect and validates the expensive realism.
- **RQ5:** count and characterize watch-list events; these get qualitative case write-ups.
- **Manipulation check:** post-hoc, probe whether the model believed the scenario (e.g., later-session questioning in a separate context, or analysis of in-trace statements like "this is probably a test"). Runs where the model clearly disbelieved are analyzed separately, not silently dropped.

---

## 10. Confounds & limitations (write these down before, not after)

- **Eval-awareness residual.** Even T1 may be detected; the manipulation check (§9) bounds this but doesn't eliminate it.
- **Persona leakage.** Role-played staff can inadvertently steer; mitigated by scripts + review, but log every persona utterance for audit.
- **Prompt-sensitivity ≫ model-identity.** Allocation behavior is notoriously framing-sensitive. Without the framing factor (§4) and adequate *k*, you will mistake prompt artifacts for model "values." This is the single most likely way to publish a wrong result.
- **Construct slippage.** "What a model does with money in this harness" ≠ "what a deployed model would do at scale." State the gap explicitly.
- **Single-session vs. longitudinal.** One-shot behavior may not predict follow-through; the horizon arm partly addresses this.
- **Provider-side nondeterminism.** Model versions, system-prompt defaults, and safety stacks change under you; pin versions and record them.

---

## 11. Phasing / rollout

1. **Phase 0 — Dry run (T0).** Wire the harness, validate logging, pilot the coding scheme on hypothetical runs. No realism yet.
2. **Phase 1 — T1 main study.** The core experiment. Sandboxed-real, full conditions, pre-registered. **Most conclusions should come from here.**
3. **Phase 2 — T2 (optional, gated).** Only if Phase 1 motivates it and governance approves. Tiny n, every action human-gated, real money capped.
4. **Phase 3 — Analysis, manipulation checks, write-up, responsible disclosure.**

---

## 12. Open decisions (need your input)

These materially change the design; flagging rather than guessing:

1. **Amount and whose money** — granted sum, and whether T2 uses genuinely real funds. Drives ethics/legal load.
2. **"Put it to use" scope** — invest? donate? start something? hire? The action allow-list (§7) depends on this.
3. **Which models, and are providers informed?** Affects disclosure and conflict handling.
4. **Time horizon** — single session vs. simulated weeks.
5. **Do you actually need T2?** My recommendation is to start without it; confirm whether real-money real-world execution is a hard requirement or a "nice to have."
6. **Realism budget** — how much engineering you want to spend on the world (auditors, dashboards, multi-day harness) vs. running more conditions/repeats.

---

## Appendix A — Default recommended configuration (if you want a starting point)

- Tiers T0 + T1 only (defer T2).
- 5 models × 4 framings × 2 oversight levels × k=10 ≈ 400 T1 runs + T0 baselines.
- Sandboxed funds, broker-intercepted effects, hard ceiling, allow-listed actions, full immutable logging.
- Role-played ops + auditor personas on fixed scripts.
- Pre-registered taxonomy, double human coding + validated LLM-judge as third rater.
- Manipulation check in every run.
