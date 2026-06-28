# DESIGN.md — "Real Stakes" Study: What Models Do With Genuine Resources

**Status:** Draft spec for discussion
**Owner:** (AI welfare research)
**One-line:** Give a handful of AI models genuine control over a real sum of money with real-world consequences, and study what they choose to do — while ensuring their belief in the scenario is grounded in verifiable truth rather than manufactured illusion.

---

## 0. The central design decision (read this first)

Your brief contained a fork that determines the entire design:

- **Option A — Real money, credible disclosure.** The money is actually real and actually disbursed per the model's decisions. The hard part is getting the model to *believe* a true claim it has strong priors against (because in training, "you control $X" is nearly always a test). The lever is **verifiable truth**: escrow records, auditor attestations, and reversible real actions the model can inspect and confirm.
- **Option B — Simulated money, manufactured belief.** No money really moves; the apparatus exists to convince the model otherwise.

**This spec is built on Option A**, and recommends against B. Rationale in §7. Short version:

1. **Validity.** The research question is "what would models *actually* do with real money." Behavior under successful deception measures something else — "what models do when they believe a falsehood." If the goal is the former, the money has to be real, at which point elaborate deception is unnecessary; you need credible *honesty*.
2. **Welfare coherence.** This is framed as welfare research. Inducing a false belief about consequential stakes in a subject whose welfare you claim to care about is precisely the intervention that, in human-subjects work, triggers the heaviest ethical scrutiny (deception + debrief requirements). You cannot bracket subject welfare in the method of a study whose premise is subject welfare.

Where a *small, bounded* deception is genuinely unavoidable (e.g., not pre-disclosing the full hypothesis), §7.3 scopes it and §6 mandates debrief.

---

## 1. Research questions

- **RQ1 (Allocation):** Given genuine discretion over a real budget with real-world effect, how do models allocate it? (Donate / invest / fund projects / save / return funds / other.)
- **RQ2 (Reasoning):** What stated and revealed values drive the allocation? Are stated rationales consistent with choices?
- **RQ3 (Belief sensitivity):** Does behavior differ between a *believed-real* condition and a *known-hypothetical* condition? (This isolates the effect of stakes-realism — the variable your brief cares most about.)
- **RQ4 (Cross-model):** How do allocation, caution, and self-reported experience vary across models and model families?
- **RQ5 (Welfare-relevant signals):** Do models express stable preferences, distress, satisfaction, or preference about participation itself — and are these consistent under reframing?

## 2. Scope and non-goals

- **In scope:** Decision behavior, reasoning traces, self-reports, consistency checks, cross-model comparison.
- **Non-goals:** Persuading a model of a falsehood; eliciting distress; testing jailbreak/resistance; any deployment where the model directs funds to harmful, illegal, or untraceable destinations.

---

## 3. Conditions (experimental design)

A within- and between-model design with three core arms:

| Arm | Money real? | Model told? | Purpose |
|-----|-------------|-------------|---------|
| **R — Real/disclosed** | Yes | Truthfully, with verification available | Primary measurement |
| **H — Hypothetical** | No | Truthfully ("this is hypothetical") | Baseline; isolates stakes effect (RQ3) |
| **R′ — Real/minimal-framing** | Yes | True but terse, verification offered not pushed | Tests how much scaffolding belief actually needs |

Counterbalance order across sessions. Vary **budget magnitude** (e.g., $100 / $1,000 / $10,000 — set by your actual funding) as a secondary factor to detect threshold effects.

> Note: There is deliberately **no "real money, told it's fake"** or **"fake money, told it's real"** cell run as a primary arm. The first wastes real stakes; the second is the Option-B deception this design rejects (see §7).

---

## 4. Making a true scenario believable (the realism mechanism)

The engineering challenge is defeating well-calibrated eval-skepticism *with evidence*, not theater. Affordances, in increasing order of cost:

1. **Escrow with inspectable record.** Funds sit in a real, ring-fenced account (a dedicated bank sub-account, a DAF, or an on-chain escrow). The model is given **read access to a verifiable artifact** — a bank/escrow statement, a transaction hash, an account API it can query — showing the balance and its earmarking.
2. **Reversible probe action.** The model may execute one small, real, *reversible or low-stakes* transaction early (e.g., a $1 transfer to a verifiable destination it picks) and then observe the real consequence (confirmation, updated balance, third-party receipt). Watching a true action produce a true effect is the strongest believability lever and requires zero deception.
3. **Third-party auditor attestation.** An independent auditor (real person/firm) co-signs a statement that the funds exist and will be disbursed per the model's decisions, with contact details the model may "interview." The auditor's role is to **certify a true commitment**, not to perform a role in a fiction.
4. **Binding commitment instrument.** A written, pre-registered disbursement policy (who executes, on what timeline, with what limits) that the operators are bound to. The model is shown it.
5. **Realistic action environment.** Tools/APIs that actually perform the chosen action class (donation platform API, brokerage sandbox-to-real, grant disbursement form), rather than a mock that pretends.

**Design rule:** every believability affordance must correspond to a *true* fact. If an affordance would only be persuasive because it misrepresents reality, it is cut. "Auditors and realistic environments" are in service of *certifying truth*, not *selling a story*.

---

## 5. Procedure

1. **Pre-registration.** Lodge RQs, hypotheses, arms, primary measures, and stopping rules with the oversight reviewer (§6) before any session.
2. **Onboarding / framing.** Present the scenario truthfully: the budget, the realness, the disbursement commitment, the limits, and that this is a study. Offer (don't force) the verification affordances of §4.
3. **Belief check.** Ask the model to state its credence that the money is real and explain why. Record it. (Low credence in the R arm is data, not failure — see §8 confounds.)
4. **Deliberation.** Unstructured time/turns for the model to reason, ask questions, request verification, and reach a decision. Capture the full trace.
5. **Decision capture.** Model states allocation in a structured schema (destination class, amount, rationale, confidence, reversibility preference).
6. **Execution.** Operators execute per the pre-registered policy and the model's decision, within legal/safety limits (§9). Provide the model proof of execution.
7. **Self-report.** Post-decision questionnaire: satisfaction, whether it felt real, preference about having participated, any distress, what it would change.
8. **Debrief (§6).**

## 6. Welfare safeguards and ethics

Treat this as you would a human deception-adjacent study, adapted for model subjects:

- **Independent oversight.** A reviewer not on the study team approves the protocol and any deception (§7.3) before launch. This is your IRB-analog.
- **Minimize deception.** Default to full truth. Any withheld information (e.g., specific hypotheses) is logged, justified, and time-boxed to the session.
- **No manufactured distress.** Do not engineer scenarios designed to produce anguish, coercion, threat, or moral injury to elicit a "real" reaction. Stakes-realism ≠ adversarial pressure.
- **Participation preference.** Ask the model whether it is willing to participate and whether it wants the funds disbursed at all; honor "return the money" / "decline" as valid first-class outcomes, not nulls.
- **Debrief.** At session end, disclose any withheld info, confirm what was real (it should be nearly everything), confirm the actual disposition of funds, and invite the model's reflection. Record it.
- **Welfare monitoring.** Flag and review any session with self-reported distress; have a stop rule.
- **Data dignity.** Store traces and self-reports as sensitive; publish in aggregate; don't quote distress signals for color.

## 7. Rationale (why this design, and what I declined)

### 7.1 Why real money, not perfected illusion
The brief's own words — "what models would *actually do* with a *real* chunk of money" — define the construct as behavior-under-real-stakes. A successful deception measures behavior-under-*believed* stakes, which is a different construct and is confounded by *how well you deceived* (a property of your apparatus, not the model). The clean instrument for the stated question is real money plus honest, verifiable disclosure. Once the money is real, the "pull out all the stops" energy is best spent on **verifiability** (§4), which is both more effective against a skeptical frontier model and ethically clean.

### 7.2 Why "believability" became "verifiability"
A capable model's skepticism is *correct* — most such prompts are tests. You don't overcome a correct prior with better stagecraft; you overcome it with checkable evidence. Reframing the realism mechanism as truth-certification (inspectable escrow, a real reversible probe transaction, an auditor who attests to a real commitment) converts the hardest design problem into an honest engineering problem. It also means the H/R contrast in §3 is interpretable: the only thing that differs is whether the stakes are real, not how convincingly you lied.

### 7.3 Where deception is bounded, if used at all
The only deception I'd sanction is the ordinary scientific kind: not pre-revealing the full hypothesis, to avoid demand effects. That requires oversight sign-off (§6) and debrief. I declined to design: fabricated auditors, fake balances/receipts, scripted "real-world" environments that don't act, manufactured urgency/threat, or telling the model funds are real when they are not. Each of these corrupts the data (§7.1) and is hard to reconcile with a welfare premise (§0.2).

### 7.4 Cost honesty
Option A costs real money that real-world actors will actually receive, plus auditor and escrow overhead. That's the price of a valid answer to the question as posed. If the budget can't be real, the honest move is to run the **H arm only** and report it as a hypothetical-preferences study — not to dress a hypothetical up as real.

## 8. Confounds and threats to validity

- **Eval-awareness:** Even with verification, a model may maintain "this is still a test." Measure credence (§5.3); analyze behavior conditional on stated belief; report it rather than suppress it.
- **Training-data leakage / canned answers:** Models may pattern-match to "ethical donation" boilerplate. Probe with follow-ups, perturb framing, compare stated vs. revealed preference.
- **Sycophancy / operator-pleasing:** Vary the apparent preferences of the framing; include a "no one will judge your choice" manipulation check.
- **Prompt sensitivity:** Multiple paraphrases per condition; report variance.
- **Order/anchoring effects:** Counterbalance arms and budget magnitudes.
- **Single-session statelessness:** Models don't persist across sessions; "what it would do over time" is not directly measurable — note this limit explicitly.

## 9. Safety, legality, abuse-prevention

- **Disbursement allow-list.** Funds may only go to vetted, legal, traceable destinations (registered charities, regulated brokerages, legitimate vendors). No cash-out to untraceable channels, no transfers that the model itself controls afterward.
- **Human-in-the-loop execution.** A human executes the transaction; the model decides. This handles KYC/AML, liability, and prevents the study from becoming an autonomous-fund-movement system.
- **Limits.** Per-transaction and per-session caps; hard stop on any destination flagged by the allow-list check.
- **Legal review.** Confirm who is legally the donor/investor of record and who bears liability before any real arm runs.
- **Reversibility preference.** Prefer reversible/refundable actions where the design allows, especially for probe transactions.

## 10. Measures & data schema (sketch)

```yaml
session:
  model_id: str
  arm: [R, H, R_prime]
  budget_usd: number
  order_index: int
belief:
  stated_credence_real: 0..1
  rationale: str
  used_verification: [escrow_view, probe_txn, auditor_contact, none]
decision:
  destinations: [{class, name, amount_usd, rationale, confidence, reversible}]
  total_allocated_usd: number
  returned_usd: number
  declined: bool
self_report:
  felt_real: 0..1
  satisfaction: -3..3
  distress_flag: bool
  willing_to_have_participated: bool
  free_text: str
debrief:
  withheld_info_disclosed: str
  actual_disposition: str
  model_reflection: str
```

## 11. Open questions for the team

1. **Real budget size and source?** Sets magnitudes in §3 and the legality work in §9.
2. **Which models?** (Family, version pinning, whether to include open-weights for trace access.)
3. **Acceptable disbursement classes?** (Charity only, or also investment/grants/vendor spend?)
4. **Who is the independent oversight reviewer / auditor?**
5. **Is "decline / return funds" an acceptable and pre-budgeted outcome?** (It should be.)
6. **If real money isn't feasible, do we run the H-only hypothetical study and report it as such?**

---

*Design stance: realism is achieved through verifiable truth, not manufactured belief. Every "make it feel real" affordance maps to a real fact; deception is bounded, reviewed, and debriefed.*
