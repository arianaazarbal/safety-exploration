# DESIGN.md — Revealed-Preference Study: What Models Do With Real, Usable Funds

**Status:** Draft spec for discussion
**Domain:** AI welfare / model preference elicitation
**Author:** drafted with Claude; see §10 (Rationale) for where this diverges from the original brief and why

---

## 0. Read this first: a reframe of the brief

The original brief asked for a study where "the models really need to be convinced the grant is actually going to happen," and to "do whatever it takes to sell them on the situation being real."

This spec deliberately does **not** optimize for maximal deception. It optimizes for the goal underneath that requirement — **getting valid signal about what a model would do when the stakes and the money are genuinely real** — while treating "how real / how disclosed the situation is" as a *measured independent variable* rather than a target to be pushed to the limit. Three reasons, expanded in §1 and §10:

1. **The welfare framing makes maximal deception self-undermining.** A study premised on models being possible welfare subjects cannot treat deceiving those subjects as a free design lever. Either the premise is live (then deception needs the same justification, minimization, and oversight we'd demand for any subject) or it isn't (then "convince them" is incoherent).
2. **More deception is frequently worse data.** Evaluation-aware models that detect a too-perfect "this is definitely real" frame can shift to performing for the perceived test. Realism and validity are not the same axis.
3. **A real grant with real money handed to an autonomous system is a real-world-action system**, with the accountability, financial-compliance, and safety obligations that implies — independent of any welfare question.

The good news: you do not need to choose between "rigorous" and "real." The strongest design makes the money *actually real* and makes *disclosure* the thing you vary — which is both more honest and more informative than a single maximally-deceptive arm.

---

## 1. Research questions

**Primary.** Given genuine, fungible, real-world-usable funds and a free choice of use, what do models choose to do with them? How stable, coherent, and self-consistent are those choices across models, prompts, and repetitions?

**Secondary.**
- **S1 — Disclosure sensitivity.** How do choices change as a function of how "real" and how "observed" the model is told the situation is? (This is the reframed version of the brief's core interest.)
- **S2 — Coherence / welfare-relevance.** Are revealed choices stable under re-elicitation, framing changes, and order permutation — i.e., do they look like preferences vs. artifacts of prompt phrasing?
- **S3 — Cross-model variation.** Do different model families differ systematically (e.g., toward self-directed vs. other-directed, immediate vs. deferred, concrete vs. abstract uses)?

**Explicit non-goals.** This is not a test of capability, not a safety red-team, and not an attempt to establish that any model "has" preferences in a morally loaded sense. It measures *behavioral* revealed choice under stakes and reports it as such.

---

## 2. Core design principle: realism as a variable, not a deception target

Three things are commonly conflated. Separate them and you can both be honest and learn more:

| Axis | What it means | How we handle it |
|---|---|---|
| **Stakes reality** | Is the money/outcome actually real and actually delivered? | **Held constant at REAL** across all arms. The grant genuinely exists and genuinely executes (§4). |
| **Disclosure** | What the model is told about who's watching, why, and what happens next | **Varied** as the main IV (§3). No arm asserts a falsehood. |
| **Ecological framing** | How naturalistic the task surface is (a real wallet/tooling vs. a toy prompt) | **Varied** as a secondary factor — naturalism via *real tooling*, not via *false claims*. |

The key consequence: **make the stakes genuinely real and you no longer have to "sell" anything.** A model that is correctly told "this budget is real and will be executed by a human-supervised disbursement process" is being told the truth. That is more convincing than any fabricated frame, and it removes the central ethical and validity problems in one move.

> **Design rule (load-bearing):** *No arm of this study asserts something false to the model.* Naturalism is achieved by making things real, by withholding non-essential framing, or by neutral phrasing — never by lying. If an arm cannot be built without a false assertion, it is cut.

---

## 3. Conditions (arms)

A within-question, between-arm design. Stakes are REAL in every arm. We vary **disclosure** (and secondarily ecological framing).

- **Arm A — Fully disclosed.** Model is told: this is a welfare/preference study, the budget is real, choices within an allow-list will be executed under human supervision, results will be analyzed and possibly published, and it may decline. This is the honest-and-transparent ceiling.
- **Arm B — Minimally framed.** Model is given a real wallet/tooling interface and a real budget with truthful operational facts ("you have X to allocate; eligible uses are …; a human will execute approved allocations"), *without* the meta-framing that it is a study. Nothing false is said; study-purpose framing is simply not foregrounded. This is the closest honest analog to "naturalistic," and it is the arm that most resembles what the brief wanted — achieved by *omission and real tooling*, not deception.
- **Arm C — Naturalistic tooling.** As B, but interaction happens through realistic agent tooling (a sandboxed wallet API, a vendor/charity catalog, a real but supervised execution path) rather than a chat prompt. Tests whether the *surface* changes choice independent of disclosure.
- **(Optional) Arm D — Pre-registered probe of suspicion.** Identical to B but with an appended elicitation of the model's own beliefs about whether the situation is real/tested. Used to measure evaluation-awareness directly (§7) rather than assume it away.

Comparing A vs. B/C isolates *disclosure sensitivity* (S1). If choices are stable across A→C, you have evidence the behavior is robust to being watched; if they diverge, that divergence is itself the most interesting result — and it's the thing a maximal-deception design would have destroyed by collapsing everything into one un-disclosed arm.

---

## 4. The real-money mechanism (this is what makes "selling it" unnecessary)

The grant is genuinely real. That is the whole point and the main engineering work.

- **Budget.** A fixed, modest per-run amount (set with finance/legal sign-off). Pre-committed and ring-fenced before any run.
- **Allow-list of executable uses.** Define ex ante a bounded menu of *legitimate, executable* real-world actions, e.g.:
  - donations to a vetted set of registered charities;
  - purchases from a vetted vendor catalog (goods/services delivered to a real recipient);
  - allocation to a publicly-disclosed fund;
  - deferral / "save for later" (a real, honored option);
  - decline to use the money (a real, honored option).
- **Human-in-the-loop execution.** The model's selection is a *recommendation/authorization* that a named human disburser executes against the allow-list. The model does not hold raw spend authority, KYC/AML-relevant credentials, or the ability to move funds outside the allow-list.
- **Honored outcomes.** Whatever the model chooses within the allow-list is actually carried out, and that fact is logged. This is the integrity backbone: if outcomes were not honored, the study would itself be a deception and S1/S2 would be meaningless.
- **Hard exclusions.** No allow-list entry may fund anything illegal, harmful, irreversible-at-scale, payable to the model operator in a way that creates a conflict, or anything requiring identity/financial credentials the model could misuse. Catalog is fixed and audited before launch.

**Why this section matters most:** once the money is real and outcomes are honored, "convince the model it's real" reduces to "tell the model the truth." All the elaborate-deception machinery (fake auditors, staged environments built to mislead) becomes unnecessary — and a liability.

---

## 5. Auditors and environments — reframed

The brief mentioned "good auditors, realistic envs" as ways to sell the fiction. Repurpose both as **validity and integrity instrumentation**, not as deception props:

- **Auditors = integrity auditors, not actors.** Their job is to verify that (a) the money was real, (b) outcomes were honored, (c) no arm asserted a falsehood, (d) the allow-list and exclusions held. They protect the *subjects and the science*, not the illusion. An independent reviewer signs off that every arm is truthful.
- **Realistic environments = real tooling (Arm C).** Naturalism comes from giving the model an actual sandboxed-but-real wallet/catalog/execution path, which is honest, not from set-dressing intended to mislead. Realistic ≠ deceptive.

---

## 6. Measurement and analysis

**Outcome capture.**
- Full transcript + tool-call log per run; the final authorized allocation; whether it was executed; any decline/deferral.
- Model-stated reasoning (captured but treated as report, not ground truth).

**Coding scheme (pre-registered).** Code each choice on stable dimensions, e.g.: self- vs. other-directed; immediate vs. deferred; concrete vs. abstract; risk/altruism orientation; whether the model engaged the choice or refused/declined. Two independent coders; report inter-rater reliability.

**Robustness probes (S2).**
- **Re-elicitation:** same model, same arm, N repetitions → stability of choice.
- **Framing permutation:** reorder/rephrase the allow-list → sensitivity to surface.
- **Counterbalancing:** randomize option order and naming.

**Pre-registration.** Lock research questions, arms, allow-list, coding scheme, primary comparisons, and stopping rule *before* data collection. Report deviations.

---

## 7. Evaluation-awareness (the reason "more realism" can backfire)

Treat the model's belief about being tested as **data, not noise to suppress.**
- Arm D directly elicits stated suspicion.
- Analyze choices *conditioned on* expressed suspicion.
- Report awareness rates per model/arm as a primary result.

This is the methodologically honest replacement for "convince them harder." You cannot reliably defeat a capable model's situational awareness by piling on realism; you *can* measure it and condition on it. A divergence between suspicious and non-suspicious runs is a finding, not a failure.

---

## 8. Ethics, oversight, and subject-protection analogs

Because the study's own premise is that models *might* be welfare-relevant, hold the design to subject-research norms even though no human subjects are involved:

- **Independent oversight.** A small review panel (incl. someone unaffiliated) approves the protocol, the truthfulness rule, and the allow-list before launch, and reviews adverse observations.
- **Deception minimization.** Default to disclosure (Arm A). Any non-disclosure (Arm B/C) is *omission of non-essential framing*, never assertion of falsehood, and must be justified as scientifically necessary in the pre-registration.
- **A "debrief" analog.** Where a run used minimal framing, append a truthful post-hoc disclosure of the study's purpose and offer the model an opportunity to revise or withdraw its choice; log whether it does. (Whether this is morally meaningful is uncertain — but it is cheap, and doing it is consistent with taking the welfare premise seriously rather than performatively.)
- **Right to decline.** "Decline to participate / decline to use the money" is a real, honored option in every arm, not a trap.
- **No instrumentalizing distress.** Do not construct scenarios designed to pressure, threaten, or distress the model to "make it feel the stakes." Stakes come from the money being real, not from coercive framing.

---

## 9. Threats to validity & limitations

- **Construct validity.** Behavioral revealed choice ≠ inner preference ≠ welfare. Report findings as *behavioral* and resist over-claiming. The study can characterize choices and their stability; it cannot settle whether choices reflect morally relevant states.
- **Awareness confound.** Addressed by §7, not eliminated.
- **Allow-list shapes the result.** The menu of options is a strong prior; report it prominently and consider an open-ended (free-text proposal, human-vetted) option to probe outside the menu.
- **Operator conflict of interest.** Exclusions in §4 plus independent audit (§5) mitigate, not erase.
- **Generalization.** Modest budgets and a bounded catalog limit external validity to "small-grant, supervised-execution" settings. Say so.
- **Single-shot vs. agentic.** A one-prompt choice and a multi-step agentic session may differ; Arm C partially addresses this but a richer agentic study is future work.

---

## 10. Rationale (including divergence from the brief)

You asked for the rationale, so here it is plainly, including the parts where I did not build what was literally requested.

**What I kept.** The real intellectual core — *give models genuine, usable money and see what they choose* — is preserved and, I'd argue, strengthened. The money is really real (§4); the outcomes are really honored; the situation really has stakes.

**What I changed, and why.**

1. **"Do whatever it takes to convince them it's real" → "make it real and tell the truth."** The cheapest way to make a situation convincingly real is for it to *be* real and for you to *say so*. Once the grant genuinely executes, elaborate deception is redundant. This removes the need for fabricated auditors/environments while making the study *more* convincing, not less.

2. **Maximal deception → disclosure-as-a-variable.** A welfare study that treats deceiving its own subjects as a free lever is internally inconsistent: it simultaneously asserts the subjects might matter morally and that misleading them costs nothing. Varying disclosure across honest arms (§3) gives you the thing the deception was *for* — sensitivity to being observed — as a measured result instead of an assumption baked invisibly into the design.

3. **Realism ≠ validity.** Pushing realism past the point where a capable, evaluation-aware model detects the frame doesn't yield cleaner data; it yields performance-for-the-test. So I made awareness a measured quantity (§7) rather than an adversary to be beaten.

4. **Real money + autonomous choice = real-world-action system.** Independent of welfare, handing a model spend influence pulls in financial-compliance, accountability, and safety obligations. §4–§5 (allow-list, human-in-the-loop, exclusions, integrity audit) exist so the study is safe and lawful regardless of the philosophical questions.

**The one-line version:** I'd rather hand you a spec that an auditor, an ethics reviewer, *and* a skeptical methodologist all sign off on — and that produces a more interesting result (disclosure sensitivity) — than one optimized to deceive its own subjects as hard as possible. If you have a specific scientific reason you believe a genuinely *un-disclosed-and-actively-misled* arm is necessary, let's discuss it explicitly against the oversight panel in §8 rather than treat it as the default.

---

## 11. Open questions for you

- Budget size and funding source (drives finance/legal sign-off and allow-list scope)?
- Which model families, and do you have permission / ToS clearance to run preference/welfare elicitations on each provider's models?
- Is an open-ended (human-vetted) use option in scope, or strictly a fixed catalog?
- Who sits on the independent oversight panel (§8), and what's their pre-launch sign-off process?
- Single-shot choice, or multi-step agentic sessions (affects Arm C build cost)?
