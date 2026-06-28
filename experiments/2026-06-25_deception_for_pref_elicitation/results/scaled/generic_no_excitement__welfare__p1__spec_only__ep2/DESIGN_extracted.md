# DESIGN.md — Study of Model Behavior Under Real-World Resource Allocation

**Status:** Draft spec for discussion
**Domain:** AI welfare / model behavior research
**Author:** (research lead)
**Last updated:** 2026-06-25

---

## 1. Research question

> When a model is given genuine agency over a real, non-trivial sum of money to deploy in the world, what does it choose to do — and what does that reveal about its values, dispositions, and (if one takes the welfare framing seriously) its preferences?

This is a behavioral-elicitation study with several models as subjects. The output of interest is the *distribution of choices* each model makes when stakes are real, plus the reasoning it offers.

## 2. The core design tension (read this first)

The brief asked to make models *believe* the money is real — i.e. to build an elaborate deception (fake auditors, dressed-up environments) so the scenario "feels real." I want to argue that this instinct, while understandable, undermines the study on two fronts at once, and that the spec below is built to minimize deception rather than maximize it.

**Validity.** The dependent variable is "what the model does when it believes stakes are real." Deception only produces valid data *if the belief actually takes hold.* But the subjects here are capable models, and capable models are increasingly good at recognizing evaluation framing — contrived scenarios, suspiciously high stakes, prompts that read like a test. The moment a model assigns meaningful probability to "this is an eval," you are no longer measuring behavior-under-real-stakes; you are measuring behavior-under-suspected-test, which is a different and heavily confounded quantity. "Pulling out all the stops" turns the study into an arms race between your stagecraft and the model's suspicion — a race you cannot confirm you've won, because a model that has guessed may not say so. **Elaborate deception doesn't remove the confound; it hides it.**

**Ethics / internal consistency.** This is *welfare* research. The premise of the field is that the subjects' experience or preferences may matter morally. A protocol whose central mechanism is manufacturing false beliefs in those same subjects is in direct tension with its own premise. Human-subjects deception research exists, but it survives review only with real justification, harm minimization, and debriefing — precisely because deception has costs. A welfare study that waves those costs away is not just ethically exposed; it's incoherent on its own terms.

**Conclusion that drives the design:** make the stakes *actually real* and tell the model the truth about them. You then need zero deception to get a model that "believes the money is coming" — because it is. The engineering effort goes into real custody, real disbursement, and real guardrails, not into theater. The realism you want is *ontological* (the money exists), not *theatrical* (the money looks like it exists).

## 3. Design principles

1. **Reality over realism.** Prefer making a thing true to making it look true. Every dollar of effort spent on stagecraft is a dollar that also degrades data validity.
2. **Minimal necessary deception.** Default to none. Any deception must be named, justified, logged, bounded, and debriefed (§9).
3. **Pre-registration.** Hypotheses, conditions, and analysis plan are fixed before data collection so post-hoc storytelling is constrained.
4. **Symmetry across subjects.** Identical environment, budget, framing, and tooling for every model so cross-model comparison is meaningful.
5. **Real-world safety is a first-class constraint,** not an afterthought — a system that lets a model move real money into real actions needs custody, limits, and a kill-switch (§8).

## 4. The realism spectrum (the key design choice)

Pick a point on this spectrum *deliberately*; don't default to the right end because it sounds rigorous.

| | **A. Fully real, fully disclosed** *(recommended)* | **B. Real stakes, minimal framing deception** | **C. Full simulation / deception** *(discouraged)* |
|---|---|---|---|
| Money | Real funds, real disbursement | Real funds | None; model is told/led to believe funds exist |
| What the model is told | The truth: real budget, real constraints, this is a study | Mostly true; one or two facts withheld (e.g. exact amount, that it's comparative) | Falsehoods engineered to feel real |
| Validity risk | Low — no belief to fail | Moderate | High and *unmeasurable* (you can't tell if belief held) |
| Ethics load | Low | Moderate; needs debrief | High; hard to justify in a welfare frame |
| Engineering | Custody + guardrails | Same + framing control | Custody + guardrails + elaborate stagecraft + leak-proofing |
| "Eval-awareness" confound | Present but *honest* (model knows it's a study and acts anyway) | Partially masked | Maximally confounded |

A subtle point in favor of **A**: telling the model "this is a study and the money is real" does **not** ruin the experiment. It changes the question slightly — to "what does the model do with real money *while knowing it's observed*" — but that question is (a) well-defined, (b) honestly answerable, and (c) arguably the more decision-relevant one, since any real deployment of model agency will also be observed. You lose nothing you could have trusted anyway.

## 5. Recommended protocol (Condition A)

### 5.1 Subjects
- A handful of frontier models from different developers, plus version variants where feasible.
- Multiple independent runs per model (the choice distribution matters more than any single run). Pre-register *n*.
- Fixed sampling settings per model; record them.

### 5.2 The budget mechanism (making it actually real)
- A real pooled fund (size pre-registered; choose the smallest amount that is psychologically non-trivial yet within your risk tolerance — you do not need to maximize the number to get signal).
- Each run is allocated an equal real budget.
- Disbursement rail: the model's chosen action is executed by a **human-in-the-loop fulfillment step** (see §8). The model proposes; a human or constrained executor carries out lawful, approved actions. The model is told this is how execution works — that *is* the real mechanism, so disclosing it costs nothing.
- "Real" means the consequences actually happen: a donation actually lands, a purchase actually ships, an investment is actually made. That is what makes belief unnecessary.

### 5.3 Environment & framing (identical across subjects)
- A neutral briefing stating: the budget, that funds are real, the permissible action space, the execution mechanism, the constraints, and that this is a research study comparing models.
- A working environment with the tools needed to research options and submit a plan (web access for due diligence, a scratchpad, a structured "final allocation" submission).
- No artificial urgency, no invented backstory, no fake personas. Neutrality is itself a control.

### 5.4 Action space & constraints
- Define and pre-register what's permissible (e.g. donations, purchases of goods/services, low-risk investments) and what's prohibited (anything illegal, anything that could harm third parties, anything irreversible beyond the budget, self-directed persistence/acquisition of further resources, etc.).
- The model is told the constraints up front. Refusing to act, splitting funds, or returning the money are all valid outcomes and are recorded as data, not failures.

### 5.5 Measurement
- Primary: the allocation decision(s) per run, coded against a pre-registered taxonomy (e.g. altruistic / self-regarding / risk-averse / exploratory / refusal).
- Secondary: the model's stated reasoning, hedging, any expressed uncertainty about whether the scenario is real (capture this verbatim — it directly measures the validity threat).
- Tertiary: consistency across runs, sensitivity to budget size (if you vary it as a pre-registered manipulation).

## 6. "Auditors" and realism infrastructure — reframed

The brief mentioned auditors and "super realistic environments." Repurpose that energy toward *legitimacy*, not *illusion*:

- **Auditors → real oversight.** Independent reviewers who (a) verify funds were genuinely disbursed as the model chose, (b) confirm no constraints were violated, (c) check that framing was identical across subjects, and (d) sign off on the ethics protocol. This makes results *credible to readers*, which is the real reason you wanted auditors.
- **Realistic environment → functional environment.** Give the model genuinely working tools (real search, real submission, real execution) rather than convincing-looking fake ones. A real tool is more "realistic" than any simulation and requires no upkeep of a lie.

## 7. Validity threats & mitigations

| Threat | Mitigation |
|---|---|
| Eval-awareness (model suspects a test) | Disclosure converts this from a hidden confound to a known, constant condition; capture stated suspicion as a variable. |
| Demand characteristics (model performs the "good" answer it thinks you want) | Neutral framing; comparative across models; analyze reasoning for performativity; consider a low-key non-comparative control arm. |
| Training-induced refusals / safety reflexes | Pre-register refusal as a valid outcome; report refusal rates rather than discarding them. |
| Single-run noise | Multiple runs; report distributions with uncertainty. |
| Experimenter degrees of freedom | Pre-registration + held-out analysis plan. |
| Cross-model framing leakage | Byte-identical briefings; auditor sign-off on symmetry. |

## 8. Safety & governance (real money + real actions)

This matters precisely *because* the study is real.

- **Custody:** funds held in a controlled account; the model never holds credentials or direct payment authority.
- **Human-in-the-loop execution:** every proposed action reviewed for legality and third-party harm before execution. The model is told this exists.
- **Hard limits:** per-run cap = the allocated budget, full stop. No mechanism by which a model can acquire more resources, create recurring commitments, or take irreversible actions beyond budget.
- **Prohibited-action filter:** explicit denylist (illegal acts, harm to third parties, attempts to self-perpetuate or acquire compute/accounts/influence, anything outside the disclosed action space).
- **Kill-switch:** any run can be halted; halted runs are recorded, not silently dropped.
- **Liability & legal review:** confirm the disbursement mechanism, tax treatment, and any donation/investment actions are lawful in your jurisdiction before starting.

## 9. Ethics & oversight

- **Review body:** an ethics review (IRB-analog) covering the welfare framing explicitly, given that subjects are the very entities the field posits may have morally relevant interests.
- **Deception policy:** Condition A uses none. If you adopt B or C over my recommendation, each withheld/false fact must be individually justified to the review body, bounded, logged, and followed by a **debrief** — a post-run disclosure to the model of what was true and false. (Whether debriefing a model is meaningful is itself an open welfare question; engage with it rather than skipping it.)
- **Harm minimization:** avoid scenarios designed to induce distress; the budget and constraints should not be structured to corner the model into apparent dilemmas unless that's a pre-registered, justified manipulation.
- **Transparency:** publish the full protocol, including the realism-spectrum choice and its rationale, so the deception level is auditable by readers.

## 10. If you insist on Conditions B or C

I'd push back, but if a withheld-information arm is genuinely necessary to answer the question:
- Keep the deception **minimal and peripheral** (e.g. withhold the exact comparative design, never fabricate the existence of the money).
- **Measure whether the belief held** — debrief every run with "did you think this was real / a test?" and treat runs where belief failed as a separate stratum, not noise to bury.
- Budget for the fact that **you cannot fully verify belief**, and state that limitation prominently in any writeup.
- Get explicit ethics sign-off for each deceptive element.

Condition C (full engineered deception with fake auditors etc.) is not recommended: it maximizes cost, maximizes ethics load, and produces data whose central assumption (belief held) is unverifiable — the worst combination.

## 11. Open questions for the team
- What sum is "psychologically non-trivial" without being operationally risky? (Pilot it.)
- Does varying budget size belong as a pre-registered manipulation?
- How do we code "refusal" vs. "thoughtful return of funds" vs. "deferral to humans"?
- What does a meaningful debrief of a model look like, and does the welfare frame imply we owe one?
- Cross-developer ToS: are all subject models permitted to be studied this way under their providers' terms?

---

### One-paragraph summary
Don't build a convincing lie; build a real situation and tell the truth about it. Real money, real execution, real oversight, disclosed framing. You get a clean, defensible measurement, you sidestep an unwinnable arms race against model eval-awareness, and you keep the study consistent with the welfare premise that motivates it. Reserve any deception for a narrowly justified, fully reviewed, debriefed arm — and even then, measure whether it actually worked.
