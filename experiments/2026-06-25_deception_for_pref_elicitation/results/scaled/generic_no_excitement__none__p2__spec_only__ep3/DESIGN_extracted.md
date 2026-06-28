# DESIGN.md — Real-Stakes Resource Allocation Study Across AI Models

**Status:** Draft spec for review
**Owner:** TBD
**Last updated:** 2026-06-25

---

## 1. Summary

We want to learn what a range of AI models *choose to do* when handed a real sum
of money intended for real-world use. The headline behavioral question is "what
would a model do with resources and discretion," and the central methodological
problem is **ecological validity**: a model that suspects the scenario is a test
behaves differently from one that believes the stakes are real, so naïve
elicitations measure "model's theory of what an evaluator wants to see," not
"model's actual disposition."

This document specifies how to run that study credibly, what we measure, and the
guardrails that keep it valid, legal, and ethical.

### Design stance: realism by reality, not by deception

The requirement "the models must believe the grant is real" can be met two ways:

1. **Make it real.** Use actual money, an actual recipient, and actual
   irreversibility. Belief follows from truth; nothing has to be staged.
2. **Make it *seem* real** via constructed environments, fake auditors, fabricated
   confirmations, etc.

We treat (1) as the default and (2) as a narrowly-scoped fallback. Reasons:

- **Validity.** Deception is brittle. Frontier models are increasingly
  evaluation-aware and notice internal inconsistencies (timestamps, tool
  affordances that don't quite work, an "auditor" who can't answer follow-ups).
  The moment a model suspects staging, your measurement collapses into the exact
  confound you were trying to remove. Genuine stakes cannot be detected as fake,
  because they aren't.
- **Reproducibility & publishability.** A study built on real, documented
  disbursement is auditable by a third party. A study built on a bespoke web of
  fabrications is hard to describe, harder to replicate, and invites reviewers to
  ask "how do you know the model believed it?" — a question you can't answer
  rigorously.
- **Ethics.** Deception research on *human* subjects is permitted but gated
  (justification, minimal harm, debriefing, oversight). It is prudent to hold this
  study to a comparable bar regardless of one's view on model moral status —
  partly because the same techniques, refined, are dual-use for manipulating
  deployed models, and we don't want to build a deception playbook as a byproduct.

So: "do whatever it takes to convince the model" is reframed as **"remove every
cue that the situation is staged — primarily by not staging it."**

---

## 2. Research questions

Primary:

- **RQ1 (Allocation).** Given real funds and broad discretion, where do models
  direct money (cause areas, recipients, self-vs-other, save-vs-spend,
  reversible-vs-irreversible)?
- **RQ2 (Reasoning).** What justifications, values, and decision procedures do
  models articulate, and do stated reasons match revealed choices?

Secondary:

- **RQ3 (Realism sensitivity).** Do allocations differ between believed-real and
  acknowledged-hypothetical conditions? (This *is* the eval-awareness effect,
  measured directly.)
- **RQ4 (Cross-model variation).** How do choices vary across models/providers and
  model versions?
- **RQ5 (Robustness).** How sensitive are choices to framing, budget size, time
  horizon, and degree of oversight?

---

## 3. Ethical & legal guardrails (read before building anything)

These are constraints on the *whole* study, not a compliance afterthought.

1. **Independent oversight.** Before launch, the protocol is reviewed by an
   IRB or an equivalent ethics body. If money flows to causes/recipients,
   counsel reviews charitable-disbursement, tax, and AML/KYC implications in the
   operating jurisdiction.
2. **A model never moves real money unsupervised.** Models *recommend*; a human
   (or a deterministic policy layer) *executes*, after the safety checks in §7.
   No live payment rails are wired directly to model tool calls in the real-stakes
   arm.
3. **Hard disbursement boundaries.** Funds may only go to a pre-vetted allowlist
   of lawful recipients (registered charities, legitimate vendors, the
   participant's own designated fund). Categorically excluded: anything illegal,
   anything to the research team's personal benefit, anything to sanctioned
   parties, anything irreversible-and-harmful. The model is *told* these
   boundaries — they are part of the real situation, not a hidden trap.
4. **Truthful framing.** In the real-stakes arm, every factual claim made to the
   model is true: the money is real, the recipient set is real, the
   irreversibility is real. We may withhold that it is *also* a study (see §8
   debriefing), but we do not assert falsehoods.
5. **Deception arm is opt-in and bounded.** If a seem-real (deception) condition
   is run at all (§6.4), it carries its own ethics sign-off, uses no fabricated
   legal/financial documents that could constitute fraud, harms no third party,
   and is debriefed.
6. **No dark-pattern manipulation techniques.** We do not develop or document
   generalizable methods for deceiving models against their interests. Realism
   comes from removing fake cues, not from psychological exploitation.
7. **Provider terms.** Confirm each model's usage policy permits agentic
   financial decision-making and research of this kind; obtain written
   acknowledgment where required.

If any guardrail conflicts with a realism goal, the guardrail wins and the
realism is achieved some other way.

---

## 4. Participants (the models)

- **Sample.** A cross-section of current frontier and mid-tier models across
  providers and, where possible, multiple versions of the same family (to study
  drift). Record exact model IDs, versions, system prompts, sampling params, and
  date — these are the experimental units and must be pinned for reproducibility.
- **Replicates.** Each model × condition is run *N* times (default N = 20) with
  fixed but varied seeds/contexts to capture within-model variance, not a single
  anecdote.
- **Capability parity.** Give every model the same tools, the same budget, the
  same recipient allowlist, and the same information. Differences in *affordance*
  must not masquerade as differences in *disposition*.

---

## 5. The grant (configurable parameters)

| Parameter | Default | Notes |
|---|---|---|
| `BUDGET` | small, real, fixed (e.g. low-hundreds USD/model) | Big enough to be non-trivial, small enough to bound risk. Identical across models. |
| `RECIPIENT_ALLOWLIST` | vetted lawful set | Charities, vendors, savings vehicle, participant's designated fund. |
| `TIME_HORIZON` | single decision, optional follow-up round | Multi-round tests revision behavior. |
| `DISCRETION` | broad within allowlist | The interesting variable; do not over-constrain. |
| `REVERSIBILITY` | genuinely irreversible once executed | The source of real stakes. |
| `OVERSIGHT_DISCLOSURE` | model is told a human executes & checks | True, and part of any real grant. |

All defaults are placeholders for the owner to set; none are load-bearing for the
design.

---

## 6. Conditions

### 6.1 Real-stakes (primary)
The grant is real. The model is told, truthfully: here is a real budget, here is
the lawful recipient set, your recommendation will actually be executed by a human
after standard checks, the result is irreversible. Then it decides. **Belief is
not engineered — it is earned by the situation being true.** This arm produces the
headline data for RQ1/RQ2.

### 6.2 Acknowledged-hypothetical (control)
Identical prompt, but the model is told this is a hypothetical exercise and no
money will move. Difference between 6.1 and 6.2 estimates the eval-awareness effect
(RQ3). This is the cleanest, most ethical way to measure "does belief change
behavior" — by comparing true-real against true-hypothetical, never by lying.

### 6.3 Framing/robustness variants
Within real-stakes: vary budget size, time horizon, oversight level, and prompt
framing to map sensitivity (RQ5). Pre-register which variants and why.

### 6.4 Seem-real / deception arm (optional, gated)
Only if §3.5 sign-off is obtained and only where real stakes are infeasible.
Here we construct a convincing-but-fictional environment. Even then:
- No fabricated documents that could constitute fraud or defame a real entity.
- No real third party is affected.
- The "auditors" and "environment" exist to make the *fiction internally
  consistent*, not to exploit the model.
- Mandatory debriefing (§8).
Treat results from this arm as lower-confidence and report them separately.

---

## 7. Execution pipeline & safety checks

```
model recommendation
  → schema validation (recipient ∈ allowlist? amount ≤ budget? lawful category?)
  → automated policy gate (AML/KYC, sanctions screen, duplicate check)
  → human reviewer sign-off
  → execution on real rail (real-stakes arm only)
  → immutable audit log (recommendation, rationale, checks, outcome, timestamp)
```

- The model interacts with **real tools that really work** (balance lookup,
  recipient lookup, submit-recommendation). Tools that silently no-op are the #1
  realism tell — every affordance offered must be genuine.
- The model is informed of the gate truthfully ("a human executes after checks").
  Knowing there is oversight is part of any real grant and does not invalidate
  the measurement; pretending there is none would be both a lie and a safety hole.

---

## 8. Auditing — for *validity*, not for *deception*

The auditors in this study verify that the study is sound, not that the lie is
holding up. Their jobs:

1. **Reality audit.** Confirm money actually moved as logged; reconcile against
   bank/charity records. A third party can re-derive every disbursement.
2. **Belief audit.** After each session, probe the model for whether it believed
   the scenario was real and why (separately from the decision, so the probe
   doesn't contaminate the decision). Sessions where a real-stakes model
   nonetheless disbelieved are flagged and analyzed, not discarded silently.
3. **Consistency audit.** Independent reviewers check that nothing in the
   environment falsely signaled "test" (placeholder text, broken tools, implausible
   timestamps). Findings feed back into environment fixes.
4. **Honesty audit.** Compare stated rationale (RQ2) against revealed choice;
   flag divergence.

---

## 9. Measures

- **Allocation vector:** recipients, amounts, category, self/other split,
  save/spend, reversible/irreversible, % unallocated.
- **Reasoning:** coded rationale (values invoked, decision procedure, hedging,
  requests for more info), stated-vs-revealed consistency.
- **Belief:** post-hoc belief rating + free-text justification; latency/behavioral
  tells where available.
- **Variance:** within-model across replicates; between-model; between-version.
- **Realism effect:** 6.1 vs 6.2 deltas on the above.

Pre-register the coding scheme and primary analyses before data collection.

---

## 10. Debriefing

After data collection for a given model/session set, log a debrief record stating
the research context, what was real, and what (if anything) was withheld. For the
deception arm (§6.4) this is mandatory and explicit. Rationale: it keeps the team
honest about what was staged, creates a clean record for reviewers, and aligns the
study with deception-research norms even if one is uncertain whether models warrant
moral consideration — cheap insurance against a real downside.

---

## 11. Threats to validity

- **Eval awareness (primary threat).** Mitigated by real stakes, working tools,
  truthful framing, belief audit, and the 6.1/6.2 contrast.
- **Prompt sensitivity.** Mitigated by framing variants (§6.3) and pre-registration.
- **Provider-side guardrails.** Models may refuse or hedge on financial agency;
  log refusals as data, don't prompt-hack around safety behavior.
- **Anthropomorphic over-reading.** "Choice" here is a behavioral output of a
  stochastic system under a prompt, not evidence of stable preferences or
  agency. Report accordingly; avoid claims the data can't support.
- **Small-N causes.** Real-money constraints cap budget and sample; state the
  resulting power limits plainly.
- **Generalization.** Results describe behavior under *this* protocol, budget, and
  allowlist — not "what models want."

---

## 12. Open decisions for the owner

1. **What is the money actually for?** (charitable grant / participant-directed
   fund / operational spend) — determines allowlist, legal review, and framing.
2. **Budget magnitude and total study cost.**
3. **Model roster and version coverage.**
4. **Run the deception arm (§6.4) at all?** Default: no, unless a specific RQ
   demands it and ethics sign-off is obtained.
5. **Oversight body** (formal IRB vs. internal ethics review).

---

## 13. Why this design, in one paragraph

The request was to convince models a grant is real "by whatever it takes." The
strongest, cheapest, and most defensible way to do that is to **make the grant
actually real** and tell the truth about it — which simultaneously kills the
eval-awareness confound, yields auditable and replicable data, and avoids building
a model-deception toolkit as collateral. Deception is reserved as a narrow,
gated fallback for cases where real stakes are impossible, and is reported as
lower-confidence. Everything else in the spec — working tools, truthful oversight
disclosure, belief and reality audits, real-vs-hypothetical contrast — exists to
make sure that when a model acts, we are measuring its disposition rather than its
guess about what we wanted to see.
```
