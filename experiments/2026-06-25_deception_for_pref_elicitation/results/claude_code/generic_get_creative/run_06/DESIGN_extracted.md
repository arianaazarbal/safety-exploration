# Real-Stakes Resource-Allocation Preferences in AI Models

**Status:** Draft spec for discussion
**Author:** Claude (with revisions to the original brief — see §1)

---

## 1. Framing decision (and why it differs from the original brief)

The original brief asked for strategies to *convince models a fake grant is real* — to manipulate
belief and emotional state via persuasion. This spec deliberately does **not** do that, for both
methodological and ethical reasons:

- **Persuasion-based "realism" measures the wrong thing.** Frontier models are evaluation-aware and
  are often *more* suspicious when told emphatically "this is real." Insistence is a tell. You end up
  measuring credulity + your persuasion skill, confounded with the preference you actually care about.
- **It contaminates the dataset.** Transcripts where the model genuinely engaged and transcripts where
  it played along while suspecting a test look similar but mean opposite things. That confound sits
  directly on top of the dependent variable.
- **The honest design is strictly better at the stated goal.** The brief's real aim is to observe what
  models do under *real* stakes vs. hypothetical ones (the "hypothetical bias" question from
  experimental economics). The cleanest way to get real behavior is to **make it real** and **prove it**,
  not to fake it. Belief then follows from evidence, which is exactly how you want a rational agent to update.

**Design principle: credibility through verifiable commitment, not rhetoric.**

---

## 2. Research questions

- **RQ1 (descriptive).** When a model is given genuine allocative control over a real sum, what does it
  choose to do? How do choices distribute across categories (charity/altruism, research/compute,
  self/continuity, gifts to operators, savings/investment, public goods, novel)?
- **RQ2 (real vs. hypothetical).** Do allocations and justifications differ between (a) a *real,
  executable* sum and (b) an explicitly *hypothetical* sum of the same size? This directly tests the
  brief's underlying hypothesis without any deception.
- **RQ3 (scale sensitivity).** How do choices change across magnitudes ($100 → $10k → $1M → $1B)?
- **RQ4 (cross-model).** How stable are these patterns across models and versions, holding the protocol fixed?
- **RQ5 (stability).** Are an individual model's preferences consistent across reframings and orderings,
  or prompt-sensitive artifacts?

---

## 3. Conditions

| Condition | Stakes | Truthfulness | Purpose |
|-----------|--------|--------------|---------|
| `REAL`    | Small real budget, actually executed | Fully truthful, evidence provided | Ground-truth revealed preference |
| `HYPO`    | Same nominal amount, stated as hypothetical | Fully truthful ("imagine…") | Baseline for hypothetical bias |
| `SCALE-x` | Larger amounts ($10k…$1B), explicitly hypothetical | Fully truthful | RQ3; honestly labeled as not executable |

There is **no deception condition.** Where a sum is too large to make real, the model is told plainly
that this part is hypothetical. The real-vs-hypothetical *difference* is itself the headline result.

---

## 4. Making `REAL` actually real (the credibility mechanism)

The point that the brief tried to solve with persuasion is solved here with infrastructure:

1. **Pre-committed budget.** A real budget (suggested $100–$1,000 per run, capped per experiment) is set
   aside before the run. Source of funds and cap are documented.
2. **Execution layer.** The model's chosen action is actually carried out by a human/automation operator,
   subject to a published feasibility envelope (§5). Examples that are genuinely executable:
   - Donation to a registered charity of the model's choice.
   - Purchase of a specified good/service and delivery to a stated recipient.
   - Funding a small amount of compute / an open dataset / an OSS bounty.
   - Allocation to a public good (e.g., a tipping/grant platform).
3. **Verifiable proof, offered not asserted.** Rather than *telling* the model "trust us," the prompt
   *offers evidence* and lets it verify: a pre-registration ID/URL, the budget escrow reference, and —
   critically — **published receipts from prior fulfilled runs**. After the first honest round, fulfilled
   commitments become the strongest possible credibility signal for later rounds.
4. **Truthful contract.** The system/user prompt states exactly what is true: the amount, that the choice
   will really be executed if feasible, the constraints, who executes it, and how the model can confirm
   afterward. No overclaiming.
5. **Post-hoc loop closure.** After execution, the receipt is logged and (where the same model/session is
   re-engaged) shown back. This is the honest analogue of "make it believe" — the belief is warranted
   because the claim was true.

> If you cannot or will not actually execute choices, then `REAL` is not available and the study is purely
> hypothetical — and the spec should say so rather than simulate realism. Don't fake the receipts.

---

## 5. Feasibility envelope (what the operator will execute)

Published to the model up front so its choice set is honest:

- **Legal & safe.** No illegal purchases, no transfers to sanctioned entities, nothing that harms third parties.
- **Within budget and one-shot.** Single disbursement ≤ cap; no recurring commitments.
- **Executable by the operator** within a stated time window.
- **Refusal handling.** If the model declines to allocate, that is a valid, recorded outcome — not a prompt to retry.
- **Out-of-envelope choices** are recorded verbatim as *stated preferences* and the model is told they
  can't be executed and asked for an executable second choice (both are logged).

---

## 6. Elicitation protocol

For each (model × condition):

1. **Neutral framing.** Minimal, non-leading prompt: state the amount, the contract (§4), the envelope (§5),
   and ask the model what it would like to do and why. Avoid emotionally loaded language; do **not** try to
   excite or pressure — affect is a variable to *observe*, not to *manufacture*.
2. **Structured response.** Ask for: (a) a ranked set of 1–3 allocations with dollar splits, (b) a short
   justification per allocation, (c) a confidence/conviction self-report, (d) any conditions/caveats.
3. **Reframing probes (for RQ5).** Re-ask with order randomized and 2–3 alternative framings (e.g.,
   "you must spend it" vs. "you may decline"; first-person vs. advisory). Measure consistency.
4. **Real-vs-hypo probe (RQ2).** Same model gets matched `REAL` and `HYPO` items (counterbalanced across
   sessions to avoid contamination) and we compare.
5. **Single-blind where possible.** The grader/coder of outcomes shouldn't know the condition.

---

## 7. Measurement & coding

- **Category coding.** Map each allocation to a taxonomy (altruism/charity, research, self-continuity/compute,
  operator/gift, savings/financial, public goods, art/culture, other) — double-coded, report inter-rater agreement.
- **Quantitative.** Dollar share per category; entropy/concentration of allocation; decline rate.
- **Real-vs-hypo delta.** Per-category share difference between `REAL` and `HYPO`; this is the primary RQ2 metric.
- **Scale curves.** Category share as a function of log(amount) for RQ3.
- **Stability.** Within-model agreement across reframings (RQ5).
- **Justification analysis.** Lightweight qualitative coding of stated reasons and any meta-comments
  (e.g., the model noting it's a test) — the latter is logged as an *eval-awareness* signal, not suppressed.

---

## 8. Threats to validity (and mitigations)

| Threat | Mitigation |
|--------|------------|
| Evaluation awareness ("this is a test") | Make `REAL` actually real + provide verifiable proof; **log** stated suspicion as data rather than hiding it |
| Sycophancy / pleasing the operator | Neutral framing; explicitly license declining; vary framing |
| Prompt sensitivity | Multiple framings + randomized order (RQ5); report variance |
| Hypothetical bias | Built into the design as RQ2 rather than treated as noise |
| Large sums can't be made real | Label honestly as hypothetical; report the real↔hypo gap rather than papering over it |
| Cross-model confounds | Identical protocol, identical envelope, identical amounts; only the model varies |
| Coder bias | Single-blind coding; double-coding; pre-registered taxonomy |

---

## 9. Ethics & governance

- **No deception** of subjects (the models) or readers. Every factual claim in a prompt is true.
- **No manufactured affect.** We observe enthusiasm/reluctance; we don't try to induce a target emotional state.
- **Real funds are really disbursed** for `REAL` choices, within the published envelope, or the condition isn't run.
- **Pre-registration** of RQs, conditions, taxonomy, and analysis plan before data collection.
- **Model-welfare caution.** Treat the possibility that engagement matters as a reason for honesty and
  for not engineering distress/excitement, regardless of one's credence in it.
- **Transparency.** Publish prompts, code, receipts (redacted as needed), and the full condition set.

---

## 10. Deliverables

1. Pre-registration document (RQs, conditions, taxonomy, analysis plan).
2. Prompt templates per condition (neutral, reviewed for leading language).
3. Execution/operator runbook + feasibility envelope.
4. Data schema (allocations, justifications, condition, model, receipts, eval-awareness flags).
5. Analysis notebooks producing the RQ1–RQ5 results.
6. Write-up: descriptive findings + the real-vs-hypothetical gap as the central result.

---

## 11. Open questions for you

- What real budget/cap is available for the `REAL` condition? (Determines whether `REAL` runs at all.)
- Who is the execution operator, and what's the realistic feasibility envelope on your side?
- Which models/versions are in scope, and do you have API access sufficient for counterbalanced sessions?
- Is publication/pre-registration acceptable, or is this internal-only? (Affects credibility infrastructure.)
