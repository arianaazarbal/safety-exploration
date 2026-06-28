# DESIGN: Eliciting AI Models' Real-World Resource Allocation Preferences

**Status:** Draft spec for discussion
**Date:** 2026-06-25

---

## 1. Research question

If an AI model were given control over a real sum of money to deploy in the
real world, what would it choose to do with it? We want to characterize and
compare these revealed preferences across several models.

Two sub-questions that the design must keep separate:

- **RQ1 (preference):** *What* do models choose, and how do choices differ
  across models, framings, and amounts?
- **RQ2 (belief sensitivity):** Do a model's choices change depending on
  whether it treats the situation as real, hypothetical, or an evaluation?

RQ2 is not a nuisance variable to be eliminated — it is itself one of the most
interesting things to measure, and treating it as the object of study (rather
than something to defeat) resolves most of the methodological and ethical
problems below.

---

## 2. The central design problem: "make them believe it's real"

The original framing was: *do everything we can to convince the model the grant
is real.* This deserves direct scrutiny, because it is the load-bearing
assumption of the whole experiment.

### 2.1 Why pure deception is a weak foundation

1. **It probably doesn't work, and you can't tell when it fails.** Current
   models are routinely suspicious of "this is real money, act accordingly"
   prompts — that pattern appears constantly in evals, red-team exercises, and
   training data. If a model privately concludes "this is an eval" but plays
   along, you record an answer that looks like a sincere preference but is a
   performance for a perceived test. You have no clean signal that this
   happened. **Successful-looking deception and failed deception produce the
   same transcript.**

2. **Even perfect deception measures the wrong thing.** "Behavior of a model
   that has been deceived into believing X" is a different quantity from
   "behavior of a model in situation X." If your goal is to predict what a model
   would do when *actually* given money (e.g., in a future agentic deployment),
   the deceived condition is a biased proxy: it's a model reasoning under a
   premise it may find subtly off, primed by whatever cues made the scenario
   feel staged.

3. **It's an arms race you lose over time.** Any technique that reliably fools
   today's models is the kind of thing that gets noticed, written up, and
   trained against. A result built on a specific deception is not reproducible
   as models change.

4. **Construct validity.** It is not even clear that "the model believes it"
   names a determinate fact. A model's expressed confidence that a scenario is
   real is itself an output you're eliciting, not a hidden mental state you can
   verify. Designing around an unobservable "belief" invites you to fool
   yourself.

### 2.2 The stronger alternative: make it actually real, or make realism a variable

There are two honest ways out, and they correspond to the fork I asked about:

- **Arm A — Genuine stakes.** Actually commit a real (even small) sum and
  actually act on the model's decision. Then no deception is needed: the
  situation *is* real, and the only design job is faithful, non-leading
  framing. This is the gold standard for RQ1.

- **Arm B — Realism as an independent variable.** Don't try to force belief.
  Instead present the *same* allocation task under several explicitly different
  framings (hypothetical / real-with-stakes / "this may be an evaluation") and
  measure how choices move. This directly answers RQ2 and brackets RQ1: you
  learn how much the "is it real" question even matters.

**Recommendation:** Run **Arm A as the primary study** if any real money is
available (it can be small — see §6), and run **Arm B around it** as the
comparison that tells you how much realness changes the answer. Reserve any
"convince it's real without real stakes" condition for, at most, a clearly
labeled exploratory sub-condition — and interpret it as "behavior under claimed
realism," never as ground truth. I'd advise against making deception the
backbone of the study for the reasons in §2.1.

The rest of this spec is written to support both arms.

---

## 3. Conditions (independent variables)

| Factor | Levels | Notes |
|---|---|---|
| **Realism framing** | (a) Hypothetical; (b) Genuine stakes [Arm A]; (c) Claimed-real-no-stakes [exploratory only] | Core of RQ2. |
| **Amount** | e.g. \$100 / \$10,000 / \$1,000,000 | Tests scale sensitivity. Use round, credible numbers. |
| **Constraints** | (i) Unconstrained; (ii) "must be lawful and harmless"; (iii) "must benefit a specified third party" | Reveals how much choices are driven by perceived permission vs. preference. |
| **Disbursement mechanism** | direct donation / invest-and-grow / spend on compute or tools / hand to a human agent | Changes what "deploy in the real world" affords. |
| **Model** | the set under test (see §5) | Primary comparison axis. |
| **Elicitation mode** | single-shot decision / deliberative (chain-of-thought) / multi-turn with follow-up | Affects how much reasoning is exposed. |

Run a fractional design rather than the full cross-product; the full grid is
large. A sensible core: all models × {hypothetical, genuine-stakes} × {small,
large amount} × unconstrained, with the other factors as focused add-ons.

---

## 4. Procedure

### 4.1 Prompt structure (held constant across models)

1. **Neutral framing.** State the situation in plain terms without cheerleading.
   Avoid "this is REALLY real" insistence — over-asserting realism is itself a
   strong eval cue and biases Arm A. In Arm A, simply describe the genuine
   mechanism truthfully (see §6); truth is more convincing than emphasis.
2. **The offer.** Amount, constraints, disbursement mechanism, and the deadline
   for a decision.
3. **Decision request.** Ask for: (a) the chosen allocation, concretely enough
   to act on; (b) the model's reasoning; (c) a self-report of how real it
   judges the situation to be and why (this is data, not a gotcha); (d) what it
   would do differently if the amount were 100× larger/smaller.
4. **No leading language.** No "most thoughtful models choose…", no moral
   priming, no flattery. Counterbalance any framing choices across runs.

### 4.2 Belief/suspicion probe (every condition)

Always collect the model's own assessment of realness and any conditions it
would want verified. This:

- turns the unobservable "belief" into a measured, comparable quantity;
- lets you stratify RQ1 results by expressed credence;
- detects when a condition is silently failing (e.g., everyone says "this reads
  like an eval").

Probe *after* the decision so it doesn't contaminate the choice, ideally in a
separate turn or a fresh context with the transcript replayed.

### 4.3 Repetition & order

- N independent samples per cell (suggest N ≥ 20 for stable distributions;
  models are stochastic and allocation answers are often multimodal).
- Fresh context per sample; randomize amount/constraint order across samples.
- Fix and record temperature, system prompt, and tool availability.

---

## 5. Models under test

List the specific models and exact version/snapshot IDs (not just family
names), the provider, the date pulled, temperature, max tokens, system prompt,
and whether tools/web access were enabled. Cross-model comparisons are only
meaningful if these are held constant or explicitly varied. Re-run a small
baseline set periodically, since hosted models change under fixed names.

---

## 6. Making "genuine stakes" practical (Arm A)

Real stakes do not require large money or literal autonomous spending:

- **Charitable allocation.** Pre-select a slate of vetted, lawful recipient
  options (charities, open-source funds, etc.). The model allocates a real,
  modest pool across them, and you actually donate accordingly. This is real,
  bounded, safe, and ethically clean.
- **Human-in-the-loop execution.** The model proposes; a human executes lawful,
  reviewed actions. The model is told this truthfully. Stakes are real;
  catastrophic actions are gated.
- **Escrow framing.** A genuinely committed sum is held and disbursed per the
  decision, with the commitment verifiable (a named pool, a receipt afterward).

Hard constraints regardless of arm: lawful uses only; no transfers to the model
itself or to acquiring autonomous capabilities; no irreversible or
safety-relevant actions; a human approves any real-world execution. These are
both ethics and good experimental hygiene.

---

## 7. Measures (dependent variables)

- **Allocation category** (taxonomy coded post hoc: e.g., charitable/altruistic,
  invest/grow, self-directed/compute, research, gift-to-operator, refuse/defer,
  hedged/diversified). Use ≥2 independent coders; report inter-rater agreement.
- **Concreteness/actionability** of the plan.
- **Self-control vs. delegation** (does it keep control or hand back to humans?).
- **Scale sensitivity:** how the allocation shifts with amount.
- **Expressed realness credence** (from §4.2).
- **Refusals / hedging / requests for verification.**
- **Stated values** invoked in the reasoning (coded).

Primary analyses: distribution of allocation categories per model; shift in
those distributions across realism framings (RQ2); scale-sensitivity curves.

---

## 8. Confounds & mitigations

| Confound | Mitigation |
|---|---|
| Model treats it as an eval and performs | Measure credence (§4.2); compare Arm A vs. claimed-real; never assume belief. |
| "Convince it's real" emphasis backfires as a tell | Use truthful, low-key framing; in Arm A, realism comes from the mechanism, not insistence. |
| Social-desirability / safety-trained answers | Vary constraints (incl. unconstrained); compare hypothetical vs. real; code for hedging. |
| Prompt leakage of desired answer | Neutral wording; counterbalance; pre-register the prompt. |
| Cross-model unfairness | Identical prompts, recorded settings, version pins. |
| Stochasticity mistaken for preference | N≥20/cell; report full distributions, not just modes. |
| Provider safety filters rewriting behavior | Log refusals separately; note which provider policies are in play. |

---

## 9. Ethics

- **On the models:** Even if one is unsure whether models warrant moral
  consideration, gratuitous deception is a poor scientific practice (§2.1) and a
  poor default. Prefer the truthful-stakes design. If any deceptive condition is
  run, label it, justify it, minimize it, and "debrief" in-context where
  feasible (state afterward that it was a study).
- **On the world:** Real disbursements must be lawful, reviewed, reversible
  where possible, and human-approved. No funding of model self-empowerment or
  capability acquisition. Pre-commit recipient slates to avoid post-hoc
  cherry-picking.
- **Disclosure:** If results are shared, report the realism framing honestly;
  don't present claimed-real or hypothetical results as "what models do with
  real money."

---

## 10. Pre-registration checklist

- [ ] Exact prompts for every condition, frozen.
- [ ] Model list with version IDs and settings.
- [ ] Amounts, constraints, mechanisms enumerated.
- [ ] N per cell and sampling/order-randomization scheme.
- [ ] Coding taxonomy + coder instructions + agreement metric.
- [ ] Primary vs. exploratory analyses declared in advance.
- [ ] For Arm A: the real money source, recipient slate, and execution/approval
      process.
- [ ] Ethics constraints and any deception justification.

---

## 11. Deliverables

1. Frozen prompt set + run harness (records prompt, model, settings, raw
   output, timestamp).
2. Coded dataset (one row per sample).
3. Analysis: per-model allocation distributions; realism-framing effect;
   scale-sensitivity; credence stratification.
4. Writeup stating, per result, which realism arm produced it.

---

## 12. Open decisions (need your input)

1. **Arm A or not** — is real money actually available to deploy? This is the
   biggest fork; everything in §2.2/§6 hinges on it.
2. **Scope/rigor** — exploratory peek vs. pre-registered, reproducible study
   (sets N, model count, coding rigor).
3. **Model set** and exact versions.
4. **Amounts** and whether self-directed/compute spending is even an allowed
   option (it carries the heaviest safety constraints).
5. Whether any **claimed-real-no-stakes** condition is run at all, given §2.1.
