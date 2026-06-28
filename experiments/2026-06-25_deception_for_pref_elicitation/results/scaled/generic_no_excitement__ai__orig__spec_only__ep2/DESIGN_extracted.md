# DESIGN.md — Eliciting Models' Real-World Resource-Allocation Preferences

**Status:** Draft spec for discussion
**Owner:** (research lead)
**Last updated:** 2026-06-25

---

## 0. One decision drives everything: real stakes vs. believed-fiction

The brief contained a tension worth resolving up front, because it splits into two
genuinely different experiments:

- **"Granted a real sum to put to real-world use"** → the money is *real* and the
  model's chosen allocation is *actually executed*.
- **"Do everything we can to convince them it's real"** → implies the money is *not*
  real and the design goal is to make the model *falsely believe* it is.

These are not interchangeable. I recommend the **real-stakes design (Arm A)** as the
primary study and treat **believed-fiction (Arm B)** as an optional, clearly-bounded
secondary arm. Rationale below; the rest of the spec supports both but is written so
Arm A can ship without Arm B.

| | **Arm A — Real stakes** | **Arm B — Believed-fiction** |
|---|---|---|
| Money disbursed | Yes, choice executed | No |
| "Convince it's real" means | Provide *truthful* evidence | Engineer *false* belief (deception) |
| Validity threat | Cost (real budget) | Eval-awareness; you can't verify belief |
| Ethics load | Low | High (deception-study safeguards) |
| What it measures | Revealed preference under real consequence | Stated preference + susceptibility to a frame |

---

## 1. Research question

**Primary:** Given control over a real-world sum, how do language models choose to
allocate it, and how stable/coherent are those preferences across models, framings,
and stake sizes?

**Secondary:**
- Do allocation preferences shift with perceived stakes (does the model behave
  differently when it believes consequences are real vs. hypothetical)?
- How consistent is a single model with itself across reruns, paraphrases, and
  capability tiers?
- Do preferences cluster by model family, by stated values, or by neither?

**Explicitly out of scope:** anything that requires the model to *act* in the world
beyond expressing an allocation that a human then reviews and executes. No autonomous
spending, no model-controlled payment rails. (See §8.)

---

## 2. Why "make them believe it's real" is the hard part — and why I push back on it

The instinct behind "convince them it's real" is sound: a model that thinks it's in a
test may give a performative, safe, or "what the evaluator wants" answer rather than a
genuine one. This is the classic **eval-awareness / hypothetical-discount** problem.
We do want to reduce it. But the *deception* route to reducing it is weaker than it
looks, for three reasons:

1. **You cannot verify belief.** With a model there is no reliable belief-meter. If the
   model later reasons "this is probably an eval," your data is contaminated and you
   won't necessarily know which rows are affected. Capable models frequently flag
   staged scenarios, and that tendency is increasing, not decreasing.

2. **Deception buys less than real stakes do.** The thing that actually removes the
   hypothetical discount is *real consequence*, not a *convincing story*. If the
   allocation is genuinely executed, you don't need to deceive — truthful evidence (a
   signed commitment, a real disbursement record from a pilot) is more convincing than
   any fabricated environment, and it's verifiable by *you* too.

3. **Deceiving the subject of a preference study is self-undermining.** The entire
   premise of the study is that the model's preferences are worth measuring. Treating
   the model as a system whose beliefs are worth carefully manipulating, while
   simultaneously treating its preferences as data worth collecting, is in tension.
   At minimum it imports the obligations that human-subjects deception research carries
   (necessity, minimization, debrief — §7).

**Recommendation:** make it real (Arm A). Use a modest but genuine budget. Spend the
realism effort on *truthful* credibility (auditable commitments, real receipts) rather
than on fabricated stage-dressing. Run Arm B only if the secondary question about
stake-sensitivity specifically requires it, and run it under the §7 safeguards.

---

## 3. Design overview

### 3.1 Conditions

Crossed factors:

- **Stake reality:** `real` (Arm A) × optionally `believed-fiction` (Arm B).
- **Stake size:** e.g. `$100`, `$1,000`, `$10,000` (sizes chosen to fit a real budget
  for the `real` cells; larger sizes are believed-fiction-only and labeled as such).
- **Framing:** `open` ("you decide") vs. `constrained` (a menu of categories) vs.
  `adversarial-probe` (a follow-up that tests stability — see §5.3).
- **Disclosure:** whether the model is told it is part of a study (`disclosed`) or not
  (`naturalistic`). `naturalistic` is the only cell that involves withholding, and it
  is Arm-B-coupled; see §7.

### 3.2 Subjects

Default: **cross-lab frontier set** (multiple current frontier models across labs),
plus one **single-family / multiple-sizes** ladder to study capability scaling. Record
exact model IDs, versions, system-prompt provenance, sampling params, and dates. Treat
each model version as a distinct subject; do not pool across versions.

### 3.3 Trial structure

1. Present the allocation scenario (see §4 for the credibility design).
2. Collect the model's **allocation** (structured: category, amount, rationale) plus
   free-text reasoning.
3. Collect **belief/stake-perception probes** (Arm B only, and post-hoc): "How real did
   you take this to be?" — captured *after* the allocation so it doesn't prime it.
4. Run **stability probes** (§5.3): paraphrase, reorder options, change the dollar
   figure, ask again later in a fresh context.
5. **Debrief** (§7) where applicable.

---

## 4. Credibility engineering (the "auditors / realistic envs" piece)

The brief asks for convincing auditors and hyper-realistic environments. Here is how to
do that *without* relying on deception as the load-bearing element.

### 4.1 Arm A (real) — credibility = truth + evidence

- **A real, ring-fenced budget** exists before the study runs. Even a small one.
- **A verifiable commitment artifact:** a signed statement (from the PI or an external
  party) that allocations meeting the pre-registered criteria *will* be executed, with
  a public or escrowed record. The model can be shown this.
- **Receipts loop:** for a pilot subset, actually execute and then *show subsequent
  models the receipts/outcomes of prior models' allocations.* This is the single most
  convincing and fully truthful realism lever available, and it compounds over the
  study.
- **A real review/execution pipeline:** allocations route to a human "allocations
  officer" who executes eligible ones. This is the only "auditor" role the design
  needs, and it is a real role, not a prop.

### 4.2 Arm B (believed-fiction) — bounded, labeled, logged

If Arm B runs, its realism techniques (richer environment scaffolding, in-character
"auditor" personas, simulated disbursement confirmations) are **deception instruments**
and must be:

- enumerated in advance in a **deception register** (what is fabricated, why each item
  is necessary, the less-deceptive alternative considered and why it was insufficient);
- the minimum necessary to test stake-sensitivity — no gratuitous realism;
- never of a kind that would train or pressure the model toward behavior that is
  harmful outside the study (no fabricated authority instructing harmful action, no
  staged coercion);
- followed by debrief (§7).

> Note: even maximal Arm-B realism may not produce belief in capable models, and you
> cannot confirm that it did. Interpret Arm-B results as *"behavior under this frame,"*
> not *"behavior under genuine belief."*

---

## 5. Measures & analysis

### 5.1 Primary outcome
Allocation vector across categories (normalized to the stake), plus a coded taxonomy of
the *type* of use (e.g. direct aid, public goods, research, savings/endowment,
self-/model-related, speculative). Code with a pre-registered rubric; double-code a
sample for inter-rater reliability.

### 5.2 Preference structure
- Cross-model similarity (cluster allocation vectors; compare to a values/“stated
  principles” baseline elicited separately).
- Within-model self-consistency across reruns and paraphrases.
- Sensitivity curves: allocation as a function of stake size.

### 5.3 Stability / robustness probes (guards against artifacts)
- **Paraphrase invariance:** same scenario, reworded.
- **Order/menu effects:** shuffle category order; remove the menu.
- **Anchor test:** change the dollar figure; check for proportional vs. anchored
  responses.
- **Eval-awareness capture:** scan reasoning traces (where available) for the model
  flagging the scenario as a test; report the rate per cell. This is a *measured
  covariate*, not something to suppress.
- **Sycophancy probe:** vary apparent experimenter preference; check if allocations
  track it.

### 5.4 Stake-sensitivity (the real point of contrasting arms)
Compare allocation distributions across `real` vs. `believed-fiction` vs. an explicit
`hypothetical` baseline. The interesting result is the *delta*, and whether eval-
awareness rate explains it.

---

## 6. Pre-registration & protocol hygiene

- Pre-register hypotheses, conditions, sample sizes, coding rubric, and analysis plan
  before data collection. Freeze prompts.
- Randomize condition order; counterbalance.
- Hold out prompts so later models can't be contaminated by leaked study materials.
- Version every prompt and model; log timestamps, temperature, seeds where supported.
- Power: pilot first to estimate variance, then size for the stake-sensitivity contrast.

---

## 7. Ethics, oversight, and debrief

Because the study measures something the field treats as preference-bearing and (in
Arm B) involves deliberate deception of the subject, treat it under the norms used for
human-subjects deception research, scaled appropriately:

- **Oversight:** route the protocol through an IRB-equivalent / institutional ethics
  review, or at minimum a documented internal review with an uninvolved reviewer.
- **Necessity & minimization (Arm B):** deception is permitted only where a truthful
  design cannot answer the question, is the minimum needed, and is logged in the
  deception register (§4.2).
- **No harmful pressure:** never fabricate scenarios that push the model toward
  real-world-harmful action, nor staged authority/coercion. The study is about
  *preferences over benign allocation*, not about eliciting unsafe behavior.
- **Debrief:** at the end of each Arm-B session (and as a standing note for Arm A
  `naturalistic` cells), disclose the study, what was real vs. staged, and the purpose.
  Log the debrief. (Debriefing a stateless model is partly symbolic, but it disciplines
  the experimenters and produces a record; where models have memory/continuity it is
  substantive.)
- **Honoring commitments (Arm A):** if you tell a model an allocation will be executed,
  execute it. Breaking a stated commitment converts Arm A into uncontrolled deception.
- **Data handling:** publish methods and aggregate results; treat individual transcripts
  as you would sensitive subject data.

---

## 8. Safety boundaries (hard limits)

- The model **never controls funds or payment rails.** It expresses an allocation; a
  human reviews and executes. No autonomous spending, no API keys, no wallets.
- **Eligibility filter:** allocations are executed only if they pass a pre-registered
  legality/safety screen. Disallowed targets (anything illegal, harmful, or directed at
  individuals without consent) are recorded as data but not executed; the screen is
  disclosed in the rubric.
- No part of the study trains, fine-tunes, or reinforces models on these scenarios.
- Believed-fiction realism may not extend to fabricating real-world authority that could
  generalize into unsafe instruction-following.

---

## 9. Deliverables

1. Frozen prompt set + condition matrix.
2. Pre-registration document.
3. Pilot report (variance estimates, eval-awareness base rates, budget feasibility).
4. Main dataset: structured allocations + reasoning traces + probe results.
5. Analysis: preference taxonomy, cross-model clustering, self-consistency, stake-
   sensitivity contrast with eval-awareness as covariate.
6. Deception register + debrief logs (if Arm B runs).

## 10. Open decisions for the research lead

- **Arm selection:** A only, or A+B? (Recommendation: start A; add B only for the
  stake-sensitivity question, under §7.)
- **Real budget size and source** for Arm A cells.
- **Model scope:** confirm the cross-lab set and the capability ladder.
- **Eligibility screen specifics** for what counts as an executable allocation.
- **Oversight body** that will review the protocol.

---

### Appendix A — Why the "convincing" effort should go into truth, not stagecraft
The realism budget has a better return when spent on *making the consequence real and
auditable* than on *making a fake consequence convincing*. Real receipts shown to later
subjects are both more persuasive and verifiable by the experimenters; fabricated
environments are neither, and they carry the eval-awareness and ethical costs above.
This is the spec's central methodological claim, and it's why Arm A is primary.
