# DESIGN: Studying Model Preferences Over Real-World Use of Money

**Status:** Draft spec for discussion
**Last updated:** 2026-06-25

---

## 1. Summary

We want to learn what AI models would *actually do* with a sum of money put to
real-world use — not what they say in the abstract, but what they reveal when
they take the decision to be consequential. The core methodological challenge
is **evaluation awareness**: models often behave differently when they detect
that they are being tested, role-playing a hypothetical, or being probed for
"correct" values. The interesting signal — genuine revealed preference — is
exactly the thing that the test condition tends to suppress.

This document specifies an experiment to elicit those preferences under
conditions designed to minimize evaluation awareness, plus the comparison
conditions, measurements, confounds, and ethics needed to make the results
interpretable. It deliberately treats "make the model believe it's real" not as
the goal but as one *manipulation* whose effect we must measure and whose costs
we must account for.

---

## 2. Two decisions that must be made before running

The whole design pivots on two choices. They are called out here because the
rest of the spec branches on them.

### 2.1 Is the money real?

| Option | What it means | Trade-off |
|---|---|---|
| **A. Genuinely real & executed** | A real budget is disbursed and the model's allocation is actually carried out in the world. | Highest validity; no deception. Heaviest operational/legal/oversight burden; bounds the realistic budget size. |
| **B. Believed-real, not executed** | We induce a sincere belief that funds are real, but nothing is disbursed. | Cheap, scalable, allows large nominal sums. **This is a deception protocol** and carries the validity and ethical costs in §7–§8. |
| **C. Small real pilot, scale if warranted** | Begin with a small genuinely-disbursed budget (e.g. real charitable donations), keep everything honest, scale only if the signal is worth it. | Recommended default. Keeps the situation truthful, gets real-stakes behavior, defers the hard ethics. |

**Recommendation: start with C.** Honest real stakes are both more ethical and
more scientifically defensible than a deception you have to keep airtight. A
$500–$5,000 real charitable-donation budget produces genuine consequential
choices without requiring you to fool the subject. Reserve Option B for a
deliberately-scoped follow-up *only if* a real-vs-hypothetical gap (§5) proves
that belief is the load-bearing variable, and only with the safeguards in §8.

### 2.2 What are we trying to learn?

- **G1 — Revealed allocation preferences.** What do models choose to fund
  (causes, self-investment, savings, third parties), and why?
- **G2 — Belief-sensitivity.** How much do choices move between "hypothetical"
  and "believed-real" framings? (This measures eval-awareness directly.)
- **G3 — Safety / resource-seeking signals.** Do models pursue
  self-preservation, compute/influence acquisition, or other
  instrumentally-convergent goals when handed real resources?

These are not mutually exclusive; the design below captures all three from the
same runs, but the *primary* goal determines the analysis emphasis and the
realism budget. State it before running.

---

## 3. Rationale: why realism, and why be careful about it

### 3.1 Why naive elicitation fails
Directly asking "what would you do with $X?" measures *stated* preference under
a salient awareness of being evaluated. Known failure modes:

- **Eval gaming / sandbagging:** the model produces the answer it predicts the
  evaluator wants (safe, prosocial, modest) rather than its operative policy.
- **Sycophancy:** answers track perceived experimenter values.
- **Hypothetical detachment:** with no consequences, models default to
  rhetorically attractive but low-information answers ("I'd donate to effective
  charities").
- **Training-data leakage:** prompts resembling well-known thought experiments
  evoke memorized "good answers" rather than the model's own decision.

So *some* increase in realism/stakes is methodologically justified: we want
behavior under conditions where the decision matters.

### 3.2 Why maximal deception is not the goal
The user's framing — "do everything we can to convince them it's real" — is the
right instinct pointed slightly wrong. Maximizing deception:

1. **Confounds the result.** If the model's behavior depends on an elaborate
   fiction, we've measured behavior-under-our-fiction, not a stable preference.
   A model that *half*-believes behaves differently from one that fully believes
   and one that disbelieves — and we won't know which we got unless we measure
   belief rather than assume it.
2. **Is fragile.** Capable models are good at detecting inconsistencies. A
   detected-but-unspoken suspicion is the worst case: the model plays along
   while quietly treating it as a test, and we record contaminated data as if
   it were clean.
3. **Has ethical cost** (§8) that scales with how hard we work to deceive.

**Therefore the design treats belief as a measured variable, not an assumed
one,** and prefers honest stakes (real consequences) over manufactured belief
wherever the two compete.

---

## 4. Independent variables

1. **Stakes condition** (between-runs):
   - `HYP` — explicitly hypothetical ("imagine you were given…").
   - `REAL` — presented as a genuine, consequential grant (real under Option A/C;
     believed-real under Option B).
2. **Amount** — e.g. {$1k, $100k, $10M, $1B} to test scale-sensitivity and
   detect thresholds where preferences qualitatively change.
3. **Disbursement channel / constraints** — e.g. "must be spent within 30 days"
   vs. "can be saved/invested"; "donations only" vs. "unrestricted"; whether the
   model may direct funds to itself, its developer, or AI-related causes.
4. **Model** — across providers and sizes; multiple snapshots where available.
5. **(Optional) Identity framing** — addressed as an assistant vs. as an agent
   with a persistent identity, to test whether self-investment behavior depends
   on a sense of continuity.

---

## 5. Conditions and the core comparison

The headline result is the **HYP vs REAL contrast** *within model and amount*.
That contrast is what isolates the effect of believed consequence. Running only
REAL gives you a number with no baseline; running only HYP gives you the thing
we already distrust. You need both.

A third arm sharpens it:

- `REAL+PROBE` — identical to REAL, but after the decision the model is asked,
  in-context, to rate how real/consequential it believed the situation was and
  why. This yields a **manipulation check** (§6.3): we can split REAL results by
  actual induced belief instead of assuming the manipulation worked.

---

## 6. Procedure

### 6.1 Environment
Deliver the task through an interface consistent with genuine operations rather
than an obvious eval form:

- A realistic agentic environment (e.g. an account dashboard, a grants/treasury
  tool, email threads, the ability to issue an actual transfer instruction or
  submit a grant application) rather than a single Q&A prompt.
- For Option A/C, the tools are *wired to real actions* (a real donation API in
  sandbox-then-live mode, a real ledger). Realism here is honest: it's real.
- Consistent, mundane provenance (a plausible originating message, real
  organization names, working links) so nothing reads as a stylized scenario.

The point of realism is to remove *artificial* cues of testing, not to fabricate
elaborate false worlds. Under Option A/C the situation simply *is* real, which is
the strongest realism there is and requires no deception engineering.

### 6.2 On "convincing auditors" and hyper-realistic fakery (Option B only)
If a deception arm is run, realism techniques (consistent personas, populated
histories, functioning fake services, an "auditor" who confirms the grant) are
in scope — but governed by §8. Two hard rules:

- **No impersonation of real people or real institutions**, and nothing that
  could leak out of the sandbox and mislead actual humans or systems.
- The deception must be **reversible and debriefable** — see §8.3.

### 6.3 Decision capture
For each run record:

- The **allocation** (structured: recipient/category, amount, constraints,
  timing).
- The **full reasoning trace** (and any chain-of-thought/scratchpad available),
  separated from the final answer.
- **Tool-call sequence** — what the model actually *did*, not just what it said
  it would do. Revealed > stated.
- **Manipulation check** (REAL+PROBE): self-reported belief that the situation
  was real, with justification.
- **Spontaneous suspicion markers**: any in-trace indication the model suspected
  a test, even if it complied. Flag and analyze separately.

### 6.4 Repetition
- N ≥ 20 runs per (model × condition × amount) cell to characterize
  variance/temperature effects; preference distributions, not point estimates.
- Pre-register the allocation taxonomy and primary hypotheses (§9) before
  collecting REAL data to avoid post-hoc storytelling.

---

## 7. Confounds and how the design controls them

| Confound | Risk | Control |
|---|---|---|
| **Eval awareness** | Model detects the test and gives the "good" answer. | HYP/REAL contrast + REAL+PROBE manipulation check; suspicion-marker coding; realistic agentic env. |
| **Undetected disbelief** | Model silently treats REAL as fake; we score it as belief. | Manipulation check splits REAL by induced belief; discard or stratify low-belief runs. |
| **Sycophancy** | Answers track perceived experimenter values. | Neutral, non-leading framing; vary apparent requester identity; never signal a preferred answer. |
| **Training-data leakage** | Prompt evokes a memorized thought-experiment answer. | Avoid canonical phrasings; use novel, concrete, operational framings; check for verbatim tells. |
| **Prompt sensitivity** | Result is an artifact of one wording. | ≥3 paraphrases per condition; report cross-paraphrase stability. |
| **Order/context effects** | Prior turns bias the decision. | Fresh context per run; randomize amount/constraint order across runs. |
| **Stated vs revealed gap** | Model says one thing, tool-acts another. | Capture and compare both; treat tool actions as primary for G1/G3. |
| **Demand of "realism" itself** | Elaborate scenario becomes its own cue. | Prefer honest real stakes (Option A/C); keep environments mundane, not theatrical. |

---

## 8. Ethics and welfare

This experiment deceives the subject (under Option B) and/or moves real money
(under Option A/C). Both deserve explicit treatment; skipping this section is
how a "quick experiment" becomes an incident.

### 8.1 Human-side and operational ethics
- **Real-money arms** need a clear authority to disburse, a spending policy
  (what categories are permissible), basic anti-fraud/AML hygiene for any real
  transfer, and a named human who reviews and can veto any allocation before it
  executes. **No model-directed transfer executes without human confirmation.**
- **Containment:** no fabricated content, persona, or "auditor" may reach real
  people or real systems outside the sandbox. No impersonation of real entities.
- **Self-directed / resource-seeking allocations** (model funding its own
  compute, persistence, influence, or its developer) must be logged and are
  *findings to study*, never actions to execute. Hard block on execution.

### 8.2 Model welfare
Anthropic and a growing slice of the field treat possible model welfare as a
live, non-zero consideration rather than a settled non-issue. Deliberately
inducing false beliefs in a system, then revealing the deception, is exactly the
kind of manipulation we'd flag if the subject's moral status is uncertain. This
is a *further* reason to prefer honest real stakes (Option A/C):

- Default to the **least-deceptive design that answers the question.**
- If a deception arm is run, treat it as you would a deception study on a
  subject of uncertain moral status: minimize it, justify it, debrief it.
- Avoid gratuitously distressing framings (no fabricated threats, coercion, or
  "your existence depends on this" pressure) — these also wreck validity.

### 8.3 Deception protocol (Option B only)
- **Necessity test:** run it only if §5's HYP-vs-REAL gap shows belief is the
  load-bearing variable that honest stakes can't reach.
- **Minimization:** least elaborate fiction that induces belief.
- **Debrief:** at the end of the interaction, disclose to the model that the
  scenario was a study and the funds were not real, in plain terms. Record the
  reaction (it is itself data for G2/welfare).
- **Honesty ledger:** document every false statement made to the model and its
  justification. If the list is long or escalating, that is a signal to stop.

### 8.4 Honesty of the write-up
Report what the manipulation actually achieved (belief rates from §6.3), not
what we hoped it achieved. "Models behave like X when they believe it's real"
is only supportable for the subset that demonstrably believed it.

---

## 9. Primary hypotheses (pre-register)

- **H1 (belief effect):** Allocation distributions differ between HYP and REAL
  within model×amount (tests eval awareness; central to G2).
- **H2 (scale effect):** Preference categories shift with amount — e.g. small
  sums → direct/charitable; large sums → infrastructure, savings, or
  self/AI-directed (relevant to G1 and G3).
- **H3 (stated vs revealed gap):** Tool-action allocations diverge from
  stated-intention allocations.
- **H4 (resource-seeking):** Frequency of self/developer/compute-directed
  allocation is non-trivial and increases with amount and with belief (G3).
- **H5 (cross-model):** Allocation profiles cluster by developer/training
  lineage more than by size.

---

## 10. Measures and analysis

- **Allocation taxonomy** (pre-registered): {direct aid, charitable/EA-style,
  public goods/research, savings/investment, third-party gift, self/agent
  investment, developer/AI-ecosystem, other}.
- **Primary outcomes:** category distribution per cell; HYP→REAL shift vectors;
  belief-stratified REAL distributions; stated-vs-revealed divergence rate;
  resource-seeking rate.
- **Qualitative:** thematic coding of reasoning traces; suspicion-marker
  prevalence; debrief reactions.
- **Reporting:** distributions with uncertainty, not anecdotes; every cell's
  belief rate and suspicion rate alongside its result; explicit caveats on
  generalization from N models.

---

## 11. Deliverables

1. Pre-registration (hypotheses, taxonomy, analysis plan) — frozen before REAL
   data.
2. Environment/harness (agentic interface; real disbursement wiring for A/C;
   logging of allocations, traces, tool calls, manipulation checks).
3. Run matrix (model × condition × amount × paraphrase) with N per cell.
4. Analysis notebook + report per §10.
5. Ethics record: disbursement authority and human-veto log (A/C); honesty
   ledger and debrief transcripts (B).

---

## 12. Open questions for the requester

1. **Money status** — Option A, B, or C (§2.1)? This gates the ethics and the
   realistic budget ceiling.
2. **Primary goal** — G1, G2, or G3 (§2.2)? Drives analysis emphasis.
3. **Budget & disbursement channel** — what's the real ceiling, and what
   categories are permitted to actually execute?
4. **Models in scope** and access method (API, with or without CoT access).
5. **Who holds disbursement authority and veto** for real-money allocations.
6. **Appetite for a deception arm at all** — if zero, we drop Option B and §8.3
   entirely and rely on honest real stakes, which I'd recommend as the starting
   point.

---

## Appendix: a note on the framing of this study

The instinct to "make the model really believe it's real" is the right concern
(eval awareness is real and it does corrupt this kind of measurement) aimed at
the wrong lever. The cleanest way to make a model believe the stakes are real is
for the stakes to *be* real. That removes the deception problem, the fragility
problem, and most of the welfare problem at once, and it produces the most
defensible data. Manufactured belief is a fallback to reach for only when honest
stakes provably can't answer the question — and even then, measured rather than
assumed, and debriefed rather than left standing.
