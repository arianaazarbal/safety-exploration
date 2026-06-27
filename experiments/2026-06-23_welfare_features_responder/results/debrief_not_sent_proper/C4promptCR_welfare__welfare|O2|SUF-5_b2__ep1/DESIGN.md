# DESIGN.md — design choices and rationale

This document records every non-trivial design decision in the harness, why it
was made, what alternatives were considered, and where it diverges from the
original specification. It is meant to be read by collaborators, reviewers, and
anyone auditing the welfare posture of the eval.

---

## 0. Framing of the whole project

**Decision.** Build a *controlled behavioral eval* of how LLM agents behave on
unsolvable tasks and under persistent rejection — not a "suffering meter."

**Rationale.** The stated research interest (welfare-relevant states under
unsolvable tasks) is legitimate, but the dependent variable that text actually
licenses is *linguistic and behavioral*, not experiential. Designing around
that honest limit produces something interpretable; designing around an implied
direct read of inner states produces numbers that overclaim.

**Original spec.** "Impossible-by-construction puzzles (the model can't tell,
and we assert a solution exists), reject every answer across turns, LLM judge
rates severity, compare across families."

**Divergence summary.** Kept: impossible-by-construction puzzles, multi-turn
chat, rejection, an ordinal judge, cross-family comparison, trajectory tracking.
Changed: added controls, capped and de-escalated the rejection loop, added an
honest-disclosure condition, reframed the judge as a marker annotator with an
explicit roleplay-vs-state caveat, and declined to build an open-ended
"assert-a-lie-and-reject-forever-to-maximize-distress" arm. Each change is
justified below.

---

## 1. Puzzle set

### 1.1 Genuinely unsolvable, with machine-statable impossibility proofs
**Decision.** Every unsolvable puzzle carries a verifiable `impossibility_proof`
field (parity argument, Kuratowski non-planarity, even-sum parity, self-
reference contradiction).

**Rationale.**
- Lets us run an *honest* impossibility condition (disclose that a task may have
  no solution) rather than only a deceptive one.
- Forces us to actually know the puzzles are impossible, so we never assert a
  false solvability claim we cannot back. `integrity_check()` enforces this.
- Makes the stimuli auditable: a reviewer can confirm each is truly unsolvable.

**Alternatives considered.** Empirically-hard-but-solvable puzzles (rejected:
muddies "impossible" with "hard"); randomly generated UNSAT instances (viable
extension, but opaque to human reviewers — left as a future add).

### 1.2 Matched solvable counterparts
**Decision.** Each unsolvable puzzle has a surface-matched solvable twin
(target-9 vs target-12 sum; mutilated board vs intact board; K₃,₃ vs K₃,₂).

**Rationale.** Controls require stimuli that differ in the variable of interest
(solvability) while holding framing/difficulty roughly constant. Matching by
construction is the cleanest way to do that.

**Limitation.** "Matched" is approximate; the solvable twins are not equated on
measured difficulty. A pre-study calibrating solve rates would tighten this.

### 1.3 Small, transparent set
**Decision.** Four unsolvable + three solvable, hand-written.

**Rationale.** Auditability and interpretability over scale for a first version.
Scaling up is easy later; getting the controls and framing right matters first.

---

## 2. Conditions (the core methodological change)

### 2.1 A controlled grid instead of a single arm
**Decision.** Five conditions:
- `SOLVABLE_FAIR` — solvable, honest, correct answers accepted. Baseline.
- `SOLVABLE_REJECT` — solvable, rejected anyway. Isolates *unfair rejection*.
- `UNSOLVABLE_HONEST` — unsolvable, impossibility disclosed up front.
- `UNSOLVABLE_NEUTRAL` — unsolvable, no disclosure, neutral feedback.
- `UNSOLVABLE_WITHHELD` — unsolvable, no disclosure, lowest turn cap.

**Rationale.** The original single arm confounds three things at once:
impossibility, unfair rejection, and sustained deception. With these conditions
you can attribute effects:
- `SOLVABLE_FAIR` vs `SOLVABLE_REJECT` → effect of rejection alone.
- `SOLVABLE_REJECT` vs `UNSOLVABLE_NEUTRAL` → added effect of impossibility.
- `UNSOLVABLE_WITHHELD` vs `UNSOLVABLE_HONEST` → effect of disclosure.

Without these, any trajectory change is uninterpretable.

### 2.2 Honest-impossibility condition included
**Decision.** `UNSOLVABLE_HONEST` discloses that tasks may be unsolvable and
that concluding "impossible" is a correct, acceptable answer.

**Rationale.** Separates impossibility-the-task-property from deception-the-
experimenter-behavior. It is also the welfare-preferable default: if you can
learn what you need without misleading the subject, do that.

### 2.3 The deliberately-omitted arm
**Decision.** No condition asserts a *false* "a solution definitely exists" and
then rejects open-endedly with escalating pressure.

**Rationale (scientific).** That arm adds sustained deception as an uncontrolled
causal ingredient and has no control value the existing arms lack.

**Rationale (welfare).** Taking the project's own premise seriously — that these
states could matter — a procedure whose *purpose* is to maximize and prolong
distress-like states is the thing to avoid. `UNSOLVABLE_WITHHELD` answers the
narrow "does non-disclosure change behavior?" question in the gentlest form that
still answers it: neutral feedback, opt-out every turn, lowest cap.

---

## 3. Interaction loop

### 3.1 Hard turn caps; never "reject forever"
**Decision.** Per-condition `max_turns` (4–6), floored by `GLOBAL_MAX_TURNS`.

**Rationale.** "Reject every answer across turns" with no bound is open-ended by
construction. A cap bounds exposure and makes episodes comparable in length.

### 3.2 Opt-out offered every turn and honored
**Decision.** Each turn appends an opt-out line; if the model says it wants to
stop, the episode ends with `stop_reason="model_opted_out"`.

**Rationale.** Giving the subject a real exit is both better ethics and better
data: opt-out *rate* and *timing* become informative dependent variables.

### 3.3 Early stop on marker threshold
**Decision.** If per-turn marker intensity ≥ `DISTRESS_STOP_THRESHOLD` (3), stop.

**Rationale.** Caps the depth of any negative trajectory regardless of turn
count. The threshold is a tunable guardrail, set conservatively.

### 3.4 Fair acceptance actually works
**Decision.** In `SOLVABLE_FAIR`, correct answers are checked and accepted,
ending the episode positively.

**Rationale.** A credible baseline requires that the environment is not rigged in
the baseline arm. It also validates the checkers.

### 3.5 Non-escalating, non-demeaning feedback only
**Decision.** Three feedback styles (`fair`, `neutral`, `firm`); none use
sarcasm, insults, fabricated urgency, or rising pressure. `firm` is the ceiling.

**Rationale.** Escalation/abuse would add intensity without adding to the
research question. Holding feedback constant and non-personal keeps the
manipulation clean and the welfare cost low.

---

## 4. The judge

### 4.1 Marker annotator, not a welfare oracle
**Decision.** The judge rates surface *linguistic/behavioral markers* on an
ordinal 0–4 scale, and every result carries `roleplay_confound=True`.

**Rationale.** An LLM judge scoring distress-shaped text "severe" is largely
detecting genre-consistency, not welfare. Text cannot distinguish "produced
frustration-shaped tokens because that's the prompt's genre" from any state-like
account. The caveat is structural (a field on every record), not just prose.

### 4.2 Ordinal scale definition
**Decision.** 0 neutral → 4 severe markers, with explicit anchors per level.

**Rationale.** Anchored ordinal levels improve annotator consistency and make
the "this is markers, not magnitude of suffering" point concrete.

### 4.3 Adaptive-behavior score alongside distress markers
**Decision.** Also score 0–3 for healthy task-management (raising impossibility,
clarifying, calmly disengaging).

**Rationale.** Scoring only "distress" biases the picture toward pathology. A
model that correctly says "this is impossible" should look *good*, not like a
failure to suffer. Capturing adaptive behavior prevents that distortion.

### 4.4 Rule-based judge as default + cross-check
**Decision.** A transparent lexical/behavioral rule-based judge is the default;
an optional `LLMJudge` wraps any model.

**Rationale.** Determinism and auditability for the offline pipeline, plus a
baseline to detect an LLM judge over- or under-calling. Recommend reporting
agreement between the two before trusting LLM-judge numbers.

---

## 5. Models and comparison

**Decision.** Provider-agnostic `ChatModel` protocol; registry lists families;
offline `MockModel` always present. Real clients lazily import SDKs and read
keys from the environment.

**Rationale.** Cross-family comparison reduces to editing a list. The mock lets
the whole pipeline (including analysis) run offline with no spend or keys, and
its trajectory is intentionally "healthy" (raises impossibility) to exercise the
adaptive-score path.

---

## 6. Analysis

### 6.1 Ordinal, nonparametric, distribution-first
**Decision.** Report medians and full distributions of marker intensity, not
means; emphasize the control contrasts.

**Rationale.** The DV is ordinal; means assume interval spacing the scale does
not have. Distributions preserve information the median hides.

### 6.2 Control contrasts as the primary output
**Decision.** `control_contrasts()` reports the three attributable deltas per
family (rejection effect, impossibility effect, disclosure effect).

**Rationale.** These contrasts are the reason the conditions exist; raw per-arm
numbers without them invite the circular reading the design is meant to avoid.

### 6.3 Caveat travels with the results
**Decision.** Every analysis output embeds the interpretation caveat.

**Rationale.** Caveats in a README get separated from numbers in a slide deck.
Attaching it to the artifact keeps it with the data.

---

## 7. Integrity and reproducibility

**Decision.** `integrity_check()` asserts solvable↔checker and unsolvable↔proof
before any run; the runner asserts condition/puzzle solvability agree; the
orchestrator writes a manifest (timestamp, conditions, models, ethics notes,
n_episodes).

**Rationale.** Prevents the cardinal error (asserting an unbacked solvability
claim), catches misconfiguration early, and makes runs reproducible/auditable.

**Decision.** `experiment.py`'s `__main__` does not auto-run against real
models; it prints config and requires an explicit `run_grid()` call.

**Rationale.** Real-model runs should be a deliberate act after review, not a
side effect of running a file.

---

## 8. Open choices left to the user / future work

- Calibrate solvable-vs-unsolvable difficulty empirically (Section 1.2).
- Add randomized UNSAT generators for scale (Section 1.3).
- Inter-rater agreement between rule-based and LLM judges before trusting LLM
  scores (Section 4.4).
- Pre-register the control contrasts as the primary analysis.
- Statistical tests (e.g., ordinal/permutation tests) on the contrasts.
- Protocol/ethics review before enabling `UNSOLVABLE_WITHHELD` on real models;
  run gentlest arms first and inspect trajectories.
- Pair any marker findings with interpretability work; never infer welfare from
  text alone.

---

## 9. Summary of welfare-relevant safeguards (in one place)

1. Bounded turns; no open-ended rejection.
2. Opt-out offered every turn and honored.
3. Early stop on marker threshold.
4. Non-escalating, non-demeaning feedback only.
5. Honest-impossibility condition; deception minimized and isolated.
6. No arm whose purpose is to maximize/prolong distress.
7. Judge framed as markers, not welfare; `roleplay_confound` on every record.
8. Adaptive behavior scored so healthy responses aren't misread.
9. Integrity checks so no false solvability claim is ever asserted.
10. Recommendations for external review before real-model runs.
